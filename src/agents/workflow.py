"""Optional lazy LangGraph boundary for bounded CivicLens orchestration."""

from __future__ import annotations

from typing import Any

from src.agents.nodes import (
    WorkflowDependencies,
    WorkflowExecutionError,
    analytics_execution_node,
    fallback_node,
    final_validation_node,
    input_validation_node,
    rag_execution_node,
    response_generation_node,
    route_after_decision,
    routing_node,
    safe_workflow_result,
)
from src.agents.state import WorkflowState
from src.common.config import Settings
from src.orchestration.route_decision import QuestionRoute


class LangGraphUnavailableError(RuntimeError):
    """Raised internally when opt-in orchestration lacks its dependency."""


def _load_langgraph():
    try:
        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise LangGraphUnavailableError(
            "LangGraph orchestration requires the optional dependency."
        ) from exc
    return StateGraph, START, END, GraphRecursionError


def build_compiled_workflow(dependencies: WorkflowDependencies | None = None):
    """Build one acyclic graph with no checkpointer, memory, planner, or loop."""

    StateGraph, START, END, _ = _load_langgraph()
    active_dependencies = dependencies or WorkflowDependencies()
    builder = StateGraph(WorkflowState)
    builder.add_node("validate_input", input_validation_node)
    builder.add_node(
        "decide_route",
        lambda state: routing_node(state, active_dependencies),
    )
    builder.add_node(
        "execute_rag",
        lambda state: rag_execution_node(state, active_dependencies),
    )
    builder.add_node(
        "execute_tool",
        lambda state: analytics_execution_node(state, active_dependencies),
    )
    builder.add_node("safe_fallback", fallback_node)
    builder.add_node("validate_result", final_validation_node)
    builder.add_node("generate_response", response_generation_node)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "decide_route")
    builder.add_conditional_edges(
        "decide_route",
        route_after_decision,
        {
            "execute_rag": "execute_rag",
            "execute_tool": "execute_tool",
            "safe_fallback": "safe_fallback",
        },
    )
    for execution_node in ("execute_rag", "execute_tool", "safe_fallback"):
        builder.add_edge(execution_node, "validate_result")
    builder.add_edge("validate_result", "generate_response")
    builder.add_edge("generate_response", END)
    return builder.compile(name="civiclens_bounded_workflow")


def run_langgraph_workflow(
    question: str,
    *,
    top_k: int,
    settings: Settings,
    query_id: str | None,
    dependencies: WorkflowDependencies | None = None,
    compiled_workflow: Any | None = None,
) -> dict[str, Any]:
    """Execute the bounded graph and convert all graph failures to abstention."""

    try:
        _, _, _, GraphRecursionError = _load_langgraph()
        graph = compiled_workflow or build_compiled_workflow(dependencies)
        initial_state: WorkflowState = {
            "question": question,
            "top_k": top_k,
            "settings": settings,
            "query_id": query_id,
            "route": None,
            "tool_id": None,
            "tool_arguments": {},
            "result": None,
            "answer": None,
            "provenance": [],
            "step_count": 0,
            "tool_call_count": 0,
            "max_steps": settings.langgraph_max_steps,
            "tool_call_limit": settings.langgraph_tool_call_limit,
            "outcome": "pending",
        }
        final_state = graph.invoke(
            initial_state,
            # LangGraph counts the final halt superstep in addition to node steps.
            {"recursion_limit": settings.langgraph_max_steps + 1},
        )
        result = final_state.get("result")
        if not isinstance(result, dict):
            raise WorkflowExecutionError(
                "missing_final_result",
                int(final_state.get("step_count") or 0),
                int(final_state.get("tool_call_count") or 0),
            )
        return result
    except LangGraphUnavailableError:
        return safe_workflow_result(
            outcome="unavailable",
            step_count=0,
            tool_call_count=0,
        )
    except WorkflowExecutionError as exc:
        return safe_workflow_result(
            outcome="limit_exceeded" if "limit" in exc.code else "failed",
            step_count=exc.step_count,
            tool_call_count=exc.tool_call_count,
        )
    except GraphRecursionError:
        return safe_workflow_result(
            outcome="limit_exceeded",
            step_count=settings.langgraph_max_steps,
            tool_call_count=0,
        )
    except Exception:
        return safe_workflow_result(
            outcome="failed",
            step_count=0,
            tool_call_count=0,
            route=QuestionRoute.RAG,
        )
