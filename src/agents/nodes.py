"""Bounded graph nodes that reuse existing CivicLens capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.analytics.simple_analytics import (
    execute_analytics_decision,
    fallback_response,
)
from src.common.config import LANGGRAPH_TOOL_CALL_LIMIT
from src.generation.answer_question import answer_question
from src.generation.schemas import AnswerStatus, NO_ANSWER
from src.orchestration.route_decision import (
    QuestionRoute,
    RouteDecision,
    decide_question_route,
    normalize_question,
)
from src.agents.state import WorkflowState


GRAPH_FALLBACK_MESSAGE = (
    "I could not safely complete the bounded workflow with the available route."
)
GRAPH_FALLBACK_CONFIDENCE = (
    "No answer was produced because bounded orchestration validation failed."
)


class WorkflowExecutionError(RuntimeError):
    """Controlled internal error converted to a safe workflow fallback."""

    def __init__(self, code: str, step_count: int, tool_call_count: int) -> None:
        super().__init__(code)
        self.code = code
        self.step_count = step_count
        self.tool_call_count = tool_call_count


RouteDecider = Callable[[str], RouteDecision]
RagAnswerer = Callable[..., dict[str, Any]]
AnalyticsExecutor = Callable[[RouteDecision], dict[str, Any]]


@dataclass(frozen=True)
class WorkflowDependencies:
    route_decider: RouteDecider = decide_question_route
    rag_answerer: RagAnswerer = answer_question
    analytics_executor: AnalyticsExecutor = execute_analytics_decision


def _next_step(state: WorkflowState) -> int:
    next_step = int(state["step_count"]) + 1
    if next_step > int(state["max_steps"]):
        raise WorkflowExecutionError(
            "step_limit_exceeded",
            next_step,
            int(state["tool_call_count"]),
        )
    return next_step


def safe_workflow_result(
    *,
    outcome: str,
    step_count: int,
    tool_call_count: int,
    route: QuestionRoute | None = None,
) -> dict[str, Any]:
    """Return a provider-neutral abstention without internal exception details."""

    mode = "analytics" if route is QuestionRoute.ANALYTICS else "rag"
    return {
        "answer": GRAPH_FALLBACK_MESSAGE,
        "sources": [],
        "confidence_note": GRAPH_FALLBACK_CONFIDENCE,
        "retrieved_chunks": [],
        "sample_rows": [],
        "mode": mode,
        "answer_status": AnswerStatus.ABSTAINED.value,
        "orchestration_mode": "langgraph",
        "orchestration_step_count": max(0, int(step_count)),
        "orchestration_tool_call_count": max(0, int(tool_call_count)),
        "orchestration_outcome": outcome,
    }


def input_validation_node(state: WorkflowState) -> dict[str, Any]:
    step_count = _next_step(state)
    normalized = normalize_question(state["question"])
    return {"question": normalized, "step_count": step_count}


def routing_node(
    state: WorkflowState,
    dependencies: WorkflowDependencies,
) -> dict[str, Any]:
    step_count = _next_step(state)
    decision = dependencies.route_decider(state["question"])
    return {
        "route": decision.route,
        "tool_id": decision.tool_id,
        "tool_arguments": dict(decision.tool_arguments),
        "step_count": step_count,
    }


def route_after_decision(state: WorkflowState) -> str:
    route = state.get("route")
    if route is QuestionRoute.RAG:
        return "execute_rag"
    if route is QuestionRoute.ANALYTICS:
        return "execute_tool"
    return "safe_fallback"


def rag_execution_node(
    state: WorkflowState,
    dependencies: WorkflowDependencies,
) -> dict[str, Any]:
    step_count = _next_step(state)
    response = dependencies.rag_answerer(
        state["question"],
        top_k=state["top_k"],
        settings=state["settings"],
        query_id=state["query_id"],
    )
    response = dict(response)
    response["mode"] = "rag"
    response.setdefault("sample_rows", [])
    return {"result": response, "step_count": step_count}


def analytics_execution_node(
    state: WorkflowState,
    dependencies: WorkflowDependencies,
) -> dict[str, Any]:
    step_count = _next_step(state)
    tool_call_count = int(state["tool_call_count"]) + 1
    if (
        tool_call_count > int(state["tool_call_limit"])
        or tool_call_count > LANGGRAPH_TOOL_CALL_LIMIT
    ):
        raise WorkflowExecutionError(
            "tool_call_limit_exceeded",
            step_count,
            tool_call_count,
        )
    decision = RouteDecision(
        route=QuestionRoute.ANALYTICS,
        tool_id=state.get("tool_id"),
        tool_arguments=state.get("tool_arguments") or {},
    )
    response = dependencies.analytics_executor(decision)
    return {
        "result": dict(response),
        "step_count": step_count,
        "tool_call_count": tool_call_count,
    }


def fallback_node(state: WorkflowState) -> dict[str, Any]:
    step_count = _next_step(state)
    response = fallback_response()
    response["answer_status"] = AnswerStatus.ABSTAINED.value
    return {
        "result": response,
        "step_count": step_count,
        "outcome": "fallback",
    }


def final_validation_node(state: WorkflowState) -> dict[str, Any]:
    step_count = _next_step(state)
    result = state.get("result")
    if not isinstance(result, dict):
        raise WorkflowExecutionError(
            "missing_result",
            step_count,
            int(state["tool_call_count"]),
        )
    if result.get("mode") not in {"rag", "analytics", "fallback"}:
        raise WorkflowExecutionError(
            "invalid_result_route",
            step_count,
            int(state["tool_call_count"]),
        )
    sources = result.get("sources")
    if not isinstance(sources, list) or any(
        not isinstance(source, dict) for source in sources
    ):
        raise WorkflowExecutionError(
            "invalid_result_provenance",
            step_count,
            int(state["tool_call_count"]),
        )
    for source in sources:
        required_values = (
            source.get("source_name"),
            source.get("source_path"),
            source.get("chunk_id"),
        )
        if any(not isinstance(value, str) or not value for value in required_values):
            raise WorkflowExecutionError(
                "invalid_result_provenance",
                step_count,
                int(state["tool_call_count"]),
            )
        if source.get("section_title") is not None and not isinstance(
            source["section_title"], str
        ):
            raise WorkflowExecutionError(
                "invalid_result_provenance",
                step_count,
                int(state["tool_call_count"]),
            )
        citation_number = source.get("citation_number")
        if citation_number is not None and (
            not isinstance(citation_number, int)
            or isinstance(citation_number, bool)
            or citation_number < 1
        ):
            raise WorkflowExecutionError(
                "invalid_result_provenance",
                step_count,
                int(state["tool_call_count"]),
            )
    raw_answer = result.get("answer")
    if raw_answer is not None and not isinstance(raw_answer, str):
        raise WorkflowExecutionError(
            "invalid_result_answer",
            step_count,
            int(state["tool_call_count"]),
        )
    answer = raw_answer or NO_ANSWER
    raw_status = result.get("answer_status")
    if raw_status is not None and raw_status not in {
        AnswerStatus.ANSWERED.value,
        AnswerStatus.ABSTAINED.value,
    }:
        raise WorkflowExecutionError(
            "invalid_result_status",
            step_count,
            int(state["tool_call_count"]),
        )
    outcome = state.get("outcome")
    if outcome != "fallback":
        outcome = (
            "abstained"
            if raw_status == AnswerStatus.ABSTAINED.value
            else "answered"
        )
    return {
        "answer": answer,
        "provenance": list(sources),
        "step_count": step_count,
        "outcome": outcome,
    }


def response_generation_node(state: WorkflowState) -> dict[str, Any]:
    step_count = _next_step(state)
    result = dict(state["result"] or {})
    result.update(
        {
            "answer": state["answer"] or NO_ANSWER,
            "sources": [dict(source) for source in state["provenance"]],
            "orchestration_mode": "langgraph",
            "orchestration_step_count": step_count,
            "orchestration_tool_call_count": int(state["tool_call_count"]),
            "orchestration_outcome": state["outcome"],
        }
    )
    return {"result": result, "step_count": step_count}
