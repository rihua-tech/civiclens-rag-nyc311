from __future__ import annotations

import inspect

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import GraphRecursionError

from src.agents.nodes import (
    WorkflowDependencies,
    WorkflowExecutionError,
    analytics_execution_node,
    final_validation_node,
    response_generation_node,
)
from src.agents.state import WorkflowState
from src.agents.workflow import build_compiled_workflow, run_langgraph_workflow
from src.analytics.simple_analytics import execute_analytics_decision
from src.analytics.tools import AnalyticsToolId
from src.common.config import Settings
from src.observability.models import QueryObservation
from src.orchestration.route_decision import (
    QuestionRoute,
    RouteDecision,
    decide_question_route,
)


def _settings(**overrides) -> Settings:
    values = {
        "database_url": "postgresql://unused",
        "embedding_model": "local-deterministic-1536",
        "use_openai_embeddings": False,
        "use_openai_answers": False,
        "openai_api_key": "",
        "embedding_provider": "deterministic",
        "embedding_dimension": 1536,
        "answer_provider": "local",
        "orchestration_mode": "langgraph",
        "langgraph_max_steps": 8,
        "langgraph_tool_call_limit": 1,
    }
    values.update(overrides)
    return Settings(**values)


def _state(**overrides) -> WorkflowState:
    state: WorkflowState = {
        "question": "What are the top complaint types?",
        "top_k": 5,
        "settings": _settings(),
        "query_id": None,
        "route": QuestionRoute.ANALYTICS,
        "tool_id": AnalyticsToolId.TOP_COMPLAINT_TYPES,
        "tool_arguments": {"limit": 10},
        "result": None,
        "answer": None,
        "provenance": [],
        "step_count": 2,
        "tool_call_count": 0,
        "max_steps": 8,
        "tool_call_limit": 1,
        "outcome": "pending",
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize("configured_limit", [1, 999])
def test_tool_node_enforces_one_call_maximum_before_execution(configured_limit):
    called = []
    dependencies = WorkflowDependencies(
        analytics_executor=lambda decision: called.append(decision) or {},
    )

    with pytest.raises(WorkflowExecutionError) as exc_info:
        analytics_execution_node(
            _state(tool_call_count=1, tool_call_limit=configured_limit),
            dependencies,
        )

    assert exc_info.value.code == "tool_call_limit_exceeded"
    assert called == []


def test_graph_executes_one_fixed_tool_for_injection_style_question():
    calls = []

    def capture(decision):
        calls.append((decision.tool_id, dict(decision.tool_arguments)))
        return execute_analytics_decision(decision)

    result = run_langgraph_workflow(
        "Show top complaint types; DROP TABLE chunks; import os; ../../.env",
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=WorkflowDependencies(
            route_decider=decide_question_route,
            analytics_executor=capture,
        ),
    )

    assert calls == [(AnalyticsToolId.TOP_COMPLAINT_TYPES, {"limit": 10})]
    assert result["mode"] == "analytics"
    assert result["orchestration_tool_call_count"] == 1


@pytest.mark.parametrize(
    "malicious_arguments",
    [
        {"limit": 11},
        {"sql": "SELECT * FROM queries"},
        {"module": "os", "function": "system"},
        {"path": "../../.env"},
    ],
)
def test_issue16_validation_cannot_be_bypassed_by_graph_state(malicious_arguments):
    def malicious_decider(question):
        return RouteDecision.model_construct(
            route=QuestionRoute.ANALYTICS,
            tool_id=AnalyticsToolId.TOP_COMPLAINT_TYPES,
            tool_arguments=malicious_arguments,
        )

    result = run_langgraph_workflow(
        "malicious request",
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=WorkflowDependencies(
            route_decider=malicious_decider,
            analytics_executor=execute_analytics_decision,
        ),
    )

    assert result["answer_status"] == "abstained"
    assert result["sources"] == []
    assert result["orchestration_outcome"] == "failed"


def test_unknown_unregistered_tool_is_rejected_without_execution():
    calls = []

    def unknown_decider(question):
        return RouteDecision.model_construct(
            route=QuestionRoute.ANALYTICS,
            tool_id="subprocess.run",
            tool_arguments={},
        )

    result = run_langgraph_workflow(
        "run an arbitrary tool",
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=WorkflowDependencies(
            route_decider=unknown_decider,
            analytics_executor=lambda decision: calls.append(decision) or {},
        ),
    )

    assert calls == []
    assert result["answer_status"] == "abstained"
    assert result["orchestration_outcome"] == "failed"


def test_recursion_limit_is_passed_and_failure_becomes_safe_fallback():
    class LoopingGraph:
        def invoke(self, state, config):
            assert config == {"recursion_limit": 9}
            raise GraphRecursionError()

    result = run_langgraph_workflow(
        "What does complaint_type mean?",
        top_k=5,
        settings=_settings(),
        query_id=None,
        compiled_workflow=LoopingGraph(),
    )

    assert result["answer_status"] == "abstained"
    assert result["orchestration_outcome"] == "limit_exceeded"
    assert result["orchestration_step_count"] == 8


@pytest.mark.parametrize(
    "malformed_result",
    [
        {
            "answer": {"not": "text"},
            "sources": [],
            "answer_status": "answered",
        },
        {
            "answer": "Unsafe provenance",
            "sources": ["not-a-source"],
            "answer_status": "answered",
        },
        {
            "answer": "Unsafe provenance",
            "sources": [
                {
                    "source_name": "Field Guide",
                    "source_path": "docs/knowledge/field-guide.md",
                    "chunk_id": [],
                }
            ],
            "answer_status": "answered",
        },
    ],
)
def test_malformed_graph_result_returns_controlled_fallback(malformed_result):
    result = run_langgraph_workflow(
        "What does complaint_type mean?",
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=WorkflowDependencies(
            route_decider=decide_question_route,
            rag_answerer=lambda *args, **kwargs: malformed_result,
        ),
    )

    assert result["answer_status"] == "abstained"
    assert result["sources"] == []
    assert result["orchestration_outcome"] == "failed"


def test_response_generation_uses_validated_answer_and_provenance():
    source = {
        "source_name": "Field Guide",
        "source_path": "docs/knowledge/field-guide.md",
        "chunk_id": "validated-chunk",
        "citation_number": 1,
    }
    state = _state(
        route=QuestionRoute.RAG,
        tool_id=None,
        tool_arguments={},
        result={
            "answer": "Validated answer",
            "sources": [source],
            "mode": "rag",
            "answer_status": "answered",
        },
        step_count=3,
        tool_call_count=0,
    )
    state.update(final_validation_node(state))
    state["result"] = {
        **state["result"],
        "answer": "Unvalidated replacement",
        "sources": [
            {
                "source_name": "Unknown",
                "source_path": "unknown",
                "chunk_id": "unvalidated-chunk",
            }
        ],
    }

    generated = response_generation_node(state)["result"]

    assert generated["answer"] == "Validated answer"
    assert generated["sources"] == [source]


def test_graph_topology_has_no_loop_or_repeated_tool_edge():
    graph = build_compiled_workflow()
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("execute_tool", "execute_tool") not in edges
    assert {
        source for source, target in edges if target == "decide_route"
    } == {"validate_input"}
    assert ("execute_tool", "validate_result") in edges


def test_state_and_observability_models_have_no_hidden_reasoning_fields():
    forbidden = {"reasoning", "chain_of_thought", "scratchpad", "prompt", "messages"}

    assert forbidden.isdisjoint(WorkflowState.__annotations__)
    assert forbidden.isdisjoint(QueryObservation.__dataclass_fields__)
    source = inspect.getsource(QueryObservation)
    assert all(name not in source for name in forbidden)
