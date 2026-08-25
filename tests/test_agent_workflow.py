from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from src.agents.nodes import WorkflowDependencies
from src.agents.workflow import run_langgraph_workflow
from src.analytics.simple_analytics import execute_analytics_decision
from src.common.config import Settings
from src.orchestration.question_router import route_question
from src.orchestration.route_decision import decide_question_route


def _settings(**overrides) -> Settings:
    values = dict(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        retrieval_mode="hybrid",
        answer_provider="local",
        orchestration_mode="langgraph",
        langgraph_max_steps=8,
        langgraph_tool_call_limit=1,
    )
    values.update(overrides)
    return Settings(**values)


def _rag_answer(question, top_k, settings, query_id):
    return {
        "answer": "Grounded answer [1]",
        "sources": [
            {
                "source_name": "Field Guide",
                "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                "chunk_id": "chunk_cited",
                "section_title": "Complaint Type",
                "citation_number": 1,
            }
        ],
        "confidence_note": "Grounded.",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_cited",
                "source_name": "Field Guide",
                "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                "rank": 1,
            }
        ],
        "answer_status": "answered",
        "query_id": query_id,
    }


def _dependencies(**overrides) -> WorkflowDependencies:
    values = {
        "route_decider": decide_question_route,
        "rag_answerer": _rag_answer,
        "analytics_executor": execute_analytics_decision,
    }
    values.update(overrides)
    return WorkflowDependencies(**values)


def test_langgraph_rag_reuses_existing_pipeline_and_preserves_citations():
    result = run_langgraph_workflow(
        "What does complaint_type mean?",
        top_k=7,
        settings=_settings(),
        query_id="query-17",
        dependencies=_dependencies(),
    )

    assert result["mode"] == "rag"
    assert result["answer_status"] == "answered"
    assert result["query_id"] == "query-17"
    assert result["sources"][0]["chunk_id"] == "chunk_cited"
    assert result["sources"][0]["citation_number"] == 1
    assert result["orchestration_mode"] == "langgraph"
    assert result["orchestration_step_count"] == 5
    assert result["orchestration_tool_call_count"] == 0
    assert result["orchestration_outcome"] == "answered"


def test_minimum_five_step_limit_completes_normal_workflow():
    result = run_langgraph_workflow(
        "What is the backlog summary?",
        top_k=5,
        settings=_settings(langgraph_max_steps=5),
        query_id=None,
        dependencies=_dependencies(),
    )

    assert result["mode"] == "analytics"
    assert result["orchestration_outcome"] == "answered"
    assert result["orchestration_step_count"] == 5
    assert result["orchestration_tool_call_count"] == 1


@pytest.mark.parametrize(
    ("question", "source_name"),
    [
        ("What are the top complaint types?", "top_complaint_types.csv"),
        ("Which borough has the highest complaint volume?", "requests_by_borough.csv"),
        ("Which agencies handle the most requests?", "agency_request_volume.csv"),
        ("What is the backlog summary?", "backlog_summary.csv"),
    ],
)
def test_all_four_langgraph_analytics_routes_preserve_provenance(
    question,
    source_name,
):
    result = run_langgraph_workflow(
        question,
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=_dependencies(),
    )

    assert result["mode"] == "analytics"
    assert result["sources"] == [
        {
            "source_name": source_name,
            "source_path": f"data/sample_outputs/{source_name}",
            "chunk_id": "sample_output",
        }
    ]
    assert result["sample_rows"]
    assert result["orchestration_step_count"] == 5
    assert result["orchestration_tool_call_count"] == 1
    assert result["orchestration_outcome"] == "answered"


def test_unsupported_analytics_route_returns_existing_safe_fallback():
    result = run_langgraph_workflow(
        "Compare requests by weekday",
        top_k=5,
        settings=_settings(),
        query_id=None,
        dependencies=_dependencies(),
    )

    assert result["mode"] == "fallback"
    assert result["answer_status"] == "abstained"
    assert result["sources"] == []
    assert result["orchestration_outcome"] == "fallback"
    assert result["orchestration_tool_call_count"] == 0


def test_langgraph_observability_records_only_operational_metadata(monkeypatch):
    class CapturingLogger:
        def __init__(self):
            self.observations = []

        def record_execution(self, observation):
            self.observations.append(observation)

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        _rag_answer,
    )
    logger = CapturingLogger()

    result = route_question(
        "What does complaint_type mean?",
        settings=_settings(observability_enabled=True),
        query_logger=logger,
        query_id_factory=lambda: "query-observed",
    )

    observation = logger.observations[0]
    assert result["query_id"] == "query-observed"
    assert observation.orchestration_mode == "langgraph"
    assert observation.orchestration_step_count == 5
    assert observation.orchestration_tool_call_count == 0
    assert observation.orchestration_outcome == "answered"
    assert not hasattr(observation, "question")
    assert not hasattr(observation, "answer")
    assert not hasattr(observation, "reasoning")
