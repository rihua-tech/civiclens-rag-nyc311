from __future__ import annotations

import subprocess
import sys

import pytest

from src.analytics.tools import AnalyticsToolId
from src.common.config import Settings
from src.orchestration.question_router import route_question
from src.orchestration.route_decision import (
    QuestionRoute,
    decide_question_route,
)


def _settings(mode: str = "direct") -> Settings:
    return Settings(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        retrieval_mode="hybrid",
        answer_provider="local",
        orchestration_mode=mode,
    )


@pytest.mark.parametrize(
    ("question", "route", "tool_id"),
    [
        (
            "What are the top complaint types?",
            QuestionRoute.ANALYTICS,
            AnalyticsToolId.TOP_COMPLAINT_TYPES,
        ),
        (
            "Which borough has the highest complaint volume?",
            QuestionRoute.ANALYTICS,
            AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
        ),
        (
            "Which agencies handle the most requests?",
            QuestionRoute.ANALYTICS,
            AnalyticsToolId.AGENCY_REQUEST_VOLUME,
        ),
        (
            "What is the backlog summary?",
            QuestionRoute.ANALYTICS,
            AnalyticsToolId.BACKLOG_SUMMARY,
        ),
        ("Compare requests by weekday", QuestionRoute.UNSUPPORTED_ANALYTICS, None),
        ("What does complaint_type mean?", QuestionRoute.RAG, None),
    ],
)
def test_shared_deterministic_route_decision(question, route, tool_id):
    decision = decide_question_route(question)

    assert decision.route is route
    assert decision.tool_id is tool_id


def test_direct_mode_is_default_and_never_loads_langgraph(monkeypatch):
    def unexpected_graph(*args, **kwargs):
        raise AssertionError("Direct mode must not load or invoke LangGraph")

    monkeypatch.setattr(
        "src.orchestration.question_router.run_langgraph_workflow",
        unexpected_graph,
    )

    result = route_question(
        "What are the top complaint types?",
        settings=_settings(),
    )

    assert result["mode"] == "analytics"
    assert result["orchestration_mode"] == "direct"
    assert result["orchestration_tool_call_count"] == 1


def test_legacy_analytics_detection_exports_use_shared_rules():
    from src.analytics.simple_analytics import (
        looks_like_analytics_question,
        looks_like_field_definition_question,
    )

    assert looks_like_analytics_question("What is the backlog summary?") is True
    assert looks_like_field_definition_question("What does complaint_type mean?") is True


def test_direct_mode_imports_and_runs_when_langgraph_is_blocked():
    script = r'''
import sys

class BlockLangGraph:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langgraph" or fullname.startswith("langgraph."):
            raise ImportError("blocked optional dependency")
        return None

sys.meta_path.insert(0, BlockLangGraph())
from src.common.config import Settings
from src.orchestration.question_router import route_question

settings = Settings(
    database_url="postgresql://unused",
    embedding_model="local-deterministic-1536",
    use_openai_embeddings=False,
    use_openai_answers=False,
    openai_api_key="",
    embedding_provider="deterministic",
    embedding_dimension=1536,
    orchestration_mode="direct",
)
result = route_question("What is the backlog summary?", settings=settings)
assert result["mode"] == "analytics"
assert result["orchestration_mode"] == "direct"
print("direct-without-langgraph-ok")
'''

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "direct-without-langgraph-ok"


def test_opt_in_mode_without_langgraph_returns_controlled_fallback(monkeypatch):
    from src.agents import workflow

    def unavailable():
        raise workflow.LangGraphUnavailableError("optional dependency unavailable")

    monkeypatch.setattr(workflow, "_load_langgraph", unavailable)

    result = workflow.run_langgraph_workflow(
        "What does complaint_type mean?",
        top_k=5,
        settings=_settings("langgraph"),
        query_id=None,
    )

    assert result["answer_status"] == "abstained"
    assert result["sources"] == []
    assert result["orchestration_mode"] == "langgraph"
    assert result["orchestration_outcome"] == "unavailable"
