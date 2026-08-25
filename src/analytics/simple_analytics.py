"""Predefined analytics routing backed by safe typed tools.

Natural-language detection selects only fixed registry IDs. Tool execution is
read-only, CSV-backed, and never accepts SQL, code, modules, or file paths.
"""

from __future__ import annotations

from typing import Any

from src.analytics.tools import (
    DEFAULT_ANALYTICS_RESULT_LIMIT,
    AnalyticsToolId,
    AnalyticsToolResult,
    execute_analytics_tool,
)
from src.analytics.tools.definitions import load_sample_output as _load_sample_output
from src.orchestration.route_decision import (
    QuestionRoute,
    RouteDecision,
    decide_question_route,
    looks_like_analytics_question as looks_like_analytics_question,
    looks_like_field_definition_question as looks_like_field_definition_question,
)


ANALYTICS_FALLBACK = (
    "I can answer only the predefined sample analytics questions for this local demo. "
    "Try asking about top complaint types, borough complaint volume, agency request volume, "
    "or the backlog summary."
)
def load_sample_output(file_name: str) -> list[dict[str, str]]:
    """Compatibility loader restricted to the four allowlisted sample files."""

    return _load_sample_output(file_name)


def _legacy_row(row: Any) -> dict[str, Any]:
    serialized = row.model_dump(mode="json")
    if "request_count" in serialized:
        serialized["request_count"] = str(serialized["request_count"])
    return serialized


def tool_result_to_analytics_response(result: AnalyticsToolResult) -> dict[str, Any]:
    """Adapt typed tool output to the existing application result contract."""

    sources = [
        {
            "source_name": item.source_name,
            "source_path": item.source_path,
            "chunk_id": item.chunk_id,
        }
        for item in result.provenance
    ]
    return {
        "answer": result.summary,
        "sources": sources,
        "confidence_note": result.disclaimer,
        "retrieved_chunks": [],
        "sample_rows": [_legacy_row(row) for row in result.rows],
        "mode": "analytics",
    }


def execute_analytics_decision(decision: RouteDecision) -> dict[str, Any]:
    """Execute exactly one tool selected by the shared deterministic router."""

    if decision.route is not QuestionRoute.ANALYTICS or decision.tool_id is None:
        raise ValueError("A registered analytics route is required.")
    result = execute_analytics_tool(decision.tool_id, decision.tool_arguments)
    return tool_result_to_analytics_response(result)


def top_complaint_types_answer() -> dict[str, Any]:
    result = execute_analytics_tool(
        AnalyticsToolId.TOP_COMPLAINT_TYPES,
        {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
    )
    return tool_result_to_analytics_response(result)


def borough_volume_answer() -> dict[str, Any]:
    result = execute_analytics_tool(
        AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
        {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
    )
    return tool_result_to_analytics_response(result)


def agency_volume_answer() -> dict[str, Any]:
    result = execute_analytics_tool(
        AnalyticsToolId.AGENCY_REQUEST_VOLUME,
        {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
    )
    return tool_result_to_analytics_response(result)


def backlog_summary_answer() -> dict[str, Any]:
    result = execute_analytics_tool(AnalyticsToolId.BACKLOG_SUMMARY, {})
    return tool_result_to_analytics_response(result)


def answer_analytics_question(question: str) -> dict[str, Any]:
    try:
        decision = decide_question_route(question)
    except ValueError:
        return fallback_response()
    if decision.route is QuestionRoute.ANALYTICS:
        return execute_analytics_decision(decision)
    return fallback_response()


def is_analytics_question(question: str) -> bool:
    return answer_analytics_question(question)["mode"] == "analytics"


def fallback_response() -> dict[str, Any]:
    return {
        "answer": ANALYTICS_FALLBACK,
        "sources": [],
        "confidence_note": "No predefined sample analytics route matched the question.",
        "retrieved_chunks": [],
        "sample_rows": [],
        "mode": "fallback",
    }
