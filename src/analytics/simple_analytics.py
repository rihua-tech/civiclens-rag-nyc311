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


ANALYTICS_FALLBACK = (
    "I can answer only the predefined sample analytics questions for this local demo. "
    "Try asking about top complaint types, borough complaint volume, agency request volume, "
    "or the backlog summary."
)
ANALYTICS_CUES = (
    "top complaint",
    "complaint type",
    "complaint volume",
    "borough",
    "agency request",
    "agencies handle",
    "request volume",
    "requests by",
    "backlog",
    "overdue",
)
FIELD_DEFINITION_CUES = (
    "what does",
    "what do",
    "mean",
    "means",
    "definition",
    "define",
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
    normalized = " ".join(question.lower().split())

    if not normalized:
        return fallback_response()
    if looks_like_field_definition_question(normalized):
        return fallback_response()
    if "complaint" in normalized and ("top" in normalized or "type" in normalized):
        return top_complaint_types_answer()
    if "borough" in normalized and (
        "highest" in normalized
        or "volume" in normalized
        or "most" in normalized
        or "requests" in normalized
        or "complaint" in normalized
    ):
        return borough_volume_answer()
    if "agenc" in normalized and (
        "most" in normalized or "volume" in normalized or "requests" in normalized
    ):
        return agency_volume_answer()
    if "backlog" in normalized or "overdue" in normalized:
        return backlog_summary_answer()

    return fallback_response()


def is_analytics_question(question: str) -> bool:
    return answer_analytics_question(question)["mode"] == "analytics"


def looks_like_analytics_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if looks_like_field_definition_question(normalized):
        return False
    return any(cue in normalized for cue in ANALYTICS_CUES)


def looks_like_field_definition_question(normalized_question: str) -> bool:
    return any(cue in normalized_question for cue in FIELD_DEFINITION_CUES)


def fallback_response() -> dict[str, Any]:
    return {
        "answer": ANALYTICS_FALLBACK,
        "sources": [],
        "confidence_note": "No predefined sample analytics route matched the question.",
        "retrieved_chunks": [],
        "sample_rows": [],
        "mode": "fallback",
    }
