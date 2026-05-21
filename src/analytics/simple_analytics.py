"""Predefined sample analytics answers for the local hybrid RAG UI.

This module intentionally uses small static CSV outputs. It is not a
text-to-SQL router and does not query production or raw NYC 311 data.
"""

from __future__ import annotations

import csv
from pathlib import Path


SAMPLE_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_outputs"
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
    """Load one of the checked-in sample analytics CSV outputs."""

    path = SAMPLE_OUTPUT_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Sample analytics output not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_count(value: str) -> int:
    return int(value.replace(",", "").strip())


def format_count(value: str | int) -> str:
    return f"{int(value):,}"


def source(file_name: str) -> dict[str, str]:
    return {
        "source_name": file_name,
        "source_path": f"data/sample_outputs/{file_name}",
        "chunk_id": "sample_output",
    }


def top_complaint_types_answer() -> dict:
    rows = load_sample_output("top_complaint_types.csv")
    top_rows = rows[:5]
    summary = "; ".join(
        f"{row['complaint_type']} ({format_count(row['request_count'])})"
        for row in top_rows
    )
    answer = f"The top sample complaint types are: {summary}."
    return analytics_response(answer, [source("top_complaint_types.csv")], rows)


def borough_volume_answer() -> dict:
    rows = load_sample_output("requests_by_borough.csv")
    sorted_rows = sorted(rows, key=lambda row: parse_count(row["request_count"]), reverse=True)
    leader = sorted_rows[0]
    summary = "; ".join(
        f"{row['borough']} ({format_count(row['request_count'])})"
        for row in sorted_rows[:5]
    )
    answer = (
        f"{leader['borough']} has the highest sample complaint volume "
        f"with {format_count(leader['request_count'])} requests. Borough totals: {summary}."
    )
    return analytics_response(answer, [source("requests_by_borough.csv")], sorted_rows)


def agency_volume_answer() -> dict:
    rows = load_sample_output("agency_request_volume.csv")
    sorted_rows = sorted(rows, key=lambda row: parse_count(row["request_count"]), reverse=True)
    summary = "; ".join(
        f"{row['agency']} ({format_count(row['request_count'])})"
        for row in sorted_rows[:5]
    )
    answer = f"The agencies handling the most sample requests are: {summary}."
    return analytics_response(answer, [source("agency_request_volume.csv")], sorted_rows)


def backlog_summary_answer() -> dict:
    rows = load_sample_output("backlog_summary.csv")
    counts = {row["status"].lower(): parse_count(row["request_count"]) for row in rows}
    open_count = counts.get("open", 0)
    in_progress_count = counts.get("in progress", 0)
    overdue_count = counts.get("overdue", 0)
    closed_recently = counts.get("closed last 7 days", 0)
    answer = (
        "The sample backlog summary shows "
        f"{format_count(open_count)} open requests, "
        f"{format_count(in_progress_count)} in progress, "
        f"{format_count(overdue_count)} overdue, and "
        f"{format_count(closed_recently)} closed in the last 7 days."
    )
    return analytics_response(answer, [source("backlog_summary.csv")], rows)


def analytics_response(answer: str, sources: list[dict], rows: list[dict]) -> dict:
    return {
        "answer": answer,
        "sources": sources,
        "confidence_note": (
            "Sample analytics answer from checked-in CSV outputs only; "
            "not live NYC 311 data and not a production text-to-SQL result."
        ),
        "retrieved_chunks": [],
        "sample_rows": rows,
        "mode": "analytics",
    }


def answer_analytics_question(question: str) -> dict:
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
    if "agenc" in normalized and ("most" in normalized or "volume" in normalized or "requests" in normalized):
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


def fallback_response() -> dict:
    return {
        "answer": ANALYTICS_FALLBACK,
        "sources": [],
        "confidence_note": "No predefined sample analytics route matched the question.",
        "retrieved_chunks": [],
        "sample_rows": [],
        "mode": "fallback",
    }
