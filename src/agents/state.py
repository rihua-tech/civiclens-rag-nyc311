"""Typed, privacy-conscious state for the bounded CivicLens LangGraph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.analytics.tools import AnalyticsToolId
from src.common.config import Settings
from src.orchestration.route_decision import QuestionRoute


WorkflowOutcome = Literal[
    "pending",
    "answered",
    "abstained",
    "fallback",
    "failed",
    "limit_exceeded",
    "unavailable",
]


class WorkflowState(TypedDict):
    question: str
    top_k: int
    settings: Settings
    query_id: str | None
    route: QuestionRoute | None
    tool_id: AnalyticsToolId | None
    tool_arguments: dict[str, Any]
    result: dict[str, Any] | None
    answer: str | None
    provenance: list[dict[str, Any]]
    step_count: int
    tool_call_count: int
    max_steps: int
    tool_call_limit: int
    outcome: WorkflowOutcome

