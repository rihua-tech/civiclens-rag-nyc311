"""Shared deterministic route decisions for direct and LangGraph execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analytics.tools import DEFAULT_ANALYTICS_RESULT_LIMIT, AnalyticsToolId


MAX_QUESTION_LENGTH = 2000

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


class QuestionRoute(StrEnum):
    RAG = "rag"
    ANALYTICS = "analytics"
    UNSUPPORTED_ANALYTICS = "unsupported_analytics"


class InvalidQuestionError(ValueError):
    """Raised when a local caller bypasses API-level question validation."""


class RouteDecision(BaseModel):
    """Application-owned deterministic route with one optional approved tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    route: QuestionRoute
    tool_id: AnalyticsToolId | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_route(self) -> "RouteDecision":
        if self.route is QuestionRoute.ANALYTICS and self.tool_id is None:
            raise ValueError("analytics routes require a registered tool ID")
        if self.route is not QuestionRoute.ANALYTICS and (
            self.tool_id is not None or self.tool_arguments
        ):
            raise ValueError("non-analytics routes cannot select a tool")
        return self


def normalize_question(question: str) -> str:
    if not isinstance(question, str):
        raise InvalidQuestionError("question must be a string")
    normalized = " ".join(question.split())
    if not normalized:
        raise InvalidQuestionError("question must not be blank")
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise InvalidQuestionError(
            f"question must be at most {MAX_QUESTION_LENGTH} characters"
        )
    return normalized


def looks_like_field_definition_question(normalized_question: str) -> bool:
    return any(cue in normalized_question.lower() for cue in FIELD_DEFINITION_CUES)


def decide_question_route(question: str) -> RouteDecision:
    """Select one fixed route without executing RAG, tools, SQL, or model calls."""

    normalized = normalize_question(question).lower()
    if looks_like_field_definition_question(normalized):
        return RouteDecision(route=QuestionRoute.RAG)
    if "complaint" in normalized and ("top" in normalized or "type" in normalized):
        return RouteDecision(
            route=QuestionRoute.ANALYTICS,
            tool_id=AnalyticsToolId.TOP_COMPLAINT_TYPES,
            tool_arguments={"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
        )
    if "borough" in normalized and (
        "highest" in normalized
        or "volume" in normalized
        or "most" in normalized
        or "requests" in normalized
        or "complaint" in normalized
    ):
        return RouteDecision(
            route=QuestionRoute.ANALYTICS,
            tool_id=AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
            tool_arguments={"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
        )
    if "agenc" in normalized and (
        "most" in normalized or "volume" in normalized or "requests" in normalized
    ):
        return RouteDecision(
            route=QuestionRoute.ANALYTICS,
            tool_id=AnalyticsToolId.AGENCY_REQUEST_VOLUME,
            tool_arguments={"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
        )
    if "backlog" in normalized or "overdue" in normalized:
        return RouteDecision(
            route=QuestionRoute.ANALYTICS,
            tool_id=AnalyticsToolId.BACKLOG_SUMMARY,
        )
    if any(cue in normalized for cue in ANALYTICS_CUES):
        return RouteDecision(route=QuestionRoute.UNSUPPORTED_ANALYTICS)
    return RouteDecision(route=QuestionRoute.RAG)


def looks_like_analytics_question(question: str) -> bool:
    try:
        decision = decide_question_route(question)
    except InvalidQuestionError:
        return False
    return decision.route in {
        QuestionRoute.ANALYTICS,
        QuestionRoute.UNSUPPORTED_ANALYTICS,
    }
