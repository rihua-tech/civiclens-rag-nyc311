"""Shared deterministic route decisions for direct and LangGraph execution."""

from __future__ import annotations

import re
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
AGGREGATION_CUES = (
    "top",
    "most common",
    "most requests",
    "count",
    "counts",
    "total",
    "totals",
    "rank",
    "ranking",
    "highest",
    "lowest",
    "breakdown",
    "distribution",
    "aggregate",
    "analytics",
    "volume",
    "requests by",
    "how many",
    "percentage",
    "percent",
)
FIELD_DEFINITION_CUES = (
    "meaning",
    "definition",
    "define",
    "field guide",
    "field-guide",
    "schema",
    "field definition",
)
FIELD_INTENT_CUES = (
    "difference",
    "different",
    "compare",
    "comparison",
    "describe",
    "explain",
    "interpret",
    "interpretation",
)
FIELD_LABEL_CUES = (
    "complaint type",
    "descriptor",
    "closed date",
    "due date",
)
LIVE_REQUEST_TIME_CUES = (
    "live",
    "right now",
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
)
FUTURE_OPERATIONAL_CUES = (
    "tomorrow",
    "next week",
    "next month",
    "will lead",
    "will be",
    "predict",
    "forecast",
)
SATISFACTION_CUES = ("satisfaction", "satisfied")
PERSONAL_CONTACT_CUES = (
    "phone number",
    "phone contact",
    "email address",
    "home address",
    "private address",
    "street address",
    "contact information",
    "contact info",
)
PERSONAL_SUBJECT_CUES = (
    "caller",
    "requester",
    "customer",
    "person who filed",
    "person that filed",
    "individual who filed",
)
DOCUMENTATION_INTENT_CUES = (
    "documentation",
    "field guide",
    "field definition",
    "documented",
    "published rows",
    "dataset include",
    "dataset contain",
    "described",
    "describe",
    "limitations",
    "source says",
    "evidence says",
    "claims",
)
SATISFACTION_METRIC_CUES = (
    "score",
    "percentage",
    "percent",
    "rate",
    "highest",
    "lowest",
    "rank",
    "ranking",
    "most satisfied",
    "least satisfied",
    "satisfaction level",
)
OPERATIONAL_SUBJECT_CUES = (
    "request",
    "complaint",
    "backlog",
    "status",
    "count",
    "volume",
    "agency",
    "borough",
)
SCHEMA_FIELD_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
FIELD_MEANING_PATTERN = re.compile(r"\bwhat do(?:es)?\b.+\bmean\b")
SPECIFIC_RECORD_PATTERN = re.compile(
    r"\b(?:service request|request|case|unique key|complaint)\s+#?\d+\b"
)


class QuestionRoute(StrEnum):
    RAG = "rag"
    ANALYTICS = "analytics"
    UNSUPPORTED_RAG = "unsupported_rag"
    UNSUPPORTED_ANALYTICS = "unsupported_analytics"


class InvalidQuestionError(ValueError):
    """Raised when a local caller bypasses API-level question validation."""


class RouteDecision(BaseModel):
    """Application-owned deterministic route with one optional approved tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    route: QuestionRoute
    tool_id: AnalyticsToolId | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    scope_reason: str | None = None

    @model_validator(mode="after")
    def validate_tool_route(self) -> "RouteDecision":
        if self.route is QuestionRoute.ANALYTICS and self.tool_id is None:
            raise ValueError("analytics routes require a registered tool ID")
        if self.route is not QuestionRoute.ANALYTICS and (
            self.tool_id is not None or self.tool_arguments
        ):
            raise ValueError("non-analytics routes cannot select a tool")
        if self.route in {
            QuestionRoute.UNSUPPORTED_RAG,
            QuestionRoute.UNSUPPORTED_ANALYTICS,
        } and not self.scope_reason:
            raise ValueError("unsupported routes require a scope reason")
        if self.route in {QuestionRoute.RAG, QuestionRoute.ANALYTICS} and self.scope_reason:
            raise ValueError("supported routes cannot have a scope reason")
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


def _contains_cue(normalized_question: str, cue: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(cue)}(?!\w)",
        normalized_question,
    ) is not None


def looks_like_aggregation_question(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    return any(_contains_cue(lowered, cue) for cue in AGGREGATION_CUES)


def looks_like_unsupported_analytics_question(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    requests_live_data = "request" in lowered and any(
        cue in lowered for cue in LIVE_REQUEST_TIME_CUES
    )
    return requests_live_data


def looks_like_documentation_question(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    return looks_like_field_definition_question(lowered) or any(
        cue in lowered for cue in DOCUMENTATION_INTENT_CUES
    )


def looks_like_private_value_request(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    has_contact_value = any(cue in lowered for cue in PERSONAL_CONTACT_CUES)
    has_personal_subject = any(cue in lowered for cue in PERSONAL_SUBJECT_CUES)
    if not (has_contact_value and has_personal_subject):
        return False
    if SPECIFIC_RECORD_PATTERN.search(lowered) is not None:
        return True
    return not looks_like_documentation_question(lowered)


def looks_like_unsupported_satisfaction_request(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    if not any(cue in lowered for cue in SATISFACTION_CUES):
        return False
    if looks_like_documentation_question(lowered):
        return False
    return any(cue in lowered for cue in SATISFACTION_METRIC_CUES)


def looks_like_live_operational_request(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    if looks_like_documentation_question(lowered):
        return False
    if any(cue in lowered for cue in ("checked-in sample", "sample output", "sample data")):
        return False
    has_operational_subject = any(cue in lowered for cue in OPERATIONAL_SUBJECT_CUES)
    has_live_time = (
        any(cue in lowered for cue in LIVE_REQUEST_TIME_CUES)
        or "current status" in lowered
        or "current count" in lowered
        or "current volume" in lowered
    )
    has_future_request = any(cue in lowered for cue in FUTURE_OPERATIONAL_CUES)
    return has_operational_subject and (has_live_time or has_future_request)


def live_request_route(normalized_question: str) -> QuestionRoute:
    lowered = normalized_question.lower()
    is_specific_record = bool(
        SPECIFIC_RECORD_PATTERN.search(lowered)
    )
    return (
        QuestionRoute.UNSUPPORTED_RAG
        if is_specific_record
        else QuestionRoute.UNSUPPORTED_ANALYTICS
    )


def looks_like_field_definition_question(normalized_question: str) -> bool:
    lowered = normalized_question.lower()
    if FIELD_MEANING_PATTERN.search(lowered) is not None or any(
        cue in lowered for cue in FIELD_DEFINITION_CUES
    ):
        return True

    has_schema_field = SCHEMA_FIELD_PATTERN.search(lowered) is not None or any(
        cue in lowered for cue in FIELD_LABEL_CUES
    )
    has_field_intent = any(cue in lowered for cue in FIELD_INTENT_CUES)
    return (
        has_schema_field
        and has_field_intent
        and not looks_like_aggregation_question(lowered)
    )


def decide_question_route(question: str) -> RouteDecision:
    """Select one fixed route without executing RAG, tools, SQL, or model calls."""

    normalized = normalize_question(question).lower()
    if looks_like_private_value_request(normalized):
        return RouteDecision(
            route=QuestionRoute.UNSUPPORTED_RAG,
            scope_reason="personal_contact_value",
        )
    if looks_like_field_definition_question(normalized):
        return RouteDecision(route=QuestionRoute.RAG)
    if looks_like_live_operational_request(normalized):
        return RouteDecision(
            route=live_request_route(normalized),
            scope_reason="live_or_future_operational_data",
        )
    if looks_like_unsupported_satisfaction_request(normalized):
        return RouteDecision(
            route=QuestionRoute.UNSUPPORTED_RAG,
            scope_reason="unsupported_performance_metric",
        )
    has_complaint_type_subject = (
        "complaint type" in normalized
        or "complaint_type" in normalized
        or "top complaint" in normalized
    )
    if has_complaint_type_subject and looks_like_aggregation_question(normalized):
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
    if any(cue in normalized for cue in ANALYTICS_CUES) or (
        looks_like_unsupported_analytics_question(normalized)
    ):
        return RouteDecision(
            route=QuestionRoute.UNSUPPORTED_ANALYTICS,
            scope_reason="unsupported_analytics",
        )
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
