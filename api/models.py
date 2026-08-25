"""Provider-neutral public HTTP contracts."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.retrieval.retrieve_context import DEFAULT_TOP_K, MAX_CANDIDATES
from src.observability.feedback import (
    MAX_FEEDBACK_COMMENT_LENGTH,
    normalize_feedback_comment,
)
from src.observability.models import FeedbackRating
from src.orchestration.route_decision import MAX_QUESTION_LENGTH


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_CANDIDATES)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AnswerSource(BaseModel):
    source_name: str
    source_path: str
    chunk_id: str
    section_title: str | None = None
    citation_number: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    route: Literal["rag", "analytics"]
    status: Literal["answered", "abstained"]
    sources: list[AnswerSource]
    confidence_note: str | None = None
    query_id: UUID | None = None


class FeedbackRequest(BaseModel):
    query_id: UUID
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=MAX_FEEDBACK_COMMENT_LENGTH)

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        return normalize_feedback_comment(value)


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    query_id: UUID
    rating: FeedbackRating
    status: Literal["recorded"] = "recorded"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
