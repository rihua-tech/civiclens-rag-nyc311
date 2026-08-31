"""Thin HTTP adapter for question orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from api.errors import SafeAPIError
from api.models import AnswerRequest, AnswerResponse, AnswerSource, ErrorResponse
from src.generation.schemas import AnswerStatus, NO_ANSWER
from src.orchestration.question_router import route_question


router = APIRouter(prefix="/api/v1", tags=["answers"])
QuestionRouter = Callable[..., dict[str, Any]]
SIMPLE_MARKDOWN_EMPHASIS_PATTERN = re.compile(
    r"\*\*(?P<text>\S(?:.*?\S)?)\*\*"
)


def get_question_router() -> QuestionRouter:
    return route_question


def normalize_answer_display_text(answer: str) -> str:
    """Remove paired asterisk emphasis markers for plain-text answer clients."""

    return SIMPLE_MARKDOWN_EMPHASIS_PATTERN.sub(
        lambda match: match.group("text"),
        answer,
    )


def _public_source(source: dict[str, Any]) -> AnswerSource:
    return AnswerSource(
        source_name=str(source.get("source_name") or "Unknown source"),
        source_path=str(source.get("source_path") or "Unknown path"),
        chunk_id=str(source.get("chunk_id") or "unknown"),
        section_title=(
            str(source["section_title"])
            if source.get("section_title") is not None
            else None
        ),
        citation_number=(
            int(source["citation_number"])
            if source.get("citation_number") is not None
            else None
        ),
    )


def public_answer_response(result: dict[str, Any]) -> AnswerResponse:
    mode = str(result.get("mode", ""))
    if mode == "backend_error":
        raise SafeAPIError(
            503,
            "backend_unavailable",
            "The local question-answering backend is unavailable.",
        )
    if mode not in {"rag", "analytics", "fallback"}:
        raise RuntimeError("orchestrator returned an unsupported route")

    route = "rag" if mode == "rag" else "analytics"
    raw_status = result.get("answer_status")
    if mode == "fallback" or raw_status == AnswerStatus.ABSTAINED.value:
        status = AnswerStatus.ABSTAINED.value
    else:
        status = AnswerStatus.ANSWERED.value

    answer = normalize_answer_display_text(str(result.get("answer") or NO_ANSWER))
    raw_sources = result.get("sources") or []
    if not isinstance(raw_sources, list):
        raise RuntimeError("orchestrator returned invalid sources")

    return AnswerResponse(
        answer=answer,
        route=route,
        status=status,
        sources=[_public_source(source) for source in raw_sources],
        confidence_note=(
            str(result["confidence_note"])
            if result.get("confidence_note") is not None
            else None
        ),
        query_id=result.get("query_id"),
    )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    response_model_exclude_none=True,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def answer(
    request: AnswerRequest,
    orchestrator: QuestionRouter = Depends(get_question_router),
) -> AnswerResponse:
    result = orchestrator(request.question, top_k=request.top_k)
    return public_answer_response(result)
