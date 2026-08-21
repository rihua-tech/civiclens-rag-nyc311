"""Thin HTTP adapter for privacy-conscious query feedback."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends

from api.errors import SafeAPIError
from api.models import ErrorResponse, FeedbackRequest, FeedbackResponse
from src.observability.feedback import (
    FeedbackPersistenceError,
    FeedbackUnavailableError,
    UnknownQueryError,
    submit_feedback,
)
from src.observability.models import FeedbackRating, FeedbackRecord


router = APIRouter(prefix="/api/v1", tags=["feedback"])
FeedbackRecorder = Callable[[str, FeedbackRating, str | None], FeedbackRecord]


def get_feedback_recorder() -> FeedbackRecorder:
    return submit_feedback


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def feedback(
    request: FeedbackRequest,
    recorder: FeedbackRecorder = Depends(get_feedback_recorder),
) -> FeedbackResponse:
    try:
        record = recorder(
            str(request.query_id),
            request.rating,
            request.comment,
        )
    except UnknownQueryError as exc:
        raise SafeAPIError(
            404,
            "unknown_query_id",
            "Feedback requires a known query_id.",
        ) from exc
    except FeedbackUnavailableError as exc:
        raise SafeAPIError(
            503,
            "feedback_disabled",
            "Feedback is unavailable while observability is disabled.",
        ) from exc
    except FeedbackPersistenceError as exc:
        raise SafeAPIError(
            503,
            "feedback_unavailable",
            "Feedback could not be recorded.",
        ) from exc

    return FeedbackResponse(
        feedback_id=record.feedback_id,
        query_id=record.query_id,
        rating=record.rating,
    )
