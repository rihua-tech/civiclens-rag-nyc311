"""Liveness and local RAG readiness endpoints."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, Response, status

from api.models import HealthResponse, ReadinessResponse
from src.orchestration.readiness import ReadinessResult, check_readiness


router = APIRouter(tags=["system"])
ReadinessChecker = Callable[[], ReadinessResult]


def get_readiness_checker() -> ReadinessChecker:
    return check_readiness


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def ready(
    response: Response,
    readiness_checker: ReadinessChecker = Depends(get_readiness_checker),
) -> ReadinessResponse:
    result = readiness_checker()
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", message=result.message)
    return ReadinessResponse(status="ready", message=result.message)
