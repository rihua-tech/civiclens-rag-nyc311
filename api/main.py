"""CivicLens FastAPI application factory and public error boundary."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.errors import SafeAPIError
from api.models import ErrorBody, ErrorResponse
from api.routes.answers import router as answers_router
from api.routes.feedback import router as feedback_router
from api.routes.system import router as system_router


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def create_app() -> FastAPI:
    application = FastAPI(
        title="CivicLens RAG API",
        version="1.0.0",
        description="Local versioned API for CivicLens question orchestration.",
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request, exc
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_request",
            "Request validation failed.",
        )

    @application.exception_handler(SafeAPIError)
    async def safe_api_error_handler(
        request: Request,
        exc: SafeAPIError,
    ) -> JSONResponse:
        del request
        return _error_response(exc.status_code, exc.code, exc.message)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request, exc
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "The request could not be completed.",
        )

    application.include_router(system_router)
    application.include_router(answers_router)
    application.include_router(feedback_router)
    return application


app = create_app()
