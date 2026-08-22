"""Small, provider-neutral client for the public CivicLens answer API."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from api.models import AnswerResponse

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_API_TIMEOUT_SECONDS = 30.0
MAX_API_TIMEOUT_SECONDS = 120.0


class APIClientError(RuntimeError):
    """Base class for sanitized UI-facing API client failures."""

    def __init__(self, code: str, user_message: str) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message


class APIConfigurationError(APIClientError):
    """Raised for invalid local client configuration."""


class APIUnavailableError(APIClientError):
    """Raised when the API process cannot be reached."""


class APITimeoutError(APIClientError):
    """Raised when the API request exceeds its configured timeout."""


class BackendNotReadyError(APIClientError):
    """Raised when the API is alive but its local RAG backend is not ready."""


class APIValidationError(APIClientError):
    """Raised when the API rejects the public request contract."""


class APIServerError(APIClientError):
    """Raised for sanitized server failures."""


class MalformedAPIResponseError(APIClientError):
    """Raised when a response does not match the public answer contract."""


def _api_base_url(explicit_value: str | None = None) -> str:
    value = (explicit_value or os.getenv("CIVICLENS_API_BASE_URL") or DEFAULT_API_BASE_URL).strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise APIConfigurationError(
            "invalid_api_url",
            "The CivicLens API address is invalid. Check CIVICLENS_API_BASE_URL.",
        )
    return value.rstrip("/")


def _api_timeout(explicit_value: float | None = None) -> float:
    raw_value: str | float = (
        explicit_value
        if explicit_value is not None
        else os.getenv("CIVICLENS_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS))
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise APIConfigurationError(
            "invalid_api_timeout",
            "The CivicLens API timeout is invalid. Check CIVICLENS_API_TIMEOUT_SECONDS.",
        ) from exc
    if not 0 < value <= MAX_API_TIMEOUT_SECONDS:
        raise APIConfigurationError(
            "invalid_api_timeout",
            f"The CivicLens API timeout must be between 0 and {MAX_API_TIMEOUT_SECONDS:g} seconds.",
        )
    return value


def ask_question(
    question: str,
    *,
    top_k: int = 5,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Call ``POST /api/v1/answer`` and validate its public response."""

    request = Request(
        f"{_api_base_url(base_url)}/api/v1/answer",
        data=json.dumps({"question": question, "top_k": top_k}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=_api_timeout(timeout_seconds)) as response:
            raw_body = response.read()
    except HTTPError as exc:
        if exc.code == 503:
            raise BackendNotReadyError(
                "backend_not_ready",
                "The CivicLens backend is not ready. Run the documented bootstrap command and try again.",
            ) from None
        if exc.code in {400, 409, 422}:
            raise APIValidationError(
                "invalid_request",
                "The API rejected this question. Check the question and top-k values.",
            ) from None
        if exc.code >= 500:
            raise APIServerError(
                "server_error",
                "CivicLens could not complete the request. Please try again later.",
            ) from None
        raise APIClientError(
            "api_error",
            "The CivicLens API rejected the request.",
        ) from None
    except (TimeoutError, socket.timeout):
        raise APITimeoutError(
            "request_timeout",
            "The CivicLens API request timed out. Please try again.",
        ) from None
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise APITimeoutError(
                "request_timeout",
                "The CivicLens API request timed out. Please try again.",
            ) from None
        raise APIUnavailableError(
            "api_unavailable",
            "The CivicLens API is unavailable. Start the API and try again.",
        ) from None
    except OSError:
        raise APIUnavailableError(
            "api_unavailable",
            "The CivicLens API is unavailable. Start the API and try again.",
        ) from None

    try:
        payload = json.loads(raw_body.decode("utf-8"))
        validated = AnswerResponse.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        raise MalformedAPIResponseError(
            "malformed_response",
            "The CivicLens API returned an unexpected response.",
        ) from None
    return validated.model_dump(mode="json", exclude_none=True)

