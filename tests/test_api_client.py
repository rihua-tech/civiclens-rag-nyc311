from __future__ import annotations

import io
import json
import socket
from urllib.error import HTTPError, URLError
from uuid import uuid4

import pytest

from app.api_client import (
    APIConfigurationError,
    APIServerError,
    APITimeoutError,
    APIUnavailableError,
    APIValidationError,
    BackendNotReadyError,
    MalformedAPIResponseError,
    ask_question,
)


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


def _answer_payload(*, route="rag"):
    return {
        "answer": "Grounded answer [1].",
        "route": route,
        "status": "answered",
        "sources": [
            {
                "source_name": "NYC 311 Field Guide",
                "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                "chunk_id": "chunk_abc",
                "section_title": "Complaint Type",
                "citation_number": 1,
            }
        ],
        "confidence_note": "Validated source.",
        "query_id": str(uuid4()),
    }


def test_valid_rag_response_uses_public_contract_and_configuration():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(_answer_payload())

    result = ask_question(
        "What does complaint_type mean?",
        top_k=7,
        base_url="http://api:8000/",
        timeout_seconds=12,
        opener=opener,
    )

    assert captured == {
        "url": "http://api:8000/api/v1/answer",
        "body": {"question": "What does complaint_type mean?", "top_k": 7},
        "timeout": 12.0,
    }
    assert result["route"] == "rag"
    assert result["sources"][0]["chunk_id"] == "chunk_abc"
    assert result["query_id"]


def test_valid_analytics_response_uses_same_public_contract():
    result = ask_question(
        "What are the top complaint types?",
        opener=lambda request, timeout: FakeResponse(_answer_payload(route="analytics")),
    )

    assert result["route"] == "analytics"
    assert result["status"] == "answered"


@pytest.mark.parametrize(
    ("status", "expected_exception"),
    (
        (503, BackendNotReadyError),
        (422, APIValidationError),
        (500, APIServerError),
    ),
)
def test_http_failures_are_mapped_to_sanitized_errors(status, expected_exception):
    def opener(request, timeout):
        raise HTTPError(
            request.full_url,
            status,
            "sensitive upstream detail",
            {},
            io.BytesIO(b'{"secret":"do-not-show"}'),
        )

    with pytest.raises(expected_exception) as caught:
        ask_question("Question", opener=opener)

    assert "sensitive upstream detail" not in str(caught.value)
    assert "do-not-show" not in str(caught.value)


def test_connection_failure_is_sanitized():
    def opener(request, timeout):
        raise URLError("postgresql://user:secret@host/db")

    with pytest.raises(APIUnavailableError) as caught:
        ask_question("Question", opener=opener)

    assert "secret" not in str(caught.value)


def test_timeout_is_sanitized():
    def opener(request, timeout):
        raise socket.timeout("provider diagnostics")

    with pytest.raises(APITimeoutError) as caught:
        ask_question("Question", opener=opener)

    assert "provider diagnostics" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    (
        {"answer": "missing contract"},
        "not-an-object",
    ),
)
def test_malformed_api_response_is_rejected(payload):
    with pytest.raises(MalformedAPIResponseError):
        ask_question(
            "Question",
            opener=lambda request, timeout: FakeResponse(payload),
        )


@pytest.mark.parametrize(
    ("base_url", "timeout"),
    (("api:8000", 30), ("http://localhost:8000", 0), ("http://localhost:8000", 121)),
)
def test_invalid_client_configuration_fails_clearly(base_url, timeout):
    with pytest.raises(APIConfigurationError):
        ask_question("Question", base_url=base_url, timeout_seconds=timeout)

