import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from app.api_client import (
    APITimeoutError,
    APIUnavailableError,
    BackendNotReadyError,
    MalformedAPIResponseError,
)
from app.streamlit_app import PAGE_TITLE, render_response, route_question


def test_streamlit_app_compiles():
    py_compile.compile(Path("app/streamlit_app.py"), doraise=True)


def test_streamlit_app_bootstraps_project_root_when_run_from_app_dir():
    result = subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path('streamlit_app.py')"],
        cwd=Path("app"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_streamlit_app_import_exposes_expected_title():
    assert PAGE_TITLE == "CivicLens RAG \u2014 NYC 311 Operations Copilot"


def test_streamlit_route_calls_public_api_client(monkeypatch):
    captured = {}

    def fake_api_client(question, *, top_k):
        captured.update(question=question, top_k=top_k)
        return {
            "answer": "shared",
            "route": "rag",
            "status": "answered",
            "sources": [],
        }

    monkeypatch.setattr("app.streamlit_app.ask_question", fake_api_client)

    response = route_question("Shared question", top_k=7)

    assert response["answer"] == "shared"
    assert captured == {"question": "Shared question", "top_k": 7}


@pytest.mark.parametrize(
    "client_error",
    (
        APIUnavailableError("api_unavailable", "The CivicLens API is unavailable."),
        APITimeoutError("request_timeout", "The CivicLens API request timed out."),
        BackendNotReadyError("backend_not_ready", "The CivicLens backend is not ready."),
        MalformedAPIResponseError("malformed_response", "The API response was unexpected."),
    ),
)
def test_streamlit_route_returns_sanitized_operational_error(monkeypatch, client_error):
    def unavailable(question, *, top_k):
        raise client_error

    monkeypatch.setattr("app.streamlit_app.ask_question", unavailable)

    response = route_question("What is the no-answer rule?")

    assert response["mode"] == "backend_error"
    assert response["error_code"] == client_error.code
    assert response["answer"] == client_error.user_message
    assert "route" not in response
    assert "status" not in response
    assert "sources" not in response
    assert "Traceback" not in str(response)


def test_streamlit_uses_public_fields_without_raw_chunk_preview():
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert "query_id" in source
    assert "confidence_note" in source
    assert "render_sources" in source
    assert "retrieved_chunks" not in source
    assert "raw_provider_payload" not in source


def test_streamlit_renders_public_status_sources_confidence_and_query_id(monkeypatch):
    rendered = []
    monkeypatch.setattr("app.streamlit_app.st.subheader", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.write", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.caption", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.markdown", lambda value: rendered.append(value))

    render_response(
        {
            "answer": "Grounded answer [1].",
            "route": "rag",
            "status": "answered",
            "confidence_note": "Validated source.",
            "query_id": "77b5a698-16da-4a3a-a492-e03c26b02cc7",
            "sources": [
                {
                    "source_name": "NYC 311 Field Guide",
                    "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                    "chunk_id": "chunk_abc",
                    "citation_number": 1,
                }
            ],
        }
    )

    output = "\n".join(rendered)
    assert "Route: rag | Status: answered" in output
    assert "Validated source." in output
    assert "77b5a698-16da-4a3a-a492-e03c26b02cc7" in output
    assert "NYC 311 Field Guide" in output
    assert "chunk_abc" in output


def test_streamlit_renders_genuine_rag_abstention_as_rag_abstention(monkeypatch):
    rendered = []
    monkeypatch.setattr("app.streamlit_app.st.subheader", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.write", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.caption", lambda value: rendered.append(value))

    render_response(
        {
            "answer": "NO_ANSWER",
            "route": "rag",
            "status": "abstained",
            "sources": [],
        }
    )

    assert "Route: rag | Status: abstained" in "\n".join(rendered)


def test_streamlit_operational_failure_has_no_rag_status_or_sources(monkeypatch):
    rendered = []
    monkeypatch.setattr("app.streamlit_app.st.subheader", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.warning", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.write", lambda value: rendered.append(value))
    monkeypatch.setattr("app.streamlit_app.st.caption", lambda value: rendered.append(value))

    render_response(
        {
            "answer": "The CivicLens API is unavailable.",
            "mode": "backend_error",
            "error_code": "api_unavailable",
        }
    )

    output = "\n".join(rendered)
    assert "The CivicLens API is unavailable." in output
    assert "Route:" not in output
    assert "Status:" not in output
    assert "Source Citations" not in output
