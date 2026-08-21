import py_compile
import subprocess
import sys
from pathlib import Path

from app.streamlit_app import BACKEND_NOT_READY_MESSAGE, PAGE_TITLE, route_question, truncate_chunk_text


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


def test_chunk_preview_text_is_truncated_for_plain_text_display():
    preview = truncate_chunk_text("x" * 1300)

    assert len(preview) == 1200
    assert preview.endswith("...")


def test_route_question_returns_safe_backend_message(monkeypatch):
    def backend_error(question):
        return {
            "answer": BACKEND_NOT_READY_MESSAGE,
            "sources": [],
            "confidence_note": "Local backend unavailable.",
            "retrieved_chunks": [],
            "sample_rows": [],
            "mode": "backend_error",
            "error_detail": "RuntimeError: connection refused",
        }

    monkeypatch.setattr("app.streamlit_app.orchestrate_question", backend_error)

    response = route_question("What is the no-answer rule?")

    assert response["mode"] == "backend_error"
    assert response["answer"] == BACKEND_NOT_READY_MESSAGE
    assert response["sources"] == []
    assert response["retrieved_chunks"] == []


def test_streamlit_route_is_a_thin_shared_orchestration_wrapper(monkeypatch):
    captured = {}

    def fake_orchestrator(question):
        captured["question"] = question
        return {"answer": "shared", "mode": "rag"}

    monkeypatch.setattr("app.streamlit_app.orchestrate_question", fake_orchestrator)

    assert route_question("Shared question") == {"answer": "shared", "mode": "rag"}
    assert captured == {"question": "Shared question"}
