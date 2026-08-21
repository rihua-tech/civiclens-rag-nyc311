from src.analytics.simple_analytics import ANALYTICS_FALLBACK
from src.orchestration.question_router import (
    BACKEND_NOT_READY_MESSAGE,
    route_question,
)


def test_supported_analytics_question_uses_predefined_route(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("RAG answerer should not be called")

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        fail_if_called,
    )

    result = route_question("What are the top complaint types?")

    assert result["mode"] == "analytics"
    assert result["sources"][0]["source_path"] == (
        "data/sample_outputs/top_complaint_types.csv"
    )


def test_unsupported_analytics_like_question_preserves_fallback(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("RAG answerer should not be called")

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        fail_if_called,
    )

    result = route_question("Compare requests by weekday")

    assert result["mode"] == "fallback"
    assert result["answer"] == ANALYTICS_FALLBACK


def test_document_question_uses_grounded_answer_path_and_forwards_top_k(monkeypatch):
    captured = {}

    def fake_answer(question, top_k, settings, query_id):
        captured.update(
            question=question,
            top_k=top_k,
            settings=settings,
            query_id=query_id,
        )
        return {
            "answer": "Grounded answer [1]",
            "sources": [],
            "confidence_note": "Grounded.",
            "retrieved_chunks": [],
            "answer_status": "answered",
        }

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        fake_answer,
    )

    result = route_question("What is the no-answer rule?", top_k=7)

    assert captured == {
        "question": "What is the no-answer rule?",
        "top_k": 7,
        "settings": captured["settings"],
        "query_id": None,
    }
    assert captured["settings"].observability_enabled is False
    assert result["mode"] == "rag"
    assert result["sample_rows"] == []


def test_backend_exception_becomes_local_application_error(monkeypatch):
    def raise_backend_error(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        raise_backend_error,
    )

    result = route_question("What is the no-answer rule?")

    assert result["mode"] == "backend_error"
    assert result["answer"] == BACKEND_NOT_READY_MESSAGE
    assert result["error_detail"] == "RuntimeError: connection refused"


def test_invalid_top_k_is_rejected_before_routing():
    try:
        route_question("What is the no-answer rule?", top_k=0)
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than 0"
    else:
        raise AssertionError("Expected invalid top_k to fail")
