from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("langgraph")

from api.main import app
from api.routes.answers import get_question_router
from src.common.config import Settings
from src.orchestration.question_router import route_question


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        answer_provider="local",
        orchestration_mode="langgraph",
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_existing_answer_endpoint_hides_graph_metadata_and_preserves_citation(
    monkeypatch,
):
    def fake_answer(question, top_k, settings, query_id):
        return {
            "answer": "Grounded graph answer [1]",
            "sources": [
                {
                    "source_name": "Field Guide",
                    "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                    "chunk_id": "chunk_graph",
                    "section_title": "Complaint Type",
                    "citation_number": 1,
                }
            ],
            "confidence_note": "Grounded.",
            "retrieved_chunks": [],
            "answer_status": "answered",
        }

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        fake_answer,
    )
    app.dependency_overrides[get_question_router] = lambda: (
        lambda question, top_k: route_question(
            question,
            top_k=top_k,
            settings=_settings(),
        )
    )

    response = TestClient(app).post(
        "/api/v1/answer",
        json={"question": "What does complaint_type mean?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded graph answer [1]",
        "route": "rag",
        "status": "answered",
        "sources": [
            {
                "source_name": "Field Guide",
                "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                "chunk_id": "chunk_graph",
                "section_title": "Complaint Type",
                "citation_number": 1,
            }
        ],
        "confidence_note": "Grounded.",
    }
    assert "orchestration" not in response.text
    assert "step_count" not in response.text


def test_malformed_graph_provenance_becomes_safe_public_abstention(monkeypatch):
    def malformed_answer(question, top_k, settings, query_id):
        return {
            "answer": "This must not pass validation.",
            "sources": ["not-a-source"],
            "confidence_note": "Invalid.",
            "retrieved_chunks": [],
            "answer_status": "answered",
        }

    monkeypatch.setattr(
        "src.orchestration.question_router.answer_question",
        malformed_answer,
    )
    app.dependency_overrides[get_question_router] = lambda: (
        lambda question, top_k: route_question(
            question,
            top_k=top_k,
            settings=_settings(),
        )
    )

    response = TestClient(app).post(
        "/api/v1/answer",
        json={"question": "What does complaint_type mean?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "rag"
    assert body["status"] == "abstained"
    assert body["sources"] == []
    assert "must not pass" not in body["answer"]

