from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.routes.answers import get_question_router
from api.routes.system import get_readiness_checker
from src.common.config import Settings
from src.orchestration.readiness import (
    ChunkIdentity,
    CorpusIdentity,
    DocumentIdentity,
    ReadinessResult,
    check_readiness,
    load_current_corpus_identity,
)


SECRET_MARKERS = (
    "sk-test-secret-value",
    "postgresql://civiclens:super-secret",
    "Traceback",
    "RuntimeError",
)


def _rag_result():
    return {
        "answer": "Use retrieved evidence [1].",
        "mode": "rag",
        "answer_status": "answered",
        "confidence_note": "Citations validated.",
        "sources": [
            {
                "source_name": "NYC 311 Field Guide",
                "source_path": "docs/knowledge/nyc311-service-request-fields.md",
                "chunk_id": "chunk_abc",
                "section_title": "Complaint Type",
                "citation_number": 1,
                "chunk_text": "private raw evidence",
                "document_content_hash": "sha256:internal",
            }
        ],
        "retrieved_chunks": [{"chunk_text": "private raw evidence"}],
        "answer_provider": "openai",
        "raw_provider_payload": {"secret": "not public"},
    }


def _analytics_result():
    return {
        "answer": "The top sample complaint type is Noise.",
        "mode": "analytics",
        "confidence_note": "Checked-in sample only.",
        "sources": [
            {
                "source_name": "top_complaint_types.csv",
                "source_path": "data/sample_outputs/top_complaint_types.csv",
                "chunk_id": "sample_output",
            }
        ],
        "sample_rows": [{"complaint_type": "Noise"}],
        "retrieved_chunks": [],
    }


def _client_with_router(router):
    app.dependency_overrides[get_question_router] = lambda: router
    return TestClient(app, raise_server_exceptions=False)


def teardown_function():
    app.dependency_overrides.clear()


def test_valid_rag_request_has_typed_sanitized_response_and_forwards_top_k():
    captured = {}

    def fake_router(question, top_k):
        captured.update(question=question, top_k=top_k)
        return _rag_result()

    response = _client_with_router(fake_router).post(
        "/api/v1/answer",
        json={"question": "  What does complaint_type mean?  ", "top_k": 7},
    )

    assert response.status_code == 200
    assert captured == {"question": "What does complaint_type mean?", "top_k": 7}
    assert response.json() == {
        "answer": "Use retrieved evidence [1].",
        "route": "rag",
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
        "confidence_note": "Citations validated.",
    }
    body = response.text
    assert "private raw evidence" not in body
    assert "raw_provider_payload" not in body
    assert "answer_provider" not in body


def test_valid_analytics_request_has_same_public_contract():
    response = _client_with_router(lambda question, top_k: _analytics_result()).post(
        "/api/v1/answer",
        json={"question": "What are the top complaint types?"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "analytics"
    assert response.json()["status"] == "answered"
    assert "sample_rows" not in response.json()


def test_analytics_fallback_is_a_public_abstention():
    fallback = {
        "answer": "No predefined route matched.",
        "mode": "fallback",
        "sources": [],
        "confidence_note": "No predefined route.",
    }
    response = _client_with_router(lambda question, top_k: fallback).post(
        "/api/v1/answer",
        json={"question": "Compare requests by weekday"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "analytics"
    assert response.json()["status"] == "abstained"


def test_invalid_question_and_top_k_values_return_stable_safe_422():
    client = _client_with_router(lambda question, top_k: _rag_result())
    invalid_payloads = (
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "Valid", "top_k": 0},
        {"question": "Valid", "top_k": 101},
    )

    for payload in invalid_payloads:
        response = client.post("/api/v1/answer", json=payload)
        assert response.status_code == 422
        assert response.json() == {
            "error": {
                "code": "invalid_request",
                "message": "Request validation failed.",
            }
        }


def test_backend_unavailable_is_controlled_503_without_internal_detail():
    def unavailable(question, top_k):
        return {
            "answer": "local debug message",
            "mode": "backend_error",
            "sources": [],
            "error_detail": (
                "RuntimeError: postgresql://civiclens:super-secret@localhost/db "
                "sk-test-secret-value"
            ),
        }

    response = _client_with_router(unavailable).post(
        "/api/v1/answer",
        json={"question": "What is the runbook?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_unavailable",
            "message": "The local question-answering backend is unavailable.",
        }
    }
    assert all(marker not in response.text for marker in SECRET_MARKERS)


def test_unexpected_failure_is_controlled_500_without_exception_detail():
    def explode(question, top_k):
        raise RuntimeError("sk-test-secret-value")

    response = _client_with_router(explode).post(
        "/api/v1/answer",
        json={"question": "What is the runbook?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "The request could not be completed.",
        }
    }
    assert all(marker not in response.text for marker in SECRET_MARKERS)


def test_health_is_dependency_free():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_for_available_local_backend():
    app.dependency_overrides[get_readiness_checker] = lambda: lambda: ReadinessResult(
        True,
        "ready",
        "Local RAG backend is ready.",
    )

    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "message": "Local RAG backend is ready.",
    }


def test_ready_returns_controlled_503_for_unavailable_local_backend():
    app.dependency_overrides[get_readiness_checker] = lambda: lambda: ReadinessResult(
        False,
        "backend_unavailable",
        "Local PostgreSQL/pgvector backend is unavailable.",
    )

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "message": "Local PostgreSQL/pgvector backend is unavailable.",
    }


class FakeCursor:
    def __init__(
        self,
        corpus_identity,
        schema_row=(True, True),
        profiles=None,
        document_rows=None,
        chunk_rows=None,
    ):
        self.schema_row = schema_row
        self.profiles = (
            {
                (
                    "sentence_transformers",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    384,
                )
            }
            if profiles is None
            else profiles
        )
        self.document_rows = (
            [
                (
                    item.document_id,
                    item.content_hash,
                    item.chunking_config_hash,
                )
                for item in corpus_identity.documents
            ]
            if document_rows is None
            else document_rows
        )
        self.chunk_rows = (
            [
                (
                    item.chunk_id,
                    item.document_id,
                    item.content_hash,
                    item.document_content_hash,
                    item.chunking_config_hash,
                )
                for item in corpus_identity.chunks
            ]
            if chunk_rows is None
            else chunk_rows
        )
        self.queries = []
        self.current_query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.current_query = str(query)
        self.queries.append((self.current_query, parameters))

    def fetchone(self):
        if "to_regclass" in self.current_query:
            return self.schema_row
        raise AssertionError(f"Unexpected fetchone query: {self.current_query}")

    def fetchall(self):
        if "SELECT DISTINCT" in self.current_query:
            return list(self.profiles)
        if "FROM documents" in self.current_query:
            return list(self.document_rows)
        if "c.chunk_id" in self.current_query:
            return list(self.chunk_rows)
        raise AssertionError(f"Unexpected fetchall query: {self.current_query}")


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.fake_cursor


def _local_settings(answer_provider="local", api_key=""):
    return Settings(
        database_url="postgresql://local/test",
        embedding_provider="sentence_transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        use_openai_embeddings=False,
        use_openai_answers=answer_provider == "openai",
        openai_api_key=api_key,
        answer_provider=answer_provider,
    )


def _corpus_identity():
    return CorpusIdentity(
        documents=(
            DocumentIdentity(
                document_id="doc_current",
                content_hash="sha256:document",
                chunking_config_hash="sha256:chunking",
            ),
        ),
        chunks=(
            ChunkIdentity(
                chunk_id="chunk_current",
                document_id="doc_current",
                content_hash="sha256:chunk",
                document_content_hash="sha256:document",
                chunking_config_hash="sha256:chunking",
            ),
        ),
    )


def test_readiness_accepts_complete_current_corpus_without_openai_key():
    corpus_identity = _corpus_identity()
    cursor = FakeCursor(corpus_identity)
    captured = {}

    def connect(database_url, *, connect_timeout):
        captured.update(database_url=database_url, connect_timeout=connect_timeout)
        return FakeConnection(cursor)

    result = check_readiness(
        settings=_local_settings(answer_provider="openai", api_key=""),
        connection_factory=connect,
        corpus_identity=corpus_identity,
    )

    assert result.ready is True
    assert captured == {
        "database_url": "postgresql://local/test",
        "connect_timeout": 3,
    }
    assert "SELECT DISTINCT" in cursor.queries[1][0]
    assert "FROM documents" in cursor.queries[2][0]
    assert "semantic_embedding" in cursor.queries[3][0]
    assert cursor.queries[3][1] == (
        384,
        "sentence_transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
        384,
        ["chunk_current"],
    )


def test_readiness_rejects_stale_document_hash():
    corpus_identity = _corpus_identity()
    result = check_readiness(
        settings=_local_settings(),
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeCursor(
                corpus_identity,
                document_rows=[
                    ("doc_current", "sha256:older-document", "sha256:chunking")
                ],
            )
        ),
        corpus_identity=corpus_identity,
    )

    assert result == ReadinessResult(
        False,
        "corpus_stale",
        "Stored RAG corpus does not match the current knowledge sources.",
    )


def test_readiness_rejects_incomplete_current_chunks():
    corpus_identity = _corpus_identity()
    result = check_readiness(
        settings=_local_settings(),
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeCursor(corpus_identity, chunk_rows=[])
        ),
        corpus_identity=corpus_identity,
    )

    assert result == ReadinessResult(
        False,
        "corpus_incomplete",
        "Stored RAG corpus is incomplete for the current knowledge sources.",
    )


def test_readiness_rejects_missing_schema_and_connection_failure():
    corpus_identity = _corpus_identity()
    missing_schema = check_readiness(
        settings=_local_settings(),
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeCursor(corpus_identity, schema_row=(True, False))
        ),
        corpus_identity=corpus_identity,
    )

    def fail_connect(*args, **kwargs):
        raise RuntimeError("database secret detail")

    unavailable = check_readiness(
        settings=_local_settings(),
        connection_factory=fail_connect,
        corpus_identity=corpus_identity,
    )

    assert missing_schema.code == "schema_unavailable"
    assert unavailable == ReadinessResult(
        False,
        "backend_unavailable",
        "Local PostgreSQL/pgvector backend is unavailable.",
    )


def test_readiness_rejects_an_incompatible_stored_embedding_profile():
    corpus_identity = _corpus_identity()
    result = check_readiness(
        settings=_local_settings(),
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeCursor(
                corpus_identity,
                profiles={("deterministic", "local-deterministic-1536", 1536)},
            )
        ),
        corpus_identity=corpus_identity,
    )

    assert result == ReadinessResult(
        False,
        "embedding_profile_incompatible",
        "Stored chunks are incompatible with the configured embedding profile.",
    )


def test_readiness_does_not_build_embedding_or_answer_providers(monkeypatch):
    corpus_identity = load_current_corpus_identity()

    def unexpected_provider_call(*args, **kwargs):
        raise AssertionError("Readiness must not create or call a provider")

    monkeypatch.setattr(
        "src.embeddings.providers.create_embedding_provider",
        unexpected_provider_call,
    )
    monkeypatch.setattr(
        "src.generation.providers.build_answer_provider",
        unexpected_provider_call,
    )

    result = check_readiness(
        settings=_local_settings(answer_provider="openai", api_key=""),
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeCursor(corpus_identity)
        ),
    )

    assert result.ready is True
