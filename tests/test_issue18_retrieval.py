from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.common.config import Settings
from src.embeddings.providers import EmbeddingSpec
from src.retrieval.hybrid_retriever import retrieve_with_mode
from src.retrieval.retrieve_context import retrieve_semantic_context
from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorStoreConsistencyError,
    VectorStoreError,
)


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://local/test",
        embedding_model="fake-model",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="fake",
        embedding_dimension=3,
        retrieval_mode="hybrid",
        semantic_candidate_count=10,
        lexical_candidate_count=10,
    )


def _identity() -> VectorIdentity:
    return VectorIdentity(
        chunk_id="chunk_1",
        document_id="doc_1",
        content_hash="sha256:chunk",
        document_content_hash="sha256:document",
        chunking_config_hash="sha256:chunking",
    )


def _postgres_row(content_hash="sha256:chunk") -> tuple:
    return (
        "chunk_1",
        "doc_1",
        "PostgreSQL canonical chunk text.",
        "source.md",
        "markdown",
        "civiclens_project",
        "docs/source.md",
        "https://example.test/source",
        "Issue 18",
        "2026-08-25",
        "Retrieval",
        ["Architecture", "Retrieval"],
        4,
        content_hash,
        "sha256:document",
        "sha256:chunking",
        "2026-08-25T00:00:00Z",
    )


class Provider:
    spec = EmbeddingSpec("fake", "fake-model", 3)

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class Store:
    provider_name = "pinecone"
    target = "pinecone:test/current"

    def prepare_sync(self, *, reindex=False):
        return None

    def sync(self, records, *, reindex=False):
        raise AssertionError("sync was not expected")

    def query(self, vector, *, candidate_limit, min_similarity):
        assert vector == [0.1, 0.2, 0.3]
        assert candidate_limit == 10
        assert min_similarity == 0.25
        return [VectorMatch(_identity(), 0.84, 1)]

    def verify(self, identities):
        return None


class Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.calls.append((str(query), parameters))

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_value


def _install_psycopg(monkeypatch, rows):
    cursor = Cursor(rows)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda _: Connection(cursor)),
    )
    return cursor


def test_pinecone_ids_are_hydrated_in_postgres_before_existing_hybrid_rrf(monkeypatch):
    cursor = _install_psycopg(monkeypatch, [_postgres_row()])
    settings = _settings()

    def semantic(question, **kwargs):
        return retrieve_semantic_context(
            question,
            provider=Provider(),
            vector_store=Store(),
            **kwargs,
        )

    lexical_calls = []

    def lexical(question, **kwargs):
        lexical_calls.append((question, kwargs))
        result = retrieve_semantic_context(
            question,
            provider=Provider(),
            vector_store=Store(),
            candidate_limit=10,
            settings=settings,
        )[0]
        result["semantic_score"] = None
        result["semantic_rank"] = None
        result["similarity_score"] = None
        result["lexical_score"] = 0.7
        result["lexical_rank"] = 1
        result["retrieval_mode"] = "lexical"
        return [result]

    results = retrieve_with_mode(
        "question",
        top_k=1,
        min_similarity=0.25,
        settings=settings,
        semantic_retriever=semantic,
        lexical_retriever=lexical,
    )

    assert len(cursor.calls) == 2
    assert lexical_calls
    assert results[0]["chunk_text"] == "PostgreSQL canonical chunk text."
    assert results[0]["semantic_score"] == 0.84
    assert results[0]["lexical_score"] == 0.7
    assert results[0]["fusion_score"] > 0
    assert results[0]["retrieval_mode"] == "hybrid"


def test_hydration_rejects_stale_postgresql_metadata(monkeypatch):
    _install_psycopg(monkeypatch, [_postgres_row("sha256:stale")])

    with pytest.raises(VectorStoreConsistencyError, match="incompatible"):
        retrieve_semantic_context(
            "question",
            candidate_limit=10,
            settings=_settings(),
            provider=Provider(),
            vector_store=Store(),
        )


def test_vector_provider_failure_propagates_instead_of_becoming_no_answer():
    class FailingStore(Store):
        def query(self, vector, *, candidate_limit, min_similarity):
            raise VectorStoreError("provider unavailable")

    with pytest.raises(VectorStoreError, match="provider unavailable"):
        retrieve_semantic_context(
            "question",
            candidate_limit=10,
            settings=_settings(),
            provider=Provider(),
            vector_store=FailingStore(),
        )
