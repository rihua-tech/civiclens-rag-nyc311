from __future__ import annotations

from src.common.config import DEFAULT_SEMANTIC_MODEL, Settings
from src.embeddings.providers import EmbeddingSpec
from src.vectorstores import VectorStore
from src.vectorstores.factory import create_vector_store
from src.vectorstores.models import (
    VectorIdentity,
    VectorRecord,
    corpus_fingerprint,
)
from src.vectorstores.pgvector_store import PgVectorStore


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://user:secret@localhost:5432/civiclens",
        embedding_model=DEFAULT_SEMANTIC_MODEL,
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="sentence_transformers",
        embedding_dimension=384,
    )


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)


def _identity(chunk_id: str = "chunk_1") -> VectorIdentity:
    return VectorIdentity(
        chunk_id=chunk_id,
        document_id="doc_1",
        content_hash=f"sha256:{chunk_id}",
        document_content_hash="sha256:document",
        chunking_config_hash="sha256:chunking",
    )


class QueueCursor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.calls.append((str(query), parameters))

    def fetchall(self):
        return self.responses.pop(0)


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_value


class ConnectionQueue:
    def __init__(self, cursors):
        self.cursors = list(cursors)

    def __call__(self, database_url):
        assert database_url.startswith("postgresql://")
        return FakeConnection(self.cursors.pop(0))


def test_default_factory_returns_the_shared_pgvector_contract():
    store = create_vector_store(_settings(), _spec(), [_identity()])

    assert isinstance(store, PgVectorStore)
    assert isinstance(store, VectorStore)
    assert store.provider_name == "pgvector"
    assert "secret" not in store.target


def test_corpus_fingerprint_is_stable_and_identity_sensitive():
    first = corpus_fingerprint(_spec(), [_identity("b"), _identity("a")])
    reordered = corpus_fingerprint(_spec(), [_identity("a"), _identity("b")])
    changed = corpus_fingerprint(_spec(), [_identity("a")])

    assert first == reordered
    assert first != changed


def test_pgvector_sync_updates_only_vectors_and_verifies_current_identity():
    identity = _identity()
    sync_cursor = QueueCursor([[]])
    verify_cursor = QueueCursor(
        [
            [("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)],
            [
                (
                    identity.chunk_id,
                    identity.document_id,
                    identity.content_hash,
                    identity.document_content_hash,
                    identity.chunking_config_hash,
                )
            ],
        ]
    )
    store = PgVectorStore(
        _settings(),
        _spec(),
        connection_factory=ConnectionQueue([sync_cursor, verify_cursor]),
    )

    result = store.sync(
        [VectorRecord(identity=identity, values=tuple([0.0] * 384))]
    )

    executed = "\n".join(query for query, _ in sync_cursor.calls).lower()
    assert result.records_written == 1
    assert result.verified is True
    assert "update chunks" in executed
    assert "semantic_embedding = %s::vector" in executed
    assert "embedding = null" in executed
    assert "insert into chunks" not in executed


def test_pgvector_query_preserves_limit_threshold_cosine_and_stable_rank():
    identity = _identity()
    cursor = QueueCursor(
        [
            [("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)],
            [
                (
                    identity.chunk_id,
                    identity.document_id,
                    identity.content_hash,
                    identity.document_content_hash,
                    identity.chunking_config_hash,
                    0.72,
                )
            ],
        ]
    )
    store = PgVectorStore(
        _settings(),
        _spec(),
        connection_factory=ConnectionQueue([cursor]),
    )

    matches = store.query(
        [0.0] * 384,
        candidate_limit=17,
        min_similarity=0.31,
    )

    query, parameters = cursor.calls[1]
    assert "1 - (c.semantic_embedding <=> %s::vector)" in query
    assert "ORDER BY semantic_score DESC, chunk_id" in query
    assert parameters[-2:] == (17, 0.31)
    assert matches[0].identity == identity
    assert matches[0].score == 0.72
    assert matches[0].rank == 1
