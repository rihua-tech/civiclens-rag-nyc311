from __future__ import annotations

from src.common.config import Settings
from src.orchestration.readiness import (
    ChunkIdentity,
    CorpusIdentity,
    DocumentIdentity,
    check_readiness,
)
from src.vectorstores.models import VectorStoreConsistencyError


def _corpus() -> CorpusIdentity:
    return CorpusIdentity(
        documents=(
            DocumentIdentity("doc_1", "sha256:document", "sha256:chunking"),
        ),
        chunks=(
            ChunkIdentity(
                "chunk_1",
                "doc_1",
                "sha256:chunk",
                "sha256:document",
                "sha256:chunking",
            ),
        ),
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
        vector_store_provider="pinecone",
        pinecone_api_key="secret",
        pinecone_index_name="civiclens",
        pinecone_index_host="civiclens.svc.pinecone.io",
    )


class Cursor:
    def __init__(self, events):
        self.events = events
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.query = str(query)
        self.events.append(("postgres", self.query, parameters))

    def fetchone(self):
        return (True, True)

    def fetchall(self):
        if "FROM documents" in self.query:
            return [("doc_1", "sha256:document", "sha256:chunking")]
        return [
            (
                "chunk_1",
                "doc_1",
                "sha256:chunk",
                "sha256:document",
                "sha256:chunking",
            )
        ]


class Connection:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_value


class PineconeReadinessStore:
    provider_name = "pinecone"
    target = "pinecone:civiclens/current"

    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def prepare_sync(self, *, reindex=False):
        raise AssertionError("readiness must not prepare or mutate")

    def sync(self, records, *, reindex=False):
        raise AssertionError("readiness must not synchronize")

    def query(self, vector, *, candidate_limit, min_similarity):
        raise AssertionError("readiness must not run semantic retrieval")

    def verify(self, identities):
        self.events.append(("pinecone_verify", [item.chunk_id for item in identities]))
        if self.error is not None:
            raise self.error


def test_readiness_checks_postgres_metadata_before_selected_pinecone():
    events = []
    store = PineconeReadinessStore(events)

    result = check_readiness(
        settings=_settings(),
        connection_factory=lambda *args, **kwargs: Connection(Cursor(events)),
        corpus_identity=_corpus(),
        vector_store=store,
    )

    assert result.ready is True
    assert events[-1] == ("pinecone_verify", ["chunk_1"])
    assert all(event[0] == "postgres" for event in events[:-1])


def test_readiness_reports_incomplete_pinecone_without_mutation_or_fallback():
    events = []
    store = PineconeReadinessStore(
        events,
        error=VectorStoreConsistencyError("partial namespace"),
    )

    result = check_readiness(
        settings=_settings(),
        connection_factory=lambda *args, **kwargs: Connection(Cursor(events)),
        corpus_identity=_corpus(),
        vector_store=store,
    )

    assert result.ready is False
    assert result.code == "vector_store_incomplete"
    assert events[-1][0] == "pinecone_verify"
