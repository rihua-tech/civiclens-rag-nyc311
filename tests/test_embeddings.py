import json
import sys
from types import SimpleNamespace

import pytest

from src.common.config import (
    DEFAULT_SEMANTIC_MODEL,
    DETERMINISTIC_MODEL,
    Settings,
)
from src.embeddings.embed_chunks import (
    EMBEDDING_DIMENSIONS,
    load_chunks,
    local_deterministic_embedding,
    store_chunks,
)
from src.embeddings.providers import EmbeddingCompatibilityError, EmbeddingSpec
from src.embeddings.providers.deterministic import DeterministicEmbeddingProvider
from src.embeddings.providers.factory import create_embedding_provider
from src.embeddings.providers.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from src.vectorstores.models import VectorSyncResult
from src.vectorstores.pgvector_store import (
    validate_stored_embedding_profiles,
    vector_column_for_spec,
)
from src.vectorstores.postgres_metadata import (
    upsert_chunk_metadata,
    upsert_document,
)


class RecordingCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, parameters=None):
        self.calls.append((query, parameters))


def metadata_chunk() -> dict:
    return {
        "chunk_id": "doc_1_chunk_abc123",
        "document_id": "doc_1",
        "chunk_text": "Readable local chunk.",
        "chunk_index": 0,
        "source_name": "NYC 311 Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/field-guide.md",
        "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
        "source_version": "dataset erm2-nwe9",
        "source_retrieved_at": "2026-08-17",
        "section_title": "Status",
        "heading_path": ["Fields", "Status"],
        "word_count": 3,
        "content_hash": "sha256:chunk",
        "document_content_hash": "sha256:document",
        "chunking_config_hash": "sha256:chunking-config",
        "ingested_at": "2026-08-17T00:00:00Z",
    }


def test_local_deterministic_embedding_returns_1536_values():
    embedding = local_deterministic_embedding("sample chunk text")

    assert len(embedding) == EMBEDDING_DIMENSIONS
    assert all(isinstance(value, float) for value in embedding)


def test_local_deterministic_embedding_is_stable_for_same_text():
    first_embedding = local_deterministic_embedding("same text")
    second_embedding = local_deterministic_embedding("same text")

    assert first_embedding == second_embedding


def test_local_deterministic_embedding_differs_for_different_text():
    first_embedding = local_deterministic_embedding("first text")
    second_embedding = local_deterministic_embedding("second text")

    assert first_embedding != second_embedding


def test_load_chunks_reads_jsonl_file(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    expected_chunk = metadata_chunk()
    chunks_path.write_text(json.dumps(expected_chunk) + "\n", encoding="utf-8")

    assert load_chunks(chunks_path) == [expected_chunk]


def test_document_upsert_preserves_manifest_and_content_metadata():
    cursor = RecordingCursor()

    upsert_document(cursor, metadata_chunk())

    query, parameters = cursor.calls[0]
    assert "source_category" in query
    assert "source_url" in query
    assert "source_version" in query
    assert "source_retrieved_at" in query
    assert "content_hash" in query
    assert parameters == (
        "doc_1",
        "NYC 311 Field Guide",
        "markdown",
        "external_nyc311",
        "docs/knowledge/field-guide.md",
        "https://data.cityofnewyork.us/d/erm2-nwe9",
        "dataset erm2-nwe9",
        "2026-08-17",
        "sha256:document",
        "sha256:chunking-config",
        "2026-08-17T00:00:00Z",
    )


def test_chunk_upsert_preserves_section_hash_and_count_metadata():
    cursor = RecordingCursor()

    upsert_chunk_metadata(cursor, metadata_chunk())

    query, parameters = cursor.calls[0]
    assert query.count("%s") == 18
    assert "section_title" in query
    assert "heading_path" in query
    assert "word_count" in query
    assert "content_hash" in query
    assert "token_count" not in query
    assert parameters[11:18] == (
        "Status",
        ["Fields", "Status"],
        3,
        "sha256:chunk",
        "sha256:document",
        "sha256:chunking-config",
        "2026-08-17T00:00:00Z",
    )
    assert "embedding_provider" not in query
    assert "semantic_embedding" not in query


def test_deterministic_provider_remains_available_and_stable():
    provider = DeterministicEmbeddingProvider()

    first = provider.embed("complaint type")
    second = provider.embed("complaint type")

    assert provider.spec == EmbeddingSpec(
        "deterministic",
        DETERMINISTIC_MODEL,
        EMBEDDING_DIMENSIONS,
    )
    assert first == second
    assert len(first) == EMBEDDING_DIMENSIONS


def test_provider_selection_preserves_legacy_deterministic_configuration():
    settings = Settings(
        database_url="postgresql://example",
        embedding_model=DETERMINISTIC_MODEL,
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
    )

    provider = create_embedding_provider(settings)

    assert provider.spec.provider == "deterministic"
    assert provider.spec.model == DETERMINISTIC_MODEL
    assert provider.spec.dimension == 1536


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        assert kwargs == {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        return [[float(index == 0) for index in range(384)] for _ in texts]


def test_semantic_provider_contract_uses_fake_without_model_download():
    loaded_models = []
    provider = SentenceTransformersEmbeddingProvider(
        model_loader=lambda model_name: loaded_models.append(model_name)
        or FakeSentenceTransformer()
    )

    embeddings = provider.embed_many(["borough", "complaint_type"])

    assert provider.spec == EmbeddingSpec(
        "sentence_transformers",
        DEFAULT_SEMANTIC_MODEL,
        384,
    )
    assert loaded_models == [DEFAULT_SEMANTIC_MODEL]
    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)


def test_semantic_provider_rejects_runtime_dimension_mismatch():
    class WrongDimensionModel(FakeSentenceTransformer):
        def get_sentence_embedding_dimension(self):
            return 768

    provider = SentenceTransformersEmbeddingProvider(
        model_loader=lambda _: WrongDimensionModel()
    )

    with pytest.raises(EmbeddingCompatibilityError, match="reports 768 dimensions"):
        provider.embed("status")


def test_storage_dimension_validation_rejects_unsupported_semantic_dimension():
    with pytest.raises(EmbeddingCompatibilityError, match=r"vector\(384\)"):
        vector_column_for_spec(
            EmbeddingSpec("sentence_transformers", "different-model", 768)
        )


def test_mixed_or_incompatible_stored_profiles_fail_clearly():
    active = EmbeddingSpec("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)
    stored = {
        ("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384),
        ("deterministic", DETERMINISTIC_MODEL, 1536),
    }

    with pytest.raises(EmbeddingCompatibilityError, match="--reindex"):
        validate_stored_embedding_profiles(stored, active)


def test_existing_openai_flag_maps_to_backward_compatible_provider(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_EMBEDDINGS", "true")
    monkeypatch.setenv("EMBEDDING_MODEL", DETERMINISTIC_MODEL)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)

    settings = Settings.from_env()

    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536


def test_semantic_profile_selects_only_384_dimension_vector_column():
    spec = EmbeddingSpec("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)

    assert vector_column_for_spec(spec) == "semantic_embedding"


class ContextCursor(RecordingCursor):
    def __init__(self, profiles):
        super().__init__()
        self.profiles = profiles

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def fetchall(self):
        return self.profiles


class ContextConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self._cursor


class FakeBatchSemanticProvider:
    spec = EmbeddingSpec("sentence_transformers", DEFAULT_SEMANTIC_MODEL, 384)

    def __init__(self):
        self.calls = []

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * 384 for _ in texts]


class RecordingVectorStore:
    provider_name = "test"
    target = "test:vector-store"

    def __init__(self, prepare_error=None):
        self.prepare_error = prepare_error
        self.prepare_calls = []
        self.sync_calls = []

    def prepare_sync(self, *, reindex=False):
        self.prepare_calls.append(reindex)
        if self.prepare_error is not None:
            raise self.prepare_error

    def sync(self, records, *, reindex=False):
        self.sync_calls.append((list(records), reindex))
        return VectorSyncResult("test", self.target, None, len(records), True)

    def query(self, vector, *, candidate_limit, min_similarity):
        raise AssertionError("query was not expected")

    def verify(self, identities):
        return None


def semantic_settings() -> Settings:
    return Settings(
        database_url="postgresql://example",
        embedding_model=DEFAULT_SEMANTIC_MODEL,
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="sentence_transformers",
        embedding_dimension=384,
    )


def test_incompatible_database_profile_fails_before_model_embedding(monkeypatch):
    cursor = ContextCursor([])
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda _: ContextConnection(cursor)),
    )
    provider = FakeBatchSemanticProvider()
    vector_store = RecordingVectorStore(
        EmbeddingCompatibilityError("Stored profile is incompatible; use --reindex")
    )

    with pytest.raises(EmbeddingCompatibilityError, match="--reindex"):
        store_chunks(
            [metadata_chunk()],
            semantic_settings(),
            provider=provider,
            vector_store=vector_store,
        )

    assert provider.calls == []
    executed_sql = "\n".join(str(query) for query, _ in cursor.calls).lower()
    assert "insert into documents" in executed_sql
    assert "insert into chunks" in executed_sql
    assert "embedding_provider" not in executed_sql


def test_explicit_reindex_is_forwarded_to_the_selected_vector_store(monkeypatch):
    cursor = ContextCursor({("deterministic", DETERMINISTIC_MODEL, 1536)})
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda _: ContextConnection(cursor)),
    )
    provider = FakeBatchSemanticProvider()
    vector_store = RecordingVectorStore()

    stored = store_chunks(
        [metadata_chunk()],
        semantic_settings(),
        provider=provider,
        reindex=True,
        vector_store=vector_store,
    )

    assert stored == 1
    assert provider.calls == [["Readable local chunk."]]
    assert vector_store.prepare_calls == [True]
    records, reindex = vector_store.sync_calls[0]
    assert reindex is True
    assert records[0].identity.chunk_id == "doc_1_chunk_abc123"
    assert len(records[0].values) == 384
