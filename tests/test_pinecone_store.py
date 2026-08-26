from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.config import Settings
from src.embeddings.providers import EmbeddingSpec
from src.vectorstores.factory import create_vector_store
from src.vectorstores.models import (
    VectorIdentity,
    VectorRecord,
    VectorStoreCompatibilityError,
    VectorStoreConsistencyError,
    VectorStoreError,
)
from src.vectorstores.pinecone_store import PineconeVectorStore


def _settings(**overrides) -> Settings:
    values = {
        "database_url": "postgresql://local/test",
        "embedding_model": "fake-model",
        "use_openai_embeddings": False,
        "use_openai_answers": False,
        "openai_api_key": "",
        "embedding_provider": "fake",
        "embedding_dimension": 3,
        "vector_store_provider": "pinecone",
        "pinecone_api_key": "never-log-this-key",
        "pinecone_index_name": "civiclens-test",
        "pinecone_index_host": "civiclens-test.svc.pinecone.io",
        "pinecone_namespace_prefix": "issue18-test",
        "pinecone_timeout_seconds": 0.5,
        "pinecone_max_retries": 0,
        "pinecone_sync_max_attempts": 2,
    }
    values.update(overrides)
    return Settings(**values)


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec("fake", "fake-model", 3)


def _identity(chunk_id: str) -> VectorIdentity:
    return VectorIdentity(
        chunk_id=chunk_id,
        document_id="doc_1",
        content_hash=f"sha256:{chunk_id}",
        document_content_hash="sha256:document",
        chunking_config_hash="sha256:chunking",
    )


class FakeIndex:
    def __init__(self):
        self.namespaces = {}
        self.calls = []

    def upsert(self, *, vectors, namespace, timeout):
        self.calls.append(("upsert", namespace, timeout))
        stored = self.namespaces.setdefault(namespace, {})
        for vector in vectors:
            stored[vector["id"]] = vector
        return SimpleNamespace(upserted_count=len(vectors), has_errors=False)

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        stored = self.namespaces.get(kwargs["namespace"], {})
        matches = [
            {
                "id": chunk_id,
                "score": 0.9 - (index * 0.1),
                "metadata": vector["metadata"],
            }
            for index, (chunk_id, vector) in enumerate(
                sorted(stored.items(), reverse=True)
            )
        ]
        return {"matches": matches[: kwargs["top_k"]]}

    def fetch(self, *, ids, namespace, timeout):
        self.calls.append(("fetch", namespace, timeout))
        stored = self.namespaces.get(namespace, {})
        return {"vectors": {chunk_id: stored[chunk_id] for chunk_id in ids if chunk_id in stored}}


class FakeIndexes:
    def __init__(self, *, dimension=3, metric="cosine", host=None, error=None):
        self.dimension = dimension
        self.metric = metric
        self.host = host or "civiclens-test.svc.pinecone.io"
        self.error = error

    def describe(self, name):
        assert name == "civiclens-test"
        if self.error is not None:
            raise self.error
        return {
            "dimension": self.dimension,
            "metric": self.metric,
            "host": self.host,
            "status": {"ready": True, "state": "ready"},
        }


class FakeClient:
    def __init__(self, indexes=None, index=None):
        self.indexes = indexes or FakeIndexes()
        self.index_value = index or FakeIndex()
        self.hosts = []

    def index(self, *, host):
        self.hosts.append(host)
        return self.index_value


def _store(client, identities):
    return PineconeVectorStore(
        _settings(),
        _spec(),
        identities,
        client_factory=lambda **kwargs: client,
        sleeper=lambda _: None,
    )


def test_pinecone_sync_uses_stable_ids_civiclens_vectors_and_verified_namespace():
    identities = [_identity("chunk_b"), _identity("chunk_a")]
    client = FakeClient()
    store = _store(client, identities)
    records = [
        VectorRecord(identity=identity, values=(0.1, 0.2, 0.3))
        for identity in identities
    ]

    store.prepare_sync()
    result = store.sync(records)
    matches = store.query(
        [0.1, 0.2, 0.3],
        candidate_limit=2,
        min_similarity=0.0,
    )

    namespace_vectors = client.index_value.namespaces[store.namespace]
    assert set(namespace_vectors) == {"chunk_a", "chunk_b"}
    assert result.provider == "pinecone"
    assert result.namespace == store.namespace
    assert result.verified is True
    assert store.namespace.startswith("issue18-test-")
    assert "never-log-this-key" not in store.target
    assert all("chunk_text" not in vector["metadata"] for vector in namespace_vectors.values())
    assert [match.identity.chunk_id for match in matches] == ["chunk_b", "chunk_a"]
    query_call = client.index_value.calls[-1][1]
    assert query_call["top_k"] == 2
    assert query_call["include_metadata"] is True
    assert query_call["timeout"] == 0.5


@pytest.mark.parametrize(
    ("indexes", "message"),
    [
        (FakeIndexes(dimension=4), "dimension"),
        (FakeIndexes(metric="dotproduct"), "cosine"),
        (FakeIndexes(host="wrong.svc.pinecone.io"), "host"),
    ],
)
def test_pinecone_rejects_incompatible_index_configuration(indexes, message):
    store = _store(FakeClient(indexes=indexes), [_identity("chunk_1")])

    with pytest.raises(VectorStoreCompatibilityError, match=message):
        store.prepare_sync()


def test_pinecone_rejects_stale_or_malformed_matches():
    identity = _identity("chunk_1")
    client = FakeClient()
    store = _store(client, [identity])
    client.index_value.namespaces[store.namespace] = {
        identity.chunk_id: {
            "id": identity.chunk_id,
            "values": [0.1, 0.2, 0.3],
            "metadata": {
                **store._metadata(identity),
                "content_hash": "sha256:stale",
            },
        }
    }

    with pytest.raises(VectorStoreConsistencyError, match="incompatible"):
        store.query([0.1, 0.2, 0.3], candidate_limit=1, min_similarity=0.0)


def test_pinecone_query_rejects_a_partially_synchronized_namespace():
    identities = [_identity("chunk_1"), _identity("chunk_2")]
    client = FakeClient()
    store = _store(client, identities)
    identity = identities[0]
    client.index_value.namespaces[store.namespace] = {
        identity.chunk_id: {
            "id": identity.chunk_id,
            "values": [0.1, 0.2, 0.3],
            "metadata": store._metadata(identity),
        }
    }

    with pytest.raises(VectorStoreConsistencyError, match="incomplete"):
        store.query([0.1, 0.2, 0.3], candidate_limit=2, min_similarity=0.0)


def test_pinecone_errors_are_sanitized_and_never_fall_back_to_pgvector():
    settings = _settings()
    identity = _identity("chunk_1")
    selected = create_vector_store(settings, _spec(), [identity])
    assert isinstance(selected, PineconeVectorStore)

    store = _store(
        FakeClient(indexes=FakeIndexes(error=RuntimeError("never-log-this-key"))),
        [identity],
    )
    with pytest.raises(VectorStoreError) as exc_info:
        store.prepare_sync()

    assert "never-log-this-key" not in str(exc_info.value)
    assert "validation failed" in str(exc_info.value)


def test_real_pinecone_sdk_imports_when_optional_lane_installs_it():
    pytest.importorskip("pinecone")
