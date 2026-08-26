"""Server-side dense-vector provider selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.common.config import (
    PGVECTOR_VECTOR_STORE,
    PINECONE_VECTOR_STORE,
    Settings,
)
from src.embeddings.providers import EmbeddingSpec
from src.vectorstores.base import VectorStore
from src.vectorstores.models import VectorIdentity, VectorStoreConfigurationError
from src.vectorstores.pgvector_store import PgVectorStore


def create_vector_store(
    settings: Settings,
    spec: EmbeddingSpec,
    identities: Sequence[VectorIdentity],
    *,
    pg_connection_factory: Callable[[str], Any] | None = None,
) -> VectorStore:
    provider = settings.vector_store_provider.strip().lower()
    if provider == PGVECTOR_VECTOR_STORE:
        return PgVectorStore(
            settings,
            spec,
            connection_factory=pg_connection_factory,
        )
    if provider == PINECONE_VECTOR_STORE:
        from src.vectorstores.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(settings, spec, identities)
    raise VectorStoreConfigurationError(
        "Configured dense-vector provider is not supported"
    )
