"""Provider-neutral dense-vector identities and results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.embeddings.providers import EmbeddingSpec, validate_embedding


class VectorStoreError(RuntimeError):
    """Base class for sanitized dense-vector backend failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when the selected vector provider is not configured safely."""


class VectorStoreCompatibilityError(VectorStoreError):
    """Raised when a provider cannot serve the active embedding profile."""


class VectorStoreConsistencyError(VectorStoreError):
    """Raised when provider data is stale, incomplete, or malformed."""


@dataclass(frozen=True)
class VectorIdentity:
    chunk_id: str
    document_id: str
    content_hash: str
    document_content_hash: str
    chunking_config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "document_id",
            "content_hash",
            "document_content_hash",
            "chunking_config_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")

    @classmethod
    def from_chunk(cls, chunk: dict) -> "VectorIdentity":
        return cls(
            chunk_id=str(chunk["chunk_id"]),
            document_id=str(chunk["document_id"]),
            content_hash=str(chunk["content_hash"]),
            document_content_hash=str(chunk["document_content_hash"]),
            chunking_config_hash=str(chunk["chunking_config_hash"]),
        )


@dataclass(frozen=True)
class VectorRecord:
    identity: VectorIdentity
    values: tuple[float, ...]

    @classmethod
    def create(
        cls,
        chunk: dict,
        embedding: Sequence[float],
        spec: EmbeddingSpec,
    ) -> "VectorRecord":
        return cls(
            identity=VectorIdentity.from_chunk(chunk),
            values=tuple(validate_embedding(embedding, spec)),
        )


@dataclass(frozen=True)
class VectorMatch:
    identity: VectorIdentity
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise VectorStoreConsistencyError("Vector match score is not finite")
        if self.rank <= 0:
            raise VectorStoreConsistencyError("Vector match rank must be greater than 0")


@dataclass(frozen=True)
class VectorSyncResult:
    provider: str
    target: str
    namespace: str | None
    records_written: int
    verified: bool


def corpus_fingerprint(
    spec: EmbeddingSpec,
    identities: Iterable[VectorIdentity],
) -> str:
    """Return a deterministic fingerprint for one corpus and embedding profile."""

    payload = {
        "embedding": {
            "provider": spec.provider,
            "model": spec.model,
            "dimension": spec.dimension,
        },
        "chunks": [
            {
                "chunk_id": identity.chunk_id,
                "document_id": identity.document_id,
                "content_hash": identity.content_hash,
                "document_content_hash": identity.document_content_hash,
                "chunking_config_hash": identity.chunking_config_hash,
            }
            for identity in sorted(identities, key=lambda item: item.chunk_id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
