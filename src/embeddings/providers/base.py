"""Shared embedding-provider contract and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class EmbeddingCompatibilityError(ValueError):
    """Raised when configured, generated, or stored embeddings are incompatible."""


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("Embedding provider name must not be empty")
        if not self.model.strip():
            raise ValueError("Embedding model name must not be empty")
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be greater than 0")


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_embedding(embedding: Sequence[float], spec: EmbeddingSpec) -> list[float]:
    values = [float(value) for value in embedding]
    if len(values) != spec.dimension:
        raise EmbeddingCompatibilityError(
            f"Embedding provider {spec.provider!r} model {spec.model!r} returned "
            f"{len(values)} dimensions; expected {spec.dimension}. Reconfigure the "
            "dimension or perform the documented full re-embedding/reindex procedure."
        )
    return values
