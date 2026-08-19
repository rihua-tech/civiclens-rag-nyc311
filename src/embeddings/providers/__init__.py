"""Configurable embedding providers for CivicLens retrieval."""

from src.embeddings.providers.base import (
    EmbeddingCompatibilityError,
    EmbeddingProvider,
    EmbeddingSpec,
    validate_embedding,
)
from src.embeddings.providers.factory import create_embedding_provider

__all__ = [
    "EmbeddingCompatibilityError",
    "EmbeddingProvider",
    "EmbeddingSpec",
    "create_embedding_provider",
    "validate_embedding",
]
