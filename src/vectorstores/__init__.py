"""Dense-vector provider adapters for CivicLens."""

from src.vectorstores.base import VectorStore
from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorRecord,
    VectorStoreCompatibilityError,
    VectorStoreConfigurationError,
    VectorStoreConsistencyError,
    VectorStoreError,
    VectorSyncResult,
)

__all__ = [
    "VectorIdentity",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "VectorStoreCompatibilityError",
    "VectorStoreConfigurationError",
    "VectorStoreConsistencyError",
    "VectorStoreError",
    "VectorSyncResult",
]
