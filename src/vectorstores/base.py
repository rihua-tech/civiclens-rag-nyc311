"""Small outbound dense-vector store contract."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorRecord,
    VectorSyncResult,
)


@runtime_checkable
class VectorStore(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def target(self) -> str: ...

    def prepare_sync(self, *, reindex: bool = False) -> None: ...

    def sync(
        self,
        records: Sequence[VectorRecord],
        *,
        reindex: bool = False,
    ) -> VectorSyncResult: ...

    def query(
        self,
        vector: Sequence[float],
        *,
        candidate_limit: int,
        min_similarity: float,
    ) -> list[VectorMatch]: ...

    def verify(self, identities: Sequence[VectorIdentity]) -> None: ...
