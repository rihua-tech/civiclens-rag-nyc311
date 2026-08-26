"""Optional Pinecone dense-vector adapter using CivicLens embeddings."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.common.config import PINECONE_VECTOR_STORE, Settings
from src.embeddings.providers import EmbeddingSpec, validate_embedding
from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorRecord,
    VectorStoreCompatibilityError,
    VectorStoreConfigurationError,
    VectorStoreConsistencyError,
    VectorStoreError,
    VectorSyncResult,
    corpus_fingerprint,
)


PINECONE_UPSERT_BATCH_SIZE = 100
PINECONE_FETCH_BATCH_SIZE = 1000


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_host(host: str) -> str:
    return host.removeprefix("https://").removeprefix("http://").rstrip("/").lower()


def _default_client_factory(*, api_key: str, timeout: float, max_retries: int):
    try:
        from pinecone import Pinecone, RetryConfig
    except ImportError as exc:
        raise VectorStoreConfigurationError(
            "Pinecone support requires the optional requirements-pinecone.txt dependency"
        ) from exc
    return Pinecone(
        api_key=api_key,
        timeout=timeout,
        retry_config=RetryConfig(max_retries=max_retries),
    )


class PineconeVectorStore:
    """Synchronize one deterministic corpus namespace in an existing index."""

    def __init__(
        self,
        settings: Settings,
        spec: EmbeddingSpec,
        identities: Sequence[VectorIdentity],
        *,
        client_factory: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        required = {
            "PINECONE_API_KEY": settings.pinecone_api_key,
            "PINECONE_INDEX_NAME": settings.pinecone_index_name,
            "PINECONE_INDEX_HOST": settings.pinecone_index_host,
            "PINECONE_NAMESPACE_PREFIX": settings.pinecone_namespace_prefix,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise VectorStoreConfigurationError(
                "Pinecone configuration is incomplete; missing " + ", ".join(missing)
            )
        if not identities:
            raise VectorStoreConfigurationError(
                "Pinecone requires the deterministic current-corpus identity"
            )
        if len({identity.chunk_id for identity in identities}) != len(identities):
            raise VectorStoreConfigurationError(
                "Current-corpus identity contains duplicate chunk IDs"
            )

        self._settings = settings
        self._spec = spec
        self._expected = {identity.chunk_id: identity for identity in identities}
        self._fingerprint = corpus_fingerprint(spec, identities)
        self._namespace = (
            f"{settings.pinecone_namespace_prefix}-{self._fingerprint[:24]}"
        )
        self._client_factory = client_factory or _default_client_factory
        self._sleeper = sleeper
        self._client: Any | None = None
        self._index: Any | None = None

    @property
    def provider_name(self) -> str:
        return PINECONE_VECTOR_STORE

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def target(self) -> str:
        return f"pinecone:{self._settings.pinecone_index_name}/{self._namespace}"

    def _clients(self) -> tuple[Any, Any]:
        if self._client is None:
            self._client = self._client_factory(
                api_key=self._settings.pinecone_api_key,
                timeout=self._settings.pinecone_timeout_seconds,
                max_retries=self._settings.pinecone_max_retries,
            )
        if self._index is None:
            self._index = self._client.index(
                host=self._settings.pinecone_index_host,
            )
        return self._client, self._index

    def _validate_index(self) -> None:
        client, _ = self._clients()
        description = client.indexes.describe(self._settings.pinecone_index_name)
        dimension = _field(description, "dimension")
        metric = str(_field(description, "metric", "")).lower()
        host = str(_field(description, "host", ""))
        if int(dimension or 0) != self._spec.dimension:
            raise VectorStoreCompatibilityError(
                "Pinecone index dimension is incompatible with the active embedding profile"
            )
        if metric != "cosine":
            raise VectorStoreCompatibilityError(
                "Pinecone index metric must be cosine"
            )
        if _normalize_host(host) != _normalize_host(self._settings.pinecone_index_host):
            raise VectorStoreCompatibilityError(
                "Pinecone index host does not match PINECONE_INDEX_HOST"
            )
        status = _field(description, "status")
        ready = _field(status, "ready") if status is not None else None
        state = str(_field(status, "state", "")).lower() if status is not None else ""
        if ready is False or (state and state != "ready"):
            raise VectorStoreError("Pinecone index is not ready")

    def prepare_sync(self, *, reindex: bool = False) -> None:
        del reindex  # Pinecone namespaces are deterministic; no index lifecycle is managed.
        try:
            self._validate_index()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone index validation failed") from exc

    def _metadata(self, identity: VectorIdentity) -> dict[str, str | int]:
        return {
            "corpus_fingerprint": self._fingerprint,
            "document_id": identity.document_id,
            "content_hash": identity.content_hash,
            "document_content_hash": identity.document_content_hash,
            "chunking_config_hash": identity.chunking_config_hash,
            "embedding_provider": self._spec.provider,
            "embedding_model": self._spec.model,
            "embedding_dimension": self._spec.dimension,
        }

    def sync(
        self,
        records: Sequence[VectorRecord],
        *,
        reindex: bool = False,
    ) -> VectorSyncResult:
        del reindex
        record_ids = {record.identity.chunk_id for record in records}
        if (
            len(records) != len(self._expected)
            or record_ids != set(self._expected)
            or any(
                self._expected[record.identity.chunk_id] != record.identity
                for record in records
            )
        ):
            raise VectorStoreConsistencyError(
                "Pinecone sync records do not match the current corpus"
            )
        try:
            self._validate_index()
            _, index = self._clients()
            for start in range(0, len(records), PINECONE_UPSERT_BATCH_SIZE):
                batch = records[start : start + PINECONE_UPSERT_BATCH_SIZE]
                response = index.upsert(
                    vectors=[
                        {
                            "id": record.identity.chunk_id,
                            "values": validate_embedding(record.values, self._spec),
                            "metadata": self._metadata(record.identity),
                        }
                        for record in batch
                    ],
                    namespace=self._namespace,
                    timeout=self._settings.pinecone_timeout_seconds,
                )
                if bool(_field(response, "has_errors", False)):
                    raise VectorStoreConsistencyError(
                        "Pinecone reported a partial vector upsert"
                    )
                upserted_count = _field(response, "upserted_count")
                if upserted_count is not None and int(upserted_count) != len(batch):
                    raise VectorStoreConsistencyError(
                        "Pinecone did not accept the complete vector batch"
                    )
            self._verify_query_visibility(records)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone synchronization failed") from exc
        return VectorSyncResult(
            provider=self.provider_name,
            target=self.target,
            namespace=self._namespace,
            records_written=len(records),
            verified=True,
        )

    def _verify_query_visibility(self, records: Sequence[VectorRecord]) -> None:
        _, index = self._clients()
        first_vector = validate_embedding(records[0].values, self._spec)
        last_error: VectorStoreConsistencyError | None = None
        for attempt in range(self._settings.pinecone_sync_max_attempts):
            response = index.query(
                vector=first_vector,
                top_k=len(records),
                namespace=self._namespace,
                filter={"corpus_fingerprint": {"$eq": self._fingerprint}},
                include_values=False,
                include_metadata=True,
                timeout=self._settings.pinecone_timeout_seconds,
            )
            try:
                matches = self._parse_matches(response, min_similarity=-1.0)
                if {match.identity.chunk_id for match in matches} != set(self._expected):
                    raise VectorStoreConsistencyError(
                        "Pinecone current corpus is not fully query-visible"
                    )
                return
            except VectorStoreConsistencyError as exc:
                last_error = exc
                if attempt + 1 < self._settings.pinecone_sync_max_attempts:
                    self._sleeper(min(0.25 * (2**attempt), 1.0))
        raise last_error or VectorStoreConsistencyError(
            "Pinecone current corpus is not query-visible"
        )

    def query(
        self,
        vector: Sequence[float],
        *,
        candidate_limit: int,
        min_similarity: float,
    ) -> list[VectorMatch]:
        try:
            self.verify(tuple(self._expected.values()))
            _, index = self._clients()
            response = index.query(
                vector=validate_embedding(vector, self._spec),
                top_k=candidate_limit,
                namespace=self._namespace,
                filter={"corpus_fingerprint": {"$eq": self._fingerprint}},
                include_values=False,
                include_metadata=True,
                timeout=self._settings.pinecone_timeout_seconds,
            )
            return self._parse_matches(
                response,
                min_similarity=min_similarity,
                require_nonempty=True,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone semantic retrieval failed") from exc

    def _parse_matches(
        self,
        response: Any,
        *,
        min_similarity: float,
        require_nonempty: bool = False,
    ) -> list[VectorMatch]:
        raw_matches = _field(response, "matches")
        if raw_matches is None or isinstance(raw_matches, (str, bytes, Mapping)):
            raise VectorStoreConsistencyError("Pinecone returned malformed matches")
        raw_matches = list(raw_matches)
        if require_nonempty and not raw_matches:
            raise VectorStoreConsistencyError(
                "Pinecone returned no matches for a verified non-empty corpus"
            )
        parsed: list[tuple[VectorIdentity, float]] = []
        seen: set[str] = set()
        for raw_match in raw_matches:
            chunk_id = str(_field(raw_match, "id", ""))
            metadata = _field(raw_match, "metadata")
            score = _field(raw_match, "score")
            if (
                not chunk_id
                or chunk_id in seen
                or not isinstance(metadata, Mapping)
                or score is None
            ):
                raise VectorStoreConsistencyError("Pinecone returned a malformed match")
            seen.add(chunk_id)
            expected = self._expected.get(chunk_id)
            if expected is None:
                raise VectorStoreConsistencyError(
                    "Pinecone returned a chunk outside the current corpus"
                )
            identity = VectorIdentity(
                chunk_id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                content_hash=str(metadata.get("content_hash", "")),
                document_content_hash=str(metadata.get("document_content_hash", "")),
                chunking_config_hash=str(metadata.get("chunking_config_hash", "")),
            )
            if (
                identity != expected
                or metadata.get("corpus_fingerprint") != self._fingerprint
                or metadata.get("embedding_provider") != self._spec.provider
                or metadata.get("embedding_model") != self._spec.model
                or int(metadata.get("embedding_dimension", 0)) != self._spec.dimension
            ):
                raise VectorStoreConsistencyError(
                    "Pinecone match is incompatible with the current corpus"
                )
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise VectorStoreConsistencyError(
                    "Pinecone returned a non-finite similarity score"
                )
            if not -1.000001 <= numeric_score <= 1.000001:
                raise VectorStoreConsistencyError(
                    "Pinecone returned an invalid cosine similarity score"
                )
            if numeric_score >= min_similarity:
                parsed.append((identity, numeric_score))
        parsed.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            VectorMatch(identity=identity, score=score, rank=rank)
            for rank, (identity, score) in enumerate(parsed, start=1)
        ]

    def verify(self, identities: Sequence[VectorIdentity]) -> None:
        expected = {identity.chunk_id: identity for identity in identities}
        if expected != self._expected:
            raise VectorStoreConsistencyError(
                "Readiness corpus does not match the Pinecone namespace"
            )
        try:
            self._validate_index()
            _, index = self._clients()
            fetched: dict[str, Any] = {}
            identifiers = list(expected)
            for start in range(0, len(identifiers), PINECONE_FETCH_BATCH_SIZE):
                response = index.fetch(
                    ids=identifiers[start : start + PINECONE_FETCH_BATCH_SIZE],
                    namespace=self._namespace,
                    timeout=self._settings.pinecone_timeout_seconds,
                )
                vectors = _field(response, "vectors")
                if not isinstance(vectors, Mapping):
                    raise VectorStoreConsistencyError(
                        "Pinecone returned malformed readiness data"
                    )
                fetched.update({str(key): value for key, value in vectors.items()})
            if set(fetched) != set(expected):
                raise VectorStoreConsistencyError(
                    "Pinecone is incomplete for the current corpus"
                )
            for chunk_id, vector in fetched.items():
                metadata = _field(vector, "metadata")
                if not isinstance(metadata, Mapping):
                    raise VectorStoreConsistencyError(
                        "Pinecone readiness metadata is malformed"
                    )
                identity = expected[chunk_id]
                if self._metadata(identity) != dict(metadata):
                    raise VectorStoreConsistencyError(
                        "Pinecone readiness metadata is incompatible"
                    )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone readiness validation failed") from exc
