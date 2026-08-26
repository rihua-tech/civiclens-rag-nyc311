"""Default pgvector dense-vector adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.common.config import PGVECTOR_VECTOR_STORE, SEMANTIC_PROVIDER, Settings
from src.embeddings.providers import (
    EmbeddingCompatibilityError,
    EmbeddingSpec,
    validate_embedding,
)
from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorRecord,
    VectorStoreConsistencyError,
    VectorSyncResult,
)


LEGACY_VECTOR_COLUMN = "embedding"
SEMANTIC_VECTOR_COLUMN = "semantic_embedding"
LEGACY_VECTOR_DIMENSION = 1536


def vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".10g") for value in embedding) + "]"


def vector_column_for_spec(spec: EmbeddingSpec) -> str:
    if spec.provider == SEMANTIC_PROVIDER:
        if spec.dimension != 384:
            raise EmbeddingCompatibilityError(
                "The local semantic pgvector column is vector(384), but provider "
                f"{spec.provider!r} model {spec.model!r} is configured for "
                f"{spec.dimension} dimensions. Update the narrowly scoped Issue 9 schema "
                "before running the documented full reindex."
            )
        return SEMANTIC_VECTOR_COLUMN
    if spec.dimension != LEGACY_VECTOR_DIMENSION:
        raise EmbeddingCompatibilityError(
            f"The backward-compatible pgvector column is vector({LEGACY_VECTOR_DIMENSION}), "
            f"but provider {spec.provider!r} model {spec.model!r} is configured for "
            f"{spec.dimension} dimensions."
        )
    return LEGACY_VECTOR_COLUMN


def fetch_embedding_profiles(cursor: Any) -> set[tuple[str | None, str | None, int | None]]:
    cursor.execute(
        """
        SELECT DISTINCT
            c.embedding_provider,
            c.embedding_model,
            c.embedding_dimension
        FROM chunks AS c
        INNER JOIN documents AS d ON d.document_id = c.document_id
        WHERE (c.embedding IS NOT NULL OR c.semantic_embedding IS NOT NULL)
          AND c.content_hash IS NOT NULL
          AND c.document_content_hash = d.content_hash
          AND c.chunking_config_hash = d.chunking_config_hash
        """
    )
    return {tuple(row) for row in cursor.fetchall()}


def validate_stored_embedding_profiles(
    profiles: set[tuple[str | None, str | None, int | None]],
    active_spec: EmbeddingSpec,
) -> None:
    expected = (active_spec.provider, active_spec.model, active_spec.dimension)
    incompatible = profiles.difference({expected})
    if incompatible:
        formatted = ", ".join(repr(profile) for profile in sorted(incompatible, key=repr))
        raise EmbeddingCompatibilityError(
            "Stored current chunks use an incompatible or unrecorded embedding profile: "
            f"{formatted}. Active profile is {expected!r}. Run "
            "`python -m src.embeddings.embed_chunks --reindex` to clear all old vectors "
            "and rebuild them with one provider/model/dimension."
        )


def clear_stored_embeddings(cursor: Any) -> None:
    cursor.execute(
        """
        UPDATE chunks
        SET embedding = NULL,
            semantic_embedding = NULL,
            embedding_provider = NULL,
            embedding_model = NULL,
            embedding_dimension = NULL
        WHERE embedding IS NOT NULL
           OR semantic_embedding IS NOT NULL
           OR embedding_provider IS NOT NULL
           OR embedding_model IS NOT NULL
           OR embedding_dimension IS NOT NULL
        """
    )


def rebuild_retrieval_indexes(cursor: Any) -> None:
    cursor.execute("REINDEX INDEX idx_chunks_semantic_embedding_hnsw")
    cursor.execute("REINDEX INDEX idx_chunks_search_vector_gin")
    cursor.execute("ANALYZE chunks")


def _connect(database_url: str):
    import psycopg

    return psycopg.connect(database_url)


class PgVectorStore:
    """Persist and query dense vectors in canonical PostgreSQL."""

    def __init__(
        self,
        settings: Settings,
        spec: EmbeddingSpec,
        *,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._spec = spec
        self._vector_column = vector_column_for_spec(spec)
        self._connect = connection_factory or _connect

    @property
    def provider_name(self) -> str:
        return PGVECTOR_VECTOR_STORE

    @property
    def target(self) -> str:
        return self._settings.safe_database_target

    def prepare_sync(self, *, reindex: bool = False) -> None:
        if reindex:
            return
        with self._connect(self._settings.database_url) as connection:
            with connection.cursor() as cursor:
                validate_stored_embedding_profiles(
                    fetch_embedding_profiles(cursor),
                    self._spec,
                )

    def sync(
        self,
        records: Sequence[VectorRecord],
        *,
        reindex: bool = False,
    ) -> VectorSyncResult:
        if len({record.identity.chunk_id for record in records}) != len(records):
            raise VectorStoreConsistencyError(
                "pgvector sync records contain duplicate chunk IDs"
            )
        with self._connect(self._settings.database_url) as connection:
            with connection.cursor() as cursor:
                if reindex:
                    clear_stored_embeddings(cursor)
                else:
                    validate_stored_embedding_profiles(
                        fetch_embedding_profiles(cursor),
                        self._spec,
                    )
                for record in records:
                    values = validate_embedding(record.values, self._spec)
                    cursor.execute(
                        f"""
                        UPDATE chunks
                        SET {self._vector_column} = %s::vector,
                            {self._other_vector_column} = NULL,
                            embedding_provider = %s,
                            embedding_model = %s,
                            embedding_dimension = %s
                        WHERE chunk_id = %s
                        """,
                        (
                            vector_literal(values),
                            self._spec.provider,
                            self._spec.model,
                            self._spec.dimension,
                            record.identity.chunk_id,
                        ),
                    )
                if reindex:
                    rebuild_retrieval_indexes(cursor)
        self.verify([record.identity for record in records])
        return VectorSyncResult(
            provider=self.provider_name,
            target=self.target,
            namespace=None,
            records_written=len(records),
            verified=True,
        )

    @property
    def _other_vector_column(self) -> str:
        if self._vector_column == LEGACY_VECTOR_COLUMN:
            return SEMANTIC_VECTOR_COLUMN
        return LEGACY_VECTOR_COLUMN

    def query(
        self,
        vector: Sequence[float],
        *,
        candidate_limit: int,
        min_similarity: float,
    ) -> list[VectorMatch]:
        values = validate_embedding(vector, self._spec)
        query_vector = vector_literal(values)
        with self._connect(self._settings.database_url) as connection:
            with connection.cursor() as cursor:
                validate_stored_embedding_profiles(
                    fetch_embedding_profiles(cursor),
                    self._spec,
                )
                cursor.execute(
                    f"""
                    SELECT
                        chunk_id,
                        document_id,
                        content_hash,
                        document_content_hash,
                        chunking_config_hash,
                        semantic_score
                    FROM (
                        SELECT
                            c.chunk_id,
                            c.document_id,
                            c.content_hash,
                            c.document_content_hash,
                            c.chunking_config_hash,
                            1 - (c.{self._vector_column} <=> %s::vector) AS semantic_score
                        FROM chunks AS c
                        INNER JOIN documents AS d ON d.document_id = c.document_id
                        WHERE c.{self._vector_column} IS NOT NULL
                          AND c.embedding_provider = %s
                          AND c.embedding_model = %s
                          AND c.embedding_dimension = %s
                          AND vector_dims(c.{self._vector_column}) = %s
                          AND c.content_hash IS NOT NULL
                          AND c.document_content_hash = d.content_hash
                          AND c.chunking_config_hash = d.chunking_config_hash
                        ORDER BY c.{self._vector_column} <=> %s::vector, c.chunk_id
                        LIMIT %s
                    ) AS scored_chunks
                    WHERE semantic_score >= %s
                    ORDER BY semantic_score DESC, chunk_id
                    """,
                    (
                        query_vector,
                        self._spec.provider,
                        self._spec.model,
                        self._spec.dimension,
                        self._spec.dimension,
                        query_vector,
                        candidate_limit,
                        min_similarity,
                    ),
                )
                rows = cursor.fetchall()
        matches = [
            VectorMatch(
                identity=VectorIdentity(
                    chunk_id=str(row[0]),
                    document_id=str(row[1]),
                    content_hash=str(row[2]),
                    document_content_hash=str(row[3]),
                    chunking_config_hash=str(row[4]),
                ),
                score=float(row[5]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]
        if len({match.identity.chunk_id for match in matches}) != len(matches):
            raise VectorStoreConsistencyError("pgvector returned duplicate chunk IDs")
        return matches

    def verify(self, identities: Sequence[VectorIdentity]) -> None:
        expected = {identity.chunk_id: identity for identity in identities}
        if not expected:
            raise VectorStoreConsistencyError("Current vector corpus is empty")
        with self._connect(self._settings.database_url) as connection:
            with connection.cursor() as cursor:
                validate_stored_embedding_profiles(
                    fetch_embedding_profiles(cursor),
                    self._spec,
                )
                cursor.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.document_id,
                        c.content_hash,
                        c.document_content_hash,
                        c.chunking_config_hash
                    FROM chunks AS c
                    INNER JOIN documents AS d ON d.document_id = c.document_id
                    WHERE c.{self._vector_column} IS NOT NULL
                      AND vector_dims(c.{self._vector_column}) = %s
                      AND c.embedding_provider = %s
                      AND c.embedding_model = %s
                      AND c.embedding_dimension = %s
                      AND c.content_hash IS NOT NULL
                      AND c.document_content_hash = d.content_hash
                      AND c.chunking_config_hash = d.chunking_config_hash
                      AND c.chunk_id = ANY(%s)
                    """,
                    (
                        self._spec.dimension,
                        self._spec.provider,
                        self._spec.model,
                        self._spec.dimension,
                        list(expected),
                    ),
                )
                rows = cursor.fetchall()
        stored = {
            str(row[0]): VectorIdentity(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                content_hash=str(row[2]),
                document_content_hash=str(row[3]),
                chunking_config_hash=str(row[4]),
            )
            for row in rows
        }
        if set(stored) != set(expected):
            raise VectorStoreConsistencyError(
                "pgvector is incomplete for the current PostgreSQL corpus"
            )
        if stored != expected:
            raise VectorStoreConsistencyError(
                "pgvector is incompatible with the current PostgreSQL corpus"
            )
