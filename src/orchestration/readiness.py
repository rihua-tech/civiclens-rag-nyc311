"""Cheap, read-only readiness checks for the local RAG backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.chunking.chunk_documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    build_chunking_config_hash,
    chunk_documents,
)
from src.common.config import Settings
from src.embeddings.embed_chunks import (
    fetch_embedding_profiles,
    validate_stored_embedding_profiles,
    vector_column_for_spec,
)
from src.embeddings.providers import EmbeddingCompatibilityError, EmbeddingSpec
from src.ingestion.load_documents import load_documents


READINESS_CONNECT_TIMEOUT_SECONDS = 3


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    code: str
    message: str


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    content_hash: str
    chunking_config_hash: str


@dataclass(frozen=True)
class ChunkIdentity:
    chunk_id: str
    document_id: str
    content_hash: str
    document_content_hash: str
    chunking_config_hash: str


@dataclass(frozen=True)
class CorpusIdentity:
    documents: tuple[DocumentIdentity, ...]
    chunks: tuple[ChunkIdentity, ...]


def load_current_corpus_identity() -> CorpusIdentity:
    """Build current identities with the authoritative ingestion/chunking rules."""

    documents = load_documents(ingested_at="readiness-corpus-identity")
    chunks = chunk_documents(documents)
    chunking_config_hash = build_chunking_config_hash(
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )
    return CorpusIdentity(
        documents=tuple(
            sorted(
                (
                    DocumentIdentity(
                        document_id=str(document["document_id"]),
                        content_hash=str(document["content_hash"]),
                        chunking_config_hash=chunking_config_hash,
                    )
                    for document in documents
                ),
                key=lambda item: item.document_id,
            )
        ),
        chunks=tuple(
            sorted(
                (
                    ChunkIdentity(
                        chunk_id=str(chunk["chunk_id"]),
                        document_id=str(chunk["document_id"]),
                        content_hash=str(chunk["content_hash"]),
                        document_content_hash=str(chunk["document_content_hash"]),
                        chunking_config_hash=str(chunk["chunking_config_hash"]),
                    )
                    for chunk in chunks
                ),
                key=lambda item: item.chunk_id,
            )
        ),
    )


def _connect(database_url: str, *, connect_timeout: int):
    import psycopg

    return psycopg.connect(database_url, connect_timeout=connect_timeout)


def check_readiness(
    settings: Settings | None = None,
    connection_factory: Callable[..., Any] | None = None,
    corpus_identity: CorpusIdentity | None = None,
) -> ReadinessResult:
    """Check configuration, schema, and the complete current embedded corpus.

    This check never loads an embedding model, performs retrieval/generation,
    calls OpenAI, or mutates PostgreSQL.
    """

    try:
        active_settings = settings or Settings.from_env()
        current_corpus = corpus_identity or load_current_corpus_identity()
        if not current_corpus.documents or not current_corpus.chunks:
            raise ValueError("Current corpus identity is empty")
        active_spec = EmbeddingSpec(
            provider=active_settings.embedding_provider,
            model=active_settings.embedding_model,
            dimension=active_settings.embedding_dimension,
        )
        vector_column = vector_column_for_spec(active_spec)
    except Exception:
        return ReadinessResult(
            ready=False,
            code="configuration_unavailable",
            message="Local RAG configuration is unavailable.",
        )

    connector = connection_factory or _connect
    try:
        with connector(
            active_settings.database_url,
            connect_timeout=READINESS_CONNECT_TIMEOUT_SECONDS,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        to_regclass('public.documents') IS NOT NULL,
                        to_regclass('public.chunks') IS NOT NULL
                    """
                )
                documents_exist, chunks_exist = cursor.fetchone()
                if not documents_exist or not chunks_exist:
                    return ReadinessResult(
                        ready=False,
                        code="schema_unavailable",
                        message="Required local RAG schema is unavailable.",
                    )

                try:
                    validate_stored_embedding_profiles(
                        fetch_embedding_profiles(cursor),
                        active_spec,
                    )
                except EmbeddingCompatibilityError:
                    return ReadinessResult(
                        ready=False,
                        code="embedding_profile_incompatible",
                        message=(
                            "Stored chunks are incompatible with the configured "
                            "embedding profile."
                        ),
                    )

                expected_documents = {
                    item.document_id: (
                        item.content_hash,
                        item.chunking_config_hash,
                    )
                    for item in current_corpus.documents
                }
                cursor.execute(
                    """
                    SELECT document_id, content_hash, chunking_config_hash
                    FROM documents
                    WHERE document_id = ANY(%s)
                    """,
                    (list(expected_documents),),
                )
                stored_documents = {
                    str(document_id): (
                        str(content_hash),
                        str(chunking_config_hash),
                    )
                    for document_id, content_hash, chunking_config_hash in cursor.fetchall()
                }
                if stored_documents != expected_documents:
                    return ReadinessResult(
                        ready=False,
                        code="corpus_stale",
                        message=(
                            "Stored RAG corpus does not match the current knowledge "
                            "sources."
                        ),
                    )

                expected_chunks = {
                    item.chunk_id: (
                        item.document_id,
                        item.content_hash,
                        item.document_content_hash,
                        item.chunking_config_hash,
                    )
                    for item in current_corpus.chunks
                }
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
                    WHERE c.{vector_column} IS NOT NULL
                      AND vector_dims(c.{vector_column}) = %s
                      AND c.embedding_provider = %s
                      AND c.embedding_model = %s
                      AND c.embedding_dimension = %s
                      AND c.content_hash IS NOT NULL
                      AND c.document_content_hash = d.content_hash
                      AND c.chunking_config_hash = d.chunking_config_hash
                      AND c.chunk_id = ANY(%s)
                    """,
                    (
                        active_spec.dimension,
                        active_spec.provider,
                        active_spec.model,
                        active_spec.dimension,
                        list(expected_chunks),
                    ),
                )
                stored_chunks = {
                    str(chunk_id): (
                        str(document_id),
                        str(content_hash),
                        str(document_content_hash),
                        str(chunking_config_hash),
                    )
                    for (
                        chunk_id,
                        document_id,
                        content_hash,
                        document_content_hash,
                        chunking_config_hash,
                    ) in cursor.fetchall()
                }
    except Exception:
        return ReadinessResult(
            ready=False,
            code="backend_unavailable",
            message="Local PostgreSQL/pgvector backend is unavailable.",
        )

    if set(stored_chunks) != set(expected_chunks):
        return ReadinessResult(
            ready=False,
            code="corpus_incomplete",
            message="Stored RAG corpus is incomplete for the current knowledge sources.",
        )
    if stored_chunks != expected_chunks:
        return ReadinessResult(
            ready=False,
            code="corpus_stale",
            message="Stored RAG corpus does not match the current knowledge sources.",
        )

    return ReadinessResult(
        ready=True,
        code="ready",
        message="Local RAG backend is ready.",
    )
