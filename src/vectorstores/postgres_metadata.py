"""Canonical PostgreSQL document and chunk metadata persistence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from src.common.config import Settings


def ensure_schema(connection: Any, schema_path: Path) -> None:
    schema_sql = schema_path.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)


def upsert_document(cursor: Any, chunk: dict) -> None:
    cursor.execute(
        """
        INSERT INTO documents (
            document_id,
            source_name,
            source_type,
            source_category,
            source_path,
            source_url,
            source_version,
            source_retrieved_at,
            content_hash,
            chunking_config_hash,
            ingested_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            source_type = EXCLUDED.source_type,
            source_category = EXCLUDED.source_category,
            source_path = EXCLUDED.source_path,
            source_url = EXCLUDED.source_url,
            source_version = EXCLUDED.source_version,
            source_retrieved_at = EXCLUDED.source_retrieved_at,
            content_hash = EXCLUDED.content_hash,
            chunking_config_hash = EXCLUDED.chunking_config_hash,
            ingested_at = EXCLUDED.ingested_at
        """,
        (
            chunk["document_id"],
            chunk.get("source_name") or "unknown",
            chunk.get("source_type"),
            chunk.get("source_category"),
            chunk.get("source_path"),
            chunk.get("source_url"),
            chunk.get("source_version"),
            chunk.get("source_retrieved_at"),
            chunk.get("document_content_hash"),
            chunk.get("chunking_config_hash"),
            chunk.get("ingested_at"),
        ),
    )


def upsert_chunk_metadata(cursor: Any, chunk: dict) -> None:
    """Upsert canonical chunk data without coupling it to vector persistence."""

    cursor.execute(
        """
        INSERT INTO chunks (
            chunk_id,
            document_id,
            chunk_text,
            chunk_index,
            source_name,
            source_type,
            source_category,
            source_path,
            source_url,
            source_version,
            source_retrieved_at,
            section_title,
            heading_path,
            word_count,
            content_hash,
            document_content_hash,
            chunking_config_hash,
            ingested_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            chunk_text = EXCLUDED.chunk_text,
            chunk_index = EXCLUDED.chunk_index,
            source_name = EXCLUDED.source_name,
            source_type = EXCLUDED.source_type,
            source_category = EXCLUDED.source_category,
            source_path = EXCLUDED.source_path,
            source_url = EXCLUDED.source_url,
            source_version = EXCLUDED.source_version,
            source_retrieved_at = EXCLUDED.source_retrieved_at,
            section_title = EXCLUDED.section_title,
            heading_path = EXCLUDED.heading_path,
            word_count = EXCLUDED.word_count,
            content_hash = EXCLUDED.content_hash,
            document_content_hash = EXCLUDED.document_content_hash,
            chunking_config_hash = EXCLUDED.chunking_config_hash,
            ingested_at = EXCLUDED.ingested_at
        """,
        (
            chunk["chunk_id"],
            chunk["document_id"],
            chunk["chunk_text"],
            chunk.get("chunk_index"),
            chunk.get("source_name"),
            chunk.get("source_type"),
            chunk.get("source_category"),
            chunk.get("source_path"),
            chunk.get("source_url"),
            chunk.get("source_version"),
            chunk.get("source_retrieved_at"),
            chunk.get("section_title"),
            chunk.get("heading_path") or [],
            chunk.get("word_count", chunk.get("token_count")),
            chunk.get("content_hash"),
            chunk.get("document_content_hash"),
            chunk.get("chunking_config_hash"),
            chunk.get("ingested_at"),
        ),
    )


def _connect(database_url: str):
    import psycopg

    return psycopg.connect(database_url)


def persist_postgres_metadata(
    chunks: Iterable[dict],
    settings: Settings,
    *,
    schema_path: Path | None = None,
    connection_factory: Callable[[str], Any] | None = None,
) -> int:
    """Persist the authoritative corpus before any vector-provider operation."""

    records = list(chunks)
    connector = connection_factory or _connect
    with connector(settings.database_url) as connection:
        if schema_path is not None:
            ensure_schema(connection, schema_path)
        with connection.cursor() as cursor:
            for chunk in records:
                upsert_document(cursor, chunk)
                upsert_chunk_metadata(cursor, chunk)
    return len(records)
