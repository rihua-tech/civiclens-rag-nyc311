"""Generate embeddings for local chunks and store them in PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from src.common.config import (
    DETERMINISTIC_DIMENSION,
    DETERMINISTIC_MODEL,
    DETERMINISTIC_PROVIDER,
    SEMANTIC_PROVIDER,
    Settings,
)
from src.embeddings.providers import (
    EmbeddingCompatibilityError,
    EmbeddingProvider,
    EmbeddingSpec,
    create_embedding_provider,
    validate_embedding,
)
from src.embeddings.providers.deterministic import (
    EMBEDDING_STOPWORDS as EMBEDDING_STOPWORDS,
    TOKEN_PATTERN as TOKEN_PATTERN,
    deterministic_embedding,
    tokenize_for_embedding as tokenize_for_embedding,
)
from src.embeddings.providers.openai_provider import OpenAIEmbeddingProvider


DEFAULT_INPUT_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_SCHEMA_PATH = Path("sql/schema.sql")
LOCAL_EMBEDDING_MODEL = DETERMINISTIC_MODEL
EMBEDDING_DIMENSIONS = DETERMINISTIC_DIMENSION
LEGACY_VECTOR_COLUMN = "embedding"
SEMANTIC_VECTOR_COLUMN = "semantic_embedding"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def load_chunks(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input chunks file not found: {input_path}")

    chunks: list[dict] = []
    with input_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def local_deterministic_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    return deterministic_embedding(text, dimensions)


def openai_embedding(text: str, settings: Settings) -> list[float]:
    model = settings.embedding_model
    if model == LOCAL_EMBEDDING_MODEL:
        model = "text-embedding-3-small"
    dimension = settings.embedding_dimension or EMBEDDING_DIMENSIONS
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model_name=model,
        dimension=dimension,
    ).embed(text)


def generate_embedding(
    text: str,
    settings: Settings | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[float]:
    active_settings = settings or Settings.from_env()
    active_provider = provider or create_embedding_provider(active_settings)
    return active_provider.embed(text)


def vector_literal(embedding: Iterable[float]) -> str:
    return "[" + ",".join(format(float(value), ".10g") for value in embedding) + "]"


def vector_column_for_spec(spec: EmbeddingSpec) -> str:
    if spec.provider == SEMANTIC_PROVIDER:
        if spec.dimension != 384:
            raise EmbeddingCompatibilityError(
                f"The local semantic pgvector column is vector(384), but provider "
                f"{spec.provider!r} model {spec.model!r} is configured for "
                f"{spec.dimension} dimensions. Update the narrowly scoped Issue 9 schema "
                "before running the documented full reindex."
            )
        return SEMANTIC_VECTOR_COLUMN
    if spec.dimension != EMBEDDING_DIMENSIONS:
        raise EmbeddingCompatibilityError(
            f"The backward-compatible pgvector column is vector({EMBEDDING_DIMENSIONS}), "
            f"but provider {spec.provider!r} model {spec.model!r} is configured for "
            f"{spec.dimension} dimensions."
        )
    return LEGACY_VECTOR_COLUMN


def fetch_embedding_profiles(cursor) -> set[tuple[str | None, str | None, int | None]]:
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


def clear_stored_embeddings(cursor) -> None:
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


def rebuild_retrieval_indexes(cursor) -> None:
    cursor.execute("REINDEX INDEX idx_chunks_semantic_embedding_hnsw")
    cursor.execute("REINDEX INDEX idx_chunks_search_vector_gin")
    cursor.execute("ANALYZE chunks")


def ensure_schema(connection, schema_path: Path) -> None:
    schema_sql = schema_path.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)


def upsert_document(cursor, chunk: dict) -> None:
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


def upsert_chunk(
    cursor,
    chunk: dict,
    embedding: list[float],
    embedding_spec: EmbeddingSpec | None = None,
) -> None:
    spec = embedding_spec or EmbeddingSpec(
        DETERMINISTIC_PROVIDER,
        LOCAL_EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS,
    )
    if embedding_spec is None:
        vector_column = LEGACY_VECTOR_COLUMN
        embedding_values = [float(value) for value in embedding]
    else:
        vector_column = vector_column_for_spec(spec)
        embedding_values = validate_embedding(embedding, spec)
    legacy_vector = (
        vector_literal(embedding_values) if vector_column == LEGACY_VECTOR_COLUMN else None
    )
    semantic_vector = (
        vector_literal(embedding_values) if vector_column == SEMANTIC_VECTOR_COLUMN else None
    )

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
            ingested_at,
            embedding_provider,
            embedding_model,
            embedding_dimension,
            embedding,
            semantic_embedding
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s::vector, %s::vector
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
            ingested_at = EXCLUDED.ingested_at,
            embedding_provider = EXCLUDED.embedding_provider,
            embedding_model = EXCLUDED.embedding_model,
            embedding_dimension = EXCLUDED.embedding_dimension,
            embedding = EXCLUDED.embedding,
            semantic_embedding = EXCLUDED.semantic_embedding
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
            spec.provider,
            spec.model,
            spec.dimension,
            legacy_vector,
            semantic_vector,
        ),
    )


def store_chunks(
    chunks: Iterable[dict],
    settings: Settings,
    schema_path: Path | None = None,
    provider: EmbeddingProvider | None = None,
    reindex: bool = False,
) -> int:
    import psycopg

    chunk_records = list(chunks)
    active_provider = provider or create_embedding_provider(settings)
    active_spec = active_provider.spec
    vector_column_for_spec(active_spec)

    with psycopg.connect(settings.database_url) as connection:
        if schema_path is not None:
            ensure_schema(connection, schema_path)

        with connection.cursor() as cursor:
            if reindex:
                clear_stored_embeddings(cursor)
            else:
                profiles = fetch_embedding_profiles(cursor)
                validate_stored_embedding_profiles(profiles, active_spec)

            embeddings = active_provider.embed_many(
                [str(chunk["chunk_text"]) for chunk in chunk_records]
            )
            if len(embeddings) != len(chunk_records):
                raise RuntimeError(
                    f"Embedding provider returned {len(embeddings)} vectors for "
                    f"{len(chunk_records)} chunks"
                )

            for chunk, embedding in zip(chunk_records, embeddings, strict=True):
                upsert_document(cursor, chunk)
                upsert_chunk(cursor, chunk, embedding, active_spec)

            if reindex:
                rebuild_retrieval_indexes(cursor)

    return len(chunk_records)


def embed_chunks(
    repo_root: str | Path | None = None,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    settings: Settings | None = None,
    provider: EmbeddingProvider | None = None,
    reindex: bool = False,
) -> tuple[int, int, str]:
    root = Path(repo_root) if repo_root is not None else project_root()
    input_file = resolve_path(input_path, root)
    schema_file = resolve_path(schema_path, root)
    active_settings = settings or Settings.from_env()

    chunks = load_chunks(input_file)
    stored_count = store_chunks(
        chunks,
        active_settings,
        schema_file,
        provider=provider,
        reindex=reindex,
    )
    return len(chunks), stored_count, active_settings.safe_database_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed local chunks and store one compatible embedding profile."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear every stored vector/profile before fully re-embedding current chunks.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    provider = create_embedding_provider(settings)
    chunks_read, chunks_stored, database_target = embed_chunks(
        settings=settings,
        provider=provider,
        reindex=args.reindex,
    )
    print(f"Chunks read: {chunks_read}")
    print(f"Chunks inserted/upserted: {chunks_stored}")
    print(f"Embedding provider: {provider.spec.provider}")
    print(f"Embedding model: {provider.spec.model}")
    print(f"Embedding dimension: {provider.spec.dimension}")
    print(f"Database target: {database_target}")


if __name__ == "__main__":
    main()
