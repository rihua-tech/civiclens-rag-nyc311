"""Generate embeddings for local chunks and store them in PostgreSQL/pgvector."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

from src.common.config import Settings


DEFAULT_INPUT_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_SCHEMA_PATH = Path("sql/schema.sql")
LOCAL_EMBEDDING_MODEL = "local-deterministic-1536"
EMBEDDING_DIMENSIONS = 1536
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
EMBEDDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
}


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


def tokenize_for_embedding(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in EMBEDDING_STOPWORDS
    ]


def local_deterministic_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0")

    tokens = tokenize_for_embedding(text)
    if not tokens:
        tokens = [hashlib.sha256(text.encode("utf-8")).hexdigest()]

    embedding = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big", signed=False) % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        embedding[index] += sign

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding

    return [round(value / norm, 8) for value in embedding]


def openai_embedding(text: str, settings: Settings) -> list[float]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when USE_OPENAI_EMBEDDINGS=true")

    model = settings.embedding_model
    if model == LOCAL_EMBEDDING_MODEL:
        model = "text-embedding-3-small"

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=model, input=text)
    embedding = [float(value) for value in response.data[0].embedding]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding model returned {len(embedding)} dimensions; "
            f"expected {EMBEDDING_DIMENSIONS}"
        )

    return embedding


def generate_embedding(text: str, settings: Settings | None = None) -> list[float]:
    active_settings = settings or Settings.from_env()
    if active_settings.use_openai_embeddings:
        return openai_embedding(text, active_settings)
    return local_deterministic_embedding(text)


def vector_literal(embedding: Iterable[float]) -> str:
    return "[" + ",".join(format(float(value), ".10g") for value in embedding) + "]"


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
            source_path,
            ingested_at
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            source_type = COALESCE(documents.source_type, EXCLUDED.source_type),
            source_path = EXCLUDED.source_path
        """,
        (
            chunk["document_id"],
            chunk.get("source_name") or "unknown",
            chunk.get("source_type"),
            chunk.get("source_path"),
            chunk.get("ingested_at"),
        ),
    )


def upsert_chunk(cursor, chunk: dict, embedding: list[float]) -> None:
    cursor.execute(
        """
        INSERT INTO chunks (
            chunk_id,
            document_id,
            chunk_text,
            chunk_index,
            source_name,
            source_path,
            token_count,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            chunk_text = EXCLUDED.chunk_text,
            chunk_index = EXCLUDED.chunk_index,
            source_name = EXCLUDED.source_name,
            source_path = EXCLUDED.source_path,
            token_count = EXCLUDED.token_count,
            embedding = EXCLUDED.embedding
        """,
        (
            chunk["chunk_id"],
            chunk["document_id"],
            chunk["chunk_text"],
            chunk.get("chunk_index"),
            chunk.get("source_name"),
            chunk.get("source_path"),
            chunk.get("token_count"),
            vector_literal(embedding),
        ),
    )


def store_chunks(
    chunks: Iterable[dict],
    settings: Settings,
    schema_path: Path | None = None,
) -> int:
    import psycopg

    chunk_records = list(chunks)
    with psycopg.connect(settings.database_url) as connection:
        if schema_path is not None:
            ensure_schema(connection, schema_path)

        with connection.cursor() as cursor:
            for chunk in chunk_records:
                embedding = generate_embedding(chunk["chunk_text"], settings)
                upsert_document(cursor, chunk)
                upsert_chunk(cursor, chunk, embedding)

    return len(chunk_records)


def embed_chunks(
    repo_root: str | Path | None = None,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    settings: Settings | None = None,
) -> tuple[int, int, str]:
    root = Path(repo_root) if repo_root is not None else project_root()
    input_file = resolve_path(input_path, root)
    schema_file = resolve_path(schema_path, root)
    active_settings = settings or Settings.from_env()

    chunks = load_chunks(input_file)
    stored_count = store_chunks(chunks, active_settings, schema_file)
    return len(chunks), stored_count, active_settings.safe_database_target


def main() -> None:
    chunks_read, chunks_stored, database_target = embed_chunks()
    print(f"Chunks read: {chunks_read}")
    print(f"Chunks inserted/upserted: {chunks_stored}")
    print(f"Database target: {database_target}")


if __name__ == "__main__":
    main()
