"""Generate embeddings after persisting canonical PostgreSQL metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from src.common.config import (
    DETERMINISTIC_DIMENSION,
    DETERMINISTIC_MODEL,
    Settings,
)
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider
from src.embeddings.providers.deterministic import (
    EMBEDDING_STOPWORDS as EMBEDDING_STOPWORDS,
    TOKEN_PATTERN as TOKEN_PATTERN,
    deterministic_embedding,
    tokenize_for_embedding as tokenize_for_embedding,
)
from src.embeddings.providers.openai_provider import OpenAIEmbeddingProvider
from src.vectorstores.base import VectorStore
from src.vectorstores.factory import create_vector_store
from src.vectorstores.models import VectorIdentity, VectorRecord, VectorSyncResult
from src.vectorstores.postgres_metadata import persist_postgres_metadata


DEFAULT_INPUT_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_SCHEMA_PATH = Path("sql/schema.sql")
LOCAL_EMBEDDING_MODEL = DETERMINISTIC_MODEL
EMBEDDING_DIMENSIONS = DETERMINISTIC_DIMENSION


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


def local_deterministic_embedding(
    text: str,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
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


def synchronize_chunks(
    chunks: Iterable[dict],
    settings: Settings,
    schema_path: Path | None = None,
    provider: EmbeddingProvider | None = None,
    reindex: bool = False,
    vector_store: VectorStore | None = None,
) -> VectorSyncResult:
    """Persist PostgreSQL metadata, then synchronize exactly one vector provider."""

    chunk_records = list(chunks)
    if not chunk_records:
        raise ValueError("No chunks are available to synchronize")

    # PostgreSQL is authoritative even when vector-provider setup or sync fails.
    persist_postgres_metadata(
        chunk_records,
        settings,
        schema_path=schema_path,
    )

    active_provider = provider or create_embedding_provider(settings)
    active_spec = active_provider.spec
    identities = [VectorIdentity.from_chunk(chunk) for chunk in chunk_records]
    active_store = vector_store or create_vector_store(
        settings,
        active_spec,
        identities,
    )
    active_store.prepare_sync(reindex=reindex)

    embeddings = active_provider.embed_many(
        [str(chunk["chunk_text"]) for chunk in chunk_records]
    )
    if len(embeddings) != len(chunk_records):
        raise RuntimeError(
            f"Embedding provider returned {len(embeddings)} vectors for "
            f"{len(chunk_records)} chunks"
        )
    records = [
        VectorRecord.create(chunk, embedding, active_spec)
        for chunk, embedding in zip(chunk_records, embeddings, strict=True)
    ]
    result = active_store.sync(records, reindex=reindex)
    if result.records_written != len(records) or not result.verified:
        raise RuntimeError("Dense-vector synchronization did not verify the full corpus")
    return result


def store_chunks(
    chunks: Iterable[dict],
    settings: Settings,
    schema_path: Path | None = None,
    provider: EmbeddingProvider | None = None,
    reindex: bool = False,
    vector_store: VectorStore | None = None,
) -> int:
    """Backward-compatible count-returning wrapper over provider synchronization."""

    return synchronize_chunks(
        chunks,
        settings,
        schema_path,
        provider=provider,
        reindex=reindex,
        vector_store=vector_store,
    ).records_written


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
    result = synchronize_chunks(
        chunks,
        active_settings,
        schema_file,
        provider=provider,
        reindex=reindex,
    )
    return len(chunks), result.records_written, result.target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist metadata and synchronize one compatible vector provider."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear pgvector values before fully rebuilding the selected provider.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    provider = create_embedding_provider(settings)
    chunks_read, chunks_stored, vector_target = embed_chunks(
        settings=settings,
        provider=provider,
        reindex=args.reindex,
    )
    print(f"Chunks read: {chunks_read}")
    print(f"Chunks synchronized: {chunks_stored}")
    print(f"Embedding provider: {provider.spec.provider}")
    print(f"Embedding model: {provider.spec.model}")
    print(f"Embedding dimension: {provider.spec.dimension}")
    print(f"Vector target: {vector_target}")


if __name__ == "__main__":
    main()
