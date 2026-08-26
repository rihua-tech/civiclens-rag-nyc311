"""Prepare the current CivicLens corpus and retrieval index safely."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.chunking.chunk_documents import create_chunks
from src.common.config import Settings
from src.embeddings.embed_chunks import embed_chunks
from src.ingestion.load_documents import ingest_documents
from src.observability.migrations import MigrationResult, apply_migrations


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of one completed bootstrap run."""

    migrations_applied: tuple[str, ...]
    migrations_skipped: tuple[str, ...]
    documents_loaded: int
    chunks_created: int
    chunks_embedded: int
    documents_path: Path
    chunks_path: Path
    embedding_target: str


def run_bootstrap(
    *,
    settings: Settings | None = None,
    migration_runner: Callable[..., MigrationResult] = apply_migrations,
    ingestion_runner: Callable[[], tuple[list[dict], Path]] = ingest_documents,
    chunking_runner: Callable[[], tuple[list[dict], Path]] = create_chunks,
    embedding_runner: Callable[..., tuple[int, int, str]] = embed_chunks,
    reporter: Callable[[str], None] | None = None,
) -> BootstrapResult:
    """Run migrations, ingestion, chunking, and embedding in that order.

    Every stage delegates to the existing project implementation. Embedding
    reindexing is intentionally disabled: an incompatible stored profile fails
    clearly and requires the documented explicit operator action.
    """

    active_settings = settings or Settings.from_env()
    report = reporter or (lambda _message: None)

    report("1/4 Applying ordered database migrations...")
    migration_result = migration_runner(
        active_settings.database_url,
        connect_timeout_seconds=active_settings.observability_connect_timeout_seconds,
    )

    report("2/4 Loading and validating the source manifest...")
    documents, documents_path = ingestion_runner()

    report("3/4 Building deterministic current chunks...")
    chunks, chunks_path = chunking_runner()

    report("4/4 Persisting PostgreSQL metadata and synchronizing dense vectors...")
    chunks_read, chunks_stored, embedding_target = embedding_runner(
        settings=active_settings,
        reindex=False,
    )
    if chunks_read != len(chunks):
        raise RuntimeError(
            "Bootstrap chunk count changed between chunking and embedding; "
            "rerun bootstrap after checking the processed chunk artifact."
        )

    return BootstrapResult(
        migrations_applied=tuple(migration_result.applied),
        migrations_skipped=tuple(migration_result.skipped),
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_embedded=chunks_stored,
        documents_path=documents_path,
        chunks_path=chunks_path,
        embedding_target=embedding_target,
    )


def main() -> None:
    """CLI entry point for host-local and Docker bootstrap workflows."""

    result = run_bootstrap(reporter=print)
    print(
        "Bootstrap complete: "
        f"{result.documents_loaded} documents, "
        f"{result.chunks_created} chunks, "
        f"{result.chunks_embedded} embeddings stored in {result.embedding_target}."
    )
    print(
        "Migrations: "
        f"applied={list(result.migrations_applied)}, "
        f"already_applied={list(result.migrations_skipped)}"
    )


if __name__ == "__main__":
    main()

