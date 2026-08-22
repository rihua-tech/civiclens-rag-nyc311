from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bootstrap import run_bootstrap
from src.common.config import Settings
from src.observability.migrations import MigrationResult


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://user:secret@postgres:5432/civiclens",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
    )


def _successful_fakes(events):
    def migrate(database_url, *, connect_timeout_seconds):
        events.append(("migrations", database_url, connect_timeout_seconds))
        return MigrationResult(("0002",), ("0001",))

    def ingest():
        events.append(("ingestion",))
        return ([{"document_id": "doc_1"}], Path("documents.jsonl"))

    def chunk():
        events.append(("chunking",))
        return (
            [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}],
            Path("chunks.jsonl"),
        )

    def embed(*, settings, reindex):
        events.append(("embeddings", settings.embedding_provider, reindex))
        return (2, 2, "postgres:5432/civiclens")

    return migrate, ingest, chunk, embed


def test_bootstrap_reuses_project_stages_in_required_order_without_reindex():
    events = []
    reports = []
    migrate, ingest, chunk, embed = _successful_fakes(events)

    result = run_bootstrap(
        settings=_settings(),
        migration_runner=migrate,
        ingestion_runner=ingest,
        chunking_runner=chunk,
        embedding_runner=embed,
        reporter=reports.append,
    )

    assert [event[0] for event in events] == [
        "migrations",
        "ingestion",
        "chunking",
        "embeddings",
    ]
    assert events[-1] == ("embeddings", "deterministic", False)
    assert result.documents_loaded == 1
    assert result.chunks_created == 2
    assert result.chunks_embedded == 2
    assert [message[:3] for message in reports] == ["1/4", "2/4", "3/4", "4/4"]


def test_bootstrap_failure_stops_later_stages():
    events = []

    def migrate(database_url, *, connect_timeout_seconds):
        events.append("migrations")
        return MigrationResult((), ("0001", "0002"))

    def ingest():
        events.append("ingestion")
        raise RuntimeError("manifest invalid")

    def must_not_run():
        events.append("unexpected")
        raise AssertionError("later bootstrap stage ran")

    with pytest.raises(RuntimeError, match="manifest invalid"):
        run_bootstrap(
            settings=_settings(),
            migration_runner=migrate,
            ingestion_runner=ingest,
            chunking_runner=must_not_run,
            embedding_runner=must_not_run,
        )

    assert events == ["migrations", "ingestion"]


def test_normal_bootstrap_is_rerun_safe_and_never_requests_destructive_reindex():
    events = []
    migrate, ingest, chunk, embed = _successful_fakes(events)

    for _ in range(2):
        run_bootstrap(
            settings=_settings(),
            migration_runner=migrate,
            ingestion_runner=ingest,
            chunking_runner=chunk,
            embedding_runner=embed,
        )

    embedding_events = [event for event in events if event[0] == "embeddings"]
    assert embedding_events == [
        ("embeddings", "deterministic", False),
        ("embeddings", "deterministic", False),
    ]


def test_bootstrap_rejects_a_processed_chunk_race():
    events = []
    migrate, ingest, chunk, embed = _successful_fakes(events)

    def inconsistent_embed(*, settings, reindex):
        return (1, 1, "postgres:5432/civiclens")

    with pytest.raises(RuntimeError, match="chunk count changed"):
        run_bootstrap(
            settings=_settings(),
            migration_runner=migrate,
            ingestion_runner=ingest,
            chunking_runner=chunk,
            embedding_runner=inconsistent_embed,
        )


def test_bootstrap_module_has_no_database_reset_or_destructive_sql_path():
    source = Path("scripts/bootstrap.py").read_text(encoding="utf-8").lower()

    assert "truncate table" not in source
    assert "drop table" not in source
    assert "reindex=true" not in source
