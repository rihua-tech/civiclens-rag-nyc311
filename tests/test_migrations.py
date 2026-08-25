from pathlib import Path

import pytest

from src.observability.migrations import (
    MIGRATION_TABLE_SQL,
    MigrationError,
    apply_migrations,
    discover_migrations,
)


class FakeCursor:
    def __init__(self, applied):
        self.applied = applied
        self.executions = []
        self.current_query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.current_query = str(query)
        self.executions.append((self.current_query, parameters))
        if "INSERT INTO schema_migrations" in self.current_query:
            version, name, checksum = parameters
            self.applied[version] = (name, checksum)

    def fetchall(self):
        if "SELECT version, name, checksum" not in self.current_query:
            raise AssertionError(f"Unexpected fetchall: {self.current_query}")
        return [
            (version, name, checksum)
            for version, (name, checksum) in sorted(self.applied.items())
        ]


class FakeConnection:
    def __init__(self, applied, all_executions):
        self.cursor_instance = FakeCursor(applied)
        self.all_executions = all_executions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.all_executions.extend(self.cursor_instance.executions)
        return False

    def cursor(self):
        return self.cursor_instance


def _normalize_sql(sql):
    return " ".join(sql.lower().split())


def test_migration_discovery_is_ordered_and_versioned(tmp_path):
    (tmp_path / "0002_second.sql").write_text("SELECT 2;", encoding="utf-8")
    (tmp_path / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")

    migrations = discover_migrations(tmp_path)

    assert [migration.version for migration in migrations] == ["0001", "0002"]
    assert [migration.name for migration in migrations] == ["first", "second"]
    assert all(len(migration.checksum) == 64 for migration in migrations)


def test_invalid_migration_filename_is_rejected(tmp_path):
    (tmp_path / "1_bad.sql").write_text("SELECT 1;", encoding="utf-8")

    with pytest.raises(MigrationError, match="Invalid migration filename"):
        discover_migrations(tmp_path)


def test_applied_versions_are_tracked_and_not_reapplied(tmp_path):
    (tmp_path / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0002_second.sql").write_text("SELECT 2;", encoding="utf-8")
    applied = {}
    executions = []

    def connect(database_url, *, connect_timeout):
        assert database_url == "postgresql://unused"
        assert connect_timeout == 4
        return FakeConnection(applied, executions)

    first = apply_migrations(
        "postgresql://unused",
        tmp_path,
        connection_factory=connect,
        connect_timeout_seconds=4,
    )
    second = apply_migrations(
        "postgresql://unused",
        tmp_path,
        connection_factory=connect,
        connect_timeout_seconds=4,
    )

    assert first.applied == ("0001", "0002")
    assert first.skipped == ()
    assert second.applied == ()
    assert second.skipped == ("0001", "0002")
    assert sum(query.strip() == "SELECT 1;" for query, _ in executions) == 1
    assert sum(query.strip() == "SELECT 2;" for query, _ in executions) == 1
    migration_inserts = [
        parameters
        for query, parameters in executions
        if "INSERT INTO schema_migrations" in query
    ]
    assert migration_inserts == [
        ("0001", "first", discover_migrations(tmp_path)[0].checksum),
        ("0002", "second", discover_migrations(tmp_path)[1].checksum),
    ]


def test_changed_applied_migration_is_rejected(tmp_path):
    path = tmp_path / "0001_first.sql"
    path.write_text("SELECT 1;", encoding="utf-8")
    migration = discover_migrations(tmp_path)[0]
    applied = {"0001": (migration.name, "different-checksum")}

    with pytest.raises(MigrationError, match="does not match"):
        apply_migrations(
            "postgresql://unused",
            tmp_path,
            connection_factory=lambda *args, **kwargs: FakeConnection(applied, []),
        )


def test_checked_in_migrations_baseline_and_upgrade_existing_tables():
    migrations = discover_migrations(Path("sql/migrations"))
    baseline = Path("sql/migrations/0001_issues_8_9_baseline.sql").read_text(
        encoding="utf-8"
    ).lower()
    upgrade = Path("sql/migrations/0002_observability_and_feedback.sql").read_text(
        encoding="utf-8"
    ).lower()
    orchestration = Path(
        "sql/migrations/0003_bounded_orchestration_metadata.sql"
    ).read_text(encoding="utf-8").lower()

    assert [migration.version for migration in migrations] == ["0001", "0002", "0003"]
    assert migrations[-1].name == "bounded_orchestration_metadata"
    assert "create table if not exists documents" in baseline
    assert "create table if not exists chunks" in baseline
    assert "create table if not exists queries" in baseline
    assert "create table if not exists retrieval_results" in baseline
    assert "alter table queries alter column question drop not null" in upgrade
    assert "alter table queries add column if not exists route" in upgrade
    assert "alter table retrieval_results add column if not exists retrieval_mode" in upgrade
    assert "create table if not exists feedback" in upgrade
    assert "create table if not exists query_events" not in upgrade
    assert "create table if not exists retrieval_events" not in upgrade
    for field in (
        "orchestration_mode text",
        "orchestration_step_count integer",
        "orchestration_tool_call_count integer",
        "orchestration_outcome text",
    ):
        assert f"alter table queries add column if not exists {field}" in orchestration
    assert "drop table" not in orchestration
    assert "truncate " not in orchestration


def test_fresh_schema_and_runner_share_migration_tracking_contract():
    schema_sql = _normalize_sql(
        Path("sql/schema.sql").read_text(encoding="utf-8")
    )
    runner_table_sql = _normalize_sql(MIGRATION_TABLE_SQL)

    assert "create table if not exists schema_migrations" in schema_sql
    assert "create table if not exists schema_migrations" in runner_table_sql
    for field in (
        "version text primary key",
        "name text not null",
        "checksum text not null",
        "applied_at timestamptz not null default now()",
    ):
        assert field in schema_sql
        assert field in runner_table_sql
