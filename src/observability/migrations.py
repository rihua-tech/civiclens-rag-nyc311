"""Small ordered SQL migration runner for the local psycopg architecture."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import Settings


DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "sql" / "migrations"
MIGRATION_FILENAME = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")
MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    sql: str
    checksum: str


@dataclass(frozen=True)
class MigrationResult:
    applied: tuple[str, ...]
    skipped: tuple[str, ...]


def discover_migrations(migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> list[Migration]:
    migrations: list[Migration] = []
    seen_versions: set[str] = set()
    for path in sorted(migrations_dir.glob("*.sql")):
        match = MIGRATION_FILENAME.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Invalid migration filename: {path.name}")
        version = match.group("version")
        if version in seen_versions:
            raise MigrationError(f"Duplicate migration version: {version}")
        sql = path.read_text(encoding="utf-8").strip()
        if not sql:
            raise MigrationError(f"Migration is empty: {path.name}")
        seen_versions.add(version)
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                path=path,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    if not migrations:
        raise MigrationError(f"No migrations found in {migrations_dir}")
    return migrations


def apply_migrations(
    database_url: str,
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR,
    connection_factory: Callable[..., Any] | None = None,
    connect_timeout_seconds: int = 5,
) -> MigrationResult:
    migrations = discover_migrations(migrations_dir)
    if connection_factory is None:
        import psycopg

        connection_factory = psycopg.connect

    applied_now: list[str] = []
    skipped: list[str] = []
    with connection_factory(
        database_url,
        connect_timeout=connect_timeout_seconds,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION_TABLE_SQL)
            cursor.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            )
            applied = {
                str(version): (str(name), str(checksum))
                for version, name, checksum in cursor.fetchall()
            }
            for migration in migrations:
                existing = applied.get(migration.version)
                if existing is not None:
                    if existing != (migration.name, migration.checksum):
                        raise MigrationError(
                            f"Applied migration {migration.version} does not match "
                            "the checked-in file"
                        )
                    skipped.append(migration.version)
                    continue
                cursor.execute(migration.sql)
                cursor.execute(
                    """
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES (%s, %s, %s)
                    """,
                    (migration.version, migration.name, migration.checksum),
                )
                applied_now.append(migration.version)

    return MigrationResult(tuple(applied_now), tuple(skipped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CivicLens SQL migrations.")
    parser.parse_args()
    settings = Settings.from_env()
    result = apply_migrations(
        settings.database_url,
        connect_timeout_seconds=settings.observability_connect_timeout_seconds,
    )
    print(f"Database target: {settings.safe_database_target}")
    print(f"Applied migrations: {', '.join(result.applied) or 'none'}")
    print(f"Already applied: {', '.join(result.skipped) or 'none'}")


if __name__ == "__main__":
    main()
