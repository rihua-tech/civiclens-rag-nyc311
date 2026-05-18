from pathlib import Path


def test_schema_enables_pgvector_extension():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    assert "create extension if not exists vector" in schema_sql


def test_schema_contains_required_tables():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    required_tables = [
        "documents",
        "chunks",
        "queries",
        "retrieval_results",
    ]

    for table_name in required_tables:
        assert f"create table if not exists {table_name}" in schema_sql
