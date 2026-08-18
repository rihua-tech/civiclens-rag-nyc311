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


def test_schema_contains_issue_8_document_and_chunk_metadata_columns():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    required_columns = [
        "source_category",
        "source_url",
        "source_version",
        "source_retrieved_at",
        "section_title",
        "heading_path",
        "word_count",
        "content_hash",
        "document_content_hash",
        "chunking_config_hash",
        "ingested_at",
    ]

    for column_name in required_columns:
        assert column_name in schema_sql

    assert "token_count integer" not in schema_sql


def test_schema_has_narrow_idempotent_issue_8_upgrade_statements():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    required_upgrades = [
        "alter table documents add column if not exists source_category text",
        "alter table documents add column if not exists content_hash text",
        "alter table chunks add column if not exists section_title text",
        "alter table chunks add column if not exists heading_path text[]",
        "alter table chunks add column if not exists word_count integer",
        "alter table chunks add column if not exists content_hash text",
        "alter table chunks add column if not exists document_content_hash text",
        "alter table chunks add column if not exists chunking_config_hash text",
        "alter table chunks add column if not exists ingested_at timestamptz",
    ]

    for statement in required_upgrades:
        assert statement in schema_sql
