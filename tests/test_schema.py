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


def test_schema_adds_narrow_issue_9_embedding_profile_and_dimension_columns():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    assert "embedding vector(1536)" in schema_sql
    assert "semantic_embedding vector(384)" in schema_sql
    assert "embedding_provider text" in schema_sql
    assert "embedding_model text" in schema_sql
    assert "embedding_dimension integer" in schema_sql
    assert "alter table chunks add column if not exists semantic_embedding vector(384)" in schema_sql


def test_schema_adds_postgresql_full_text_and_semantic_indexes():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    assert "search_vector tsvector generated always as" in schema_sql
    assert "to_tsvector('english', coalesce(chunk_text, ''))" in schema_sql
    assert "idx_chunks_search_vector_gin" in schema_sql
    assert "using gin(search_vector)" in schema_sql
    assert "idx_chunks_semantic_embedding_hnsw" in schema_sql
    assert "semantic_embedding vector_cosine_ops" in schema_sql
