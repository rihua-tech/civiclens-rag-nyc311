from pathlib import Path


def _normalized_sql(path: str) -> str:
    return " ".join(Path(path).read_text(encoding="utf-8").lower().split())


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
        "feedback",
        "schema_migrations",
    ]

    for table_name in required_tables:
        assert f"create table if not exists {table_name}" in schema_sql


def test_schema_makes_raw_question_optional_and_adds_observability_fields():
    schema_sql = Path("sql/schema.sql").read_text(encoding="utf-8").lower()

    assert "question text not null" not in schema_sql
    assert "alter table queries alter column question drop not null" in schema_sql
    for field in (
        "question_length",
        "route",
        "retrieval_strategy",
        "answer_status",
        "latency_ms",
        "semantic_score",
        "lexical_score",
        "fusion_score",
        "reranker_score",
        "source_path",
    ):
        assert field in schema_sql


def test_fresh_schema_and_issue_13_upgrade_share_the_same_contract():
    schema_sql = _normalized_sql("sql/schema.sql")
    upgrade_sql = _normalized_sql(
        "sql/migrations/0002_observability_and_feedback.sql"
    )

    query_fields = (
        "question_length integer",
        "route text",
        "retrieval_strategy text",
        "embedding_provider text",
        "embedding_model text",
        "answer_provider text",
        "answer_model text",
        "answer_status text",
        "reranking_enabled boolean",
        "top_k integer",
        "latency_ms double precision",
        "observability_version text",
    )
    retrieval_fields = (
        "document_id text references documents(document_id)",
        "retrieval_mode text",
        "semantic_score double precision",
        "semantic_rank integer",
        "lexical_score double precision",
        "lexical_rank integer",
        "fusion_score double precision",
        "reranker_score double precision",
        "pre_rerank_rank integer",
        "source_name text",
        "source_type text",
        "source_category text",
        "source_path text",
        "source_url text",
        "section_title text",
        "heading_path text[]",
        "content_hash text",
        "document_content_hash text",
    )
    for field in query_fields:
        assert field in schema_sql
        assert f"alter table queries add column if not exists {field}" in upgrade_sql
    for field in retrieval_fields:
        assert field in schema_sql
        assert (
            f"alter table retrieval_results add column if not exists {field}"
            in upgrade_sql
        )

    assert "create table if not exists feedback" in schema_sql
    assert "create table if not exists feedback" in upgrade_sql
    for sql in (schema_sql, upgrade_sql):
        assert (
            "query_id text not null references queries(query_id) on delete cascade"
            in sql
        )
        assert (
            "comment text check (comment is null or char_length(comment) <= 1000)"
            in sql
        )
        assert "create index if not exists idx_feedback_query_id" in sql
        assert "on feedback(query_id)" in sql
        assert "create index if not exists idx_queries_created_at" in sql
        assert "on queries(created_at)" in sql

    assert "question text," in schema_sql
    assert "question text not null" not in schema_sql
    assert "alter table queries alter column question drop not null" in upgrade_sql
    assert "drop table" not in upgrade_sql
    assert "truncate " not in upgrade_sql


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
