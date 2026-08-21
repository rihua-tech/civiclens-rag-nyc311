CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT,
    source_category TEXT,
    source_path TEXT,
    source_url TEXT,
    source_version TEXT,
    source_retrieved_at DATE,
    content_hash TEXT,
    chunking_config_hash TEXT,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(document_id),
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER,
    source_name TEXT,
    source_type TEXT,
    source_category TEXT,
    source_path TEXT,
    source_url TEXT,
    source_version TEXT,
    source_retrieved_at DATE,
    section_title TEXT,
    heading_path TEXT[],
    word_count INTEGER,
    content_hash TEXT,
    document_content_hash TEXT,
    chunking_config_hash TEXT,
    ingested_at TIMESTAMPTZ,
    embedding_provider TEXT,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    embedding vector(1536),
    semantic_embedding vector(384),
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', COALESCE(source_name, '')), 'A') ||
        setweight(to_tsvector('simple', COALESCE(section_title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(chunk_text, '')), 'B')
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Issue 8 metadata upgrade for existing local databases. These statements are
-- intentionally narrow and idempotent. Issue 13 adds ordered migrations for
-- existing databases; this file remains the complete fresh-database schema.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_category TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_version TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_retrieved_at DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunking_config_hash TEXT;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_type TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_category TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_version TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_retrieved_at DATE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_title TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS heading_path TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS word_count INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_content_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunking_config_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;

-- Issue 9 retrieval upgrade. The existing vector(1536) column remains for the
-- deterministic and opt-in OpenAI paths. Real local semantic vectors use a
-- separate fixed vector(384) column so incompatible profiles cannot be mixed.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_provider TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS semantic_embedding vector(384);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', COALESCE(source_name, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(section_title, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(chunk_text, '')), 'B')
) STORED;

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    question TEXT,
    question_length INTEGER,
    route TEXT,
    retrieval_strategy TEXT,
    embedding_provider TEXT,
    embedding_model TEXT,
    answer_provider TEXT,
    answer_model TEXT,
    answer_status TEXT,
    reranking_enabled BOOLEAN,
    top_k INTEGER,
    latency_ms DOUBLE PRECISION,
    observability_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    retrieval_id TEXT PRIMARY KEY,
    query_id TEXT REFERENCES queries(query_id),
    chunk_id TEXT REFERENCES chunks(chunk_id),
    document_id TEXT REFERENCES documents(document_id),
    similarity_score DOUBLE PRECISION,
    rank INTEGER,
    retrieval_mode TEXT,
    semantic_score DOUBLE PRECISION,
    semantic_rank INTEGER,
    lexical_score DOUBLE PRECISION,
    lexical_rank INTEGER,
    fusion_score DOUBLE PRECISION,
    reranker_score DOUBLE PRECISION,
    pre_rerank_rank INTEGER,
    source_name TEXT,
    source_type TEXT,
    source_category TEXT,
    source_path TEXT,
    source_url TEXT,
    section_title TEXT,
    heading_path TEXT[],
    content_hash TEXT,
    document_content_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    helpful BOOLEAN NOT NULL,
    comment TEXT CHECK (comment IS NULL OR char_length(comment) <= 1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Issue 13 in-place upgrade clauses for environments that still apply the
-- complete schema directly. Ordered local upgrades use sql/migrations/.
ALTER TABLE queries ALTER COLUMN question DROP NOT NULL;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS question_length INTEGER;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS route TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS retrieval_strategy TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS embedding_provider TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS answer_provider TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS answer_model TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS answer_status TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS reranking_enabled BOOLEAN;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS top_k INTEGER;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS observability_version TEXT;

ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS document_id TEXT
    REFERENCES documents(document_id);
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS retrieval_mode TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS semantic_score DOUBLE PRECISION;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS semantic_rank INTEGER;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS lexical_score DOUBLE PRECISION;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS lexical_rank INTEGER;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS fusion_score DOUBLE PRECISION;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS reranker_score DOUBLE PRECISION;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS pre_rerank_rank INTEGER;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS source_name TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS source_type TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS source_category TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS source_path TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS section_title TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS heading_path TEXT[];
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE retrieval_results ADD COLUMN IF NOT EXISTS document_content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector_gin
    ON chunks USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_semantic_embedding_hnsw
    ON chunks USING HNSW (semantic_embedding vector_cosine_ops)
    WHERE semantic_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_retrieval_results_query_id
    ON retrieval_results(query_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_chunk_id
    ON retrieval_results(chunk_id);

CREATE INDEX IF NOT EXISTS idx_feedback_query_id
    ON feedback(query_id);

CREATE INDEX IF NOT EXISTS idx_queries_created_at
    ON queries(created_at);
