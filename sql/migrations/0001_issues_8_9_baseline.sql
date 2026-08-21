-- Baseline: schema produced by Issues 8 and 9 before Issue 13 observability.
-- Every statement is safe for the existing local database and for a fresh one.

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
    question TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    retrieval_id TEXT PRIMARY KEY,
    query_id TEXT REFERENCES queries(query_id),
    chunk_id TEXT REFERENCES chunks(chunk_id),
    similarity_score DOUBLE PRECISION,
    rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector_gin
    ON chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_semantic_embedding_hnsw
    ON chunks USING HNSW (semantic_embedding vector_cosine_ops)
    WHERE semantic_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_retrieval_results_query_id
    ON retrieval_results(query_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_results_chunk_id
    ON retrieval_results(chunk_id);
