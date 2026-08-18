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
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Issue 8 metadata upgrade for existing local databases. These statements are
-- intentionally narrow and idempotent; the general migration framework belongs
-- to Issue 13. Re-run ingestion, chunking, and embedding after applying them.
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

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_query_id
    ON retrieval_results(query_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_chunk_id
    ON retrieval_results(chunk_id);
