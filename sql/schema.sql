CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT,
    source_path TEXT,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(document_id),
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER,
    source_name TEXT,
    source_path TEXT,
    token_count INTEGER,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
