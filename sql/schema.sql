CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    source_url TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(document_id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    source_name TEXT,
    section_title TEXT,
    token_count INTEGER,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    user_question TEXT NOT NULL,
    generated_answer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    result_id TEXT PRIMARY KEY,
    query_id TEXT REFERENCES queries(query_id),
    chunk_id TEXT REFERENCES chunks(chunk_id),
    similarity_score DOUBLE PRECISION,
    rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id TEXT PRIMARY KEY,
    query_id TEXT REFERENCES queries(query_id),
    citation_found BOOLEAN,
    answer_grounded_score DOUBLE PRECISION,
    retrieval_relevance_score DOUBLE PRECISION,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
