-- Issue 13: migrate the existing query/retrieval tables in place and add feedback.

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

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    helpful BOOLEAN NOT NULL,
    comment TEXT CHECK (comment IS NULL OR char_length(comment) <= 1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);
