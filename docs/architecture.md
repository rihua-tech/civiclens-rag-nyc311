# Architecture

## CivicLens RAG — Hybrid RAG Architecture

```text
NYC 311 Documentation
+ NYC 311 Data Dictionary
+ NYC 311 Lakehouse README / Runbooks
+ Gold Mart Sample Summaries
        ↓
Ingestion Pipeline
        ↓
Text Cleaning + Chunking
        ↓
Metadata Tagging
        ↓
Embedding Generation
        ↓
PostgreSQL + pgvector
        ↓
Retriever
        ↓
LLM Answer Generator
        ↓
Cited Answer UI
```

## Design Principle

Documents and metadata are stored for retrieval. Structured metrics should remain in SQL tables or small sample CSV marts instead of being dumped into the vector database.
