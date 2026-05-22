# Architecture

## CivicLens RAG - Hybrid RAG Architecture

```mermaid
flowchart TD
    docs["NYC 311 documentation<br/>Data dictionary notes<br/>Runbooks"]
    samples["Sample analytics CSV outputs"]

    docs --> ingestion["Document ingestion"]
    ingestion --> chunking["Text cleaning and chunking"]
    chunking --> embeddings["Embedding generation"]
    embeddings --> store["PostgreSQL + pgvector"]
    store --> retrieval["Retriever"]
    retrieval --> answer["Context-only answer generator"]
    answer --> ui["Cited Streamlit UI"]

    samples --> router["Simple analytics router"]
    router --> analyticsAnswer["Predefined analytics answer"]
    analyticsAnswer --> ui
```

## Design Principle

Documents and metadata are stored for retrieval. Structured metrics remain in SQL tables or small sample CSV outputs instead of being dumped into the vector database.

This is a local development architecture, not a production deployment.
