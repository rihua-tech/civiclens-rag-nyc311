# Hybrid RAG Architecture

```mermaid
flowchart TD
    docs["NYC 311 documentation<br/>Data dictionary notes<br/>Runbooks"]
    samples["Sample analytics CSV outputs"]

    docs --> ingestion["Document ingestion"]
    ingestion --> chunking["Text cleaning and chunking"]
    chunking --> embeddings["Local embeddings by default"]
    embeddings --> store["PostgreSQL + pgvector"]
    store --> retrieval["Vector retrieval"]
    retrieval --> answer["Context-only cited answer generation"]
    answer --> ui["Cited Streamlit UI"]

    samples --> router["Simple analytics router"]
    router --> analyticsAnswer["Predefined analytics answer"]
    analyticsAnswer --> ui
```

This architecture uses vector retrieval for documentation questions and predefined sample analytics outputs for structured analytics questions.

Evaluation, pytest, and GitHub Actions validate retrieval behavior, citation coverage, analytics routing, and safe no-answer responses.

## Design Principle

Documents and metadata are stored for retrieval. Structured metrics remain in SQL tables or small sample CSV outputs instead of being dumped into the vector database.

This is a local development architecture, not a production deployment. It is not connected to live NYC 311 data, OpenAI is optional and disabled by default, and the analytics path uses predefined sample CSV outputs rather than production text-to-SQL.
