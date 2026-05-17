# CivicLens RAG — NYC 311 Operations Copilot

AI Data Engineering / Hybrid RAG portfolio project that extends the NYC 311 Lakehouse with a cited RAG assistant for service request documentation, data definitions, pipeline runbooks, and operational analytics questions.

## Project Goal

Build a reproducible RAG pipeline that:

1. Ingests NYC 311 documentation, data dictionary notes, and project runbooks
2. Cleans and chunks source documents
3. Creates embeddings
4. Stores searchable vectors in PostgreSQL + pgvector
5. Retrieves relevant context for user questions
6. Generates LLM answers with source citations
7. Provides a simple Streamlit UI

## Planned Architecture

```text
NYC 311 Docs / Data Dictionary / Project Runbooks
        ↓
Document Ingestion
        ↓
Cleaning + Chunking + Metadata
        ↓
Embeddings
        ↓
PostgreSQL + pgvector
        ↓
Retriever
        ↓
LLM Answer Generator
        ↓
Cited Answer UI
```

## Project Status

In Progress.

This repository is initially scaffolded for development. Do not claim completed, deployed, production-ready, or cloud execution proof until real implementation and evidence are added.

## Tech Stack

- Python
- FastAPI
- Streamlit
- PostgreSQL
- pgvector
- OpenAI API
- Embeddings
- Hybrid RAG
- Vector Search
- SQL
- Docker
- GitHub Actions

## Folder Structure

```text
civiclens-rag-nyc311/
├── README.md
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── common/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_outputs/
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── rag-design.md
│   └── evaluation-notes.md
├── sql/
│   ├── schema.sql
│   └── sample_queries.sql
└── tests/
```
