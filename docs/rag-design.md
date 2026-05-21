# RAG Design

## Retrieval Scope

The assistant should answer questions using retrieved source context from:

- NYC 311 documentation
- Data dictionary notes
- Project README files
- Pipeline runbooks
- Selected analytics summaries

## Answer Requirements

Each generated answer should include:

- A clear answer
- Source citations
- A note when the retrieved context is insufficient

## No-Answer Rule

If the retrieved context is weak, the assistant should say it does not have enough source context to answer confidently.

## Local Embedding Storage Flow

Issue 3 stores local chunks and embeddings in PostgreSQL with pgvector. This section describes only the database and embedding storage step.

```bash
python -m src.ingestion.load_documents
python -m src.chunking.chunk_documents
docker compose up -d
python -m src.embeddings.embed_chunks
```

By default, embeddings are generated offline with the deterministic local `local-deterministic-1536` model. OpenAI embeddings are opt-in only with `USE_OPENAI_EMBEDDINGS=true` and a configured `OPENAI_API_KEY`.

## Local Retrieval and Cited Answer Flow

Issue 4 adds the first local retrieval and cited answer layer. This remains a development workflow, not a production deployment or complete assistant.

```text
Question
    -> local question embedding
    -> PostgreSQL/pgvector chunk retrieval
    -> context-only cited answer
    -> safe no-answer response when context is weak
```

The default answer generator is local and uses only retrieved chunk text. OpenAI answer generation is opt-in with `USE_OPENAI_ANSWERS=true` and a configured `OPENAI_API_KEY`.

## Local Streamlit Hybrid Flow

Issue 5 adds a local Streamlit browser UI and small predefined analytics support. Run it locally with:

```bash
docker compose up -d
python -m src.ingestion.load_documents
python -m src.chunking.chunk_documents
python -m src.embeddings.embed_chunks
streamlit run app/streamlit_app.py
```

Hybrid RAG in this repo is intentionally simple:

- Documentation questions use vector retrieval from PostgreSQL/pgvector and return cited local answers.
- Simple analytics questions use predefined sample CSV outputs from `data/sample_outputs/`.
- This is not a production text-to-SQL agent, not deployed, and not connected to live NYC 311 data.
