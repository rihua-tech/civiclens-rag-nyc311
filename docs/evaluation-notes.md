# Evaluation Notes

Issue 6 adds a small local evaluation harness for the CivicLens RAG development workflow. It is intended to catch obvious regressions before showing the project locally; it is not a production benchmark.

## What It Checks

The evaluation dataset lives at `data/evaluation/rag_test_questions.csv`. Each row includes a question, category, expected behavior, and optional source hint.

`python -m src.evaluation.evaluate_rag` checks:

- Answers are not empty.
- Cited document questions return source citations when expected.
- Expected source hints appear in returned sources.
- Analytics questions route to predefined CSV sample outputs under `data/sample_outputs/`.
- No-answer questions return the safe local no-answer response.
- Main answers do not expose raw markdown clutter such as `##` headings or code fences.

## What It Does Not Check

- It does not grade full semantic correctness.
- It does not use an LLM judge.
- It does not call OpenAI by default.
- It does not require network access.
- It does not validate live NYC 311 data or raw production datasets.
- It does not prove deployment, cloud execution, or production readiness.

## Local Evaluation Flow

Document/RAG questions use PostgreSQL + pgvector retrieval, so local evaluation is an integration check. Start the local database and refresh the local processed documents before running it:

```bash
docker compose up -d
python -m src.ingestion.load_documents
python -m src.chunking.chunk_documents
python -m src.embeddings.embed_chunks
python -m src.evaluation.evaluate_rag
```

By default, embeddings and answers use local deterministic behavior. OpenAI-backed embeddings or answers remain opt-in through environment flags and are not required for this evaluation.

## CI Scope

GitHub Actions runs only offline-safe checks:

```bash
python -m pytest -q
python -m compileall app src tests
```

CI does not require Docker, `.env`, OpenAI credentials, a live database, external APIs, or raw NYC 311 datasets.
