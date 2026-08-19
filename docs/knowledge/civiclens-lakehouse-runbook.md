# CivicLens Lakehouse Context and Local RAG Runbook

## Source Classification

This is CivicLens project documentation. It describes the repository's local workflow and portfolio architecture; it is not an official City of New York operational runbook.

## Architecture Boundary

CivicLens demonstrates an AI application built on trusted data-engineering assets. Curated documentation follows this local path:

1. The source manifest authorizes a small inventory of repository documents.
2. Ingestion normalizes content and records source provenance and a content hash.
3. Markdown chunking stays within heading sections and carries the heading path into chunk metadata.
4. The configurable provider uses real local Sentence Transformers embeddings by default; deterministic embeddings remain the offline test/CI fallback.
5. PostgreSQL with pgvector stores document, chunk, provenance, embedding-profile, and vector fields.
6. Native CivicLens retrieval combines pgvector semantic search with PostgreSQL full-text search and returns cited chunks for context-only answers.

Structured sample analytics remain in `data/sample_outputs/` and use predefined routing. Large raw NYC 311 service-request datasets do not belong in the RAG knowledge corpus.

## Local Refresh Runbook

From the repository root:

```bash
python -m src.ingestion.load_documents
python -m src.chunking.chunk_documents
docker compose up -d
python -m src.embeddings.embed_chunks --reindex
```

The ingestion command uses `docs/knowledge/source-manifest.json` by default. A manifest content-hash mismatch is a deliberate failure: review the source change and update its manifest hash before ingesting it.

The embedding command applies the idempotent schema in `sql/schema.sql` before upserting documents and chunks. Existing local databases can therefore add Issue 8 metadata and narrowly scoped Issue 9 retrieval columns without a general migration framework. The first Issue 9 semantic run uses `--reindex` to clear incompatible old vectors, rebuild all current chunks with one recorded provider/model/dimension, and rebuild the retrieval indexes. Later refreshes with the same profile can omit `--reindex`. Retrieval excludes legacy rows without current content hashes.

## Troubleshooting Checks

### Ingestion fails before writing documents

- Confirm every manifest path exists inside the repository.
- Confirm the manifest source type matches the file extension.
- Recalculate the documented normalized SHA-256 hash after an intentional source edit.
- Do not bypass a mismatch by silently skipping a manifest entry.

### Chunks are missing section context

- Confirm the source is marked as `markdown` in the manifest.
- Confirm headings use ATX Markdown syntax such as `## Heading`.
- Inspect `section_title` and `heading_path` in `data/processed/chunks.jsonl`.

### PostgreSQL rows lack Issue 8 metadata

- Apply `sql/schema.sql` through the embedding command or a PostgreSQL client.
- Re-run the full ingestion and chunking flow so the processed JSONL contains the new fields.
- Re-run embedding storage so document and current stable chunk upserts refresh the database rows.

### Retrieval returns no useful context

- Verify the relevant document is present in the manifest and processed files.
- Verify embeddings were refreshed after content or chunking changes.
- Verify the printed embedding provider/model/dimension matches the single profile stored in PostgreSQL.
- Use `RETRIEVAL_MODE=semantic` to isolate dense retrieval or `RETRIEVAL_MODE=hybrid` to include PostgreSQL lexical retrieval and RRF.
- Keep reranking disabled while diagnosing base retrieval; when enabled, it only processes the configured bounded candidate set.

## Safety and Limitations

- The real semantic and reranker models may download weights on their first explicit local use.
- Automated tests and CI explicitly use the deterministic provider and never download model weights.
- OpenAI integrations remain optional and are not required by ingestion or CI.
- The knowledge base is curated documentation, not a live NYC 311 feed.
- The schema upgrade is intentionally limited to Issue 8 metadata and Issue 9 retrieval columns; the general migration framework belongs to Issue 13.
