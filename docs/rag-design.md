# RAG Design

## Retrieval Scope

The assistant should answer questions using retrieved source context from:

- The curated external NYC 311 field guide
- CivicLens project README and design files
- The CivicLens lakehouse context and local RAG runbook
- Selected analytics summaries

`docs/knowledge/source-manifest.json` is the authoritative default documentation inventory. Its `source_category` metadata distinguishes external NYC 311 knowledge from CivicLens project documentation. Large raw service-request datasets are not RAG sources.

## Ingestion, IDs, and Hashes

Default ingestion validates every manifest entry before writing `data/processed/documents.jsonl`. An explicit `source_paths` argument remains available for tests and compatible local workflows.

Document IDs use the canonical repository-relative source path and do not contain ingestion time. Content hashes use normalized UTF-8 text: LF line endings, trailing whitespace removed per line, outer whitespace removed, then SHA-256 stored with a `sha256:` prefix.

Markdown documents are split at ATX headings before word windows are applied. A chunk never spans two parsed sections. Each stored chunk prefixes its body with the heading path and also carries `section_title` and the full `heading_path` as metadata, so retrieval can use section context without reconstructing it. Plain-text documents use the existing word-window fallback with empty section metadata. Chunk size and overlap remain configurable and overlap resets at each section boundary.

Chunk IDs are deterministic over document identity, section location, section-local position, normalized content hash, and a deterministic hash of the chunking algorithm/version, size, and overlap. The approximate whitespace count is named `word_count`, not `token_count`.

## Metadata Flow

The following metadata is preserved from source manifest through document and chunk JSONL into PostgreSQL:

- document and chunk IDs
- source name, type, category, local path, URL, version, and retrieval date
- section title and heading path where available
- ingestion timestamp
- normalized document and chunk content hashes
- deterministic chunking-configuration hash
- chunk `word_count`

Retrieval results return this provenance alongside the existing similarity score and rank so citations and later evaluation can identify the exact source section.

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

`sql/schema.sql` includes narrowly scoped, idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements for Issue 8 metadata. Applying the schema and rerunning ingestion, chunking, and embedding safely upgrades an existing local database. A legacy `token_count` column or stale Phase 1 chunk row may remain, but the Issue 8 pipeline writes `word_count` and retrieval excludes legacy rows without a current content hash. A general migration framework remains deferred to Issue 13.

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
