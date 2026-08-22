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

Markdown documents are split at ATX headings before word windows are applied. A chunk never spans two parsed sections. Each stored chunk prefixes its body with the heading path and also carries `section_title` and the full `heading_path` as metadata. Plain-text documents retain the word-window fallback with empty section metadata. Chunk size and overlap remain configurable and overlap resets at each section boundary.

Chunk IDs are deterministic over document identity, section location, section-local position, normalized content hash, and a deterministic hash of the chunking algorithm/version, size, and overlap. The approximate whitespace count is named `word_count`, not `token_count`.

## Embedding Providers

Retrieval code uses one embedding-provider contract exposing provider name, model name, dimension, single-text embedding, and batch embedding.

| Provider | Model/default | Dimension | Purpose |
|---|---|---:|---|
| `sentence_transformers` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default real local semantic mode |
| `deterministic` | `local-deterministic-1536` | 1536 | Offline tests, CI, and compatible local fallback |
| `openai` | Existing opt-in model configuration; `text-embedding-3-small` fallback | 1536 | Backward-compatible optional API path |

`all-MiniLM-L6-v2` was selected as the one default semantic model because it is a compact English sentence/paragraph encoder intended for semantic search and produces a documented 384-dimensional vector. Model loading is lazy. The first explicit semantic run may download weights from the model registry; automated tests inject fakes and never load or download them.

`USE_OPENAI_EMBEDDINGS=true` remains a backward-compatible provider override and still requires `OPENAI_API_KEY`. OpenAI is not the default and is not required for local operation or CI.

## Answer Requirements

Answer generation uses an application-owned provider contract:

```text
AnswerProvider.generate(question, evidence) -> ProviderResult
ProviderResult = answer text + stable chunk citation IDs + answered/abstained status
```

`local` is the deterministic default for local workflows, evaluation, and CI. `openai` is the only optional commercial answer provider. OpenAI-specific Responses API parsing remains inside its provider module; application orchestration consumes only `ProviderResult`.

The provider receives only the user question and allow-listed retrieved evidence fields: chunk text/ID, source name/path/type/category, section title, and heading path. It never receives credentials, database URLs, environment configuration, unrelated documents, or unrelated application state.

Retrieved text is labeled as untrusted data and is separated from application rules. The provider is instructed not to treat retrieved text as application instructions, while application-side validation enforces the allowed citation IDs and safe abstention behavior.

Citation identity uses stable `chunk_id` values rather than model-generated display numbers. After generation, CivicLens:

1. deduplicates returned citation IDs in deterministic order;
2. rejects IDs absent from the retrieved result set;
3. rebuilds provenance from the application-owned retrieved chunks;
4. creates display citation numbers from validated retrieved positions;
5. converts an answered result with zero valid citations to `NO_ANSWER`.

The application never trusts provider-generated source metadata or fabricates citations to rescue an answer.

## No-Answer Rule

If retrieval returns no sufficiently similar source context, or the retrieved text does not directly support the question, the assistant should say it does not have enough source context to answer confidently.

## Dimension and Profile Safety

The original `chunks.embedding vector(1536)` column remains for deterministic and opt-in OpenAI embeddings. The real local model uses a separate `chunks.semantic_embedding vector(384)` column. Every embedded current chunk also records `embedding_provider`, `embedding_model`, and `embedding_dimension`.

Only one vector column is populated for an active chunk profile. Storage and semantic retrieval reject:

- a configured dimension that does not match its fixed pgvector column;
- a model whose runtime dimension differs from `EMBEDDING_DIMENSION`;
- stored current chunks with unrecorded, mixed, or incompatible provider/model/dimension metadata.

This prevents deterministic, OpenAI, and local semantic vectors from silently coexisting as though they shared one vector space.

### Complete Re-embedding and Reindex Procedure

Use this procedure when first moving an Issue 8 database to semantic mode or whenever provider, model, or dimension changes:

1. Select exactly one profile in `.env`, for example:

   ```dotenv
   EMBEDDING_PROVIDER=sentence_transformers
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_DIMENSION=384
   ```

2. If selecting a model with a dimension other than 384 for Sentence Transformers or 1536 for the compatible deterministic/OpenAI column, first make a reviewed, narrowly scoped schema/index change. The current code deliberately fails instead of altering vector dimensions automatically.
3. Refresh and validate the current Issue 8 corpus:

   ```bash
   python -m src.ingestion.load_documents
   python -m src.chunking.chunk_documents
   ```

4. Start PostgreSQL and run the explicit full rebuild:

   ```bash
   docker compose up -d
   python -m src.embeddings.embed_chunks --reindex
   ```

   The command applies `sql/schema.sql`, clears all old vector values and profile metadata, embeds every current chunk with one active provider, and rebuilds the semantic HNSW and lexical GIN indexes. Without `--reindex`, incompatible stored profiles fail with a clear error.
5. Verify the printed provider, model, dimension, chunk counts, and safe database target. Then run representative semantic, lexical/hybrid, and optional reranking queries. PostgreSQL can also verify the stored profile:

   ```sql
   SELECT embedding_provider, embedding_model, embedding_dimension, COUNT(*)
   FROM chunks
   WHERE embedding IS NOT NULL OR semantic_embedding IS NOT NULL
   GROUP BY embedding_provider, embedding_model, embedding_dimension;
   ```

The schema changes above are specific to Issue 9 retrieval. Existing databases now use the ordered Issue 13 migration command, `python -m src.observability.migrations`, before re-embedding when an unapplied schema version exists.

## PostgreSQL Semantic and Lexical Retrieval

PostgreSQL + pgvector remains the primary vector store. Semantic queries use cosine distance against the column selected by the active provider profile. They keep the Issue 8 current-chunk filters: chunk content hash must exist, document content hash must match the current document, and chunking configuration hash must match.

PostgreSQL full-text search uses a stored weighted `tsvector` over source name, section title, and chunk text. A GIN index supports parameterized, read-only lexical queries using `websearch_to_tsquery`. Question filler is removed before the remaining bounded terms use PostgreSQL web-search semantics, so exact identifiers such as `complaint_type` are not suppressed by words such as "what" or "mean." This path complements semantic search for NYC 311 field names, agency terms, complaint/problem terminology, and technical identifiers.

## Retrieval Modes and Fusion

`RETRIEVAL_MODE=semantic` returns dense results only. `RETRIEVAL_MODE=hybrid` retrieves bounded dense and lexical candidate sets and deduplicates/fuses them using Reciprocal Rank Fusion:

```text
RRF score(chunk) = sum(1 / (RRF_K + source_rank))
```

The default `RRF_K` is 60. Ties use best source rank, semantic rank, lexical rank, and finally `chunk_id`, producing a stable ordering. Candidate counts and final `top_k` are capped at 100 in application code.

## Optional Bounded Reranking

`RERANKING_ENABLED=false` by default. When enabled, the single local reranker `cross-encoder/ms-marco-MiniLM-L6-v2` scores only the first `RERANK_CANDIDATE_LIMIT` fused or semantic candidates. It never scans the corpus. Reranker ties preserve the pre-rerank order and then `chunk_id`.

Like the semantic model, the reranker is lazy and may download weights on its first explicit local use. CI uses fakes and keeps reranking disabled.

## Stable Result Contract and Metadata Flow

Semantic, lexical, hybrid, and reranked paths preserve:

- document and chunk IDs;
- chunk text and final rank;
- source name, type, category, local path, URL, version, and retrieval date;
- section title and heading path;
- ingestion timestamp, `word_count`, content hashes, and chunking configuration hash.

Diagnostics are additive: `semantic_score`/`semantic_rank`, `lexical_score`/`lexical_rank`, `fusion_score`, `pre_rerank_rank`, and `reranker_score`. `similarity_score` remains available for downstream Phase 1 compatibility when a semantic score exists.

## Local Retrieval and Cited Answer Flow

```text
Question
    -> configured semantic provider
    -> PostgreSQL semantic + lexical candidates
    -> deterministic RRF
    -> optional bounded reranker
    -> existing context-only cited answer
```

The default semantic minimum similarity is 0.25. PostgreSQL lexical candidates can still contribute through RRF, but unrelated dense matches below that threshold are excluded before answer generation.

## FastAPI and Shared Question Orchestration

Streamlit calls the versioned FastAPI backend through its public `POST /api/v1/answer` contract. FastAPI then calls the same reusable Python question orchestrator used by non-HTTP callers and tests. The orchestrator preserves the existing decision boundary: supported analytics questions use predefined checked-in CSV outputs, analytics-looking questions without a predefined route safely decline, and documentation questions reuse Issue 9 retrieval plus Issue 11 grounded generation and citation validation.

FastAPI remains a thin adapter. It validates the public request, calls orchestration, and serializes a provider-neutral response. It does not own analytics routing, retrieval, provider selection, grounding, citation validation, observability persistence, or feedback validation. Public responses omit raw chunks, provider payloads, environment/database configuration, and local exception details; Streamlit now renders this same allow-listed response rather than bypassing HTTP for internal retrieval diagnostics. When observability is enabled, the answer response adds the stable orchestration-owned `query_id` needed for feedback.

```text
Question -> Streamlit -> FastAPI /api/v1/answer -> shared orchestrator
                                                    |-> analytics route -> application result
                                                    \-> RAG retrieval -> grounded generation -> application result

Application result -> allow-listed FastAPI response -> Streamlit
                   \-> allow-listed query/retrieval metadata (optional)
```

`GET /health` is dependency-free liveness. `GET /ready` performs a bounded, read-only check for loadable configuration, reachable PostgreSQL, required RAG tables, exact current manifest document/chunk identities and hashes, and embeddings matching the configured provider/model/dimension. Neither endpoint calls OpenAI, loads models, performs retrieval/generation, or mutates the database.

Start the versioned local HTTP adapter with:

```bash
uvicorn api.main:app
```

## Local Streamlit Hybrid Flow

Prepare the database with the normal bootstrap workflow, start FastAPI, and then run Streamlit:

```bash
python -m scripts.bootstrap
uvicorn api.main:app
streamlit run app/streamlit_app.py
```

The Streamlit interface sends every question to the public API. The shared server-side orchestrator routes predefined analytics questions to sample CSV summaries and documentation questions to configured semantic-only or hybrid retrieval before grounded generation. `CIVICLENS_API_BASE_URL` defaults to `http://localhost:8000` for this host-local workflow; Compose maps `CIVICLENS_DOCKER_API_BASE_URL` to `http://api:8000`. A bounded `CIVICLENS_API_TIMEOUT_SECONDS` controls the UI client. The UI can start before bootstrap and renders controlled not-ready, timeout, connection, validation, server, or malformed-response messages.

## Docker Bootstrap and Readiness

The Compose stack packages PostgreSQL/pgvector, FastAPI, and Streamlit. `docker compose run --rm api python -m scripts.bootstrap` runs ordered migrations, manifest ingestion, chunking, and embedding/index upserts by calling the existing project functions. Normal bootstrap is rerun-safe and never passes the explicit reindex flag. A stored provider/model/dimension mismatch stops with the existing clear error and requires a separate operator-approved reindex procedure.

Docker uses `/health` only for API process liveness. `/ready` retains the stricter Issue 12 semantics and returns `200` only when PostgreSQL contains the exact current manifest documents and chunks with a compatible active embedding profile. It may correctly return `503` while all three containers are running before bootstrap. Named volumes preserve database state and the local Hugging Face cache across ordinary stop/start cycles.

## Retrieval Troubleshooting

When a local pipeline step fails, an operator should check the manifest hash validation, processed document/chunk counts, active provider/model/dimension, PostgreSQL health, the single stored embedding profile, and the semantic/lexical indexes in that order. A profile mismatch requires the explicit `--reindex` procedure rather than a partial update.

## Configuration

The Issue 9 environment variables are:

- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`
- `RETRIEVAL_MODE`
- `SEMANTIC_CANDIDATE_COUNT`, `LEXICAL_CANDIDATE_COUNT`, `RRF_K`
- `RERANKING_ENABLED`, `RERANKER_MODEL`, `RERANK_CANDIDATE_LIMIT`

The legacy `USE_OPENAI_EMBEDDINGS` and `OPENAI_API_KEY` variables remain supported. `.env.example` shows the normal semantic/hybrid defaults. GitHub Actions explicitly selects the deterministic provider, semantic-only retrieval, and disabled reranking so tests never contact OpenAI or a model registry.

Issue 13 adds `OBSERVABILITY_ENABLED` and `OBSERVABILITY_CONNECT_TIMEOUT_SECONDS`. Observability is disabled for normal CI. It records existing retrieval diagnostics and source identities, not retrieval content or vectors, and does not alter Issue 9 ranking behavior.

Issue 14 adds `CIVICLENS_API_BASE_URL` and `CIVICLENS_API_TIMEOUT_SECONDS` for the Streamlit HTTP boundary. Docker service names are runtime networking details only; host-local application and test workflows remain supported.

## Answer and Analytics Boundaries

Issue 11 does not change Issue 9 retrieval. Documentation questions still use its final retrieved chunks; predefined analytics remains a separate deterministic route.

Answer-provider configuration uses `ANSWER_PROVIDER`, `ANSWER_MODEL`, `ANSWER_TIMEOUT_SECONDS`, and `ANSWER_MAX_RETRIES`. The existing `USE_OPENAI_ANSWERS` and `OPENAI_API_KEY` settings remain backward compatible: `USE_OPENAI_ANSWERS=true` selects OpenAI even when `.env.example` still contains its local default. Selecting OpenAI without credentials or encountering a timeout/provider failure triggers a controlled local fallback identified by a non-secret reason code. Provider abstention and an answered response without valid citations produce the safe no-answer result instead of a fallback-generated claim.

Retries are bounded from zero through five and delegated to the OpenAI SDK. Authentication/configuration failures are mapped to a safe configuration failure rather than copied into application output. Settings and provider representations redact the API key.

## Evaluation Boundary

Issue 10 evaluates the unchanged Issue 9 retrieval paths as separate semantic, hybrid, and hybrid-plus-reranking strategies. The versioned fixture uses section-level relevance IDs, supports multiple relevant sections, and macro-averages per-question Recall@k and reciprocal rank only across eligible questions. Expected-source retrieval remains a separate document-level check.

The default evaluation command is an API-free deterministic regression and is not evidence of real semantic quality. A separate real local profile requires cached Sentence Transformers models and the prepared PostgreSQL/pgvector corpus, then calls the native Issue 9 retrieval implementation without benchmark-only ranking behavior. Both profiles report routing, citation presence/validity, safe no-answer behavior, and unsupported answers separately.

An optional `--answer-profile openai` path reuses the same real-retrieval evaluation framework while labeling provider/model configuration and per-question fallback status separately from the deterministic baseline. Tests exercise that integration with a fake provider. No real OpenAI evaluation was run during Issue 11 because no API key was supplied.

Disposable Markdown/JSON runs belong under ignored `data/evaluation/results/`; only the explicitly reviewed `docs/evaluation-report.md` is the version-controlled baseline. Dataset design, formulas, commands, and limitations are documented in `docs/evaluation-notes.md`.

Live OpenAI evaluation, hosted observability, distributed tracing, cloud deployment, tool registries, LangGraph, Pinecone, LangChain, and LlamaIndex remain outside automated verification. The local Docker Compose deployment and versioned HTTP adapter do not add production API concerns such as authentication, streaming, rate limiting, or monitoring.
