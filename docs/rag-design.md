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

The existing answer path remains context-only. Each answer should include a clear response, source citations, and a note when retrieved context is insufficient.

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

The schema changes above are specific to Issue 9 retrieval. They are not a general migration framework; that remains Issue 13 work.

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

## Local Streamlit Hybrid Flow

Prepare the database with the complete reindex procedure, then run:

```bash
streamlit run app/streamlit_app.py
```

The application continues to route predefined analytics questions to sample CSV summaries. Documentation questions use the configured semantic-only or hybrid retrieval mode before the unchanged answer generator.

## Retrieval Troubleshooting

When a local pipeline step fails, an operator should check the manifest hash validation, processed document/chunk counts, active provider/model/dimension, PostgreSQL health, the single stored embedding profile, and the semantic/lexical indexes in that order. A profile mismatch requires the explicit `--reindex` procedure rather than a partial update.

## Configuration

The Issue 9 environment variables are:

- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`
- `RETRIEVAL_MODE`
- `SEMANTIC_CANDIDATE_COUNT`, `LEXICAL_CANDIDATE_COUNT`, `RRF_K`
- `RERANKING_ENABLED`, `RERANKER_MODEL`, `RERANK_CANDIDATE_LIMIT`

The legacy `USE_OPENAI_EMBEDDINGS` and `OPENAI_API_KEY` variables remain supported. `.env.example` shows the normal semantic/hybrid defaults. GitHub Actions explicitly selects the deterministic provider, semantic-only retrieval, and disabled reranking so tests never contact OpenAI or a model registry.

## Answer and Analytics Boundaries

Issue 9 changes retrieval only. The default answer generator remains the existing local context-only cited path, and the opt-in OpenAI answer path is unchanged. At the project level, documentation RAG and predefined analytics routing remain separate branches.

Formal Recall@K/MRR reporting, new evaluation report architecture, LLM provider redesign, APIs, observability, general migrations, deployment, tool registries, LangGraph, Pinecone, LangChain, and LlamaIndex are not implemented here.
