# Data Sources

## Purpose

CivicLens uses a small, curated, version-controlled documentation corpus. It does not copy raw NYC 311 service-request rows into the RAG knowledge base.

`docs/knowledge/source-manifest.json` is the default authoritative ingestion inventory. Every default source must be explicitly listed with its name, type, category, local path, source URL, version or retrieval date, and normalized content hash.

## Source Categories

The manifest distinguishes two categories:

- `external_nyc311`: curated domain knowledge derived from official City of New York sources.
- `civiclens_project`: repository documentation about the local CivicLens architecture, runbook, evaluation, and limitations.

This distinction is preserved in document, chunk, and PostgreSQL metadata. Project documentation must not be presented as an official NYC operational source.

## Curated Inventory

| Curated source | Category | Local path | Primary provenance | Purpose |
|---|---|---|---|---|
| NYC 311 Service Request Field Guide | `external_nyc311` | `docs/knowledge/nyc311-service-request-fields.md` | Official NYC Open Data dataset `erm2-nwe9`, its metadata API, the official 2025 update notice, and NYC311 | Definitions and interpretation limits for complaint/problem type, closed date, status, agency, borough/location, and request timestamps |
| CivicLens Lakehouse Context and Local RAG Runbook | `civiclens_project` | `docs/knowledge/civiclens-lakehouse-runbook.md` | This repository | Local architecture, refresh steps, metadata checks, and troubleshooting boundaries |
| CivicLens README and design notes | `civiclens_project` | `README.md`, `docs/data-sources.md`, `docs/architecture.md`, `docs/rag-design.md`, `docs/evaluation-notes.md` | This repository | Project behavior, architecture, evaluation, and limitations |

The official NYC references used by the field guide are:

- <https://data.cityofnewyork.us/d/erm2-nwe9>
- <https://data.cityofnewyork.us/api/views/erm2-nwe9>
- <https://opendata.cityofnewyork.us/311-service-requests-from-2010-to-present-updates/>
- <https://portal.311.nyc.gov/about-nyc-311/>

The official pages were reviewed on 2026-08-17. The source manifest records that retrieval date and the locally curated file's hash; it does not claim the daily-updated dataset itself is frozen at that date.

## Manifest and Local Overrides

Running `python -m src.ingestion.load_documents` with no source arguments loads only manifest entries and validates their hashes before producing documents.

The Python API retains an explicit `source_paths` override for tests and compatible local workflows. Override files receive `local_override` source categorization unless the caller provides manifest metadata through the default path. Missing or unsupported override files continue to be skipped as in Phase 1; missing, unsupported, duplicate, or hash-mismatched manifest entries fail loudly because the manifest is authoritative.

## Stable IDs and Content Hashes

All text uses the same documented normalization before hashing:

1. Convert CRLF and CR line endings to LF.
2. Remove trailing whitespace from each line.
3. Remove leading and trailing whitespace around the complete document or chunk.
4. Encode the normalized text as UTF-8.
5. Calculate SHA-256 and store it as `sha256:<lowercase hex digest>`.

Document IDs are `doc_` plus the first 16 hexadecimal characters of SHA-256 over the canonical repository-relative POSIX source path. Chunk IDs are deterministic hashes of the document ID, heading path, section position, section-local chunk position, normalized chunk content hash, and deterministic chunking-configuration hash. Neither ID includes the ingestion timestamp.

## Scope Rules

- Keep the manifest and curated Markdown sources in version control.
- Keep structured metrics in SQL tables or small CSV summaries under `data/sample_outputs/`.
- Keep raw and processed service-request records out of Git; `.gitignore` excludes `data/raw/*` and `data/processed/*` except their placeholders.
- Do not include secrets, private files, copied customer information, or unsupported operational claims.
- Treat field values as evolving and non-exhaustive where the official dataset says so.
- This corpus supplies documentation evidence, not live NYC 311 data or production analytics.
