# Data Sources

## Purpose

This document defines the trusted source materials for CivicLens RAG — NYC 311 Operations Copilot.

The first version uses a small curated set of documents and sample outputs. It should not ingest millions of raw NYC 311 records into the vector database.

## Source Inventory

| Source name | Source type | Planned local path | How it will be used | Questions it can help answer |
|---|---|---|---|---|
| NYC 311 Data Dictionary | Dataset documentation / field reference | `data/raw/nyc311_data_dictionary.md` | Provides field definitions and data context for NYC 311 service request records. This source will be chunked, embedded, and stored in the vector database. | What does `complaint_type` mean? What does `closed_date` mean? Which fields describe request status, agency, location, or timestamps? |
| NYC 311 Lakehouse README | Project README / overview documentation | `data/raw/nyc311_lakehouse_readme.md` | Connects this RAG project to the NYC 311 Lakehouse scope, goals, architecture, and data engineering context. This source will be chunked, embedded, and stored in the vector database. | How does CivicLens relate to the NYC 311 Lakehouse? What is the intended role of the RAG assistant? |
| NYC 311 Architecture Docs | Architecture documentation | `data/raw/nyc311_architecture_docs.md` | Provides context about the lakehouse design, data flow, layers, and system boundaries. This source will be chunked, embedded, and stored in the vector database. | What are the planned data layers? Where do retrieval and cited answers fit in the project architecture? |
| NYC 311 Runbooks | Operational documentation | `data/raw/nyc311_runbooks.md` | Provides procedural context for pipeline operation, troubleshooting, and maintenance questions. This source will be chunked, embedded, and stored in the vector database. | What should an operator check when a pipeline step fails? Which runbook explains a recurring issue? |
| Optional Sample Analytics Outputs | Small summary CSVs or result files | `data/sample_outputs/` | Provides compact analytics examples without loading raw NYC 311 records into the vector store. These outputs should stay as CSV/SQL sample data, not vector-only data. | What patterns are visible in the sample summaries? Which metrics or dimensions are represented in example outputs? |

## Scope Rules

- Use a small curated set of documents for the MVP.
- Store documentation, runbooks, and field definitions in the vector database.
- Keep structured metrics in SQL tables or small CSV summary files.
- Do not ingest millions of raw NYC 311 records into the vector database.
- Do not include secrets, private files, or unsupported claims.

## First MVP Source Priority

1. NYC 311 Data Dictionary
2. NYC 311 Lakehouse README
3. NYC 311 Architecture Docs
4. NYC 311 Runbooks
5. Optional sample analytics outputs
