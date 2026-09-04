# Portfolio Card Copy

## Project Title

CivicLens RAG — NYC 311 Operations Copilot

## Positioning

Applied AI / RAG Evaluation

**Status:** NON-PRODUCTION PORTFOLIO DEMO

## Short Description

Hosted hybrid RAG application for grounded NYC 311 documentation Q&A with semantic + PostgreSQL full-text retrieval, deterministic Reciprocal Rank Fusion (RRF), validated citations, safe abstention, and bounded sample analytics.

## Recruiter-Friendly Bullets

- Built and deployed a hybrid RAG system combining semantic search and PostgreSQL full-text retrieval with deterministic Reciprocal Rank Fusion over curated NYC 311 knowledge.
- Validated grounded answers with backend-owned citation provenance and safe abstention; in an approved small local portfolio evaluation, hybrid retrieval achieved 83.9% Recall@5 and 92.9% expected-source retrieval across 14 retrieval-eligible questions.
- Delivered the recruiter-facing Next.js UI through Vercel with Render FastAPI and Neon PostgreSQL + pgvector, while keeping approved analytics bounded to four typed, allowlisted, read-only tools over checked-in sample data.

## Architecture at a Glance

```text
Browser
→ Vercel Next.js (presentation only)
→ Render FastAPI (AI application boundary)
→ CivicLens orchestration
→ Hybrid RAG or Approved Analytics
```

RAG branch:

```text
Curated knowledge
→ manifest-controlled ingestion / section-aware chunking
→ Neon PostgreSQL + pgvector
→ semantic search + PostgreSQL full-text search
→ deterministic RRF
→ optional bounded reranking (disabled in the hosted profile)
→ grounded OpenAI answer generation
→ CivicLens citation validation / safe abstention
```

PostgreSQL remains authoritative for text, metadata, provenance, hashes, and lexical retrieval. OpenAI receives retrieved evidence for hosted answer generation; CivicLens owns citation validation and provenance.

## Evaluation Proof

- **83.9% Recall@5**
- **92.9% Expected-Source Retrieval**
- **14 Retrieval-Eligible Questions**

Approved local portfolio evaluation. This small curated benchmark does not establish production-scale performance or statistical significance.

## Tech Stack Tags

Python, FastAPI, Next.js, TypeScript, PostgreSQL, pgvector, Hybrid Retrieval, RRF, OpenAI, Docker, GitHub Actions, RAG Evaluation

Streamlit remains engineering/debug tooling. Pinecone, bounded LangGraph, and LangChain Core are optional engineering capabilities, not the primary hosted architecture.

## Recommended Portfolio Actions

- [Live Demo](https://civiclens-rag-nyc311.vercel.app)
- [Case Study](../README.md)
- [GitHub Repo](https://github.com/rihua-tech/civiclens-rag-nyc311)
- [Architecture](assets/civiclens-rag-architecture-overview.jpg)

## Public / Internal Links

- Live Demo: [https://civiclens-rag-nyc311.vercel.app](https://civiclens-rag-nyc311.vercel.app)
- Repository: [https://github.com/rihua-tech/civiclens-rag-nyc311](https://github.com/rihua-tech/civiclens-rag-nyc311)
- Architecture overview graphic: [docs/assets/civiclens-rag-architecture-overview.jpg](assets/civiclens-rag-architecture-overview.jpg)
- Architecture deep dive: [docs/architecture.md](architecture.md)
- Evaluation report: [docs/evaluation-report.md](evaluation-report.md)
- Evaluation methodology: [docs/evaluation-notes.md](evaluation-notes.md)
- Deployment proof: [docs/deployment.md](deployment.md)
- Frontend setup: [frontend/README.md](../frontend/README.md)
- Grounded RAG screenshot: [docs/screenshots/issue20-vercel-grounded-rag.png](screenshots/issue20-vercel-grounded-rag.png)
- Approved analytics screenshot: [docs/screenshots/issue20-vercel-analytics.png](screenshots/issue20-vercel-analytics.png)
- Safe abstention screenshot: [docs/screenshots/issue20-vercel-safe-abstention.png](screenshots/issue20-vercel-safe-abstention.png)

## Honest Scope / Limitations

CivicLens is a non-production portfolio demo over curated documentation and checked-in sample analytics. It is not connected to live NYC 311 operations, does not expose unrestricted text-to-SQL, and makes no claim of production availability, reliability, SLA, high availability, authentication, or autoscaling. The hosted Render backend may experience cold starts.
