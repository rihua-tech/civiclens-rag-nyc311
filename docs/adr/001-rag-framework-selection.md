# ADR 001: Select LangChain Core for the Issue 18 Retrieval Adapter

- Status: Accepted
- Date: 2026-08-25
- Scope: Issue 18 ecosystem portability only

## Context

CivicLens already owns ingestion, chunking, embeddings, PostgreSQL metadata,
semantic and lexical retrieval, Reciprocal Rank Fusion, reranking, answer
generation, citation validation, evaluation, API contracts, and bounded
LangGraph orchestration. Issue 18 requires one optional framework compatibility
adapter without moving any of those responsibilities into a framework.

The adapter needs only to accept a text query, call the existing synchronous
CivicLens retrieval boundary, and map the returned chunks into a supported
framework document/retriever representation. Stable chunk IDs, provenance,
rank, and score must survive the mapping.

## Options Considered

### LangChain Core

`langchain-core` provides `BaseRetriever` and `Document`. A custom retriever
implements `_get_relevant_documents()` and returns `Document` objects. This is
a direct fit for the required outbound adapter: CivicLens `chunk_text` becomes
`Document.page_content`, while stable IDs, provenance, rank, and scores remain
application-owned document metadata.

Benefits:

- the adapter can depend on `langchain-core` without installing the full
  `langchain` package or a framework-owned RAG chain;
- Issue 17's optional LangGraph dependency already uses `langchain-core`, so
  the selected adapter overlaps with an existing optional ecosystem boundary;
- the synchronous custom-retriever method matches CivicLens's current
  synchronous retrieval and FastAPI execution model;
- tests can inject the existing CivicLens retrieval callable and require no
  model, database, hosted tracing, or network service.

Tradeoffs:

- `Document` has no dedicated score field, so CivicLens score and rank must be
  preserved in metadata;
- `BaseRetriever` also exposes asynchronous runnable behavior, but Issue 18
  does not add an asynchronous CivicLens retrieval implementation;
- `langchain-core` has its own Pydantic/runnable lifecycle and must remain
  isolated behind a lazy optional import.

References:

- https://reference.langchain.com/python/langchain_core/retrievers/
- https://reference.langchain.com/python/langchain-core/documents
- https://docs.langchain.com/oss/python/integrations/retrievers/index

### LlamaIndex Core

`llama-index-core` provides `BaseRetriever`, node types, and `NodeWithScore`.
Its first-class scored-node representation maps rank/score naturally, and its
retriever contract supports synchronous and asynchronous implementations.

Benefits:

- score is represented directly by `NodeWithScore`;
- node IDs and metadata align well with stable CivicLens chunk identities;
- the library is purpose-built around data and RAG abstractions.

Tradeoffs:

- it introduces a separate, broader RAG-core dependency with no existing
  overlap in CivicLens;
- its node, callback, settings, and query-bundle abstractions add more mapping
  surface than this retrieval-only adapter needs;
- selecting it alongside the existing LangGraph ecosystem would increase the
  optional dependency and maintenance matrix without improving CivicLens's
  native pipeline.

References:

- https://docs.llamaindex.ai/en/stable/api_reference/retrievers/router/
- https://docs.llamaindex.ai/en/stable/api_reference/schema/

## Decision

Select **LangChain Core** and implement exactly one optional
`CivicLensRetriever` adapter over the existing CivicLens retrieval callable.

The implementation will:

- import `langchain-core` lazily so native CivicLens imports and runs without
  it;
- return LangChain `Document` objects containing chunk text plus allow-listed
  CivicLens identity, provenance, rank, and score metadata;
- use the existing synchronous CivicLens retrieval boundary and expose async
  compatibility by moving that same call to a worker thread, not by creating a
  second async retrieval stack;
- avoid LangChain vector stores, embeddings, generation chains, agents,
  citations, evaluation, hosted tracing, and orchestration.

## Consequences

The optional adapter makes CivicLens retrieval consumable by LangChain callers
without creating a second CivicLens RAG backend. Native CivicLens remains the
only path that guarantees CivicLens citation validation and safe abstention.
External LangChain-generated answers do not automatically inherit those
guarantees.

Core/native installations do not install `langchain-core`. The selected
dependency is kept in a separate requirements file and receives a separate
offline CI test job. Future framework API changes are confined to this thin
adapter and its focused tests.
