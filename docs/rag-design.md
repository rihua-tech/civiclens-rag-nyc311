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
