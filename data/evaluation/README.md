# Evaluation Data

`rag_test_questions.csv` is the version-controlled Issue 10 evaluation fixture. It upgrades the Phase 1 cases in place, gives every question a stable ID and dataset version, and marks legacy versus Advanced RAG additions explicitly.

Retrieval relevance is evaluated at section level for every eligible question. A relevant ID has the form `document_id::section_title`; pipe-separated values represent multiple relevant sections. Questions without relevance labels, including analytics and safe no-answer cases, are excluded from retrieval-metric denominators rather than counted as failures.

The fixture records expected routes, answer behaviors, source document IDs, section titles, and source paths. It is a small curated portfolio benchmark, not a production-scale or large human-annotated dataset.

Generated Markdown and JSON runs are written to `data/evaluation/results/`. That directory is disposable and ignored by Git. The explicitly reviewed portfolio baseline belongs at `docs/evaluation-report.md` and is never overwritten automatically by an evaluation run.
