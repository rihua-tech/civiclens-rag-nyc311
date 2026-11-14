# Evaluation Data

`rag_test_questions.csv` is the version-controlled Issue 10 evaluation fixture. It upgrades the Phase 1 cases in place, gives every question a stable ID and dataset version, and marks legacy versus Advanced RAG additions explicitly.

For the Phase 0/P0 abstention work, the original 24 questions remain the regression dataset. `p0_development_questions.csv` contains 26 answerable and unsupported edge cases used for implementation feedback, including six scope-guard paraphrases added in `p0-dev-v2`. `p0_holdout_nonblind_questions.csv` was created and SHA-256 frozen before behavioral changes, but it is explicitly **non-blind** because this environment could not isolate its contents from the implementation agent. It must not be described as an independent test. Its immutable metadata and digest are recorded in `p0_holdout_manifest.json`; any content change requires a new dataset version and manifest rather than updating the existing digest.

Retrieval relevance is evaluated at section level for every eligible question. A relevant ID has the form `document_id::section_title`; pipe-separated values represent multiple relevant sections. Questions without relevance labels, including analytics and safe no-answer cases, are excluded from retrieval-metric denominators rather than counted as failures.

The fixture records expected routes, answer behaviors, source document IDs, section titles, and source paths. It is a small curated portfolio benchmark, not a production-scale or large human-annotated dataset.

Evaluation prefers structured `answer_status=abstained`. Exact no-answer text is only a labeled compatibility inference when an older response has no status. Reports separate abstention recall (unsupported questions), false-abstention rate (answerable questions), abstention integrity, routing, citations, unsupported answers, retrieval, and provider/fallback usage.

Generated Markdown and JSON runs are written to `data/evaluation/results/`. That directory is disposable and ignored by Git. The explicitly reviewed portfolio baseline belongs at `docs/evaluation-report.md` and is never overwritten automatically by an evaluation run.
