# Evaluation Notes

Issue 10 provides a repeatable evaluation framework for the retrieval and deterministic application behavior implemented through Issue 9. It preserves the useful Phase 1 questions, adds explicit Advanced RAG cases, and keeps retrieval quality separate from routing, citations, and safe no-answer behavior.

This is a curated portfolio benchmark. It is not production-scale, is not a large human-annotated benchmark, does not use an LLM as judge, and does not establish production readiness or statistical significance.

## Dataset and Relevance Labels

The version-controlled fixture is `data/evaluation/rag_test_questions.csv`. Every row has a stable question ID, one dataset version, a legacy/advanced marker, question category, expected route, expected answer behavior, and explicit ground truth where applicable.

All retrieval-eligible Issue 10 questions use **section-level** relevance. Relevant IDs have the form `document_id::section_title`, and pipe-separated fields support multiple relevant IDs. Questions without retrieval labels are excluded from Recall@k, MRR, and expected-source denominators; they are not automatic retrieval failures. Reports record the selected granularity and every denominator.

Expected-source retrieval is measured independently at document level. It checks actual retrieved `document_id` values and never infers source correctness from answer wording.

The fixture covers NYC 311 field definitions, architecture, runbooks, predefined analytics, safe no-answer cases, negative cases, and adversarial/misleading questions. Phase 1 cases are retained with `phase1_legacy=true`; Issue 10 additions use `false`.

## Retrieval Metrics

For an eligible question, Recall@k is:

```text
|relevant section IDs intersect retrieved section IDs at k| / |relevant section IDs|
```

Recall is calculated per question and macro-averaged. Multiple relevant IDs therefore receive proportional credit rather than an all-or-nothing score.

Reciprocal rank is `1 / rank_of_first_relevant_result`. A miss is zero. MRR is the macro average across eligible questions, using the first retrieved relevant section when several are labeled.

The real local comparison calls the existing Issue 9 retrieval implementation for three separate strategies:

- semantic;
- hybrid (semantic + PostgreSQL FTS with RRF);
- hybrid + the optional bounded reranker.

The evaluator records provider, model, dimension, candidate limits, RRF configuration, reranker configuration, threshold, `top_k`, and a configuration hash. It does not duplicate or tune Issue 9 retrieval algorithms.

## Application Behavior Metrics

The framework reports these independently rather than combining them into a vague "RAG accuracy" score:

- routing accuracy for document RAG, predefined analytics, and expected safe paths;
- citation presence when a cited answer is expected;
- citation validity against the actual retrieved chunk/source metadata;
- safe no-answer accuracy;
- unsupported-answer count and rate.

Arbitrary or fabricated citation numbers are invalid. Machine-readable reports retain per-question results, retrieved ranks/scores, configuration, and failed-case diagnostics.

## Deterministic Offline Regression

Run the offline profile with one command:

```bash
python -m src.evaluation.evaluate_rag --profile offline
```

It uses deterministic hash embeddings and in-memory cosine search over the checked-in manifest corpus. It requires no paid API, API key, PostgreSQL server, internet access, or model download. The command sets the Hugging Face and Transformers offline flags inside its process and writes both Markdown and JSON to `data/evaluation/results/`.

These results validate repeatability, metric/reporting behavior, and offline application regressions. They are **not** a real semantic benchmark and must not be presented as evidence of Sentence Transformers retrieval quality.

## Real Local Advanced RAG Benchmark

The real profile is separate:

```bash
python -m src.evaluation.evaluate_rag --profile real
```

It requires the documented Issue 9 Sentence Transformers model and reranker to already exist in the local cache, plus a prepared PostgreSQL/pgvector database with the current corpus. The evaluator forces local-cache-only model resolution and refuses to download missing weights. It then exercises the actual semantic, PostgreSQL lexical, RRF, and optional reranking call paths.

If the cache or database is unavailable, the real run reports that condition instead of substituting deterministic embeddings or inventing results.

## Reports and Version Control

Every run produces human-readable Markdown and machine-readable JSON under ignored `data/evaluation/results/`. Disposable runs never overwrite the approved baseline automatically.

After human review, measured Issue 10 results are recorded explicitly in `docs/evaluation-report.md`. That baseline states the dataset version/date, section relevance, metric formulas, retrieval configurations, offline-versus-real distinction, actual measured results, failed cases, and limitations.

## Boundaries

The framework does not grade free-form semantic answer quality, use an LLM judge, call OpenAI, tune prompts, or claim general benchmark leadership. Real LLM evaluation is deferred until after Issue 11. Higher curated-benchmark scores do not automatically imply production quality.
