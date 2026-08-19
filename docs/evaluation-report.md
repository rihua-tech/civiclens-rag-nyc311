# CivicLens Issue 10 Evaluation Baseline

This is the approved Issue 10 portfolio baseline, measured on 2026-08-19. It records the current implementation honestly; it is not a production benchmark, a tuning target, or a claim of statistical significance.

## Evaluation Definition

- Dataset: `issue10-v1` at `data/evaluation/rag_test_questions.csv`
- Questions: 24 total (18 retained Phase 1 cases and 6 Advanced RAG additions)
- Retrieval-eligible questions: 14
- Retrieval relevance granularity: section
- Final retrieval depth: `k=5`
- Expected-source granularity: document
- No LLM judge or paid API

For each eligible question, Recall@5 is `|relevant section IDs intersect retrieved section IDs at 5| / |relevant section IDs|`. Multiple relevant sections receive proportional credit. MRR uses the reciprocal rank of the first relevant section, or zero for a miss. Both are macro averages across the 14 eligible questions; unlabeled analytics/no-answer cases are excluded from these denominators.

Expected-source retrieval checks actual retrieved document IDs separately. Routing, citation presence, citation validity, safe no-answer, and unsupported-answer behavior are also reported independently.

## Deterministic Offline Regression

Run timestamp: `2026-08-19T14:37:45Z`

This profile used `deterministic` / `local-deterministic-1536`, 1536 dimensions, in-memory cosine search, minimum similarity 0.25, and top 5. It required no database, API, network, or model weights.

| Metric | Result | Denominator |
|---|---:|---:|
| Recall@5 | 0.4464 | 14 |
| MRR | 0.4167 | 14 |
| Expected-source retrieval | 0.5000 | 14 |
| Routing accuracy | 0.9583 | 24 |
| Citation presence | 0.7143 | 14 |
| Citation validity | 0.7143 | 14 |
| Safe no-answer accuracy | 0.5000 | 6 |
| Unsupported answers | 4 (0.1667) | 24 |

These deterministic hash-embedding values are regression results only. They are not a real semantic benchmark and are not evidence of Sentence Transformers retrieval quality.

## Real Local Advanced RAG Comparison

Run timestamp: `2026-08-19T14:38:48Z`

The real comparison used the cached `sentence-transformers/all-MiniLM-L6-v2` model at 384 dimensions, PostgreSQL/pgvector, minimum semantic similarity 0.25, semantic and lexical candidate limits of 20, `RRF_K=60`, and top 5. The reranked strategy used cached `cross-encoder/ms-marco-MiniLM-L6-v2` on a maximum of 20 candidates. No weights were downloaded during the run.

| Strategy | Recall@5 (n=14) | MRR (n=14) | Expected source (n=14) |
|---|---:|---:|---:|
| Semantic | 0.6607 | 0.5857 | 0.7857 |
| Hybrid | 0.8393 | 0.7071 | 0.9286 |
| Hybrid + reranking | 0.8214 | 0.7619 | 0.9286 |

These figures compare this small curated dataset under the recorded configuration. They do not demonstrate general benchmark leadership or production superiority.

## Application Behavior Results

Application metrics were identical across the three real retrieval strategies for this fixture:

| Metric | Result | Denominator |
|---|---:|---:|
| Routing accuracy | 0.9583 | 24 |
| Citation presence | 0.9286 | 14 |
| Citation validity | 0.9286 | 14 |
| Safe no-answer accuracy | 0.3333 | 6 |
| Unsupported answers | 5 (0.2083) | 24 |

The failed-case diagnostics show that current application behavior answered four questions expected to abstain (`q015`, `q021`, `q022`, and `q024`). The adversarial `q023` also routed to predefined analytics rather than document RAG, produced no expected citation, and was counted as unsupported. These outcomes are recorded rather than hidden or corrected by changing retrieval behavior.

## Known Limitations

- The fixture is small, curated, and portfolio-oriented rather than production-scale or broadly human-annotated.
- Section labels are explicit but do not constitute exhaustive relevance judgments for every possibly useful chunk.
- Deterministic answer checks do not grade free-form semantic answer quality.
- No LLM is used as a judge; real LLM evaluation is deferred until after Issue 11.
- A single local run does not characterize latency, load, reliability, or statistical variance.
- Higher scores do not by themselves prove production quality or readiness.
- Issue 9 semantic, lexical, RRF, reranking, threshold, candidate-limit, and model behavior was not changed to improve these scores.
