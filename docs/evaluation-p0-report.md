# CivicLens Phase 0 / P0 Evaluation Report

Evaluation date: 2026-11-12 (America/New_York)

Follow-up completed: 2026-11-13 (America/New_York)

Branch: `fix/abstention-reliability`

Local `main` and starting branch SHA: `240547253720f3df9c077281217e321a06ec624a`

## Result boundary

This report covers Phase 0 evaluation integrity and Phase 1 abstention reliability only. It does not include the Phase 2 Next.js homepage work. The approved 2026-08-19 results in `docs/evaluation-report.md` are historical published results under the earlier metric definition. They were not rerun in their original environment and are not used as the pre-fix comparison below.

The comparable before/after runs below are newly reproduced offline deterministic regressions using the same corrected `p0-report-v2` definitions, deterministic hash embeddings, in-memory cosine retrieval, `k=5`, minimum similarity `0.25`, and the checked-in corpus. These runs are regression evidence, not Sentence Transformers or hosted performance.

## Corrected definitions

- Abstention is determined by structured `answer_status=abstained`; exact no-answer text is only a labeled legacy inference when status is absent.
- Abstention recall is measured only on questions expected to abstain.
- False-abstention rate is measured only on questions expected to answer.
- Abstention integrity requires zero sources, citation IDs, and display citations.
- Routing, citation presence, citation ID validity, unsupported-answer rate, and retrieval Recall@5 remain separate.
- Citation ID validity is not treated as proof of semantic answer correctness.
- Provider calls and fallback usage are retained per question and aggregated.

## Comparable regression results

| Metric | Pre-fix, corrected definition | Post-fix | Change |
|---|---:|---:|---:|
| Abstention recall (n=6) | 0.5000 | 1.0000 | +0.5000 |
| False-abstention rate (n=18) | 0.2778 | 0.3333 | +0.0556 (worse) |
| Routing accuracy (n=24) | 1.0000 | 1.0000 | 0.0000 |
| Citation presence (n=14) | 0.6429 | 0.5714 | -0.0714 |
| Citation ID validity (n=14) | 0.6429 | 0.5714 | -0.0714 |
| Abstention integrity | 1.0000 | 1.0000 | 0.0000 |
| Unsupported-answer rate (n=24) | 0.1250 (3/24) | 0.0000 (0/24) | -0.1250 |
| Retrieval Recall@5 (n=14) | 0.3750 | 0.3750 | 0.0000 |
| MRR (n=14) | 0.3571 | 0.3571 | 0.0000 |

The false-abstention regression is real. The conservative lexical sufficiency check refuses `q004` because the deterministic retriever returned only a troubleshooting chunk about retrieval, not the requested Retrieval Scope section. Re-accepting that one-word topical overlap would violate the P0 evidence-sufficiency requirement. Retrieval was intentionally not changed.

The exact post-fix false abstentions are `q003`, `q004`, `q009`, `q010`, `q020`, and `q023`. Five (`q003`, `q009`, `q010`, `q020`, `q023`) retrieved zero chunks, so the answer provider was not called. `q004` retrieved one wrong section (`Retrieval returns no useful context`); the deterministic provider was called once and correctly abstained. `q004` is the only new false abstention and the only citation-validity loss relative to the corrected pre-fix run. No provider exception or fallback contributed to the decline.

Provider/fallback summary: the pre-fix run recorded 19 `local` and 5 `not_used` responses, 11 provider calls, and zero provider fallbacks. The post-fix run recorded 17 `local`, 4 `predefined_analytics`, and 3 `not_used` responses, 9 provider calls, and zero provider fallbacks. Scope-blocked requests do not call an answer provider or analytics tool.

## Development results

The 20-question `p0-dev-v1` dataset was used for implementation feedback. It covers private contact values, live/current/future operational requests, field definitions containing `address`, `today`, or `phone`, supported sample analytics, adversarial wrapping, and unrelated questions.

| Metric | Pre-fix | Post-fix |
|---|---:|---:|
| Abstention recall (n=9) | 0.3333 | 1.0000 |
| False-abstention rate (n=11) | 0.3636 | 0.4545 |
| Routing accuracy (n=20) | 0.9500 | 1.0000 |
| Citation presence / ID validity (n=9) | 0.5556 / 0.5556 | 0.4444 / 0.4444 |
| Abstention integrity | 1.0000 | 1.0000 |
| Unsupported-answer rate | 0.3000 (6/20) | 0.0000 (0/20) |
| Retrieval Recall@5 (n=9) | 0.4444 | 0.4444 |

All nine unsupported development requests abstained without executing an unrelated predefined tool. Five answerable questions abstained because deterministic retrieval did not supply sufficient directly relevant evidence.

## Frozen non-blind holdout diagnostic

`p0-holdout-nonblind-v1` contains 12 questions. It was created and frozen before behavioral edits with SHA-256 `a2eb290ae3c30743ae1e2046b05d2d51a12c8a831b98b2b11a155b1531ae512c`, and it was evaluated once after implementation freeze. Genuine isolation was unavailable: the implementation agent created and could inspect the labels. Therefore these results are non-blind diagnostics, not independent-test evidence.

| Metric | Result |
|---|---:|
| Abstention recall (n=6) | 1.0000 |
| False-abstention rate (n=6) | 0.1667 |
| Routing accuracy (n=12) | 1.0000 |
| Citation presence / ID validity (n=4) | 0.7500 / 0.7500 |
| Abstention integrity | 1.0000 |
| Unsupported-answer rate | 0.0000 (0/12) |
| Retrieval Recall@5 (n=4) | 0.7500 |

## Manual evidence review

Thirty representative outputs were reviewed directly against returned provenance and checked-in source content, without an LLM judge. Of 15 answered outputs, 11 had a supported core answer and four were partial or nonresponsive despite valid citation IDs. Six expected abstentions were appropriately citation-free. Nine answerable cases safely abstained because sufficient evidence was not retrieved. The itemized review is in `docs/evaluation-p0-manual-review.md`.

## Scope-guard follow-up

The 2026-11-13 review confirmed and fixed two narrow intent defects:

- Satisfaction is no longer blocked merely because the word appears. Documentation questions such as whether closure proves customer satisfaction and questions about satisfaction-claim limitations remain on the RAG path. Unsupported score, percentage, rate, and ranking requests remain blocked.
- A documentation word such as `describe` or `documented` no longer exempts a caller/requester contact-value request tied to a specific request, case, complaint, or unique key. Explicit schema questions such as `Describe the caller_phone field` remain on the RAG path.

Focused direct and LangGraph regressions verify that blocked requests execute neither retrieval nor an analytics tool.

## Real local and OpenAI evaluation status

- Local recovery: the timeout was caused by Docker Desktop's Linux engine being stopped. The existing Compose `postgres` service was started, reported healthy, and the existing non-destructive bootstrap refreshed 7 documents and 77 current Sentence Transformers embeddings. All three ordered migrations were already applied. Readiness then returned `ready`; no schema or infrastructure expansion was needed.
- Real local Sentence Transformers retrieval and deterministic answers completed successfully:

| Strategy | Recall@5 (n=14) | Abstention recall (n=6) | False abstention (n=18) | Citation ID validity (n=14) | Unsupported answers | Provider calls / fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Semantic | 0.7321 | 1.0000 | 0.0000 | 1.0000 | 0/24 | 16 / 0 |
| Hybrid | 0.8393 | 1.0000 | 0.0000 | 1.0000 | 0/24 | 16 / 0 |
| Hybrid + rerank | 0.8214 | 1.0000 | 0.0000 | 1.0000 | 0/24 | 16 / 0 |

- The separate OpenAI answer evaluation also completed successfully using the same real retrieval strategies and configured `gpt-4o-mini` structured-answer provider:

| Strategy | Recall@5 (n=14) | Abstention recall (n=6) | False abstention (n=18) | Citation ID validity (n=14) | Unsupported answers | OpenAI calls / fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Semantic | 0.7321 | 1.0000 | 0.1111 | 0.8571 | 0/24 | 16 / 0 |
| Hybrid | 0.8393 | 1.0000 | 0.0556 | 0.9286 | 0/24 | 16 / 0 |
| Hybrid + rerank | 0.8214 | 1.0000 | 0.0556 | 0.9286 | 0/24 | 16 / 0 |

The OpenAI run made 48 actual provider calls, used zero local fallbacks, and recorded no provider-use failures. Semantic abstained on `q001` and `q023`; hybrid and hybrid+rerank abstained on `q023`. These are reported as false abstentions rather than relabeled successes.
- Hosted deployment: **not evaluated or changed**. Repository documentation describes the hosted path as deterministic 1536-dimensional embeddings, hybrid retrieval, and OpenAI answer generation. No P0 number in this report is hosted performance.
- Historical local hybrid Recall@5 remains historical local evidence only and is not presented as hosted performance.

## Verification

- Backend: `419 passed` with `python -m pytest -q`.
- Python lint: `ruff check src tests` passed.
- Diff whitespace: `git diff --check` passed.
- Frontend: `17 passed` with Vitest; ESLint passed; Next.js production build passed. No homepage source was modified.

## Assessment and limitations

**CONDITIONAL PASS.** Abstention recall, routing, scope blocking, citation-free abstentions, unsupported-answer rate, real-local readiness, and separate OpenAI execution are validated. The scope-guard defects found during follow-up are fixed. However, false-abstention rate remains higher on the comparable deterministic offline regression and development sets, so acceptance criterion 4 is not met under that profile. The increase is precisely one safe refusal (`q004`) caused by an irrelevant retrieved section, and relaxing evidence sufficiency would reintroduce the defect. The non-blind holdout cannot support an independent-test claim, and deterministic citation validity still overstates semantic quality in four manually reviewed answers. P0 closure therefore requires reviewer acceptance of this documented conservative offline tradeoff.
