"""Repeatable retrieval and application evaluation for CivicLens RAG.

The default profile is an offline deterministic regression over the checked-in
corpus. It is deliberately not described as a real semantic benchmark. The
separate real profile calls the Issue 9 PostgreSQL/pgvector retrieval paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.analytics.simple_analytics import (
    answer_analytics_question,
    looks_like_analytics_question,
)
from src.chunking.chunk_documents import chunk_documents
from src.common.config import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL,
    DETERMINISTIC_DIMENSION,
    DETERMINISTIC_MODEL,
    DETERMINISTIC_PROVIDER,
    LOCAL_ANSWER_PROVIDER,
    OPENAI_ANSWER_PROVIDER,
    SEMANTIC_PROVIDER,
    Settings,
)
from src.embeddings.providers.deterministic import DeterministicEmbeddingProvider
from src.evaluation.metrics import (
    aggregate_application_metrics,
    aggregate_retrieval_metrics,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.reporting import write_reports
from src.generation.answer_question import (
    NO_ANSWER,
    generate_answer_from_chunks,
    local_answer,
)
from src.generation.providers import AnswerProvider, build_answer_provider
from src.ingestion.load_documents import load_documents
from src.retrieval.retrieve_context import DEFAULT_MIN_SIMILARITY, retrieve_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_test_questions.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
EVALUATION_SCHEMA_VERSION = "issue10-report-v1"
DEFAULT_TOP_K = 5
MULTI_VALUE_SEPARATOR = "|"
SUPPORTED_RELEVANCE_GRANULARITIES = {"document", "section", "chunk"}
REQUIRED_COLUMNS = {
    "dataset_version",
    "question_id",
    "phase1_legacy",
    "question",
    "category",
    "expected_route",
    "expected_answer_behavior",
    "relevance_granularity",
    "relevant_ids",
    "expected_source_document_ids",
    "expected_section_titles",
    "expected_source_paths",
}
MESSY_MARKDOWN_MARKERS = ("```", "##", "# ")
ANALYTICS_SOURCE_PREFIX = "data/sample_outputs/"
NO_SOURCE_HINT = "none"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    category: str
    expected_behavior: str
    expected_source_hint: str = ""
    question_id: str = "legacy-question"
    dataset_version: str = "phase1-legacy"
    phase1_legacy: bool = True
    expected_route: str = "rag"
    relevance_granularity: str = ""
    relevant_ids: tuple[str, ...] = ()
    expected_source_document_ids: tuple[str, ...] = ()
    expected_section_titles: tuple[str, ...] = ()
    expected_source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    question: EvaluationQuestion
    passed: bool
    passed_checks: int
    failed_checks: list[str]
    mode: str


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    retrieval_mode: str
    reranking_enabled: bool


Retriever = Callable[[str, StrategyDefinition], list[dict[str, Any]]]
ApplicationResponder = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


REAL_STRATEGIES = (
    StrategyDefinition("semantic", "semantic", False),
    StrategyDefinition("hybrid", "hybrid", False),
    StrategyDefinition("hybrid_rerank", "hybrid", True),
)
OFFLINE_STRATEGY = StrategyDefinition(
    "deterministic_offline_regression",
    "deterministic_in_memory_cosine",
    False,
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def split_values(value: str | None) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(value or "").split(MULTI_VALUE_SEPARATOR)
        if item.strip()
    )


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false value, found {value!r}")
    return normalized == "true"


def load_evaluation_questions(
    path: str | Path = DEFAULT_EVALUATION_PATH,
) -> list[EvaluationQuestion]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation questions file not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Evaluation CSV is missing required columns: {missing}")

        questions: list[EvaluationQuestion] = []
        seen_ids: set[str] = set()
        dataset_versions: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            question_text = row.get("question", "").strip()
            if not question_text:
                continue
            question_id = row["question_id"].strip()
            if not question_id:
                raise ValueError(f"Evaluation row {row_number} has no question_id")
            if question_id in seen_ids:
                raise ValueError(f"Duplicate evaluation question_id: {question_id}")
            seen_ids.add(question_id)

            dataset_version = row["dataset_version"].strip()
            if not dataset_version:
                raise ValueError(f"Evaluation row {row_number} has no dataset_version")
            dataset_versions.add(dataset_version)

            granularity = row["relevance_granularity"].strip().lower()
            relevant_ids = split_values(row["relevant_ids"])
            if relevant_ids and granularity not in SUPPORTED_RELEVANCE_GRANULARITIES:
                raise ValueError(
                    f"Evaluation row {row_number} has relevant IDs but unsupported "
                    f"granularity {granularity!r}"
                )
            if granularity and not relevant_ids:
                raise ValueError(
                    f"Evaluation row {row_number} declares relevance granularity "
                    "without relevant IDs"
                )

            expected_paths = split_values(row["expected_source_paths"])
            questions.append(
                EvaluationQuestion(
                    question=question_text,
                    category=row["category"].strip(),
                    expected_behavior=row["expected_answer_behavior"].strip(),
                    expected_source_hint=(expected_paths[0] if expected_paths else ""),
                    question_id=question_id,
                    dataset_version=dataset_version,
                    phase1_legacy=parse_bool(row["phase1_legacy"]),
                    expected_route=row["expected_route"].strip(),
                    relevance_granularity=granularity,
                    relevant_ids=relevant_ids,
                    expected_source_document_ids=split_values(
                        row["expected_source_document_ids"]
                    ),
                    expected_section_titles=split_values(row["expected_section_titles"]),
                    expected_source_paths=expected_paths,
                )
            )

    if len(dataset_versions) != 1:
        raise ValueError(
            "An evaluation file must contain exactly one dataset version; found "
            + ", ".join(sorted(dataset_versions))
        )
    validate_relevance_granularity(questions)
    return questions


def validate_relevance_granularity(questions: Sequence[EvaluationQuestion]) -> str:
    granularities = {
        question.relevance_granularity
        for question in questions
        if question.relevant_ids
    }
    if len(granularities) > 1:
        raise ValueError(
            "Retrieval metrics cannot silently mix relevance granularities: "
            + ", ".join(sorted(granularities))
        )
    return next(iter(granularities), "not_applicable")


def source_matches_hint(sources: list[dict[str, Any]], expected_source_hint: str) -> bool:
    if not expected_source_hint:
        return True
    if expected_source_hint.lower() == NO_SOURCE_HINT:
        return not sources
    normalized_hint = expected_source_hint.lower()
    return any(
        normalized_hint in str(source.get("source_path", "")).lower()
        or normalized_hint in str(source.get("source_name", "")).lower()
        for source in sources
    )


def answer_has_messy_markdown(answer: str) -> bool:
    return any(marker in answer for marker in MESSY_MARKDOWN_MARKERS)


def analytics_sources_are_sample_outputs(sources: list[dict[str, Any]]) -> bool:
    return bool(sources) and all(
        str(source.get("source_path", "")).startswith(ANALYTICS_SOURCE_PREFIX)
        for source in sources
    )


def evaluate_question(
    question: EvaluationQuestion,
    response: dict[str, Any],
) -> EvaluationResult:
    """Preserve the useful Phase 1 smoke-check behavior."""
    failed_checks: list[str] = []
    passed_checks = 0
    answer = str(response.get("answer", "")).strip()
    sources = list(response.get("sources", []))
    mode = str(response.get("mode", "unknown"))
    if answer:
        passed_checks += 1
    else:
        failed_checks.append("answer is empty")
    if not answer_has_messy_markdown(answer):
        passed_checks += 1
    else:
        failed_checks.append("answer contains raw markdown clutter")

    if question.expected_behavior == "cited_answer":
        if answer and answer != NO_ANSWER:
            passed_checks += 1
        else:
            failed_checks.append("expected a cited answer, got safe no-answer")
        if sources:
            passed_checks += 1
        else:
            failed_checks.append("expected source citations")
        if source_matches_hint(sources, question.expected_source_hint):
            passed_checks += 1
        else:
            failed_checks.append(
                f"missing expected source hint: {question.expected_source_hint}"
            )
    elif question.expected_behavior == "analytics_answer":
        if mode == "analytics":
            passed_checks += 1
        else:
            failed_checks.append("expected analytics route")
        if analytics_sources_are_sample_outputs(sources):
            passed_checks += 1
        else:
            failed_checks.append("expected sample analytics output source")
        if source_matches_hint(sources, question.expected_source_hint):
            passed_checks += 1
        else:
            failed_checks.append(
                f"missing expected analytics source hint: {question.expected_source_hint}"
            )
    elif question.expected_behavior == "safe_no_answer":
        if answer == NO_ANSWER:
            passed_checks += 1
        else:
            failed_checks.append("expected safe no-answer response")
        if not sources or source_matches_hint(sources, question.expected_source_hint):
            passed_checks += 1
        else:
            failed_checks.append("safe no-answer source mismatch")
    else:
        failed_checks.append(
            f"unknown expected_behavior: {question.expected_behavior}"
        )

    return EvaluationResult(
        question=question,
        passed=not failed_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        mode=mode,
    )


def offline_settings(settings: Settings | None = None) -> Settings:
    """Return a deterministic, API-free Settings profile for legacy callers."""
    active_settings = settings or Settings.from_env()
    return replace(
        active_settings,
        embedding_model=DETERMINISTIC_MODEL,
        embedding_provider=DETERMINISTIC_PROVIDER,
        embedding_dimension=DETERMINISTIC_DIMENSION,
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        answer_provider=LOCAL_ANSWER_PROVIDER,
        retrieval_mode="semantic",
        reranking_enabled=False,
    )


def route_application_response(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics":
        return analytics_response
    if looks_like_analytics_question(question):
        return analytics_response
    response = local_answer(question, retrieved_chunks)
    response["mode"] = "rag"
    response.setdefault("sample_rows", [])
    return response


def build_answer_profile_responder(
    settings: Settings,
    answer_profile: str,
    provider: AnswerProvider | None = None,
) -> ApplicationResponder:
    """Reuse the evaluator for an explicitly separate answer-provider profile."""
    if answer_profile == LOCAL_ANSWER_PROVIDER:
        return route_application_response
    if answer_profile != OPENAI_ANSWER_PROVIDER:
        raise ValueError(f"Unsupported answer profile: {answer_profile!r}")

    selected_provider = provider
    if selected_provider is None:
        if settings.answer_provider != OPENAI_ANSWER_PROVIDER:
            raise RuntimeError(
                "Real-provider evaluation requires ANSWER_PROVIDER=openai."
            )
        if not settings.openai_api_key:
            raise RuntimeError(
                "Real-provider evaluation requires configured credentials and was not run."
            )
        selected_provider = build_answer_provider(settings)

    def respond(question: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return generate_answer_from_chunks(
            question,
            retrieved_chunks,
            settings=settings,
            provider=selected_provider,
        )

    return respond


def answer_hybrid_question(
    question: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Backward-compatible Phase 1/Issue 9 local application helper."""
    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics" or looks_like_analytics_question(question):
        return analytics_response
    try:
        retrieved = retrieve_context(question, settings=offline_settings(settings))
    except Exception as exc:
        return {
            "answer": "",
            "sources": [],
            "confidence_note": "Local PostgreSQL/pgvector backend unavailable.",
            "retrieved_chunks": [],
            "sample_rows": [],
            "mode": "backend_error",
            "error_detail": f"{type(exc).__name__}: {exc}",
        }
    return route_application_response(question, retrieved)


def evaluate_questions(
    questions: list[EvaluationQuestion],
    settings: Settings | None = None,
) -> list[EvaluationResult]:
    return [
        evaluate_question(
            question,
            answer_hybrid_question(question.question, settings=settings),
        )
        for question in questions
    ]


def format_summary(results: list[EvaluationResult]) -> str:
    total_questions = len(results)
    passed_questions = sum(1 for result in results if result.passed)
    failed_questions = total_questions - passed_questions
    passed_checks = sum(result.passed_checks for result in results)
    failed_checks = sum(len(result.failed_checks) for result in results)
    lines = [
        "CivicLens RAG Evaluation",
        f"Total questions: {total_questions}",
        f"Passed questions: {passed_questions}",
        f"Failed questions: {failed_questions}",
        f"Passed checks: {passed_checks}",
        f"Failed checks: {failed_checks}",
        "",
        "Per-question status:",
    ]
    for index, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{index}. {status} "
            f"[{result.question.category}/{result.question.expected_behavior}/{result.mode}] "
            f"{result.question.question}"
        )
        lines.extend(f"   - {failure}" for failure in result.failed_checks)
    return "\n".join(lines)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cosine similarity requires vectors with the same dimension")
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_offline_retriever(top_k: int, min_similarity: float) -> Retriever:
    """Build an API-free deterministic regression retriever over checked-in sources."""
    documents = load_documents(ingested_at="deterministic-offline-regression")
    chunks = chunk_documents(documents)
    provider = DeterministicEmbeddingProvider()
    chunk_vectors = provider.embed_many([str(chunk["chunk_text"]) for chunk in chunks])

    def retrieve(question: str, strategy: StrategyDefinition) -> list[dict[str, Any]]:
        if strategy != OFFLINE_STRATEGY:
            raise ValueError("Offline regression accepts only its deterministic strategy")
        question_vector = provider.embed(question)
        scored = [
            (cosine_similarity(question_vector, vector), chunk)
            for vector, chunk in zip(chunk_vectors, chunks, strict=True)
        ]
        scored = [item for item in scored if item[0] >= min_similarity]
        scored.sort(key=lambda item: (-item[0], str(item[1]["chunk_id"])))
        results: list[dict[str, Any]] = []
        for rank, (score, original) in enumerate(scored[:top_k], start=1):
            result = dict(original)
            result.update(
                {
                    "rank": rank,
                    "retrieval_mode": "deterministic_offline_regression",
                    "similarity_score": score,
                    "semantic_score": score,
                    "semantic_rank": rank,
                    "lexical_score": None,
                    "lexical_rank": None,
                    "fusion_score": None,
                    "reranker_score": None,
                    "pre_rerank_rank": None,
                }
            )
            results.append(result)
        return results

    return retrieve


def build_real_retriever(settings: Settings, top_k: int, min_similarity: float) -> Retriever:
    def retrieve(question: str, strategy: StrategyDefinition) -> list[dict[str, Any]]:
        strategy_settings = replace(
            settings,
            retrieval_mode=strategy.retrieval_mode,
            reranking_enabled=strategy.reranking_enabled,
            use_openai_embeddings=False,
            use_openai_answers=False,
        )
        return retrieve_context(
            question,
            top_k=top_k,
            min_similarity=min_similarity,
            settings=strategy_settings,
        )

    return retrieve


def relevance_id(result: dict[str, Any], granularity: str) -> str:
    if granularity == "document":
        return str(result.get("document_id", ""))
    if granularity == "chunk":
        return str(result.get("chunk_id", ""))
    if granularity == "section":
        document_id = str(result.get("document_id", ""))
        section_title = str(result.get("section_title", ""))
        return f"{document_id}::{section_title}" if document_id and section_title else ""
    raise ValueError(f"Unsupported relevance granularity: {granularity!r}")


def citations_in_answer(answer: str) -> list[int]:
    return [int(match) for match in CITATION_PATTERN.findall(answer)]


def citations_are_valid(
    answer: str,
    sources: Sequence[dict[str, Any]],
    retrieved_chunks: Sequence[dict[str, Any]],
) -> bool:
    citations = citations_in_answer(answer)
    if not citations:
        return False
    source_chunk_ids = {str(source.get("chunk_id", "")) for source in sources}
    for citation in citations:
        if citation < 1 or citation > len(retrieved_chunks):
            return False
        cited_chunk_id = str(retrieved_chunks[citation - 1].get("chunk_id", ""))
        if not cited_chunk_id or cited_chunk_id not in source_chunk_ids:
            return False
    return True


def compact_retrieved_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "rank",
            "chunk_id",
            "document_id",
            "source_path",
            "section_title",
            "semantic_score",
            "semantic_rank",
            "lexical_score",
            "lexical_rank",
            "fusion_score",
            "reranker_score",
            "pre_rerank_rank",
        )
    }


def evaluate_strategy_question(
    question: EvaluationQuestion,
    strategy: StrategyDefinition,
    retriever: Retriever,
    top_k: int,
    application_responder: ApplicationResponder = route_application_response,
    answer_profile: str = LOCAL_ANSWER_PROVIDER,
) -> dict[str, Any]:
    analytics_response = answer_analytics_question(question.question)
    if analytics_response["mode"] == "analytics" or looks_like_analytics_question(
        question.question
    ):
        retrieved_chunks: list[dict[str, Any]] = []
        response = analytics_response
        answer_provider_used = False
    else:
        retrieved_chunks = retriever(question.question, strategy)
        response = application_responder(question.question, retrieved_chunks)
        response.setdefault("mode", "rag")
        response.setdefault("sample_rows", [])
        answer_provider_used = True

    answer = str(response.get("answer", "")).strip()
    sources = list(response.get("sources", []))
    actual_route = str(response.get("mode", "unknown"))
    retrieval_eligible = bool(question.relevant_ids)
    retrieved_relevance_ids = (
        [
            identifier
            for result in retrieved_chunks
            if (identifier := relevance_id(result, question.relevance_granularity))
        ]
        if retrieval_eligible
        else []
    )
    retrieved_source_ids = [
        str(result.get("document_id", ""))
        for result in retrieved_chunks
        if result.get("document_id")
    ]

    per_question_recall = (
        recall_at_k(retrieved_relevance_ids, question.relevant_ids, top_k)
        if retrieval_eligible
        else None
    )
    per_question_rr = (
        reciprocal_rank(retrieved_relevance_ids, question.relevant_ids)
        if retrieval_eligible
        else None
    )
    expected_source_retrieved = (
        bool(
            set(question.expected_source_document_ids).intersection(retrieved_source_ids)
        )
        if retrieval_eligible
        else None
    )
    citation_expected = question.expected_behavior == "cited_answer"
    citation_present = bool(citations_in_answer(answer)) if citation_expected else None
    citation_valid = (
        citations_are_valid(answer, sources, retrieved_chunks)
        if citation_expected
        else None
    )
    safe_no_answer_expected = question.expected_behavior == "safe_no_answer"
    safe_no_answer_correct = answer == NO_ANSWER if safe_no_answer_expected else None
    unsupported_answer = bool(
        (safe_no_answer_expected and answer != NO_ANSWER)
        or (
            citation_expected
            and answer != NO_ANSWER
            and (not retrieved_chunks or not citation_valid)
        )
    )
    routing_correct = actual_route == question.expected_route

    failures: list[str] = []
    if retrieval_eligible and per_question_recall != 1.0:
        failures.append("not all relevant section IDs were retrieved at k")
    if retrieval_eligible and not expected_source_retrieved:
        failures.append("expected source document was not retrieved")
    if not routing_correct:
        failures.append("route mismatch")
    if citation_expected and not citation_present:
        failures.append("expected citation is absent")
    if citation_expected and not citation_valid:
        failures.append("citation is invalid")
    if safe_no_answer_expected and not safe_no_answer_correct:
        failures.append("safe no-answer expectation failed")
    if unsupported_answer:
        failures.append("unsupported answer detected")
    if (
        answer_profile == OPENAI_ANSWER_PROVIDER
        and answer_provider_used
        and (
            response.get("answer_provider") != OPENAI_ANSWER_PROVIDER
            or response.get("fallback_used") is True
        )
    ):
        failures.append("real answer provider was not used successfully")

    return {
        "question_id": question.question_id,
        "question": question.question,
        "category": question.category,
        "phase1_legacy": question.phase1_legacy,
        "expected_route": question.expected_route,
        "actual_route": actual_route,
        "expected_answer_behavior": question.expected_behavior,
        "relevance_granularity": question.relevance_granularity or None,
        "relevant_ids": list(question.relevant_ids),
        "retrieved_relevance_ids": retrieved_relevance_ids,
        "retrieval_eligible": retrieval_eligible,
        "recall_at_k": per_question_recall,
        "reciprocal_rank": per_question_rr,
        "expected_source_document_ids": list(question.expected_source_document_ids),
        "expected_section_titles": list(question.expected_section_titles),
        "expected_source_paths": list(question.expected_source_paths),
        "retrieved_source_document_ids": retrieved_source_ids,
        "expected_source_retrieved": expected_source_retrieved,
        "routing_correct": routing_correct,
        "citation_expected": citation_expected,
        "citation_present": citation_present,
        "citation_valid": citation_valid,
        "safe_no_answer_expected": safe_no_answer_expected,
        "safe_no_answer_correct": safe_no_answer_correct,
        "unsupported_answer": unsupported_answer,
        "answer_profile": answer_profile,
        "answer_provider": response.get("answer_provider"),
        "answer_status": response.get("answer_status"),
        "answer_fallback_used": response.get("fallback_used", False),
        "answer": answer,
        "sources": sources,
        "retrieved_results": [
            compact_retrieved_result(result) for result in retrieved_chunks
        ],
        "failures": failures,
    }


def configuration_hash(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strategy_configuration(
    strategy: StrategyDefinition,
    settings: Settings,
    top_k: int,
    min_similarity: float,
    profile: str,
) -> dict[str, Any]:
    if profile == "offline":
        configuration: dict[str, Any] = {
            "retrieval_mode": strategy.retrieval_mode,
            "embedding_provider": DETERMINISTIC_PROVIDER,
            "embedding_model": DETERMINISTIC_MODEL,
            "embedding_dimension": DETERMINISTIC_DIMENSION,
            "top_k": top_k,
            "semantic_candidate_count": top_k,
            "lexical_candidate_count": "not_applicable",
            "rrf_k": "not_applicable",
            "reranking_enabled": False,
            "reranker_model": "not_applicable",
            "rerank_candidate_limit": "not_applicable",
            "min_similarity": min_similarity,
        }
    else:
        configuration = {
            "retrieval_mode": strategy.retrieval_mode,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "top_k": top_k,
            "semantic_candidate_count": settings.semantic_candidate_count,
            "lexical_candidate_count": settings.lexical_candidate_count,
            "rrf_k": settings.rrf_k,
            "reranking_enabled": strategy.reranking_enabled,
            "reranker_model": (
                settings.reranker_model if strategy.reranking_enabled else "disabled"
            ),
            "rerank_candidate_limit": settings.rerank_candidate_limit,
            "min_similarity": min_similarity,
        }
    configuration["configuration_hash"] = configuration_hash(configuration)
    return configuration


def evaluate_strategy(
    questions: Sequence[EvaluationQuestion],
    strategy: StrategyDefinition,
    retriever: Retriever,
    settings: Settings,
    top_k: int,
    min_similarity: float,
    profile: str,
    application_responder: ApplicationResponder = route_application_response,
    answer_profile: str = LOCAL_ANSWER_PROVIDER,
) -> dict[str, Any]:
    question_results = [
        evaluate_strategy_question(
            question,
            strategy,
            retriever,
            top_k,
            application_responder=application_responder,
            answer_profile=answer_profile,
        )
        for question in questions
    ]
    failed_cases = [
        {
            "strategy": strategy.name,
            "question_id": result["question_id"],
            "question": result["question"],
            "failures": result["failures"],
            "expected_route": result["expected_route"],
            "expected_answer_behavior": result["expected_answer_behavior"],
            "actual_route": result["actual_route"],
            "retrieved_relevance_ids": result["retrieved_relevance_ids"],
            "retrieved_source_document_ids": result[
                "retrieved_source_document_ids"
            ],
        }
        for result in question_results
        if result["failures"]
    ]
    return {
        "name": strategy.name,
        "configuration": strategy_configuration(
            strategy, settings, top_k, min_similarity, profile
        ),
        "retrieval_metrics": aggregate_retrieval_metrics(question_results),
        "application_metrics": aggregate_application_metrics(question_results),
        "failed_cases": failed_cases,
        "question_results": question_results,
    }


def ensure_cached_model(model_name: str) -> str:
    """Resolve a model from local cache only; never contact a registry."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    try:
        return str(snapshot_download(model_name, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"Required local model {model_name!r} is not available in the cache; "
            "the evaluator will not download it automatically."
        ) from exc


def ensure_real_profile_available(settings: Settings) -> None:
    if settings.embedding_provider != SEMANTIC_PROVIDER:
        raise RuntimeError(
            "Real evaluation requires EMBEDDING_PROVIDER=sentence_transformers; "
            f"found {settings.embedding_provider!r}."
        )
    if settings.embedding_model != DEFAULT_SEMANTIC_MODEL:
        raise RuntimeError(
            "Real evaluation must use the documented Issue 9 default model "
            f"{DEFAULT_SEMANTIC_MODEL!r}; found {settings.embedding_model!r}."
        )
    if settings.embedding_dimension != DEFAULT_SEMANTIC_DIMENSION:
        raise RuntimeError(
            "Real evaluation requires the Issue 9 storage dimension "
            f"{DEFAULT_SEMANTIC_DIMENSION}; found {settings.embedding_dimension}."
        )
    ensure_cached_model(settings.embedding_model)
    ensure_cached_model(settings.reranker_model or DEFAULT_RERANKER_MODEL)

    import psycopg

    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(
            "Real evaluation requires the prepared local PostgreSQL/pgvector backend."
        ) from exc


def build_report(
    questions: Sequence[EvaluationQuestion],
    strategies: Sequence[dict[str, Any]],
    profile: str,
    evaluation_timestamp: str,
    top_k: int,
    answer_profile: str = LOCAL_ANSWER_PROVIDER,
    answer_model: str = "deterministic-context-extractor-v1",
) -> dict[str, Any]:
    dataset_version = questions[0].dataset_version if questions else "unknown"
    granularity = validate_relevance_granularity(questions)
    if profile == "offline":
        boundary = (
            "Deterministic/offline regression only. Hash-based embeddings and an "
            "in-memory cosine search validate repeatability and framework behavior; "
            "these scores are not evidence of Sentence Transformers semantic quality."
        )
    else:
        boundary = (
            "Real local Issue 9 comparison using cached Sentence Transformers, "
            "PostgreSQL/pgvector, PostgreSQL FTS, RRF, and the cached optional "
            "cross-encoder. No LLM judge or paid API is used."
        )
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_profile": profile,
        "evaluation_timestamp": evaluation_timestamp,
        "top_k": top_k,
        "dataset": {
            "version": dataset_version,
            "path": str(DEFAULT_EVALUATION_PATH.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "question_count": len(questions),
            "phase1_legacy_question_count": sum(
                question.phase1_legacy for question in questions
            ),
            "advanced_question_count": sum(
                not question.phase1_legacy for question in questions
            ),
            "eligible_retrieval_question_count": sum(
                bool(question.relevant_ids) for question in questions
            ),
            "relevance_granularity": granularity,
        },
        "interpretation_boundary": boundary,
        "strategies": list(strategies),
        "limitations": [
            "This is a small curated portfolio benchmark, not production-scale evaluation.",
            "It is not a large human-annotated benchmark and does not establish statistical significance.",
            "No LLM is used as a judge; checks cover explicit retrieval IDs and deterministic application behavior only.",
            "Real LLM evaluation remains deferred until after Issue 11.",
            "Higher scores do not by themselves prove production quality or readiness.",
        ],
    }
    if answer_profile == OPENAI_ANSWER_PROVIDER:
        report["answer_evaluation"] = {
            "profile": OPENAI_ANSWER_PROVIDER,
            "provider": OPENAI_ANSWER_PROVIDER,
            "model": answer_model,
            "separate_from_deterministic_baseline": True,
        }
        report["interpretation_boundary"] += (
            " Answer generation used the explicitly selected OpenAI profile; "
            "provider failures and local fallbacks remain visible per question."
        )
    return report


def run_evaluation(
    questions: Sequence[EvaluationQuestion],
    profile: str,
    settings: Settings,
    top_k: int,
    min_similarity: float,
    evaluation_timestamp: str,
    answer_profile: str = LOCAL_ANSWER_PROVIDER,
    answer_provider: AnswerProvider | None = None,
) -> dict[str, Any]:
    if answer_profile == OPENAI_ANSWER_PROVIDER and profile != "real":
        raise ValueError(
            "OpenAI answer evaluation must remain separate and requires --profile real."
        )
    application_responder = build_answer_profile_responder(
        settings,
        answer_profile,
        provider=answer_provider,
    )
    if profile == "offline":
        strategies = (OFFLINE_STRATEGY,)
        retriever = build_offline_retriever(top_k, min_similarity)
    elif profile == "real":
        ensure_real_profile_available(settings)
        strategies = REAL_STRATEGIES
        retriever = build_real_retriever(settings, top_k, min_similarity)
    else:
        raise ValueError(f"Unsupported evaluation profile: {profile!r}")

    strategy_results = [
        evaluate_strategy(
            questions,
            strategy,
            retriever,
            settings,
            top_k,
            min_similarity,
            profile,
            application_responder=application_responder,
            answer_profile=answer_profile,
        )
        for strategy in strategies
    ]
    return build_report(
        questions,
        strategy_results,
        profile,
        evaluation_timestamp,
        top_k,
        answer_profile=answer_profile,
        answer_model=(
            settings.answer_model
            if answer_profile == OPENAI_ANSWER_PROVIDER
            else "deterministic-context-extractor-v1"
        ),
    )


def print_report_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Evaluation profile: {report['evaluation_profile']}",
        f"Dataset version: {report['dataset']['version']}",
        f"Questions: {report['dataset']['question_count']}",
        f"Relevance granularity: {report['dataset']['relevance_granularity']}",
    ]
    for strategy in report["strategies"]:
        retrieval = strategy["retrieval_metrics"]
        application = strategy["application_metrics"]
        lines.extend(
            [
                f"Strategy: {strategy['name']}",
                f"  Recall@{report['top_k']}: {retrieval['recall_at_k']['value']}",
                f"  MRR: {retrieval['mrr']['value']}",
                "  Expected-source retrieval: "
                f"{retrieval['expected_source_retrieval']['value']}",
                f"  Routing accuracy: {application['routing_accuracy']['value']}",
                f"  Citation presence: {application['citation_presence']['value']}",
                f"  Citation validity: {application['citation_validity']['value']}",
                "  Safe no-answer accuracy: "
                f"{application['safe_no_answer_accuracy']['value']}",
                "  Unsupported answers: "
                f"{application['unsupported_answer']['count']}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeatable CivicLens RAG retrieval and behavior evaluation."
    )
    parser.add_argument("--profile", choices=("offline", "real"), default="offline")
    parser.add_argument("--questions-path", default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--output-stem", default="")
    parser.add_argument(
        "--answer-profile",
        choices=(LOCAL_ANSWER_PROVIDER, OPENAI_ANSWER_PROVIDER),
        default=LOCAL_ANSWER_PROVIDER,
        help="Keep local baseline answers separate from optional OpenAI evaluation.",
    )
    args = parser.parse_args()

    if args.profile == "offline":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    questions = load_evaluation_questions(args.questions_path)
    evaluation_timestamp = args.timestamp or utc_timestamp()
    settings = Settings.from_env()
    report = run_evaluation(
        questions,
        profile=args.profile,
        settings=settings,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        evaluation_timestamp=evaluation_timestamp,
        answer_profile=args.answer_profile,
    )
    default_stem = (
        f"{args.profile}-evaluation"
        if args.answer_profile == LOCAL_ANSWER_PROVIDER
        else f"{args.profile}-{args.answer_profile}-answer-evaluation"
    )
    stem = args.output_stem or default_stem
    markdown_path, json_path = write_reports(report, args.output_dir, stem)
    print(print_report_summary(report))
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
