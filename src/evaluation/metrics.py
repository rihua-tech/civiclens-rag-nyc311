"""Reusable deterministic metrics for the CivicLens evaluation framework."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def unique_ranked_ids(retrieved_ids: Iterable[str]) -> list[str]:
    """Deduplicate ranked relevance IDs without changing their first rank."""
    seen: set[str] = set()
    ordered: list[str] = []
    for relevance_id in retrieved_ids:
        normalized = str(relevance_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Return |relevant intersect retrieved@k| / |relevant|.

    Multiple relevant IDs are supported. Questions without relevance labels are
    ineligible and must be excluded by the caller instead of being assigned zero.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = {str(relevance_id).strip() for relevance_id in relevant_ids if str(relevance_id).strip()}
    if not relevant:
        raise ValueError("Recall@k requires at least one relevant ID")
    retrieved = set(unique_ranked_ids(retrieved_ids)[:k])
    return len(relevant.intersection(retrieved)) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """Return reciprocal rank of the first relevant retrieved ID, or zero."""
    relevant = {str(relevance_id).strip() for relevance_id in relevant_ids if str(relevance_id).strip()}
    if not relevant:
        raise ValueError("Reciprocal rank requires at least one relevant ID")
    for rank, relevance_id in enumerate(unique_ranked_ids(retrieved_ids), start=1):
        if relevance_id in relevant:
            return 1.0 / rank
    return 0.0


def macro_metric(values: Iterable[float]) -> dict[str, float | int | None]:
    """Aggregate per-question values with an explicit eligible denominator."""
    collected = [float(value) for value in values]
    denominator = len(collected)
    numerator = sum(collected)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def boolean_metric(values: Iterable[bool]) -> dict[str, float | int | None]:
    """Aggregate boolean outcomes with numerator and denominator metadata."""
    collected = [bool(value) for value in values]
    denominator = len(collected)
    numerator = sum(collected)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def aggregate_retrieval_metrics(
    question_results: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """Macro-average eligible per-question retrieval metrics."""
    eligible = [result for result in question_results if result.get("retrieval_eligible")]
    return {
        "recall_at_k": macro_metric(result["recall_at_k"] for result in eligible),
        "mrr": macro_metric(result["reciprocal_rank"] for result in eligible),
        "expected_source_retrieval": boolean_metric(
            result["expected_source_retrieved"] for result in eligible
        ),
    }


def aggregate_application_metrics(
    question_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Keep routing, citations, no-answer, and unsupported behavior separate."""
    citation_eligible = [
        result for result in question_results if result.get("citation_expected")
    ]
    no_answer_eligible = [
        result for result in question_results if result.get("safe_no_answer_expected")
    ]
    unsupported_count = sum(
        bool(result.get("unsupported_answer")) for result in question_results
    )
    total = len(question_results)
    return {
        "routing_accuracy": boolean_metric(
            result["routing_correct"] for result in question_results
        ),
        "citation_presence": boolean_metric(
            result["citation_present"] for result in citation_eligible
        ),
        "citation_validity": boolean_metric(
            result["citation_valid"] for result in citation_eligible
        ),
        "safe_no_answer_accuracy": boolean_metric(
            result["safe_no_answer_correct"] for result in no_answer_eligible
        ),
        "unsupported_answer": {
            "count": unsupported_count,
            "rate": unsupported_count / total if total else None,
            "denominator": total,
        },
    }
