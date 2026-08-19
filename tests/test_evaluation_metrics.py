import pytest

from src.evaluation.evaluate_rag import EvaluationQuestion, validate_relevance_granularity
from src.evaluation.metrics import (
    aggregate_application_metrics,
    aggregate_retrieval_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_first_result_relevant():
    assert recall_at_k(["relevant", "other"], ["relevant"], 2) == 1.0


def test_recall_at_k_lower_ranked_relevant():
    assert recall_at_k(["other", "relevant"], ["relevant"], 2) == 1.0
    assert recall_at_k(["other", "relevant"], ["relevant"], 1) == 0.0


def test_recall_at_k_no_relevant_result():
    assert recall_at_k(["other"], ["relevant"], 5) == 0.0


def test_recall_at_k_supports_multiple_relevant_ids():
    assert recall_at_k(["a", "other", "b"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)


def test_recall_at_k_deduplicates_retrieved_ids():
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 1.0


def test_recall_rejects_missing_relevance_labels():
    with pytest.raises(ValueError, match="at least one relevant ID"):
        recall_at_k(["a"], [], 1)


def test_mrr_uses_first_relevant_result_and_zero_for_miss():
    assert reciprocal_rank(["other", "b", "a"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["other"], ["a", "b"]) == 0.0


def test_retrieval_aggregation_excludes_ineligible_questions():
    results = [
        {
            "retrieval_eligible": True,
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "expected_source_retrieved": True,
        },
        {
            "retrieval_eligible": True,
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "expected_source_retrieved": False,
        },
        {"retrieval_eligible": False},
    ]

    metrics = aggregate_retrieval_metrics(results)

    assert metrics["recall_at_k"] == {
        "value": 0.5,
        "numerator": 1.0,
        "denominator": 2,
    }
    assert metrics["mrr"]["denominator"] == 2
    assert metrics["expected_source_retrieval"]["denominator"] == 2


def test_relevance_granularity_cannot_be_silently_mixed():
    questions = [
        EvaluationQuestion(
            "one",
            "test",
            "cited_answer",
            relevance_granularity="section",
            relevant_ids=("doc::section",),
        ),
        EvaluationQuestion(
            "two",
            "test",
            "cited_answer",
            relevance_granularity="document",
            relevant_ids=("doc",),
        ),
    ]

    with pytest.raises(ValueError, match="cannot silently mix"):
        validate_relevance_granularity(questions)


def test_application_metrics_keep_denominators_separate():
    results = [
        {
            "routing_correct": True,
            "citation_expected": True,
            "citation_present": True,
            "citation_valid": False,
            "safe_no_answer_expected": False,
            "unsupported_answer": True,
        },
        {
            "routing_correct": False,
            "citation_expected": False,
            "safe_no_answer_expected": True,
            "safe_no_answer_correct": True,
            "unsupported_answer": False,
        },
    ]

    metrics = aggregate_application_metrics(results)

    assert metrics["routing_accuracy"]["value"] == 0.5
    assert metrics["citation_presence"]["denominator"] == 1
    assert metrics["citation_validity"]["value"] == 0.0
    assert metrics["safe_no_answer_accuracy"]["denominator"] == 1
    assert metrics["unsupported_answer"] == {
        "count": 1,
        "rate": 0.5,
        "denominator": 2,
    }
