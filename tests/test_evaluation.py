from pathlib import Path

from src.evaluation.evaluate_rag import (
    DEFAULT_EVALUATION_PATH,
    REQUIRED_COLUMNS,
    EvaluationQuestion,
    analytics_sources_are_sample_outputs,
    answer_has_messy_markdown,
    evaluate_question,
    load_evaluation_questions,
    source_matches_hint,
)
from src.generation.answer_question import NO_ANSWER


def test_evaluation_csv_exists():
    assert DEFAULT_EVALUATION_PATH.is_file()


def test_evaluation_csv_has_required_columns():
    header = Path(DEFAULT_EVALUATION_PATH).read_text(encoding="utf-8").splitlines()[0].split(",")

    assert REQUIRED_COLUMNS <= set(header)


def test_evaluation_csv_has_minimum_question_count_and_categories():
    questions = load_evaluation_questions()
    categories = {question.category for question in questions}

    assert len(questions) >= 10
    assert {"architecture", "analytics", "no_answer"} <= categories


def test_evaluation_questions_have_explicit_source_hints():
    questions = load_evaluation_questions()

    assert all(question.expected_source_hint for question in questions)


def test_markdown_clutter_detection():
    assert answer_has_messy_markdown("## Raw heading")
    assert answer_has_messy_markdown("```text\nraw code\n```")
    assert not answer_has_messy_markdown("A concise answer with citations [1].")


def test_source_hint_matching():
    sources = [{"source_name": "architecture.md", "source_path": "docs/architecture.md"}]

    assert source_matches_hint(sources, "docs/architecture.md")
    assert not source_matches_hint(sources, "docs/rag-design.md")
    assert source_matches_hint([], "none")
    assert not source_matches_hint(sources, "none")


def test_analytics_source_detection():
    sources = [{"source_path": "data/sample_outputs/top_complaint_types.csv"}]

    assert analytics_sources_are_sample_outputs(sources)
    assert not analytics_sources_are_sample_outputs([{"source_path": "docs/rag-design.md"}])


def test_evaluate_question_accepts_cited_answer_response():
    question = EvaluationQuestion(
        question="What is the architecture?",
        category="architecture",
        expected_behavior="cited_answer",
        expected_source_hint="docs/architecture.md",
    )
    response = {
        "answer": "The architecture uses ingestion, embeddings, pgvector, and a cited UI. [1]",
        "sources": [{"source_name": "architecture.md", "source_path": "docs/architecture.md"}],
        "mode": "rag",
    }

    result = evaluate_question(question, response)

    assert result.passed


def test_evaluate_question_accepts_analytics_response():
    question = EvaluationQuestion(
        question="What are the top complaint types?",
        category="analytics",
        expected_behavior="analytics_answer",
        expected_source_hint="data/sample_outputs/top_complaint_types.csv",
    )
    response = {
        "answer": "The top sample complaint types are Noise - Residential and HEAT/HOT WATER.",
        "sources": [{"source_path": "data/sample_outputs/top_complaint_types.csv"}],
        "mode": "analytics",
    }

    result = evaluate_question(question, response)

    assert result.passed


def test_evaluate_question_accepts_safe_no_answer_response():
    question = EvaluationQuestion(
        question="What is zqxjv plern mivrok blasson?",
        category="no_answer",
        expected_behavior="safe_no_answer",
    )
    response = {
        "answer": NO_ANSWER,
        "sources": [],
        "mode": "rag",
    }

    result = evaluate_question(question, response)

    assert result.passed
