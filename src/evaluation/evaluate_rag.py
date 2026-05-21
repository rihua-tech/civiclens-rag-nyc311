"""Basic local evaluation checks for CivicLens RAG.

This is a lightweight integration smoke test for the local hybrid path. It
does not call OpenAI, external APIs, or live NYC 311 data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analytics.simple_analytics import answer_analytics_question, looks_like_analytics_question
from src.common.config import Settings
from src.generation.answer_question import NO_ANSWER, answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_test_questions.csv"
REQUIRED_COLUMNS = {"question", "category", "expected_behavior", "expected_source_hint"}
MESSY_MARKDOWN_MARKERS = ("```", "##", "# ")
ANALYTICS_SOURCE_PREFIX = "data/sample_outputs/"
NO_SOURCE_HINT = "none"


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    category: str
    expected_behavior: str
    expected_source_hint: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    question: EvaluationQuestion
    passed: bool
    passed_checks: int
    failed_checks: list[str]
    mode: str


def load_evaluation_questions(path: str | Path = DEFAULT_EVALUATION_PATH) -> list[EvaluationQuestion]:
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

        questions = [
            EvaluationQuestion(
                question=row["question"].strip(),
                category=row["category"].strip(),
                expected_behavior=row["expected_behavior"].strip(),
                expected_source_hint=row.get("expected_source_hint", "").strip(),
            )
            for row in reader
            if row.get("question", "").strip()
        ]

    return questions


def offline_settings(settings: Settings | None = None) -> Settings:
    active_settings = settings or Settings.from_env()
    return Settings(
        database_url=active_settings.database_url,
        embedding_model=active_settings.embedding_model,
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
    )


def answer_hybrid_question(question: str, settings: Settings | None = None) -> dict[str, Any]:
    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics":
        return analytics_response
    if looks_like_analytics_question(question):
        return analytics_response

    try:
        response = answer_question(question, settings=offline_settings(settings))
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

    response["mode"] = "rag"
    response.setdefault("sample_rows", [])
    return response


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


def evaluate_question(question: EvaluationQuestion, response: dict[str, Any]) -> EvaluationResult:
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

    expected_behavior = question.expected_behavior
    if expected_behavior == "cited_answer":
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
            failed_checks.append(f"missing expected source hint: {question.expected_source_hint}")
    elif expected_behavior == "analytics_answer":
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
            failed_checks.append(f"missing expected analytics source hint: {question.expected_source_hint}")
    elif expected_behavior == "safe_no_answer":
        if answer == NO_ANSWER:
            passed_checks += 1
        else:
            failed_checks.append("expected safe no-answer response")
        if source_matches_hint(sources, question.expected_source_hint):
            passed_checks += 1
        else:
            failed_checks.append(f"safe no-answer source hint mismatch: {question.expected_source_hint}")
    else:
        failed_checks.append(f"unknown expected_behavior: {expected_behavior}")

    return EvaluationResult(
        question=question,
        passed=not failed_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        mode=mode,
    )


def evaluate_questions(questions: list[EvaluationQuestion], settings: Settings | None = None) -> list[EvaluationResult]:
    return [
        evaluate_question(question, answer_hybrid_question(question.question, settings=settings))
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
            f"{index}. {status} [{result.question.category}/{result.question.expected_behavior}/{result.mode}] "
            f"{result.question.question}"
        )
        for failed_check in result.failed_checks:
            lines.append(f"   - {failed_check}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local CivicLens RAG evaluation checks.")
    parser.add_argument(
        "--questions-path",
        default=DEFAULT_EVALUATION_PATH,
        help="Path to the evaluation questions CSV.",
    )
    args = parser.parse_args()

    questions = load_evaluation_questions(args.questions_path)
    results = evaluate_questions(questions)
    print(format_summary(results))

    return 1 if any(not result.passed for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
