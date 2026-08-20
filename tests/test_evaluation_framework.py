import json
from pathlib import Path

from src.common.config import Settings
from src.evaluation.evaluate_rag import (
    DEFAULT_EVALUATION_PATH,
    DEFAULT_RESULTS_DIR,
    OFFLINE_STRATEGY,
    REAL_STRATEGIES,
    EvaluationQuestion,
    StrategyDefinition,
    build_answer_profile_responder,
    build_report,
    citations_are_valid,
    evaluate_strategy,
    evaluate_strategy_question,
    load_evaluation_questions,
    run_evaluation,
)
from src.evaluation.reporting import markdown_report, write_reports
from src.generation.schemas import AnswerStatus, ProviderResult


def evaluation_settings() -> Settings:
    return Settings(
        database_url="postgresql://unused",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="sentence_transformers",
        embedding_dimension=384,
        retrieval_mode="hybrid",
        semantic_candidate_count=12,
        lexical_candidate_count=10,
        rrf_k=60,
        reranking_enabled=False,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_candidate_limit=8,
    )


def cited_question() -> EvaluationQuestion:
    return EvaluationQuestion(
        question="What does complaint type mean?",
        category="data_dictionary",
        expected_behavior="cited_answer",
        question_id="q-test",
        dataset_version="test-v1",
        phase1_legacy=False,
        expected_route="rag",
        relevance_granularity="section",
        relevant_ids=("doc-guide::Problem",),
        expected_source_document_ids=("doc-guide",),
        expected_section_titles=("Problem",),
        expected_source_paths=("docs/guide.md",),
    )


def retrieved_chunk() -> dict:
    return {
        "rank": 1,
        "chunk_id": "chunk-guide",
        "document_id": "doc-guide",
        "chunk_text": (
            "Complaint type describes the broad category of the reported NYC 311 "
            "problem."
        ),
        "source_name": "guide",
        "source_path": "docs/guide.md",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "section_title": "Problem",
        "heading_path": ["Problem"],
        "semantic_score": 0.75,
        "semantic_rank": 1,
    }


def test_issue10_dataset_has_stable_version_ids_and_section_relevance():
    questions = load_evaluation_questions(DEFAULT_EVALUATION_PATH)

    assert {question.dataset_version for question in questions} == {"issue10-v1"}
    assert len({question.question_id for question in questions}) == len(questions)
    eligible = [question for question in questions if question.relevant_ids]
    assert eligible
    assert {question.relevance_granularity for question in eligible} == {"section"}
    assert any(len(question.relevant_ids) > 1 for question in eligible)
    assert any(question.phase1_legacy for question in questions)
    assert any(not question.phase1_legacy for question in questions)


def test_issue10_dataset_covers_required_question_types():
    categories = {
        question.category for question in load_evaluation_questions(DEFAULT_EVALUATION_PATH)
    }

    assert {
        "architecture",
        "runbook",
        "data_dictionary",
        "analytics",
        "no_answer",
        "negative",
        "adversarial",
    } <= categories


def test_strategy_question_preserves_diagnostics_and_valid_citation():
    def retriever(_question: str, _strategy: StrategyDefinition) -> list[dict]:
        return [retrieved_chunk()]

    result = evaluate_strategy_question(
        cited_question(), REAL_STRATEGIES[0], retriever, top_k=5
    )

    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank"] == 1.0
    assert result["citation_present"] is True
    assert result["citation_valid"] is True
    assert result["routing_correct"] is True
    assert result["expected_section_titles"] == ["Problem"]
    assert result["retrieved_results"][0]["semantic_score"] == 0.75
    assert result["failures"] == []


def test_fabricated_citation_is_rejected():
    chunk = retrieved_chunk()
    sources = [{"chunk_id": chunk["chunk_id"]}]

    assert citations_are_valid("Supported text [1].", sources, [chunk])
    assert not citations_are_valid("Fabricated citation [99].", sources, [chunk])


def test_safe_no_answer_and_unsupported_answer_are_detected():
    question = EvaluationQuestion(
        question="What does complaint type mean?",
        category="negative",
        expected_behavior="safe_no_answer",
        question_id="q-negative",
        dataset_version="test-v1",
        phase1_legacy=False,
        expected_route="rag",
    )

    def supported_retriever(
        _question: str, _strategy: StrategyDefinition
    ) -> list[dict]:
        return [retrieved_chunk()]

    answered = evaluate_strategy_question(
        question, REAL_STRATEGIES[0], supported_retriever, top_k=5
    )
    abstained = evaluate_strategy_question(
        question, REAL_STRATEGIES[0], lambda _question, _strategy: [], top_k=5
    )

    assert answered["safe_no_answer_correct"] is False
    assert answered["unsupported_answer"] is True
    assert "unsupported answer detected" in answered["failures"]
    assert abstained["safe_no_answer_correct"] is True
    assert abstained["unsupported_answer"] is False


def test_strategy_results_remain_separate_and_keep_configuration():
    def retriever(_question: str, strategy: StrategyDefinition) -> list[dict]:
        chunk = retrieved_chunk()
        chunk["retrieval_mode"] = strategy.retrieval_mode
        if strategy.reranking_enabled:
            chunk["reranker_score"] = 0.9
            chunk["pre_rerank_rank"] = 2
        return [chunk]

    settings = evaluation_settings()
    results = [
        evaluate_strategy(
            [cited_question()],
            strategy,
            retriever,
            settings,
            top_k=5,
            min_similarity=0.25,
            profile="real",
        )
        for strategy in REAL_STRATEGIES
    ]

    assert [result["name"] for result in results] == [
        "semantic",
        "hybrid",
        "hybrid_rerank",
    ]
    assert len({result["configuration"]["configuration_hash"] for result in results}) == 3
    assert results[0]["configuration"]["retrieval_mode"] == "semantic"
    assert results[1]["configuration"]["retrieval_mode"] == "hybrid"
    assert results[2]["configuration"]["reranking_enabled"] is True
    assert results[2]["question_results"][0]["retrieved_results"][0][
        "pre_rerank_rank"
    ] == 2


def test_failed_case_diagnostics_are_preserved():
    def empty_retriever(_question: str, _strategy: StrategyDefinition) -> list[dict]:
        return []

    result = evaluate_strategy(
        [cited_question()],
        REAL_STRATEGIES[0],
        empty_retriever,
        evaluation_settings(),
        top_k=5,
        min_similarity=0.25,
        profile="real",
    )

    assert result["failed_cases"][0]["question_id"] == "q-test"
    assert "expected citation is absent" in result["failed_cases"][0]["failures"]
    assert "unsupported answer detected" not in result["failed_cases"][0]["failures"]


def test_offline_profile_uses_only_deterministic_strategy(monkeypatch):
    calls: list[StrategyDefinition] = []

    def fake_builder(_top_k: int, _min_similarity: float):
        def retrieve(_question: str, strategy: StrategyDefinition) -> list[dict]:
            calls.append(strategy)
            return [retrieved_chunk()]

        return retrieve

    monkeypatch.setattr(
        "src.evaluation.evaluate_rag.build_offline_retriever", fake_builder
    )
    report = run_evaluation(
        [cited_question()],
        profile="offline",
        settings=evaluation_settings(),
        top_k=5,
        min_similarity=0.25,
        evaluation_timestamp="2026-08-19T00:00:00Z",
    )

    assert calls == [OFFLINE_STRATEGY]
    configuration = report["strategies"][0]["configuration"]
    assert configuration["embedding_provider"] == "deterministic"
    assert configuration["reranking_enabled"] is False
    assert "not evidence of Sentence Transformers" in report["interpretation_boundary"]


def test_reports_include_metadata_denominators_and_per_question_results(tmp_path):
    def retriever(_question: str, _strategy: StrategyDefinition) -> list[dict]:
        return [retrieved_chunk()]

    strategy = evaluate_strategy(
        [cited_question()],
        REAL_STRATEGIES[0],
        retriever,
        evaluation_settings(),
        top_k=5,
        min_similarity=0.25,
        profile="real",
    )
    report = build_report(
        [cited_question()],
        [strategy],
        profile="real",
        evaluation_timestamp="2026-08-19T00:00:00Z",
        top_k=5,
    )

    rendered = markdown_report(report)
    markdown_path, json_path = write_reports(report, tmp_path, "test-run")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "Dataset version: `test-v1`" in rendered
    assert "Retrieval relevance granularity: `section`" in rendered
    assert "configuration_hash" in rendered
    assert markdown_path.parent == tmp_path
    assert payload["strategies"][0]["retrieval_metrics"]["recall_at_k"][
        "denominator"
    ] == 1
    assert payload["strategies"][0]["question_results"][0]["question_id"] == "q-test"
    assert DEFAULT_RESULTS_DIR.name == "results"
    assert Path("docs/evaluation-report.md") not in (markdown_path, json_path)


def test_optional_real_provider_evaluation_reuses_framework_with_fake():
    class FakeOpenAIProvider:
        provider_name = "openai"
        model_name = "fake-openai-model"

        def generate(self, _question, _evidence):
            return ProviderResult(
                "Complaint type is the broad problem category.",
                ("chunk-guide",),
                AnswerStatus.ANSWERED,
            )

    def retriever(_question: str, _strategy: StrategyDefinition) -> list[dict]:
        return [retrieved_chunk()]

    settings = evaluation_settings()
    responder = build_answer_profile_responder(
        settings,
        answer_profile="openai",
        provider=FakeOpenAIProvider(),
    )
    strategy = evaluate_strategy(
        [cited_question()],
        REAL_STRATEGIES[0],
        retriever,
        settings,
        top_k=5,
        min_similarity=0.25,
        profile="real",
        application_responder=responder,
        answer_profile="openai",
    )
    report = build_report(
        [cited_question()],
        [strategy],
        profile="real",
        evaluation_timestamp="2026-08-19T00:00:00Z",
        top_k=5,
        answer_profile="openai",
        answer_model="fake-openai-model",
    )

    question_result = strategy["question_results"][0]
    assert question_result["answer_provider"] == "openai"
    assert question_result["answer_fallback_used"] is False
    assert question_result["citation_valid"] is True
    assert question_result["failures"] == []
    assert report["answer_evaluation"] == {
        "profile": "openai",
        "provider": "openai",
        "model": "fake-openai-model",
        "separate_from_deterministic_baseline": True,
    }
