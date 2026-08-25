from datetime import datetime, timezone

from src.common.config import Settings
from src.observability.query_logger import (
    PostgresQueryLogger,
    build_query_observation,
)
from src.orchestration.question_router import route_question


QUERY_ID = "11111111-1111-4111-8111-111111111111"


def _settings(enabled=True):
    return Settings(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        retrieval_mode="hybrid",
        answer_provider="local",
        observability_enabled=enabled,
    )


def _retrieved_chunk():
    return {
        "chunk_id": "chunk_abc",
        "document_id": "doc_abc",
        "chunk_text": "raw evidence must not be logged",
        "embedding": [0.1, 0.2],
        "source_name": "Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/field-guide.md",
        "source_url": "https://example.invalid/field-guide",
        "section_title": "Complaint Type",
        "heading_path": ["Guide", "Complaint Type"],
        "content_hash": "sha256:chunk",
        "document_content_hash": "sha256:document",
        "retrieval_mode": "hybrid",
        "rank": 1,
        "similarity_score": 0.8,
        "semantic_score": 0.8,
        "semantic_rank": 1,
        "lexical_score": 0.4,
        "lexical_rank": 2,
        "fusion_score": 0.03,
        "reranker_score": None,
        "pre_rerank_rank": None,
        "provider_payload": {"api_key": "sk-test-secret-value"},
    }


class CapturingLogger:
    def __init__(self, fail=False):
        self.fail = fail
        self.observations = []

    def record_execution(self, observation):
        if self.fail:
            raise RuntimeError("database password detail")
        self.observations.append(observation)


def test_one_query_id_traces_route_generation_and_retrieval(monkeypatch):
    captured = {}

    def fake_answer(question, top_k, settings, query_id):
        captured.update(query_id=query_id, question=question, top_k=top_k)
        return {
            "answer": "Grounded answer [1]",
            "sources": [],
            "retrieved_chunks": [_retrieved_chunk()],
            "answer_status": "answered",
            "answer_provider": "local",
            "answer_model": "deterministic-context-extractor-v1",
            "query_id": query_id,
        }

    monkeypatch.setattr("src.orchestration.question_router.answer_question", fake_answer)
    logger = CapturingLogger()
    times = iter((10.0, 10.125))

    result = route_question(
        "What does complaint_type mean?",
        top_k=5,
        settings=_settings(),
        query_logger=logger,
        query_id_factory=lambda: QUERY_ID,
        clock=lambda: next(times),
    )

    observation = logger.observations[0]
    assert result["query_id"] == QUERY_ID
    assert captured["query_id"] == QUERY_ID
    assert observation.query_id == QUERY_ID
    assert observation.retrieval_results[0].query_id == QUERY_ID
    assert observation.latency_ms == 125.0
    assert observation.route == "rag"
    assert observation.retrieval_strategy == "hybrid"
    assert observation.orchestration_mode == "direct"
    assert observation.orchestration_step_count == 2
    assert observation.orchestration_tool_call_count == 0
    assert observation.orchestration_outcome == "answered"


def test_observability_disabled_writes_nothing():
    logger = CapturingLogger(fail=True)

    result = route_question(
        "What are the top complaint types?",
        settings=_settings(enabled=False),
        query_logger=logger,
    )

    assert result["mode"] == "analytics"
    assert "query_id" not in result
    assert logger.observations == []


def test_logging_failure_does_not_break_a_successful_answer():
    result = route_question(
        "What are the top complaint types?",
        settings=_settings(),
        query_logger=CapturingLogger(fail=True),
        query_id_factory=lambda: QUERY_ID,
    )

    assert result["mode"] == "analytics"
    assert result["answer"]
    assert "query_id" not in result
    assert result["observability_status"] == "logging_failed"


def test_successful_analytics_observation_is_recorded_as_answered():
    logger = CapturingLogger()

    result = route_question(
        "What are the top complaint types?",
        settings=_settings(),
        query_logger=logger,
        query_id_factory=lambda: QUERY_ID,
    )

    assert result["mode"] == "analytics"
    assert logger.observations[0].route == "analytics"
    assert logger.observations[0].answer_status == "answered"
    assert logger.observations[0].retrieval_results == ()
    assert logger.observations[0].orchestration_mode == "direct"
    assert logger.observations[0].orchestration_tool_call_count == 1


class FakeCursor:
    def __init__(self):
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.executions.append((str(query), parameters))


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_instance


def test_postgres_logger_persists_only_allow_listed_parameterized_metadata():
    question = "private raw question sk-test-secret-value"
    answer = "private raw answer"
    result = {
        "mode": "rag",
        "answer": answer,
        "answer_status": "answered",
        "answer_provider": "openai",
        "answer_model": "gpt-test",
        "retrieved_chunks": [_retrieved_chunk()],
        "raw_provider_payload": {"authorization": "Bearer secret"},
    }
    observation = build_query_observation(
        query_id=QUERY_ID,
        question=question,
        top_k=5,
        settings=_settings(),
        result=result,
        created_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
        latency_ms=12.5,
        retrieval_id_factory=lambda: "retrieval-1",
    )
    cursor = FakeCursor()
    logger = PostgresQueryLogger(
        "postgresql://unused",
        3,
        connection_factory=lambda *args, **kwargs: FakeConnection(cursor),
    )

    logger.record_execution(observation)

    assert len(cursor.executions) == 2
    query_sql, query_parameters = cursor.executions[0]
    retrieval_sql, retrieval_parameters = cursor.executions[1]
    assert "%s" in query_sql and "%s" in retrieval_sql
    persisted = repr((query_parameters, retrieval_parameters))
    for forbidden in (
        question,
        answer,
        "raw evidence must not be logged",
        "sk-test-secret-value",
        "Bearer secret",
        "[0.1, 0.2]",
    ):
        assert forbidden not in persisted
    assert "chunk_abc" in persisted
    assert "docs/knowledge/field-guide.md" in persisted
    assert query_parameters[0] == QUERY_ID
    assert query_parameters[3:7] == ("direct", 0, 0, "answered")
    assert retrieval_parameters[1] == QUERY_ID
