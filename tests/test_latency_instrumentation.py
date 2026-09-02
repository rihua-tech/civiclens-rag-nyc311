from __future__ import annotations

import json

import pytest

from src.common.config import Settings
from src.observability.latency import (
    LATENCY_EVENT,
    RequestLatency,
    capture_request_latency,
    connect_with_latency,
    measure_latency,
)
from src.orchestration.question_router import route_question


def _settings() -> Settings:
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
        observability_enabled=False,
    )


def _retrieved_chunk() -> dict:
    return {
        "chunk_id": "chunk_latency",
        "document_id": "doc_latency",
        "chunk_text": "The CivicLens architecture uses retrieval and answer generation.",
        "source_name": "Architecture",
        "source_path": "docs/architecture.md",
        "similarity_score": 0.8,
        "rank": 1,
    }


def test_rag_request_emits_structured_stage_timings(monkeypatch):
    log_messages: list[str] = []
    monkeypatch.setattr(
        "src.generation.answer_question.retrieve_context",
        lambda question, top_k, min_similarity, settings: [_retrieved_chunk()],
    )
    monkeypatch.setattr(
        "src.observability.latency.LATENCY_LOGGER.info",
        log_messages.append,
    )
    times = iter(
        (
            0.000,  # total start
            0.001,  # routing start
            0.003,  # routing end
            0.004,  # retrieval start
            0.010,  # retrieval end
            0.011,  # generation start
            0.031,  # generation end
            0.032,  # citation ID validation start
            0.034,  # citation ID validation end
            0.035,  # display citation rebuild start
            0.036,  # display citation rebuild end
            0.040,  # total end
        )
    )

    result = route_question(
        "What does the CivicLens architecture use?",
        settings=_settings(),
        latency_clock=lambda: next(times),
    )

    assert result["mode"] == "rag"
    assert len(log_messages) == 1
    payload = json.loads(log_messages[0])
    assert payload == {
        "answer_generation_ms": 20.0,
        "citation_validation_ms": 3.0,
        "db_connection_ms": 0.0,
        "event": LATENCY_EVENT,
        "outcome": "answered",
        "query_id": None,
        "retrieval_ms": 6.0,
        "route": "rag",
        "routing_ms": 2.0,
        "total_ms": 40.0,
    }


def test_database_connection_acquisition_is_measured_separately():
    connection = object()
    times = iter((1.000, 1.007))
    timing = RequestLatency(clock=lambda: next(times))

    with capture_request_latency(timing):
        returned = connect_with_latency(lambda database_url: connection, "unused")

    assert returned is connection
    assert timing.db_connection_ms == pytest.approx(7.0)


def test_request_latency_context_is_reset_after_capture_exits():
    clock_calls: list[float] = []
    times = iter((2.000, 2.005))

    def clock() -> float:
        value = next(times)
        clock_calls.append(value)
        return value

    timing = RequestLatency(clock=clock)
    with capture_request_latency(timing):
        with measure_latency("routing_ms"):
            pass

    measured_routing_ms = timing.routing_ms
    with measure_latency("routing_ms"):
        pass

    assert measured_routing_ms == pytest.approx(5.0)
    assert timing.routing_ms == measured_routing_ms
    assert clock_calls == [2.000, 2.005]


def test_latency_logging_failure_does_not_affect_rag_answer(monkeypatch):
    monkeypatch.setattr(
        "src.generation.answer_question.retrieve_context",
        lambda question, top_k, min_similarity, settings: [_retrieved_chunk()],
    )

    def fail_to_log(message: str) -> None:
        raise RuntimeError("latency logger unavailable")

    monkeypatch.setattr(
        "src.observability.latency.LATENCY_LOGGER.info",
        fail_to_log,
    )

    result = route_question(
        "What does the CivicLens architecture use?",
        settings=_settings(),
    )

    assert result["mode"] == "rag"
    assert result["answer_status"] == "answered"
    assert result["answer"] == (
        "The CivicLens architecture uses retrieval and answer generation. [1]"
    )
    assert result["sources"][0]["chunk_id"] == "chunk_latency"


def test_analytics_request_emits_only_applicable_stage_timings(monkeypatch):
    log_messages: list[str] = []
    monkeypatch.setattr(
        "src.observability.latency.LATENCY_LOGGER.info",
        log_messages.append,
    )
    times = iter(
        (
            4.000,  # total start
            4.001,  # routing start
            4.003,  # routing end
            4.010,  # total end
        )
    )

    result = route_question(
        "What are the top complaint types?",
        settings=_settings(),
        latency_clock=lambda: next(times),
    )

    assert result["mode"] == "analytics"
    assert len(log_messages) == 1
    payload = json.loads(log_messages[0])
    assert payload["route"] == "analytics"
    assert payload["retrieval_ms"] == 0.0
    assert payload["answer_generation_ms"] == 0.0
    assert payload["citation_validation_ms"] == 0.0
    assert payload["db_connection_ms"] == 0.0
    assert payload["routing_ms"] == pytest.approx(2.0)
    assert payload["total_ms"] == pytest.approx(10.0)
