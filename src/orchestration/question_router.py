"""Shared question routing for Streamlit, FastAPI, and local callers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from src.analytics.simple_analytics import (
    answer_analytics_question,
    looks_like_analytics_question,
)
from src.common.config import Settings
from src.generation.answer_question import answer_question
from src.observability.query_logger import (
    QueryLogger,
    build_query_logger,
    build_query_observation,
)
from src.retrieval.retrieve_context import DEFAULT_TOP_K, validate_top_k


BACKEND_NOT_READY_MESSAGE = (
    "The local PostgreSQL/pgvector backend is not ready. Start Docker with "
    "`docker compose up -d`, then run ingestion, chunking, and embedding commands."
)


def route_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
    query_logger: QueryLogger | None = None,
    query_id_factory: Callable[[], str] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Route a question through predefined analytics or grounded RAG.

    Backend failures remain an application-level result so local interfaces can
    choose their own presentation. Public HTTP translation belongs in the API
    adapter and never exposes ``error_detail``.
    """

    validate_top_k(top_k)
    started_at = datetime.now(timezone.utc)
    started = clock()
    try:
        active_settings = settings or Settings.from_env()
    except Exception:
        active_settings = None

    query_id = None
    if active_settings is not None and active_settings.observability_enabled:
        id_factory = query_id_factory or (lambda: str(uuid4()))
        query_id = id_factory()

    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics":
        response = dict(analytics_response)
    elif looks_like_analytics_question(question):
        response = dict(analytics_response)
    else:
        try:
            response = answer_question(
                question,
                top_k=top_k,
                settings=active_settings or settings,
                query_id=query_id,
            )
            response = dict(response)
            response["mode"] = "rag"
            response.setdefault("sample_rows", [])
        except Exception as exc:  # pragma: no cover - interface tests exercise this path
            response = {
                "answer": BACKEND_NOT_READY_MESSAGE,
                "sources": [],
                "confidence_note": "Local backend unavailable.",
                "retrieved_chunks": [],
                "sample_rows": [],
                "mode": "backend_error",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }

    if query_id is not None and active_settings is not None:
        response["query_id"] = query_id
        try:
            observation = build_query_observation(
                query_id=query_id,
                question=question,
                top_k=top_k,
                settings=active_settings,
                result=response,
                created_at=started_at,
                latency_ms=(clock() - started) * 1000.0,
            )
            active_logger = query_logger or build_query_logger(active_settings)
            active_logger.record_execution(observation)
            response["observability_status"] = "recorded"
        except Exception:
            response.pop("query_id", None)
            response["observability_status"] = "logging_failed"

    return response
