"""Shared question routing for Streamlit, FastAPI, and local callers."""

from __future__ import annotations

from typing import Any

from src.analytics.simple_analytics import (
    answer_analytics_question,
    looks_like_analytics_question,
)
from src.common.config import Settings
from src.generation.answer_question import answer_question
from src.retrieval.retrieve_context import DEFAULT_TOP_K, validate_top_k


BACKEND_NOT_READY_MESSAGE = (
    "The local PostgreSQL/pgvector backend is not ready. Start Docker with "
    "`docker compose up -d`, then run ingestion, chunking, and embedding commands."
)


def route_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Route a question through predefined analytics or grounded RAG.

    Backend failures remain an application-level result so local interfaces can
    choose their own presentation. Public HTTP translation belongs in the API
    adapter and never exposes ``error_detail``.
    """

    validate_top_k(top_k)
    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics":
        return analytics_response
    if looks_like_analytics_question(question):
        return analytics_response

    try:
        rag_response = answer_question(question, top_k=top_k, settings=settings)
    except Exception as exc:  # pragma: no cover - interface tests exercise this path
        return {
            "answer": BACKEND_NOT_READY_MESSAGE,
            "sources": [],
            "confidence_note": "Local backend unavailable.",
            "retrieved_chunks": [],
            "sample_rows": [],
            "mode": "backend_error",
            "error_detail": f"{type(exc).__name__}: {exc}",
        }

    response = dict(rag_response)
    response["mode"] = "rag"
    response.setdefault("sample_rows", [])
    return response
