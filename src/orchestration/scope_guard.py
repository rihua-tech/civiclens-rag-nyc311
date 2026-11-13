"""Application-owned abstention response for unsupported request scope."""

from __future__ import annotations

from typing import Any

from src.generation.schemas import AnswerStatus, NO_ANSWER
from src.orchestration.route_decision import QuestionRoute, RouteDecision


def scope_abstention_response(decision: RouteDecision) -> dict[str, Any]:
    """Return a citation-free abstention without executing retrieval or a tool."""

    mode = (
        "analytics"
        if decision.route in {
            QuestionRoute.ANALYTICS,
            QuestionRoute.UNSUPPORTED_ANALYTICS,
        }
        else "rag"
    )
    return {
        "answer": NO_ANSWER,
        "sources": [],
        "confidence_note": (
            "The request is outside the supported documentation and checked-in "
            "sample analytics scope."
        ),
        "retrieved_chunks": [],
        "sample_rows": [],
        "mode": mode,
        "answer_status": AnswerStatus.ABSTAINED.value,
        "scope_blocked": True,
        "scope_reason": decision.scope_reason or "unsupported_scope",
        "provider_called": False,
        "fallback_used": False,
        "citation_ids": [],
        "rejected_citation_ids": [],
    }
