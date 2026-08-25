"""Application-owned observability models with an explicit privacy allow-list."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


OBSERVABILITY_SCHEMA_VERSION = "issue17-v2"


class FeedbackRating(str, Enum):
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class RetrievalObservation:
    """Allow-listed retrieval diagnostics; chunk text and vectors are excluded."""

    query_id: str
    retrieval_id: str
    chunk_id: str
    document_id: str | None
    rank: int
    retrieval_mode: str | None
    similarity_score: float | None
    semantic_score: float | None
    semantic_rank: int | None
    lexical_score: float | None
    lexical_rank: int | None
    fusion_score: float | None
    reranker_score: float | None
    pre_rerank_rank: int | None
    source_name: str | None
    source_type: str | None
    source_category: str | None
    source_path: str | None
    source_url: str | None
    section_title: str | None
    heading_path: tuple[str, ...]
    content_hash: str | None
    document_content_hash: str | None

    @classmethod
    def from_result(
        cls,
        query_id: str,
        retrieval_id: str,
        result: dict[str, Any],
        fallback_rank: int,
    ) -> "RetrievalObservation":
        return cls(
            query_id=query_id,
            retrieval_id=retrieval_id,
            chunk_id=str(result["chunk_id"]),
            document_id=(
                str(result["document_id"])
                if result.get("document_id") is not None
                else None
            ),
            rank=int(result.get("rank") or fallback_rank),
            retrieval_mode=(
                str(result["retrieval_mode"])
                if result.get("retrieval_mode") is not None
                else None
            ),
            similarity_score=_optional_float(result.get("similarity_score")),
            semantic_score=_optional_float(result.get("semantic_score")),
            semantic_rank=_optional_int(result.get("semantic_rank")),
            lexical_score=_optional_float(result.get("lexical_score")),
            lexical_rank=_optional_int(result.get("lexical_rank")),
            fusion_score=_optional_float(result.get("fusion_score")),
            reranker_score=_optional_float(result.get("reranker_score")),
            pre_rerank_rank=_optional_int(result.get("pre_rerank_rank")),
            source_name=(
                str(result["source_name"])
                if result.get("source_name") is not None
                else None
            ),
            source_type=(
                str(result["source_type"])
                if result.get("source_type") is not None
                else None
            ),
            source_category=(
                str(result["source_category"])
                if result.get("source_category") is not None
                else None
            ),
            source_path=(
                str(result["source_path"])
                if result.get("source_path") is not None
                else None
            ),
            source_url=(
                str(result["source_url"])
                if result.get("source_url") is not None
                else None
            ),
            section_title=(
                str(result["section_title"])
                if result.get("section_title") is not None
                else None
            ),
            heading_path=tuple(str(item) for item in result.get("heading_path") or ()),
            content_hash=(
                str(result["content_hash"])
                if result.get("content_hash") is not None
                else None
            ),
            document_content_hash=(
                str(result["document_content_hash"])
                if result.get("document_content_hash") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class QueryObservation:
    query_id: str
    created_at: datetime
    route: str
    orchestration_mode: str
    orchestration_step_count: int
    orchestration_tool_call_count: int
    orchestration_outcome: str
    retrieval_strategy: str | None
    embedding_provider: str | None
    embedding_model: str | None
    answer_provider: str | None
    answer_model: str | None
    answer_status: str
    reranking_enabled: bool
    top_k: int
    question_length: int
    latency_ms: float
    retrieval_results: tuple[RetrievalObservation, ...]
    observability_version: str = OBSERVABILITY_SCHEMA_VERSION


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    query_id: str
    rating: FeedbackRating
    comment: str | None
