"""Parameterized PostgreSQL persistence for privacy-conscious execution metadata."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from src.common.config import Settings
from src.observability.models import QueryObservation, RetrievalObservation


ConnectionFactory = Callable[..., Any]
RetrievalIdFactory = Callable[[], str]


class QueryLogger(Protocol):
    def record_execution(self, observation: QueryObservation) -> None: ...


class PostgresQueryLogger:
    def __init__(
        self,
        database_url: str,
        connect_timeout_seconds: int,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self._database_url = database_url
        self._connect_timeout_seconds = connect_timeout_seconds
        self._connection_factory = connection_factory

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory(
                self._database_url,
                connect_timeout=self._connect_timeout_seconds,
            )
        import psycopg

        return psycopg.connect(
            self._database_url,
            connect_timeout=self._connect_timeout_seconds,
        )

    def record_execution(self, observation: QueryObservation) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO queries (
                        query_id,
                        question,
                        question_length,
                        route,
                        orchestration_mode,
                        orchestration_step_count,
                        orchestration_tool_call_count,
                        orchestration_outcome,
                        retrieval_strategy,
                        embedding_provider,
                        embedding_model,
                        answer_provider,
                        answer_model,
                        answer_status,
                        reranking_enabled,
                        top_k,
                        latency_ms,
                        observability_version,
                        created_at
                    )
                    VALUES (
                        %s, NULL, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        observation.query_id,
                        observation.question_length,
                        observation.route,
                        observation.orchestration_mode,
                        observation.orchestration_step_count,
                        observation.orchestration_tool_call_count,
                        observation.orchestration_outcome,
                        observation.retrieval_strategy,
                        observation.embedding_provider,
                        observation.embedding_model,
                        observation.answer_provider,
                        observation.answer_model,
                        observation.answer_status,
                        observation.reranking_enabled,
                        observation.top_k,
                        observation.latency_ms,
                        observation.observability_version,
                        observation.created_at,
                    ),
                )
                for retrieval in observation.retrieval_results:
                    cursor.execute(
                        """
                        INSERT INTO retrieval_results (
                            retrieval_id,
                            query_id,
                            chunk_id,
                            document_id,
                            similarity_score,
                            rank,
                            retrieval_mode,
                            semantic_score,
                            semantic_rank,
                            lexical_score,
                            lexical_rank,
                            fusion_score,
                            reranker_score,
                            pre_rerank_rank,
                            source_name,
                            source_type,
                            source_category,
                            source_path,
                            source_url,
                            section_title,
                            heading_path,
                            content_hash,
                            document_content_hash
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        _retrieval_parameters(retrieval),
                    )


def _retrieval_parameters(retrieval: RetrievalObservation) -> tuple[Any, ...]:
    return (
        retrieval.retrieval_id,
        retrieval.query_id,
        retrieval.chunk_id,
        retrieval.document_id,
        retrieval.similarity_score,
        retrieval.rank,
        retrieval.retrieval_mode,
        retrieval.semantic_score,
        retrieval.semantic_rank,
        retrieval.lexical_score,
        retrieval.lexical_rank,
        retrieval.fusion_score,
        retrieval.reranker_score,
        retrieval.pre_rerank_rank,
        retrieval.source_name,
        retrieval.source_type,
        retrieval.source_category,
        retrieval.source_path,
        retrieval.source_url,
        retrieval.section_title,
        list(retrieval.heading_path),
        retrieval.content_hash,
        retrieval.document_content_hash,
    )


def build_query_observation(
    *,
    query_id: str,
    question: str,
    top_k: int,
    settings: Settings,
    result: dict[str, Any],
    created_at: datetime,
    latency_ms: float,
    retrieval_id_factory: RetrievalIdFactory | None = None,
) -> QueryObservation:
    """Build one allow-listed record without copying raw question/answer text."""
    id_factory = retrieval_id_factory or (lambda: str(uuid4()))
    mode = str(result.get("mode") or "backend_error")
    route = "rag" if mode in {"rag", "backend_error"} else "analytics"
    answer_status = str(result.get("answer_status") or "")
    if not answer_status:
        if mode == "backend_error":
            answer_status = "failed"
        elif mode == "analytics":
            answer_status = "answered"
        else:
            answer_status = "abstained"

    raw_retrievals = result.get("retrieved_chunks") or []
    retrievals = tuple(
        RetrievalObservation.from_result(
            query_id,
            id_factory(),
            retrieved,
            fallback_rank,
        )
        for fallback_rank, retrieved in enumerate(raw_retrievals, start=1)
        if retrieved.get("chunk_id")
    )
    is_rag = route == "rag"
    return QueryObservation(
        query_id=query_id,
        created_at=created_at,
        route=route,
        orchestration_mode=str(result.get("orchestration_mode") or "direct"),
        orchestration_step_count=max(
            0,
            int(result.get("orchestration_step_count") or 0),
        ),
        orchestration_tool_call_count=max(
            0,
            int(result.get("orchestration_tool_call_count") or 0),
        ),
        orchestration_outcome=str(
            result.get("orchestration_outcome") or answer_status
        ),
        retrieval_strategy=settings.retrieval_mode if is_rag else None,
        embedding_provider=settings.embedding_provider if is_rag else None,
        embedding_model=settings.embedding_model if is_rag else None,
        answer_provider=(
            str(result["answer_provider"])
            if result.get("answer_provider") is not None
            else None
        ),
        answer_model=(
            str(result["answer_model"])
            if result.get("answer_model") is not None
            else None
        ),
        answer_status=answer_status,
        reranking_enabled=settings.reranking_enabled if is_rag else False,
        top_k=top_k,
        question_length=len(question),
        latency_ms=max(0.0, float(latency_ms)),
        retrieval_results=retrievals,
    )


def build_query_logger(settings: Settings) -> QueryLogger:
    return PostgresQueryLogger(
        settings.database_url,
        settings.observability_connect_timeout_seconds,
    )
