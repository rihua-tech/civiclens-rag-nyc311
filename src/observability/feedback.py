"""Feedback validation and persistence outside the FastAPI adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from src.common.config import Settings
from src.observability.models import FeedbackRating, FeedbackRecord


MAX_FEEDBACK_COMMENT_LENGTH = 1000
ConnectionFactory = Callable[..., Any]
FeedbackIdFactory = Callable[[], str]


def normalize_feedback_comment(comment: str | None) -> str | None:
    if comment is None:
        return None
    normalized = comment.strip()
    if not normalized:
        return None
    if len(normalized) > MAX_FEEDBACK_COMMENT_LENGTH:
        raise ValueError("feedback comment exceeds the allowed length")
    return normalized


class FeedbackError(RuntimeError):
    """Base class with no database or credential details in its message."""


class FeedbackUnavailableError(FeedbackError):
    pass


class UnknownQueryError(FeedbackError):
    pass


class FeedbackPersistenceError(FeedbackError):
    pass


class FeedbackStore(Protocol):
    def record_feedback(
        self,
        query_id: str,
        rating: FeedbackRating,
        comment: str | None,
    ) -> FeedbackRecord: ...


class PostgresFeedbackStore:
    def __init__(
        self,
        database_url: str,
        connect_timeout_seconds: int,
        connection_factory: ConnectionFactory | None = None,
        feedback_id_factory: FeedbackIdFactory | None = None,
    ) -> None:
        self._database_url = database_url
        self._connect_timeout_seconds = connect_timeout_seconds
        self._connection_factory = connection_factory
        self._feedback_id_factory = feedback_id_factory or (lambda: str(uuid4()))

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

    def record_feedback(
        self,
        query_id: str,
        rating: FeedbackRating,
        comment: str | None,
    ) -> FeedbackRecord:
        normalized_comment = normalize_feedback_comment(comment)
        feedback_id = self._feedback_id_factory()
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM queries WHERE query_id = %s",
                        (query_id,),
                    )
                    if cursor.fetchone() is None:
                        raise UnknownQueryError("query_id is not known")
                    cursor.execute(
                        """
                        INSERT INTO feedback (
                            feedback_id,
                            query_id,
                            helpful,
                            comment
                        )
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            feedback_id,
                            query_id,
                            rating is FeedbackRating.HELPFUL,
                            normalized_comment,
                        ),
                    )
        except UnknownQueryError:
            raise
        except Exception as exc:
            raise FeedbackPersistenceError(
                "feedback could not be persisted"
            ) from exc
        return FeedbackRecord(
            feedback_id=feedback_id,
            query_id=query_id,
            rating=rating,
            comment=normalized_comment,
        )


def submit_feedback(
    query_id: str,
    rating: FeedbackRating,
    comment: str | None,
    settings: Settings | None = None,
    store: FeedbackStore | None = None,
) -> FeedbackRecord:
    active_settings = settings or Settings.from_env()
    if not active_settings.observability_enabled:
        raise FeedbackUnavailableError("feedback is disabled")
    active_store = store or PostgresFeedbackStore(
        active_settings.database_url,
        active_settings.observability_connect_timeout_seconds,
    )
    return active_store.record_feedback(query_id, rating, comment)
