from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.answers import get_question_router
from api.routes.feedback import get_feedback_recorder
from src.common.config import Settings
from src.observability.feedback import (
    FeedbackPersistenceError,
    FeedbackUnavailableError,
    PostgresFeedbackStore,
    UnknownQueryError,
    submit_feedback,
)
from src.observability.models import FeedbackRating, FeedbackRecord


QUERY_ID = "11111111-1111-4111-8111-111111111111"
FEEDBACK_ID = "22222222-2222-4222-8222-222222222222"


def teardown_function():
    app.dependency_overrides.clear()


def _record(query_id, rating, comment):
    return FeedbackRecord(
        feedback_id=FEEDBACK_ID,
        query_id=query_id,
        rating=rating,
        comment=comment,
    )


@pytest.mark.parametrize("rating", ["helpful", "not_helpful"])
def test_helpful_and_not_helpful_feedback_are_typed_and_recorded(rating):
    captured = {}

    def recorder(query_id, selected_rating, comment):
        captured.update(
            query_id=query_id,
            rating=selected_rating,
            comment=comment,
        )
        return _record(query_id, selected_rating, comment)

    app.dependency_overrides[get_feedback_recorder] = lambda: recorder
    response = TestClient(app).post(
        "/api/v1/feedback",
        json={"query_id": QUERY_ID, "rating": rating, "comment": "  useful  "},
    )

    assert response.status_code == 201
    assert response.json() == {
        "feedback_id": FEEDBACK_ID,
        "query_id": QUERY_ID,
        "rating": rating,
        "status": "recorded",
    }
    assert captured == {
        "query_id": QUERY_ID,
        "rating": FeedbackRating(rating),
        "comment": "useful",
    }


def test_feedback_rejects_unknown_query_id():
    def unknown(query_id, rating, comment):
        raise UnknownQueryError("database details must not escape")

    app.dependency_overrides[get_feedback_recorder] = lambda: unknown
    response = TestClient(app).post(
        "/api/v1/feedback",
        json={"query_id": QUERY_ID, "rating": "helpful"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "unknown_query_id",
            "message": "Feedback requires a known query_id.",
        }
    }
    assert "database details" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (FeedbackUnavailableError("disabled detail"), "feedback_disabled"),
        (FeedbackPersistenceError("database secret detail"), "feedback_unavailable"),
    ],
)
def test_feedback_unavailable_errors_are_sanitized(error, expected_code):
    def unavailable(query_id, rating, comment):
        raise error

    app.dependency_overrides[get_feedback_recorder] = lambda: unavailable
    response = TestClient(app).post(
        "/api/v1/feedback",
        json={"query_id": QUERY_ID, "rating": "helpful"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == expected_code
    assert "detail" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "not-a-uuid", "rating": "helpful"},
        {"query_id": QUERY_ID, "rating": "maybe"},
        {"query_id": QUERY_ID, "rating": "helpful", "comment": "x" * 1001},
    ],
)
def test_feedback_rejects_malformed_or_oversized_payloads(payload):
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/v1/feedback",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_answer_query_id_can_be_used_by_feedback():
    def router(question, top_k):
        return {
            "answer": "Grounded [1].",
            "mode": "rag",
            "answer_status": "answered",
            "sources": [],
            "query_id": QUERY_ID,
        }

    def recorder(query_id, rating, comment):
        assert query_id == QUERY_ID
        return _record(query_id, rating, comment)

    app.dependency_overrides[get_question_router] = lambda: router
    app.dependency_overrides[get_feedback_recorder] = lambda: recorder
    client = TestClient(app)

    answer = client.post("/api/v1/answer", json={"question": "What is this?"})
    feedback = client.post(
        "/api/v1/feedback",
        json={"query_id": answer.json()["query_id"], "rating": "helpful"},
    )

    assert answer.status_code == 200
    assert UUID(answer.json()["query_id"]) == UUID(QUERY_ID)
    assert feedback.status_code == 201
    assert feedback.json()["query_id"] == answer.json()["query_id"]


class FakeCursor:
    def __init__(self, known=True):
        self.known = known
        self.executions = []
        self.current_query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.current_query = str(query)
        self.executions.append((self.current_query, parameters))

    def fetchone(self):
        assert "SELECT 1 FROM queries" in self.current_query
        return (1,) if self.known else None


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_instance


def test_postgres_feedback_checks_query_and_uses_parameterized_insert():
    cursor = FakeCursor()
    store = PostgresFeedbackStore(
        "postgresql://unused",
        3,
        connection_factory=lambda *args, **kwargs: FakeConnection(cursor),
        feedback_id_factory=lambda: FEEDBACK_ID,
    )

    record = store.record_feedback(
        QUERY_ID,
        FeedbackRating.NOT_HELPFUL,
        "Needs a clearer source.",
    )

    assert record.feedback_id == FEEDBACK_ID
    assert cursor.executions[0][1] == (QUERY_ID,)
    assert cursor.executions[1][1] == (
        FEEDBACK_ID,
        QUERY_ID,
        False,
        "Needs a clearer source.",
    )
    assert "%s" in cursor.executions[0][0]
    assert "%s" in cursor.executions[1][0]


@pytest.mark.parametrize(
    ("comment", "expected_comment"),
    [
        (None, None),
        ("   ", None),
        ("  Direct caller feedback.  ", "Direct caller feedback."),
    ],
)
def test_postgres_feedback_normalizes_comments_for_direct_callers(
    comment,
    expected_comment,
):
    cursor = FakeCursor()
    store = PostgresFeedbackStore(
        "postgresql://unused",
        3,
        connection_factory=lambda *args, **kwargs: FakeConnection(cursor),
        feedback_id_factory=lambda: FEEDBACK_ID,
    )

    record = store.record_feedback(
        QUERY_ID,
        FeedbackRating.HELPFUL,
        comment,
    )

    assert record.comment == expected_comment
    assert cursor.executions[1][1][-1] == expected_comment


def test_postgres_feedback_enforces_comment_limit_for_direct_callers():
    cursor = FakeCursor()
    store = PostgresFeedbackStore(
        "postgresql://unused",
        3,
        connection_factory=lambda *args, **kwargs: FakeConnection(cursor),
    )

    with pytest.raises(ValueError, match="exceeds the allowed length"):
        store.record_feedback(
            QUERY_ID,
            FeedbackRating.HELPFUL,
            "x" * 1001,
        )

    assert cursor.executions == []


def test_postgres_feedback_rejects_unknown_query_without_insert():
    cursor = FakeCursor(known=False)
    store = PostgresFeedbackStore(
        "postgresql://unused",
        3,
        connection_factory=lambda *args, **kwargs: FakeConnection(cursor),
    )

    with pytest.raises(UnknownQueryError):
        store.record_feedback(QUERY_ID, FeedbackRating.HELPFUL, None)

    assert len(cursor.executions) == 1


def test_feedback_is_unavailable_when_observability_is_disabled():
    settings = Settings(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        observability_enabled=False,
    )

    with pytest.raises(FeedbackUnavailableError):
        submit_feedback(
            QUERY_ID,
            FeedbackRating.HELPFUL,
            None,
            settings=settings,
        )
