from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routes.answers import get_question_router
from src.common.config import cors_allowed_origins


LOCAL_ORIGIN = "http://localhost:3000"
PRODUCTION_ORIGIN = "https://civiclens-demo.vercel.app"


def _answer_result() -> dict:
    return {
        "answer": "Use retrieved evidence [1].",
        "mode": "rag",
        "answer_status": "answered",
        "sources": [
            {
                "source_name": "CivicLens Architecture",
                "source_path": "docs/architecture.md",
                "chunk_id": "chunk_cors",
                "citation_number": 1,
            }
        ],
    }


def _client(*origins: str) -> TestClient:
    application = create_app(allowed_origins=tuple(origins))
    application.dependency_overrides[get_question_router] = lambda: (
        lambda question, top_k: _answer_result()
    )
    return TestClient(application, raise_server_exceptions=False)


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/answer",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


@pytest.mark.parametrize("origin", [LOCAL_ORIGIN, PRODUCTION_ORIGIN])
def test_approved_browser_origins_receive_minimal_preflight_access(origin):
    response = _preflight(_client(LOCAL_ORIGIN, PRODUCTION_ORIGIN), origin)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-methods"] == "POST"
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_rejected_origin_is_not_granted_cors_access():
    client = _client(LOCAL_ORIGIN, PRODUCTION_ORIGIN)
    origin = "https://unapproved.example"

    preflight = _preflight(client, origin)
    request = client.post(
        "/api/v1/answer",
        headers={"Origin": origin},
        json={"question": "What is the architecture?", "top_k": 5},
    )

    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers
    assert request.status_code == 200
    assert "access-control-allow-origin" not in request.headers


def test_non_browser_api_behavior_is_unchanged():
    response = _client(LOCAL_ORIGIN).post(
        "/api/v1/answer",
        json={"question": "What is the architecture?", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "rag"
    assert response.json()["sources"][0]["chunk_id"] == "chunk_cors"
    assert "access-control-allow-origin" not in response.headers


def test_cors_origin_configuration_is_normalized_and_deduplicated():
    assert cors_allowed_origins(
        " http://LOCALHOST:3000/, https://CivicLens-Demo.Vercel.App, "
        "https://civiclens-demo.vercel.app "
    ) == (LOCAL_ORIGIN, PRODUCTION_ORIGIN)


@pytest.mark.parametrize(
    "configured",
    [
        "*",
        "ftp://example.com",
        "https://user:password@example.com",
        "https://example.com/path",
        "https://example.com?debug=true",
        "https://example.com:not-a-port",
    ],
)
def test_unsafe_or_non_origin_cors_configuration_is_rejected(configured):
    with pytest.raises(ValueError, match="CIVICLENS_CORS_ALLOWED_ORIGINS"):
        cors_allowed_origins(configured)
