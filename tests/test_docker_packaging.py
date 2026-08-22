from pathlib import Path

from src.common.config import DEFAULT_RERANKER_MODEL


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_compose_defines_three_services_and_persistent_volumes():
    compose = _read("docker-compose.yml")

    assert "  postgres:" in compose
    assert "  api:" in compose
    assert "  ui:" in compose
    assert "postgres_data:/var/lib/postgresql/data" in compose
    assert "  postgres_data:" in compose
    assert "  model_cache:" in compose
    assert "container_name:" not in compose


def test_compose_uses_internal_service_dns_and_configurable_host_ports():
    compose = _read("docker-compose.yml")

    assert "POSTGRES_HOST: postgres" in compose
    assert (
        "CIVICLENS_API_BASE_URL: ${CIVICLENS_DOCKER_API_BASE_URL:-http://api:8000}"
        in compose
    )
    assert '${API_PORT:-8000}:8000' in compose
    assert '${UI_PORT:-8501}:8501' in compose
    assert '${POSTGRES_PORT:-5432}:5432' in compose


def test_api_healthcheck_uses_liveness_not_rag_readiness():
    compose = _read("docker-compose.yml")
    api_block = compose.split("  api:", 1)[1].split("  ui:", 1)[0]

    assert "/health" in api_block
    assert "/ready" not in api_block
    assert "curl" not in api_block
    assert "service_healthy" in api_block


def test_compose_reranker_default_matches_project_configuration():
    compose = _read("docker-compose.yml")
    env_example = _read(".env.example")

    assert (
        f"RERANKER_MODEL: ${{RERANKER_MODEL:-{DEFAULT_RERANKER_MODEL}}}"
        in compose
    )
    assert f"RERANKER_MODEL={DEFAULT_RERANKER_MODEL}" in env_example
    assert "RERANKING_ENABLED: ${RERANKING_ENABLED:-false}" in compose


def test_images_bind_public_interfaces_and_do_not_bake_secrets():
    api_dockerfile = _read("Dockerfile.api")
    ui_dockerfile = _read("Dockerfile.ui")

    assert '"--host", "0.0.0.0"' in api_dockerfile
    assert '"--server.address=0.0.0.0"' in ui_dockerfile
    assert "USER appuser" in api_dockerfile
    assert "USER appuser" in ui_dockerfile
    assert "OPENAI_API_KEY" not in api_dockerfile + ui_dockerfile
    assert "ARG " not in api_dockerfile + ui_dockerfile
    assert "https://download.pytorch.org/whl/cpu" in api_dockerfile
    assert "requirements-ui.txt" in ui_dockerfile


def test_dockerignore_excludes_sensitive_generated_content_but_keeps_corpus():
    ignored = _read(".dockerignore")

    for required_exclusion in (
        ".git",
        ".env",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        "data/processed/",
        "data/evaluation/results/",
        "*review*.zip",
    ):
        assert required_exclusion in ignored

    for required_context in (
        "src/",
        "sql/",
        "docs/knowledge/",
        "requirements.txt",
        "requirements-ui.txt",
    ):
        assert required_context not in ignored
