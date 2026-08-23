from __future__ import annotations

import re
from pathlib import Path

import yaml


BLUEPRINT_PATH = Path("render.yaml")
RENDER_START_PATH = Path("scripts/render_start.sh")


def _blueprint() -> dict:
    parsed = yaml.safe_load(BLUEPRINT_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _service(blueprint: dict, name: str) -> dict:
    matches = [service for service in blueprint["services"] if service["name"] == name]
    assert len(matches) == 1
    return matches[0]


def _environment(service: dict) -> dict[str, dict]:
    return {item["key"]: item for item in service["envVars"]}


def test_blueprint_defines_only_the_two_free_oregon_docker_web_services():
    assert BLUEPRINT_PATH.is_file()
    blueprint = _blueprint()

    assert set(blueprint) == {"services"}
    assert {service["name"] for service in blueprint["services"]} == {
        "civiclens-api",
        "civiclens-ui",
    }
    for service in blueprint["services"]:
        assert service["type"] == "web"
        assert service["runtime"] == "docker"
        assert service["plan"] == "free"
        assert service["region"] == "oregon"
        assert service["dockerContext"] == "."
        assert service["autoDeployTrigger"] == "off"
        assert "branch" not in service


def test_api_uses_existing_database_and_explicit_offline_demo_profile():
    api = _service(_blueprint(), "civiclens-api")
    environment = _environment(api)

    assert api["dockerfilePath"] == "./Dockerfile.api"
    assert api["healthCheckPath"] == "/health"
    assert environment["DATABASE_URL"] == {
        "key": "DATABASE_URL",
        "fromDatabase": {
            "name": "civiclens-postgres",
            "property": "connectionString",
        },
    }
    expected_values = {
        "EMBEDDING_PROVIDER": "deterministic",
        "EMBEDDING_MODEL": "local-deterministic-1536",
        "EMBEDDING_DIMENSION": "1536",
        "RETRIEVAL_MODE": "hybrid",
        "RERANKING_ENABLED": "false",
        "ANSWER_PROVIDER": "local",
        "USE_OPENAI_ANSWERS": "false",
        "USE_OPENAI_EMBEDDINGS": "false",
        "OBSERVABILITY_ENABLED": "false",
        "PORT": "8000",
    }
    assert {
        key: environment[key]["value"] for key in expected_values
    } == expected_values
    assert "OPENAI_API_KEY" not in environment
    assert not any(key.startswith("POSTGRES_") for key in environment)


def test_ui_uses_generated_api_https_url_without_hardcoded_render_hostname():
    ui = _service(_blueprint(), "civiclens-ui")
    environment = _environment(ui)

    assert ui["dockerfilePath"] == "./Dockerfile.ui"
    assert ui["healthCheckPath"] == "/_stcore/health"
    assert environment["CIVICLENS_API_BASE_URL"] == {
        "key": "CIVICLENS_API_BASE_URL",
        "fromService": {
            "type": "web",
            "name": "civiclens-api",
            "envVarKey": "RENDER_EXTERNAL_URL",
        },
    }
    assert environment["CIVICLENS_API_TIMEOUT_SECONDS"]["value"] == "60"
    assert environment["PORT"]["value"] == "8501"
    assert "onrender.com" not in BLUEPRINT_PATH.read_text(encoding="utf-8").lower()


def test_blueprint_uses_exactly_one_free_tier_compatible_bootstrap_mechanism():
    api = _service(_blueprint(), "civiclens-api")
    assert api["dockerCommand"] == "/bin/sh scripts/render_start.sh"
    assert "initialDeployHook" not in api
    assert "preDeployCommand" not in api

    start_script = RENDER_START_PATH.read_text(encoding="utf-8")
    bootstrap = start_script.index("python -m scripts.bootstrap")
    uvicorn = start_script.index("exec python -m uvicorn api.main:app")
    assert bootstrap < uvicorn
    assert "--reindex" not in start_script


def test_blueprint_contains_no_literal_database_url_or_credential():
    source = BLUEPRINT_PATH.read_text(encoding="utf-8")

    assert re.search(r"postgres(?:ql)?://", source, flags=re.IGNORECASE) is None
    assert re.search(r"password\s*:", source, flags=re.IGNORECASE) is None
    assert re.search(r"(?:OPENAI_API_KEY|Authorization|Bearer\s+)", source) is None
