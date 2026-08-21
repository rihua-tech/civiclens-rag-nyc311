from src.common import config


CONFIG_ENV_VARS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "USE_OPENAI_EMBEDDINGS",
    "RETRIEVAL_MODE",
    "SEMANTIC_CANDIDATE_COUNT",
    "LEXICAL_CANDIDATE_COUNT",
    "RRF_K",
    "RERANKING_ENABLED",
    "RERANKER_MODEL",
    "RERANK_CANDIDATE_LIMIT",
    "ANSWER_PROVIDER",
    "ANSWER_MODEL",
    "ANSWER_TIMEOUT_SECONDS",
    "ANSWER_MAX_RETRIES",
    "USE_OPENAI_ANSWERS",
    "OPENAI_API_KEY",
    "OBSERVABILITY_ENABLED",
    "OBSERVABILITY_CONNECT_TIMEOUT_SECONDS",
)


def test_normal_local_defaults_select_one_real_semantic_hybrid_profile(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    settings = config.Settings.from_env()

    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384
    assert settings.retrieval_mode == "hybrid"
    assert settings.semantic_candidate_count == 20
    assert settings.lexical_candidate_count == 20
    assert settings.rrf_k == 60
    assert settings.reranking_enabled is False
    assert settings.rerank_candidate_limit == 20
    assert settings.answer_provider == "local"
    assert settings.answer_model == "gpt-4o-mini"
    assert settings.answer_timeout_seconds == 30.0
    assert settings.answer_max_retries == 2
    assert settings.observability_enabled is False
    assert settings.observability_connect_timeout_seconds == 3


def test_observability_configuration_is_explicit(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_CONNECT_TIMEOUT_SECONDS", "7")

    settings = config.Settings.from_env()

    assert settings.observability_enabled is True
    assert settings.observability_connect_timeout_seconds == 7


def test_legacy_deterministic_environment_remains_backward_compatible(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "local-deterministic-1536")
    monkeypatch.setenv("USE_OPENAI_EMBEDDINGS", "false")

    settings = config.Settings.from_env()

    assert settings.embedding_provider == "deterministic"
    assert settings.embedding_model == "local-deterministic-1536"
    assert settings.embedding_dimension == 1536


def test_legacy_openai_flag_overrides_semantic_env_example_profile(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("USE_OPENAI_EMBEDDINGS", "true")

    settings = config.Settings.from_env()

    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536


def test_legacy_answer_flag_overrides_local_env_example_default(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("ANSWER_PROVIDER", "local")
    monkeypatch.setenv("USE_OPENAI_ANSWERS", "true")

    settings = config.Settings.from_env()

    assert settings.answer_provider == "openai"
    assert settings.use_openai_answers is True


def test_answer_retry_limit_is_bounded(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("ANSWER_MAX_RETRIES", "6")

    try:
        config.Settings.from_env()
    except ValueError as exc:
        assert "between 0 and 5" in str(exc)
    else:
        raise AssertionError("Expected an invalid retry count to fail")
