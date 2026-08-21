"""Shared local configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlsplit, urlunsplit


DETERMINISTIC_PROVIDER = "deterministic"
DETERMINISTIC_MODEL = "local-deterministic-1536"
DETERMINISTIC_DIMENSION = 1536
SEMANTIC_PROVIDER = "sentence_transformers"
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_DIMENSION = 384
OPENAI_PROVIDER = "openai"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
LOCAL_ANSWER_PROVIDER = "local"
OPENAI_ANSWER_PROVIDER = "openai"
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"
DEFAULT_ANSWER_TIMEOUT_SECONDS = 30.0
DEFAULT_ANSWER_MAX_RETRIES = 2
MAX_ANSWER_RETRIES = 5
DEFAULT_OBSERVABILITY_CONNECT_TIMEOUT_SECONDS = 3


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def env_bounded_retry_count(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 0 <= value <= MAX_ANSWER_RETRIES:
        raise ValueError(f"{name} must be between 0 and {MAX_ANSWER_RETRIES}")
    return value


def build_database_url() -> str:
    user = os.getenv("POSTGRES_USER", "civiclens")
    password = os.getenv("POSTGRES_PASSWORD", "change_me")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "civiclens_rag")
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"


def mask_database_url(database_url: str) -> str:
    parts = urlsplit(database_url)
    if not parts.password:
        return database_url

    username = parts.username or ""
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    embedding_model: str
    use_openai_embeddings: bool
    use_openai_answers: bool
    openai_api_key: str = field(repr=False)
    embedding_provider: str = ""
    embedding_dimension: int = 0
    retrieval_mode: str = "semantic"
    semantic_candidate_count: int = 20
    lexical_candidate_count: int = 20
    rrf_k: int = 60
    reranking_enabled: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    rerank_candidate_limit: int = 20
    answer_provider: str = LOCAL_ANSWER_PROVIDER
    answer_model: str = DEFAULT_ANSWER_MODEL
    answer_timeout_seconds: float = DEFAULT_ANSWER_TIMEOUT_SECONDS
    answer_max_retries: int = DEFAULT_ANSWER_MAX_RETRIES
    observability_enabled: bool = False
    observability_connect_timeout_seconds: int = (
        DEFAULT_OBSERVABILITY_CONNECT_TIMEOUT_SECONDS
    )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_available()
        use_openai_embeddings = env_flag("USE_OPENAI_EMBEDDINGS", default=False)
        configured_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        configured_model = os.getenv("EMBEDDING_MODEL", "").strip()
        legacy_openai_override = (
            use_openai_embeddings and configured_provider != OPENAI_PROVIDER
        )

        if legacy_openai_override:
            embedding_provider = OPENAI_PROVIDER
            embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL
            default_dimension = DETERMINISTIC_DIMENSION
        elif configured_provider == OPENAI_PROVIDER:
            embedding_provider = OPENAI_PROVIDER
            embedding_model = configured_model or DEFAULT_OPENAI_EMBEDDING_MODEL
            if embedding_model == DETERMINISTIC_MODEL:
                embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL
            default_dimension = DETERMINISTIC_DIMENSION
        elif configured_provider == DETERMINISTIC_PROVIDER or (
            not configured_provider and configured_model == DETERMINISTIC_MODEL
        ):
            embedding_provider = DETERMINISTIC_PROVIDER
            embedding_model = configured_model or DETERMINISTIC_MODEL
            default_dimension = DETERMINISTIC_DIMENSION
        else:
            embedding_provider = configured_provider or SEMANTIC_PROVIDER
            embedding_model = configured_model or DEFAULT_SEMANTIC_MODEL
            default_dimension = DEFAULT_SEMANTIC_DIMENSION

        legacy_use_openai_answers = env_flag("USE_OPENAI_ANSWERS", default=False)
        configured_answer_provider = os.getenv("ANSWER_PROVIDER", "").strip().lower()
        if configured_answer_provider and configured_answer_provider not in {
            LOCAL_ANSWER_PROVIDER,
            OPENAI_ANSWER_PROVIDER,
        }:
            raise ValueError(
                "ANSWER_PROVIDER must be either "
                f"{LOCAL_ANSWER_PROVIDER!r} or {OPENAI_ANSWER_PROVIDER!r}"
            )
        answer_provider = (
            OPENAI_ANSWER_PROVIDER
            if legacy_use_openai_answers
            else configured_answer_provider or LOCAL_ANSWER_PROVIDER
        )

        return cls(
            database_url=os.getenv("DATABASE_URL") or build_database_url(),
            embedding_model=embedding_model,
            use_openai_embeddings=use_openai_embeddings,
            use_openai_answers=answer_provider == OPENAI_ANSWER_PROVIDER,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_provider=embedding_provider,
            embedding_dimension=(
                default_dimension
                if legacy_openai_override
                else env_int("EMBEDDING_DIMENSION", default_dimension)
            ),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
            semantic_candidate_count=env_int("SEMANTIC_CANDIDATE_COUNT", 20),
            lexical_candidate_count=env_int("LEXICAL_CANDIDATE_COUNT", 20),
            rrf_k=env_int("RRF_K", 60),
            reranking_enabled=env_flag("RERANKING_ENABLED", default=False),
            reranker_model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip(),
            rerank_candidate_limit=env_int("RERANK_CANDIDATE_LIMIT", 20),
            answer_provider=answer_provider,
            answer_model=os.getenv("ANSWER_MODEL", DEFAULT_ANSWER_MODEL).strip()
            or DEFAULT_ANSWER_MODEL,
            answer_timeout_seconds=env_float(
                "ANSWER_TIMEOUT_SECONDS", DEFAULT_ANSWER_TIMEOUT_SECONDS
            ),
            answer_max_retries=env_bounded_retry_count(
                "ANSWER_MAX_RETRIES", DEFAULT_ANSWER_MAX_RETRIES
            ),
            observability_enabled=env_flag(
                "OBSERVABILITY_ENABLED",
                default=False,
            ),
            observability_connect_timeout_seconds=env_int(
                "OBSERVABILITY_CONNECT_TIMEOUT_SECONDS",
                DEFAULT_OBSERVABILITY_CONNECT_TIMEOUT_SECONDS,
            ),
        )

    @property
    def safe_database_target(self) -> str:
        return mask_database_url(self.database_url)
