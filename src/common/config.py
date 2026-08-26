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
DIRECT_ORCHESTRATION_MODE = "direct"
LANGGRAPH_ORCHESTRATION_MODE = "langgraph"
DEFAULT_LANGGRAPH_MAX_STEPS = 8
MAX_LANGGRAPH_MAX_STEPS = 32
LANGGRAPH_TOOL_CALL_LIMIT = 1
PGVECTOR_VECTOR_STORE = "pgvector"
PINECONE_VECTOR_STORE = "pinecone"
DEFAULT_PINECONE_TIMEOUT_SECONDS = 10.0
MAX_PINECONE_TIMEOUT_SECONDS = 60.0
DEFAULT_PINECONE_MAX_RETRIES = 2
DEFAULT_PINECONE_SYNC_MAX_ATTEMPTS = 5
CORS_ALLOWED_ORIGINS_ENV = "CIVICLENS_CORS_ALLOWED_ORIGINS"
DEFAULT_CORS_ALLOWED_ORIGINS = ("http://localhost:3000",)


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


def env_bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = env_int(name, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_bounded_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = env_float(name, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def cors_allowed_origins(raw_value: str | None = None) -> tuple[str, ...]:
    """Return a normalized, explicit browser-origin allowlist.

    A localhost-only default supports the Issue 19 development client without
    granting cross-origin access to arbitrary sites. Production origins must be
    supplied explicitly through ``CIVICLENS_CORS_ALLOWED_ORIGINS``.
    """

    if raw_value is None:
        load_dotenv_if_available()
        configured = os.getenv(CORS_ALLOWED_ORIGINS_ENV)
    else:
        configured = raw_value
    if configured is None:
        return DEFAULT_CORS_ALLOWED_ORIGINS

    origins: list[str] = []
    for candidate in configured.split(","):
        origin = candidate.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError(
                f"{CORS_ALLOWED_ORIGINS_ENV} must not contain the wildcard origin"
            )

        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{CORS_ALLOWED_ORIGINS_ENV} must contain only HTTP(S) origins"
            )

        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(
                f"{CORS_ALLOWED_ORIGINS_ENV} contains an invalid port"
            ) from exc

        normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        if normalized not in origins:
            origins.append(normalized)

    return tuple(origins)


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
    orchestration_mode: str = DIRECT_ORCHESTRATION_MODE
    langgraph_max_steps: int = DEFAULT_LANGGRAPH_MAX_STEPS
    langgraph_tool_call_limit: int = LANGGRAPH_TOOL_CALL_LIMIT
    vector_store_provider: str = PGVECTOR_VECTOR_STORE
    pinecone_api_key: str = field(default="", repr=False)
    pinecone_index_name: str = ""
    pinecone_index_host: str = ""
    pinecone_namespace_prefix: str = "civiclens"
    pinecone_timeout_seconds: float = DEFAULT_PINECONE_TIMEOUT_SECONDS
    pinecone_max_retries: int = DEFAULT_PINECONE_MAX_RETRIES
    pinecone_sync_max_attempts: int = DEFAULT_PINECONE_SYNC_MAX_ATTEMPTS

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
        orchestration_mode = os.getenv(
            "ORCHESTRATION_MODE",
            DIRECT_ORCHESTRATION_MODE,
        ).strip().lower()
        if orchestration_mode not in {
            DIRECT_ORCHESTRATION_MODE,
            LANGGRAPH_ORCHESTRATION_MODE,
        }:
            raise ValueError(
                "ORCHESTRATION_MODE must be either "
                f"{DIRECT_ORCHESTRATION_MODE!r} or {LANGGRAPH_ORCHESTRATION_MODE!r}"
            )
        tool_call_limit = env_int(
            "LANGGRAPH_TOOL_CALL_LIMIT",
            LANGGRAPH_TOOL_CALL_LIMIT,
        )
        if tool_call_limit != LANGGRAPH_TOOL_CALL_LIMIT:
            raise ValueError("LANGGRAPH_TOOL_CALL_LIMIT must be exactly 1")
        vector_store_provider = os.getenv(
            "VECTOR_STORE_PROVIDER",
            PGVECTOR_VECTOR_STORE,
        ).strip().lower()
        if vector_store_provider not in {
            PGVECTOR_VECTOR_STORE,
            PINECONE_VECTOR_STORE,
        }:
            raise ValueError(
                "VECTOR_STORE_PROVIDER must be either "
                f"{PGVECTOR_VECTOR_STORE!r} or {PINECONE_VECTOR_STORE!r}"
            )
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
        pinecone_index_host = os.getenv("PINECONE_INDEX_HOST", "").strip()
        pinecone_namespace_prefix = os.getenv(
            "PINECONE_NAMESPACE_PREFIX",
            "civiclens",
        ).strip()
        if vector_store_provider == PINECONE_VECTOR_STORE:
            missing = [
                name
                for name, value in (
                    ("PINECONE_API_KEY", pinecone_api_key),
                    ("PINECONE_INDEX_NAME", pinecone_index_name),
                    ("PINECONE_INDEX_HOST", pinecone_index_host),
                    ("PINECONE_NAMESPACE_PREFIX", pinecone_namespace_prefix),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Pinecone configuration is incomplete; missing " + ", ".join(missing)
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
            orchestration_mode=orchestration_mode,
            langgraph_max_steps=env_bounded_int(
                "LANGGRAPH_MAX_STEPS",
                DEFAULT_LANGGRAPH_MAX_STEPS,
                minimum=5,
                maximum=MAX_LANGGRAPH_MAX_STEPS,
            ),
            langgraph_tool_call_limit=tool_call_limit,
            vector_store_provider=vector_store_provider,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            pinecone_index_host=pinecone_index_host,
            pinecone_namespace_prefix=pinecone_namespace_prefix,
            pinecone_timeout_seconds=env_bounded_float(
                "PINECONE_TIMEOUT_SECONDS",
                DEFAULT_PINECONE_TIMEOUT_SECONDS,
                minimum=0.1,
                maximum=MAX_PINECONE_TIMEOUT_SECONDS,
            ),
            pinecone_max_retries=env_bounded_retry_count(
                "PINECONE_MAX_RETRIES",
                DEFAULT_PINECONE_MAX_RETRIES,
            ),
            pinecone_sync_max_attempts=env_bounded_int(
                "PINECONE_SYNC_MAX_ATTEMPTS",
                DEFAULT_PINECONE_SYNC_MAX_ATTEMPTS,
                minimum=1,
                maximum=20,
            ),
        )

    @property
    def safe_database_target(self) -> str:
        return mask_database_url(self.database_url)
