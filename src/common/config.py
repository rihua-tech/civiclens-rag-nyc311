"""Shared local configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus, urlsplit, urlunsplit


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
    database_url: str
    embedding_model: str
    use_openai_embeddings: bool
    openai_api_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_available()
        return cls(
            database_url=os.getenv("DATABASE_URL") or build_database_url(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "local-deterministic-1536"),
            use_openai_embeddings=env_flag("USE_OPENAI_EMBEDDINGS", default=False),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    @property
    def safe_database_target(self) -> str:
        return mask_database_url(self.database_url)
