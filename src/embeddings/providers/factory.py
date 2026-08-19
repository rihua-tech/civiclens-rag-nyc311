"""Embedding-provider selection with Phase 1 configuration compatibility."""

from __future__ import annotations

from src.common.config import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL,
    DETERMINISTIC_DIMENSION,
    DETERMINISTIC_MODEL,
    DETERMINISTIC_PROVIDER,
    OPENAI_PROVIDER,
    SEMANTIC_PROVIDER,
    Settings,
)
from src.embeddings.providers.base import EmbeddingProvider
from src.embeddings.providers.deterministic import DeterministicEmbeddingProvider
from src.embeddings.providers.openai_provider import OpenAIEmbeddingProvider
from src.embeddings.providers.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)


PROVIDER_ALIASES = {
    "local": DETERMINISTIC_PROVIDER,
    "local_deterministic": DETERMINISTIC_PROVIDER,
    "sentence-transformers": SEMANTIC_PROVIDER,
    "semantic": SEMANTIC_PROVIDER,
}


def selected_provider_name(settings: Settings) -> str:
    if settings.use_openai_embeddings:
        return OPENAI_PROVIDER
    configured = settings.embedding_provider.strip().lower()
    if configured:
        return PROVIDER_ALIASES.get(configured, configured)
    if settings.embedding_model == DETERMINISTIC_MODEL:
        return DETERMINISTIC_PROVIDER
    return SEMANTIC_PROVIDER


def configured_dimension(settings: Settings, provider_name: str) -> int:
    if settings.embedding_dimension > 0:
        return settings.embedding_dimension
    if provider_name in {DETERMINISTIC_PROVIDER, OPENAI_PROVIDER}:
        return DETERMINISTIC_DIMENSION
    return DEFAULT_SEMANTIC_DIMENSION


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    active_settings = settings or Settings.from_env()
    provider_name = selected_provider_name(active_settings)
    dimension = configured_dimension(active_settings, provider_name)

    if provider_name == DETERMINISTIC_PROVIDER:
        model_name = active_settings.embedding_model or DETERMINISTIC_MODEL
        return DeterministicEmbeddingProvider(model_name=model_name, dimension=dimension)
    if provider_name == SEMANTIC_PROVIDER:
        model_name = active_settings.embedding_model or DEFAULT_SEMANTIC_MODEL
        return SentenceTransformersEmbeddingProvider(model_name=model_name, dimension=dimension)
    if provider_name == OPENAI_PROVIDER:
        model_name = active_settings.embedding_model or DEFAULT_OPENAI_EMBEDDING_MODEL
        if model_name == DETERMINISTIC_MODEL:
            model_name = DEFAULT_OPENAI_EMBEDDING_MODEL
        return OpenAIEmbeddingProvider(
            api_key=active_settings.openai_api_key,
            model_name=model_name,
            dimension=dimension,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER {provider_name!r}; expected deterministic, "
        "sentence_transformers, or openai"
    )
