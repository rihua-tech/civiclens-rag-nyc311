"""Backward-compatible opt-in OpenAI embedding provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.common.config import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DETERMINISTIC_DIMENSION,
    OPENAI_PROVIDER,
)
from src.embeddings.providers.base import EmbeddingSpec, validate_embedding


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        dimension: int = DETERMINISTIC_DIMENSION,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when USE_OPENAI_EMBEDDINGS=true "
                "or EMBEDDING_PROVIDER=openai"
            )
        self._api_key = api_key
        self._client_factory = client_factory
        self._spec = EmbeddingSpec(OPENAI_PROVIDER, model_name, dimension)

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(api_key=self._api_key)
        from openai import OpenAI

        return OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> list[float]:
        response = self._client().embeddings.create(model=self.spec.model, input=text)
        return validate_embedding(response.data[0].embedding, self.spec)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
