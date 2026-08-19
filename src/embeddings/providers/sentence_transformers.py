"""Lazy local Sentence Transformers embedding provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.common.config import (
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL,
    SEMANTIC_PROVIDER,
)
from src.embeddings.providers.base import (
    EmbeddingCompatibilityError,
    EmbeddingSpec,
    validate_embedding,
)


_MODEL_CACHE: dict[str, Any] = {}


class SentenceTransformersEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        dimension: int = DEFAULT_SEMANTIC_DIMENSION,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._spec = EmbeddingSpec(SEMANTIC_PROVIDER, model_name, dimension)
        self._model_loader = model_loader
        self._model: Any | None = None

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _load_model(self) -> Any:
        if self._model is None:
            if self._model_loader is not None:
                self._model = self._model_loader(self.spec.model)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = _MODEL_CACHE.get(self.spec.model)
                if self._model is None:
                    self._model = SentenceTransformer(self.spec.model)
                    _MODEL_CACHE[self.spec.model] = self._model

            dimension_getter = getattr(self._model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = self._model.get_sentence_embedding_dimension
            runtime_dimension = int(dimension_getter())
            if runtime_dimension != self.spec.dimension:
                raise EmbeddingCompatibilityError(
                    f"Sentence Transformers model {self.spec.model!r} reports "
                    f"{runtime_dimension} dimensions; configured dimension is "
                    f"{self.spec.dimension}. Run the documented full reindex only after "
                    "the pgvector schema dimension is made compatible."
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        model = self._load_model()
        embeddings = model.encode(
            text_list,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [validate_embedding(embedding, self.spec) for embedding in embeddings]
