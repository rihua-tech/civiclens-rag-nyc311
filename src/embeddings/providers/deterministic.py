"""Deterministic offline embedding provider retained for tests and CI."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from src.common.config import (
    DETERMINISTIC_DIMENSION,
    DETERMINISTIC_MODEL,
    DETERMINISTIC_PROVIDER,
)
from src.embeddings.providers.base import EmbeddingSpec, validate_embedding


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
EMBEDDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
}


def tokenize_for_embedding(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in EMBEDDING_STOPWORDS
    ]


def deterministic_embedding(text: str, dimensions: int = DETERMINISTIC_DIMENSION) -> list[float]:
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0")

    tokens = tokenize_for_embedding(text)
    if not tokens:
        tokens = [hashlib.sha256(text.encode("utf-8")).hexdigest()]

    embedding = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big", signed=False) % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        embedding[index] += sign

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [round(value / norm, 8) for value in embedding]


class DeterministicEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DETERMINISTIC_MODEL,
        dimension: int = DETERMINISTIC_DIMENSION,
    ) -> None:
        self._spec = EmbeddingSpec(DETERMINISTIC_PROVIDER, model_name, dimension)

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed(self, text: str) -> list[float]:
        return validate_embedding(deterministic_embedding(text, self.spec.dimension), self.spec)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
