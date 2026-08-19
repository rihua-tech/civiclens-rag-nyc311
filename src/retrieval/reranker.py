"""Optional bounded local cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from src.common.config import Settings


MAX_RERANK_CANDIDATES = 100
_MODEL_CACHE: dict[str, Any] = {}


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


class SentenceTransformersCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Reranker model name must not be empty")
        self._model_name = model_name
        self._model_loader = model_loader
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        if self._model is None:
            if self._model_loader is not None:
                self._model = self._model_loader(self.model_name)
            else:
                from sentence_transformers import CrossEncoder

                self._model = _MODEL_CACHE.get(self.model_name)
                if self._model is None:
                    self._model = CrossEncoder(self.model_name)
                    _MODEL_CACHE[self.model_name] = self._model
        return self._model

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        passage_list = list(passages)
        if not passage_list:
            return []
        pairs = [(question, passage) for passage in passage_list]
        scores = self._load_model().predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


def create_reranker(settings: Settings) -> Reranker:
    return SentenceTransformersCrossEncoderReranker(settings.reranker_model)


def validate_rerank_limit(candidate_limit: int) -> int:
    if candidate_limit <= 0:
        raise ValueError("rerank candidate limit must be greater than 0")
    if candidate_limit > MAX_RERANK_CANDIDATES:
        raise ValueError(
            f"rerank candidate limit must be less than or equal to {MAX_RERANK_CANDIDATES}"
        )
    return candidate_limit


def rerank_results(
    question: str,
    results: Sequence[dict],
    reranker: Reranker,
    candidate_limit: int,
) -> list[dict]:
    limit = validate_rerank_limit(candidate_limit)
    copied_results = [dict(result) for result in results]
    if not copied_results:
        return []

    bounded_count = min(limit, len(copied_results))
    reranked_candidates = copied_results[:bounded_count]
    untouched_candidates = copied_results[bounded_count:]
    for position, result in enumerate(copied_results, start=1):
        result["pre_rerank_rank"] = result.get("rank") or position
        result["reranker_score"] = None

    scores = reranker.score(
        question,
        [str(result["chunk_text"]) for result in reranked_candidates],
    )
    if len(scores) != len(reranked_candidates):
        raise RuntimeError(
            f"Reranker {reranker.model_name!r} returned {len(scores)} scores for "
            f"{len(reranked_candidates)} candidates"
        )
    for result, score in zip(reranked_candidates, scores, strict=True):
        result["reranker_score"] = float(score)
        result["reranker_model"] = reranker.model_name

    reranked_candidates.sort(
        key=lambda result: (
            -float(result["reranker_score"]),
            int(result["pre_rerank_rank"]),
            str(result["chunk_id"]),
        )
    )
    ordered = reranked_candidates + untouched_candidates
    for rank, result in enumerate(ordered, start=1):
        result["rank"] = rank
    return ordered
