"""Deterministic semantic/lexical fusion and retrieval-mode orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.common.config import Settings
from src.retrieval.reranker import Reranker, create_reranker, rerank_results
from src.retrieval.retrieve_context import (
    expand_schema_field_aliases,
    retrieve_lexical_context,
    retrieve_semantic_context,
    validate_candidate_limit,
    validate_top_k,
)


SemanticRetriever = Callable[..., list[dict]]
LexicalRetriever = Callable[..., list[dict]]


def reciprocal_rank_fusion(
    semantic_results: Sequence[dict],
    lexical_results: Sequence[dict],
    rrf_k: int = 60,
) -> list[dict]:
    if rrf_k <= 0:
        raise ValueError("RRF_K must be greater than 0")

    fused: dict[str, dict] = {}
    for source_name, results, rank_field, score_field in (
        ("semantic", semantic_results, "semantic_rank", "semantic_score"),
        ("lexical", lexical_results, "lexical_rank", "lexical_score"),
    ):
        for fallback_rank, original in enumerate(results, start=1):
            chunk_id = str(original["chunk_id"])
            rank = int(original.get(rank_field) or original.get("rank") or fallback_rank)
            result = fused.setdefault(chunk_id, dict(original))
            result[rank_field] = rank
            result[score_field] = original.get(score_field)
            if source_name == "semantic":
                result["similarity_score"] = original.get("semantic_score")
            result["fusion_score"] = float(result.get("fusion_score") or 0.0) + (
                1.0 / (rrf_k + rank)
            )

    for result in fused.values():
        result.setdefault("semantic_score", None)
        result.setdefault("semantic_rank", None)
        result.setdefault("lexical_score", None)
        result.setdefault("lexical_rank", None)
        result.setdefault("similarity_score", None)
        result.setdefault("reranker_score", None)
        result.setdefault("pre_rerank_rank", None)
        result["retrieval_mode"] = "hybrid"

    ordered = sorted(
        fused.values(),
        key=lambda result: (
            -float(result["fusion_score"]),
            min(
                int(result["semantic_rank"] or 10**9),
                int(result["lexical_rank"] or 10**9),
            ),
            int(result["semantic_rank"] or 10**9),
            int(result["lexical_rank"] or 10**9),
            str(result["chunk_id"]),
        ),
    )
    for rank, result in enumerate(ordered, start=1):
        result["rank"] = rank
    return ordered


def finalize_results(results: Sequence[dict], top_k: int) -> list[dict]:
    final_results = [dict(result) for result in results[:top_k]]
    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
    return final_results


def retrieve_with_mode(
    question: str,
    top_k: int,
    min_similarity: float,
    settings: Settings,
    semantic_retriever: SemanticRetriever = retrieve_semantic_context,
    lexical_retriever: LexicalRetriever = retrieve_lexical_context,
    reranker: Reranker | None = None,
) -> list[dict]:
    final_limit = validate_top_k(top_k)
    retrieval_question = expand_schema_field_aliases(question)
    semantic_limit = validate_candidate_limit(
        settings.semantic_candidate_count,
        "semantic candidate count",
    )
    mode = settings.retrieval_mode.strip().lower()
    if mode not in {"semantic", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE must be 'semantic' or 'hybrid'")

    semantic_results = semantic_retriever(
        retrieval_question,
        candidate_limit=semantic_limit,
        min_similarity=min_similarity,
        settings=settings,
    )
    if mode == "semantic":
        candidates = semantic_results
    else:
        lexical_limit = validate_candidate_limit(
            settings.lexical_candidate_count,
            "lexical candidate count",
        )
        lexical_results = lexical_retriever(
            retrieval_question,
            candidate_limit=lexical_limit,
            settings=settings,
        )
        candidates = reciprocal_rank_fusion(
            semantic_results,
            lexical_results,
            rrf_k=settings.rrf_k,
        )

    if settings.reranking_enabled:
        active_reranker = reranker or create_reranker(settings)
        candidates = rerank_results(
            retrieval_question,
            candidates,
            active_reranker,
            settings.rerank_candidate_limit,
        )

    return finalize_results(candidates, final_limit)
