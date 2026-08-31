import pytest

from src.common.config import Settings
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, retrieve_with_mode


def result(chunk_id: str, rank: int, mode: str) -> dict:
    semantic = mode == "semantic"
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "chunk_text": f"Evidence for {chunk_id}",
        "source_name": "NYC 311 Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/nyc311-service-request-fields.md",
        "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
        "source_version": "dataset erm2-nwe9",
        "source_retrieved_at": "2026-08-17",
        "section_title": "Complaint Type",
        "heading_path": ["Request Identity", "Complaint Type"],
        "word_count": 12,
        "content_hash": f"sha256:{chunk_id}",
        "document_content_hash": "sha256:document",
        "chunking_config_hash": "sha256:chunking",
        "ingested_at": "2026-08-17T00:00:00Z",
        "similarity_score": 0.9 / rank if semantic else None,
        "semantic_score": 0.9 / rank if semantic else None,
        "semantic_rank": rank if semantic else None,
        "lexical_score": 0.8 / rank if not semantic else None,
        "lexical_rank": rank if not semantic else None,
        "fusion_score": None,
        "reranker_score": None,
        "pre_rerank_rank": None,
        "retrieval_mode": mode,
        "rank": rank,
    }


def settings(**overrides) -> Settings:
    values = {
        "database_url": "postgresql://example",
        "embedding_model": "local-deterministic-1536",
        "use_openai_embeddings": False,
        "use_openai_answers": False,
        "openai_api_key": "",
        "embedding_provider": "deterministic",
        "embedding_dimension": 1536,
        "retrieval_mode": "hybrid",
        "semantic_candidate_count": 10,
        "lexical_candidate_count": 10,
        "rrf_k": 60,
        "reranking_enabled": False,
        "rerank_candidate_limit": 5,
    }
    values.update(overrides)
    return Settings(**values)


def test_rrf_combines_results_deduplicates_and_calculates_scores():
    semantic = [result("a", 1, "semantic"), result("b", 2, "semantic")]
    lexical = [result("b", 1, "lexical"), result("c", 2, "lexical")]

    fused = reciprocal_rank_fusion(semantic, lexical, rrf_k=60)

    assert [item["chunk_id"] for item in fused] == ["b", "a", "c"]
    assert len(fused) == 3
    assert fused[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1]["fusion_score"] == pytest.approx(1 / 61)
    assert fused[0]["semantic_rank"] == 2
    assert fused[0]["lexical_rank"] == 1
    assert fused[0]["semantic_score"] == pytest.approx(0.45)
    assert fused[0]["lexical_score"] == pytest.approx(0.8)
    assert fused[0]["section_title"] == "Complaint Type"
    assert fused[0]["heading_path"] == ["Request Identity", "Complaint Type"]


def test_rrf_tie_ordering_is_deterministic():
    semantic = [result("semantic", 1, "semantic")]
    lexical = [result("lexical", 1, "lexical")]

    first = reciprocal_rank_fusion(semantic, lexical)
    second = reciprocal_rank_fusion(semantic, lexical)

    assert [item["chunk_id"] for item in first] == ["semantic", "lexical"]
    assert first == second


def test_semantic_only_mode_does_not_execute_lexical_retrieval():
    calls = []

    def semantic_retriever(*args, **kwargs):
        calls.append(("semantic", kwargs["candidate_limit"]))
        return [result("a", 1, "semantic"), result("b", 2, "semantic")]

    def lexical_retriever(*args, **kwargs):
        raise AssertionError("Lexical retrieval must not run in semantic-only mode")

    results = retrieve_with_mode(
        "complaint type",
        top_k=1,
        min_similarity=0.05,
        settings=settings(retrieval_mode="semantic"),
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
    )

    assert calls == [("semantic", 10)]
    assert [item["chunk_id"] for item in results] == ["a"]
    assert results[0]["retrieval_mode"] == "semantic"


def test_hybrid_mode_executes_bounded_candidates_and_returns_stable_contract():
    calls = []

    def semantic_retriever(*args, **kwargs):
        calls.append(("semantic", kwargs["candidate_limit"]))
        return [result("a", 1, "semantic"), result("b", 2, "semantic")]

    def lexical_retriever(*args, **kwargs):
        calls.append(("lexical", kwargs["candidate_limit"]))
        return [result("b", 1, "lexical")]

    results = retrieve_with_mode(
        "complaint_type",
        top_k=2,
        min_similarity=0.05,
        settings=settings(semantic_candidate_count=8, lexical_candidate_count=6),
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
    )

    assert calls == [("semantic", 8), ("lexical", 6)]
    assert [item["chunk_id"] for item in results] == ["b", "a"]
    assert all(item["retrieval_mode"] == "hybrid" for item in results)
    assert results[0]["rank"] == 1
    assert results[0]["source_category"] == "external_nyc311"
    assert results[0]["document_content_hash"] == "sha256:document"


def test_snake_case_comparison_expands_before_retrieving_both_field_sections():
    calls = []

    def semantic_retriever(question, **kwargs):
        calls.append(("semantic", question))
        return [
            {
                **result("closed", 1, "semantic"),
                "section_title": "Closed Date",
            },
            {
                **result("due", 2, "semantic"),
                "section_title": "Due Date",
            },
        ]

    def lexical_retriever(question, **kwargs):
        calls.append(("lexical", question))
        return []

    results = retrieve_with_mode(
        "What is the difference between closed_date and due_date?",
        top_k=2,
        min_similarity=0.05,
        settings=settings(),
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
    )

    assert [mode for mode, _ in calls] == ["semantic", "lexical"]
    assert all("closed_date" in question for _, question in calls)
    assert all("due_date" in question for _, question in calls)
    assert all("Closed Date" in question for _, question in calls)
    assert all("Due Date" in question for _, question in calls)
    assert all("Field Guide" not in question for _, question in calls)
    assert {item["section_title"] for item in results} == {
        "Closed Date",
        "Due Date",
    }


def test_invalid_or_unbounded_retrieval_configuration_fails():
    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        retrieve_with_mode(
            "status",
            top_k=5,
            min_similarity=0.05,
            settings=settings(retrieval_mode="lexical"),
            semantic_retriever=lambda *args, **kwargs: [],
        )

    with pytest.raises(ValueError, match="less than or equal to 100"):
        retrieve_with_mode(
            "status",
            top_k=5,
            min_similarity=0.05,
            settings=settings(semantic_candidate_count=101),
            semantic_retriever=lambda *args, **kwargs: [],
        )
