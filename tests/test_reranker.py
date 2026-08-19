from src.common.config import Settings
from src.retrieval.hybrid_retriever import retrieve_with_mode
from src.retrieval.reranker import rerank_results


def candidate(chunk_id: str, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_1",
        "chunk_text": f"passage {chunk_id}",
        "source_name": "NYC 311 Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/nyc311-service-request-fields.md",
        "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
        "source_version": "dataset erm2-nwe9",
        "source_retrieved_at": "2026-08-17",
        "section_title": "Status",
        "heading_path": ["Responding Agency", "Status"],
        "word_count": 3,
        "content_hash": f"sha256:{chunk_id}",
        "document_content_hash": "sha256:document",
        "chunking_config_hash": "sha256:chunking",
        "ingested_at": "2026-08-17T00:00:00Z",
        "similarity_score": 1 / rank,
        "semantic_score": 1 / rank,
        "semantic_rank": rank,
        "lexical_score": None,
        "lexical_rank": None,
        "fusion_score": None,
        "reranker_score": None,
        "pre_rerank_rank": None,
        "retrieval_mode": "semantic",
        "rank": rank,
    }


class FakeReranker:
    model_name = "fake-cross-encoder"

    def __init__(self, scores):
        self.scores = scores
        self.passages = []

    def score(self, question, passages):
        assert question == "Which status means closed?"
        self.passages = list(passages)
        return self.scores


def test_reranker_reorders_only_bounded_candidates_and_preserves_provenance():
    results = [candidate("a", 1), candidate("b", 2), candidate("c", 3)]
    reranker = FakeReranker([0.1, 0.9])

    reranked = rerank_results(
        "Which status means closed?",
        results,
        reranker,
        candidate_limit=2,
    )

    assert reranker.passages == ["passage a", "passage b"]
    assert [result["chunk_id"] for result in reranked] == ["b", "a", "c"]
    assert reranked[0]["pre_rerank_rank"] == 2
    assert reranked[0]["reranker_score"] == 0.9
    assert reranked[0]["source_path"] == (
        "docs/knowledge/nyc311-service-request-fields.md"
    )
    assert reranked[0]["heading_path"] == ["Responding Agency", "Status"]
    assert reranked[2]["reranker_score"] is None


def test_reranker_ties_use_original_rank_then_chunk_id():
    results = [candidate("b", 1), candidate("a", 2)]

    reranked = rerank_results(
        "Which status means closed?",
        results,
        FakeReranker([0.5, 0.5]),
        candidate_limit=2,
    )

    assert [result["chunk_id"] for result in reranked] == ["b", "a"]


def test_disabled_reranking_leaves_normal_retrieval_path_unchanged():
    original = [candidate("a", 1), candidate("b", 2)]
    settings = Settings(
        database_url="postgresql://example",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        retrieval_mode="semantic",
        reranking_enabled=False,
    )

    results = retrieve_with_mode(
        "Which status means closed?",
        top_k=2,
        min_similarity=0.05,
        settings=settings,
        semantic_retriever=lambda *args, **kwargs: original,
        reranker=FakeReranker([0.9, 0.1]),
    )

    assert [result["chunk_id"] for result in results] == ["a", "b"]
    assert all(result["reranker_score"] is None for result in results)


def test_enabled_reranking_changes_orchestrated_order_without_model_download():
    settings = Settings(
        database_url="postgresql://example",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider="deterministic",
        embedding_dimension=1536,
        retrieval_mode="semantic",
        reranking_enabled=True,
        rerank_candidate_limit=2,
    )

    results = retrieve_with_mode(
        "Which status means closed?",
        top_k=2,
        min_similarity=0.05,
        settings=settings,
        semantic_retriever=lambda *args, **kwargs: [
            candidate("a", 1),
            candidate("b", 2),
        ],
        reranker=FakeReranker([0.1, 0.9]),
    )

    assert [result["chunk_id"] for result in results] == ["b", "a"]
