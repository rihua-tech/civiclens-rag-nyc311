from __future__ import annotations

import asyncio

import pytest

from src.integrations.langchain_retriever import create_langchain_retriever


pytest.importorskip("langchain_core")


def _result() -> dict:
    return {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "chunk_text": "Canonical CivicLens chunk text.",
        "source_name": "architecture.md",
        "source_type": "markdown",
        "source_category": "civiclens_project",
        "source_path": "docs/architecture.md",
        "source_url": "https://example.test/architecture",
        "source_version": "Issue 18",
        "source_retrieved_at": "2026-08-25",
        "section_title": "Retrieval",
        "heading_path": ["Architecture", "Retrieval"],
        "word_count": 4,
        "content_hash": "sha256:chunk",
        "document_content_hash": "sha256:document",
        "chunking_config_hash": "sha256:chunking",
        "ingested_at": "2026-08-25T00:00:00Z",
        "similarity_score": 0.82,
        "semantic_score": 0.82,
        "semantic_rank": 1,
        "lexical_score": 0.5,
        "lexical_rank": 2,
        "fusion_score": 0.03,
        "reranker_score": None,
        "pre_rerank_rank": None,
        "retrieval_mode": "hybrid",
        "rank": 1,
    }


def test_langchain_adapter_maps_native_results_without_a_second_rag_path():
    calls = []

    def retrieve(question, **kwargs):
        calls.append((question, kwargs))
        return [_result()]

    retriever = create_langchain_retriever(
        top_k=4,
        min_similarity=0.41,
        retrieval_callable=retrieve,
    )

    documents = retriever.invoke("How does retrieval work?")

    assert calls == [
        (
            "How does retrieval work?",
            {"top_k": 4, "min_similarity": 0.41, "settings": None},
        )
    ]
    assert documents[0].id == "chunk_1"
    assert documents[0].page_content == "Canonical CivicLens chunk text."
    assert documents[0].metadata["chunk_id"] == "chunk_1"
    assert documents[0].metadata["source_path"] == "docs/architecture.md"
    assert documents[0].metadata["rank"] == 1
    assert documents[0].metadata["fusion_score"] == 0.03
    assert documents[0].metadata["score"] == 0.03
    assert "chunk_text" not in documents[0].metadata


def test_langchain_adapter_async_boundary_preserves_the_same_mapping():
    retriever = create_langchain_retriever(
        retrieval_callable=lambda question, **kwargs: [_result()]
    )

    documents = asyncio.run(retriever.ainvoke("question"))

    assert documents[0].id == "chunk_1"
    assert documents[0].metadata["semantic_score"] == 0.82
