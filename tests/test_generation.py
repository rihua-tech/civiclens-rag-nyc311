from src.common.config import Settings
from src.generation.answer_question import (
    NO_ANSWER,
    answer_question,
    format_answer_response,
    local_answer,
)


def sample_chunk() -> dict:
    return {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "chunk_text": "The NYC 311 Lakehouse architecture uses ingestion, cleaning, chunking, embeddings, and pgvector storage.",
        "source_name": "architecture.md",
        "source_path": "docs/architecture.md",
        "similarity_score": 0.44,
        "rank": 1,
    }


def architecture_markdown_chunk() -> dict:
    return {
        "chunk_id": "chunk_architecture",
        "document_id": "doc_architecture",
        "chunk_text": (
            "# Architecture ## CivicLens RAG - Hybrid RAG Architecture ```text "
            "NYC 311 Documentation + NYC 311 Data Dictionary + NYC 311 Lakehouse README / Runbooks "
            "+ Gold Mart Sample Summaries -> Ingestion Pipeline -> Text Cleaning + Chunking "
            "-> Metadata Tagging -> Embedding Generation -> PostgreSQL + pgvector -> Retriever "
            "-> LLM Answer Generator -> Cited Answer UI ``` ## Design Principle "
            "Documents and metadata are stored for retrieval. Structured metrics should remain in SQL tables."
        ),
        "source_name": "architecture.md",
        "source_path": "docs/architecture.md",
        "similarity_score": 0.52,
        "rank": 1,
    }


def test_cited_answer_includes_sources_section():
    response = local_answer("What is the NYC 311 Lakehouse architecture?", [sample_chunk()])
    formatted = format_answer_response(response)

    assert response["answer"] != NO_ANSWER
    assert "Sources:" in formatted
    assert "architecture.md - docs/architecture.md - chunk chunk_1" in formatted


def test_rag_answer_removes_raw_markdown_heading_clutter():
    response = local_answer("What is the NYC 311 Lakehouse architecture?", [architecture_markdown_chunk()])

    assert response["answer"] != NO_ANSWER
    assert "##" not in response["answer"]
    assert "```" not in response["answer"]
    assert "Architecture ##" not in response["answer"]
    assert "PostgreSQL/pgvector storage" in response["answer"]
    assert response["sources"][0]["source_name"] == "architecture.md"


def test_empty_retrieval_returns_safe_no_answer():
    response = local_answer("What is the answer?", [])

    assert response["answer"] == NO_ANSWER
    assert response["sources"] == []
    assert response["retrieved_chunks"] == []


def test_weak_retrieval_returns_safe_no_answer():
    unrelated_chunk = sample_chunk()
    unrelated_chunk["chunk_text"] = "This sentence discusses unrelated local setup details."

    response = local_answer("What does complaint_type mean?", [unrelated_chunk])

    assert response["answer"] == NO_ANSWER
    assert response["sources"]


def test_answer_generation_does_not_require_openai_by_default(monkeypatch):
    monkeypatch.delenv("USE_OPENAI_ANSWERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.generation.answer_question.retrieve_context",
        lambda question, top_k, min_similarity, settings: [sample_chunk()],
    )
    settings = Settings(
        database_url="postgresql://example",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
    )

    response = answer_question("What is the architecture?", settings=settings)

    assert response["answer"] != NO_ANSWER
