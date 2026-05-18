from src.retrieval.retrieve_context import format_cli_results, format_retrieval_rows, validate_top_k


def test_retriever_result_formatting_preserves_metadata():
    rows = [
        (
            "chunk_1",
            "doc_1",
            "Chunk text about NYC 311 architecture.",
            "architecture.md",
            "docs/architecture.md",
            0.42,
        )
    ]

    results = format_retrieval_rows(rows)

    assert results == [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "chunk_text": "Chunk text about NYC 311 architecture.",
            "source_name": "architecture.md",
            "source_path": "docs/architecture.md",
            "similarity_score": 0.42,
            "rank": 1,
        }
    ]


def test_top_k_validation_accepts_positive_values():
    assert validate_top_k(3) == 3


def test_top_k_validation_rejects_non_positive_values():
    try:
        validate_top_k(0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid top_k")


def test_cli_formatting_includes_similarity_and_source_metadata():
    output = format_cli_results(
        "What is the architecture?",
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "chunk_text": "Architecture context.",
                "source_name": "architecture.md",
                "source_path": "docs/architecture.md",
                "similarity_score": 0.31,
                "rank": 1,
            }
        ],
    )

    assert "score=0.3100" in output
    assert "architecture.md" in output
    assert "chunk_1" in output
