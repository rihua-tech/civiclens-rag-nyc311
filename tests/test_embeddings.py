import json
from pathlib import Path

from src.embeddings.embed_chunks import EMBEDDING_DIMENSIONS, load_chunks, local_deterministic_embedding


def test_local_deterministic_embedding_returns_1536_values():
    embedding = local_deterministic_embedding("sample chunk text")

    assert len(embedding) == EMBEDDING_DIMENSIONS
    assert all(isinstance(value, float) for value in embedding)


def test_local_deterministic_embedding_is_stable_for_same_text():
    first_embedding = local_deterministic_embedding("same text")
    second_embedding = local_deterministic_embedding("same text")

    assert first_embedding == second_embedding


def test_local_deterministic_embedding_differs_for_different_text():
    first_embedding = local_deterministic_embedding("first text")
    second_embedding = local_deterministic_embedding("second text")

    assert first_embedding != second_embedding


def test_load_chunks_reads_jsonl_file(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    expected_chunk = {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "chunk_text": "Readable local chunk.",
        "chunk_index": 0,
        "source_name": "source.md",
        "source_path": "docs/source.md",
        "token_count": 3,
    }
    chunks_path.write_text(json.dumps(expected_chunk) + "\n", encoding="utf-8")

    assert load_chunks(chunks_path) == [expected_chunk]
