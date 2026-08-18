import json

from src.embeddings.embed_chunks import (
    EMBEDDING_DIMENSIONS,
    load_chunks,
    local_deterministic_embedding,
    upsert_chunk,
    upsert_document,
)


class RecordingCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, parameters=None):
        self.calls.append((query, parameters))


def metadata_chunk() -> dict:
    return {
        "chunk_id": "doc_1_chunk_abc123",
        "document_id": "doc_1",
        "chunk_text": "Readable local chunk.",
        "chunk_index": 0,
        "source_name": "NYC 311 Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/field-guide.md",
        "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
        "source_version": "dataset erm2-nwe9",
        "source_retrieved_at": "2026-08-17",
        "section_title": "Status",
        "heading_path": ["Fields", "Status"],
        "word_count": 3,
        "content_hash": "sha256:chunk",
        "document_content_hash": "sha256:document",
        "chunking_config_hash": "sha256:chunking-config",
        "ingested_at": "2026-08-17T00:00:00Z",
    }


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
    expected_chunk = metadata_chunk()
    chunks_path.write_text(json.dumps(expected_chunk) + "\n", encoding="utf-8")

    assert load_chunks(chunks_path) == [expected_chunk]


def test_document_upsert_preserves_manifest_and_content_metadata():
    cursor = RecordingCursor()

    upsert_document(cursor, metadata_chunk())

    query, parameters = cursor.calls[0]
    assert "source_category" in query
    assert "source_url" in query
    assert "source_version" in query
    assert "source_retrieved_at" in query
    assert "content_hash" in query
    assert parameters == (
        "doc_1",
        "NYC 311 Field Guide",
        "markdown",
        "external_nyc311",
        "docs/knowledge/field-guide.md",
        "https://data.cityofnewyork.us/d/erm2-nwe9",
        "dataset erm2-nwe9",
        "2026-08-17",
        "sha256:document",
        "sha256:chunking-config",
        "2026-08-17T00:00:00Z",
    )


def test_chunk_upsert_preserves_section_hash_and_count_metadata():
    cursor = RecordingCursor()

    upsert_chunk(cursor, metadata_chunk(), [0.1, -0.2])

    query, parameters = cursor.calls[0]
    assert query.count("%s") == 19
    assert "section_title" in query
    assert "heading_path" in query
    assert "word_count" in query
    assert "content_hash" in query
    assert "token_count" not in query
    assert parameters[11:18] == (
        "Status",
        ["Fields", "Status"],
        3,
        "sha256:chunk",
        "sha256:document",
        "sha256:chunking-config",
        "2026-08-17T00:00:00Z",
    )
