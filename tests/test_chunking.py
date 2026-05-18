import json
from pathlib import Path

from src.chunking.chunk_documents import create_chunks


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_chunking_writes_chunks_jsonl(tmp_path):
    input_path = tmp_path / "data" / "processed" / "documents.jsonl"
    output_path = tmp_path / "data" / "processed" / "chunks.jsonl"
    document = {
        "document_id": "doc_test",
        "source_name": "source.md",
        "source_type": "markdown",
        "source_path": "docs/source.md",
        "text": " ".join(f"word{i}" for i in range(95)),
        "ingested_at": "2026-05-18T00:00:00Z",
    }
    write_jsonl(input_path, [document])

    chunks, written_path = create_chunks(
        repo_root=tmp_path,
        input_path=input_path,
        output_path=output_path,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert written_path == output_path
    assert output_path.is_file()
    records = read_jsonl(output_path)
    assert records == chunks
    assert len(records) == 3


def test_chunks_preserve_source_metadata_and_text(tmp_path):
    input_path = tmp_path / "documents.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    document = {
        "document_id": "doc_metadata",
        "source_name": "architecture.md",
        "source_type": "markdown",
        "source_path": "docs/architecture.md",
        "text": "alpha beta gamma\n\n delta epsilon zeta",
        "ingested_at": "2026-05-18T00:00:00Z",
    }
    write_jsonl(input_path, [document])

    chunks, _ = create_chunks(
        repo_root=tmp_path,
        input_path=input_path,
        output_path=output_path,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks
    first_chunk = chunks[0]
    assert first_chunk["chunk_id"] == "doc_metadata_chunk_0000"
    assert first_chunk["document_id"] == "doc_metadata"
    assert first_chunk["source_name"] == "architecture.md"
    assert first_chunk["source_path"] == "docs/architecture.md"
    assert first_chunk["chunk_text"]
    assert first_chunk["chunk_text"] == "alpha beta gamma delta"
    assert first_chunk["token_count"] == 4
