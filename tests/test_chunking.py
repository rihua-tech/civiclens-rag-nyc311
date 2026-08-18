import json
from pathlib import Path

from src.chunking.chunk_documents import chunk_documents, create_chunks
from src.ingestion.load_documents import content_hash


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metadata_document(**overrides) -> dict:
    document = {
        "document_id": "doc_metadata",
        "source_name": "NYC 311 Field Guide",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "source_path": "docs/knowledge/field-guide.md",
        "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
        "source_version": "dataset erm2-nwe9",
        "source_retrieved_at": "2026-08-17",
        "content_hash": content_hash("placeholder"),
        "text": "# Placeholder\n\nplaceholder",
        "ingested_at": "2026-08-17T00:00:00Z",
    }
    document.update(overrides)
    if "text" in overrides and "content_hash" not in overrides:
        document["content_hash"] = content_hash(document["text"])
    return document


def test_chunking_writes_chunks_jsonl(tmp_path):
    input_path = tmp_path / "data" / "processed" / "documents.jsonl"
    output_path = tmp_path / "data" / "processed" / "chunks.jsonl"
    document = metadata_document(
        text="# Words\n\n" + " ".join(f"word{i}" for i in range(95)),
    )
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


def test_markdown_chunks_respect_sections_and_preserve_heading_paths():
    document = metadata_document(
        text=(
            "# Service Request Fields\n\n"
            "Overview text.\n\n"
            "## Complaint Type\n\n"
            "Complaint type is the broad problem category.\n\n"
            "### API Compatibility\n\n"
            "The API field remains complaint_type.\n\n"
            "## Closed Date\n\n"
            "Closed date is when the agency closed the request."
        )
    )

    chunks = chunk_documents([document], chunk_size=50, chunk_overlap=5)

    by_section = {chunk["section_title"]: chunk for chunk in chunks}
    assert by_section["Service Request Fields"]["heading_path"] == ["Service Request Fields"]
    assert by_section["Complaint Type"]["heading_path"] == [
        "Service Request Fields",
        "Complaint Type",
    ]
    assert by_section["API Compatibility"]["heading_path"] == [
        "Service Request Fields",
        "Complaint Type",
        "API Compatibility",
    ]
    assert by_section["Closed Date"]["heading_path"] == [
        "Service Request Fields",
        "Closed Date",
    ]
    assert "Closed date" not in by_section["Complaint Type"]["chunk_text"]


def test_chunks_preserve_complete_metadata_and_use_word_count():
    document = metadata_document(text="# Status\n\nalpha beta gamma delta")

    chunks = chunk_documents([document], chunk_size=10, chunk_overlap=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"].startswith("doc_metadata_chunk_")
    assert chunk["document_id"] == "doc_metadata"
    assert chunk["source_name"] == "NYC 311 Field Guide"
    assert chunk["source_type"] == "markdown"
    assert chunk["source_category"] == "external_nyc311"
    assert chunk["source_path"] == "docs/knowledge/field-guide.md"
    assert chunk["source_url"] == "https://data.cityofnewyork.us/d/erm2-nwe9"
    assert chunk["source_version"] == "dataset erm2-nwe9"
    assert chunk["source_retrieved_at"] == "2026-08-17"
    assert chunk["section_title"] == "Status"
    assert chunk["heading_path"] == ["Status"]
    assert chunk["ingested_at"] == "2026-08-17T00:00:00Z"
    assert chunk["document_content_hash"] == document["content_hash"]
    assert chunk["chunking_config_hash"].startswith("sha256:")
    assert chunk["chunk_text"] == "Status alpha beta gamma delta"
    assert chunk["content_hash"] == content_hash("Status alpha beta gamma delta")
    assert chunk["word_count"] == 5
    assert "token_count" not in chunk


def test_chunk_ids_do_not_depend_on_ingestion_timestamp():
    first_document = metadata_document(ingested_at="2026-05-18T00:00:00Z")
    second_document = metadata_document(ingested_at="2026-08-17T00:00:00Z")

    first_ids = [chunk["chunk_id"] for chunk in chunk_documents([first_document])]
    second_ids = [chunk["chunk_id"] for chunk in chunk_documents([second_document])]

    assert first_ids == second_ids


def test_plain_text_documents_use_sectionless_fallback():
    document = metadata_document(
        source_type="text",
        source_path="docs/knowledge/runbook.txt",
        text="alpha beta gamma delta epsilon",
    )

    chunks = chunk_documents([document], chunk_size=3, chunk_overlap=1)

    assert [chunk["chunk_text"] for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta epsilon",
    ]
    assert all(chunk["section_title"] is None for chunk in chunks)
    assert all(chunk["heading_path"] == [] for chunk in chunks)
