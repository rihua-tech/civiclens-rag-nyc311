import json
from pathlib import Path

from src.ingestion.load_documents import ingest_documents, load_documents


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_document_records_contain_required_metadata(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# CivicLens\n\nLocal project overview.", encoding="utf-8")

    documents = load_documents(
        repo_root=tmp_path,
        source_paths=("README.md", "docs/missing.md"),
        ingested_at="2026-05-18T00:00:00Z",
    )

    assert len(documents) == 1
    document = documents[0]
    assert set(document) == {
        "document_id",
        "source_name",
        "source_type",
        "source_path",
        "text",
        "ingested_at",
    }
    assert document["document_id"].startswith("doc_")
    assert document["source_name"] == "README.md"
    assert document["source_type"] == "markdown"
    assert document["source_path"] == "README.md"
    assert document["text"] == "# CivicLens\n\nLocal project overview."
    assert document["ingested_at"] == "2026-05-18T00:00:00Z"


def test_ingestion_writes_documents_jsonl(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "data-sources.md").write_text("# Data Sources\n\nTrusted notes.", encoding="utf-8")
    output_path = tmp_path / "data" / "processed" / "documents.jsonl"

    documents, written_path = ingest_documents(
        repo_root=tmp_path,
        source_paths=("docs/data-sources.md",),
        output_path=output_path,
        ingested_at="2026-05-18T00:00:00Z",
    )

    assert written_path == output_path
    assert output_path.is_file()
    records = read_jsonl(output_path)
    assert records == documents
    assert records[0]["source_path"] == "docs/data-sources.md"
