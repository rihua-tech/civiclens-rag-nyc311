import json
from pathlib import Path

import pytest

from src.ingestion.load_documents import (
    content_hash,
    ingest_documents,
    load_documents,
    load_source_manifest,
    stable_document_id,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(repo_root: Path, source_path: str, expected_hash: str) -> Path:
    manifest_path = repo_root / "docs" / "knowledge" / "source-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "hashing": "sha256 over normalized UTF-8 text",
        "sources": [
            {
                "source_name": "NYC 311 Test Field Guide",
                "source_type": "markdown",
                "source_category": "external_nyc311",
                "path": source_path,
                "source_url": "https://data.cityofnewyork.us/d/erm2-nwe9",
                "source_version": "test fixture",
                "retrieved_at": "2026-08-17",
                "content_hash": expected_hash,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_manifest_is_default_authoritative_inventory(tmp_path):
    source_path = tmp_path / "docs" / "knowledge" / "field-guide.md"
    source_path.parent.mkdir(parents=True)
    source_text = "# Complaint Type\n\nOfficial field notes."
    source_path.write_text(source_text, encoding="utf-8")
    write_manifest(tmp_path, "docs/knowledge/field-guide.md", content_hash(source_text))

    documents = load_documents(repo_root=tmp_path, ingested_at="2026-08-17T00:00:00Z")

    assert len(documents) == 1
    document = documents[0]
    assert document["source_name"] == "NYC 311 Test Field Guide"
    assert document["source_category"] == "external_nyc311"
    assert document["source_url"] == "https://data.cityofnewyork.us/d/erm2-nwe9"
    assert document["source_version"] == "test fixture"
    assert document["source_retrieved_at"] == "2026-08-17"
    assert document["content_hash"] == content_hash(source_text)


def test_explicit_source_paths_override_preserves_local_workflow(tmp_path):
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
        "source_category",
        "source_path",
        "source_url",
        "source_version",
        "source_retrieved_at",
        "content_hash",
        "text",
        "ingested_at",
    }
    assert document["document_id"] == stable_document_id("README.md")
    assert document["source_name"] == "README.md"
    assert document["source_type"] == "markdown"
    assert document["source_category"] == "local_override"
    assert document["source_path"] == "README.md"
    assert document["text"] == "# CivicLens\n\nLocal project overview."
    assert document["ingested_at"] == "2026-05-18T00:00:00Z"


def test_document_ids_and_hashes_are_stable_and_timestamp_independent(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_bytes(b"# CivicLens  \r\n\r\nStable content.\r\n")

    first = load_documents(tmp_path, ("README.md",), "2026-05-18T00:00:00Z")[0]
    second = load_documents(tmp_path, ("./README.md",), "2026-08-17T00:00:00Z")[0]

    assert first["document_id"] == second["document_id"]
    assert first["content_hash"] == second["content_hash"]
    assert first["ingested_at"] != second["ingested_at"]
    assert first["content_hash"] == content_hash("# CivicLens\n\nStable content.")


def test_manifest_rejects_content_hash_mismatch(tmp_path):
    source_path = tmp_path / "docs" / "knowledge" / "field-guide.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Current content", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path,
        "docs/knowledge/field-guide.md",
        content_hash("# Older content"),
    )

    with pytest.raises(ValueError, match="Content hash mismatch"):
        load_source_manifest(tmp_path, manifest_path)


def test_manifest_rejects_paths_outside_repository(tmp_path):
    manifest_path = write_manifest(tmp_path, "../secret.md", content_hash("secret"))

    with pytest.raises(ValueError, match="must stay inside the repository"):
        load_source_manifest(tmp_path, manifest_path)


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
