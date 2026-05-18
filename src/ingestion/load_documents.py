"""Load local source documents into a JSONL document store."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_PATHS = (
    "README.md",
    "docs/data-sources.md",
    "docs/architecture.md",
    "docs/rag-design.md",
    "docs/evaluation-notes.md",
)
DEFAULT_OUTPUT_PATH = Path("data/processed/documents.jsonl")
SUPPORTED_SOURCE_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_document_id(source_path: str) -> str:
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def relative_source_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_document_record(path: Path, repo_root: Path, ingested_at: str) -> dict[str, str]:
    source_path = relative_source_path(path, repo_root)
    source_type = SUPPORTED_SOURCE_TYPES[path.suffix.lower()]
    text = normalize_text(path.read_text(encoding="utf-8"))

    return {
        "document_id": stable_document_id(source_path),
        "source_name": path.name,
        "source_type": source_type,
        "source_path": source_path,
        "text": text,
        "ingested_at": ingested_at,
    }


def load_documents(
    repo_root: str | Path | None = None,
    source_paths: Iterable[str | Path] = DEFAULT_SOURCE_PATHS,
    ingested_at: str | None = None,
) -> list[dict[str, str]]:
    root = Path(repo_root) if repo_root is not None else project_root()
    timestamp = ingested_at or utc_timestamp()
    documents: list[dict[str, str]] = []

    for source_path in source_paths:
        path = resolve_path(source_path, root)
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SOURCE_TYPES:
            continue
        documents.append(build_document_record(path, root, timestamp))

    return documents


def write_documents(documents: Iterable[dict[str, str]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as jsonl_file:
        for document in documents:
            jsonl_file.write(json.dumps(document, ensure_ascii=False) + "\n")

    return output


def ingest_documents(
    repo_root: str | Path | None = None,
    source_paths: Iterable[str | Path] = DEFAULT_SOURCE_PATHS,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ingested_at: str | None = None,
) -> tuple[list[dict[str, str]], Path]:
    root = Path(repo_root) if repo_root is not None else project_root()
    output = resolve_path(output_path, root)
    documents = load_documents(root, source_paths, ingested_at)
    write_documents(documents, output)
    return documents, output


def main() -> None:
    documents, output_path = ingest_documents()
    print(f"Documents loaded: {len(documents)}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
