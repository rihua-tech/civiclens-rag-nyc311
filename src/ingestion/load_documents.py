"""Load manifest-authorized local source documents into a JSONL document store."""

from __future__ import annotations

import hashlib
import json
import posixpath
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_MANIFEST_PATH = Path("docs/knowledge/source-manifest.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/documents.jsonl")
SUPPORTED_SOURCE_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
CONTENT_HASH_PREFIX = "sha256:"
MANIFEST_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_source_path(source_path: str | Path) -> str:
    """Return the documented, platform-independent path used for stable IDs."""
    normalized = str(source_path).replace("\\", "/")
    canonical = posixpath.normpath(normalized)
    while canonical.startswith("./"):
        canonical = canonical[2:]
    return canonical


def stable_document_id(source_path: str | Path) -> str:
    canonical_path = canonical_source_path(source_path)
    digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def normalize_text(text: str) -> str:
    """Normalize line endings and trailing whitespace before hashing or storage."""
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in normalized_lines).strip()


def content_hash(text: str) -> str:
    normalized = normalize_text(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{CONTENT_HASH_PREFIX}{digest}"


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def relative_source_path(path: Path, repo_root: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(repo_root.resolve())
        return canonical_source_path(relative_path)
    except ValueError:
        return canonical_source_path(path.resolve())


def load_source_manifest(
    repo_root: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST_PATH,
) -> list[dict[str, object]]:
    """Load and validate the authoritative default source inventory."""
    root = Path(repo_root) if repo_root is not None else project_root()
    path = resolve_path(manifest_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported source manifest version: {manifest.get('manifest_version')!r}")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source manifest must contain a non-empty 'sources' list")

    required_fields = {
        "source_name",
        "source_type",
        "source_category",
        "path",
        "source_url",
        "source_version",
        "retrieved_at",
        "content_hash",
    }
    validated_sources: list[dict[str, object]] = []
    seen_paths: set[str] = set()

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Source manifest entry {index} must be an object")

        missing_fields = sorted(required_fields.difference(source))
        if missing_fields:
            raise ValueError(
                f"Source manifest entry {index} is missing fields: {', '.join(missing_fields)}"
            )

        source_path = canonical_source_path(str(source["path"]))
        if Path(source_path).is_absolute() or source_path == ".." or source_path.startswith("../"):
            raise ValueError(f"Manifest source path must stay inside the repository: {source_path}")
        if source_path in seen_paths:
            raise ValueError(f"Duplicate source manifest path: {source_path}")
        seen_paths.add(source_path)

        local_path = resolve_path(source_path, root)
        if not local_path.is_file():
            raise FileNotFoundError(f"Manifest source not found: {local_path}")

        expected_type = SUPPORTED_SOURCE_TYPES.get(local_path.suffix.lower())
        if expected_type is None:
            raise ValueError(f"Unsupported manifest source type for {source_path}")
        if source["source_type"] != expected_type:
            raise ValueError(
                f"Manifest source type mismatch for {source_path}: "
                f"expected {expected_type!r}, found {source['source_type']!r}"
            )

        normalized_text = normalize_text(local_path.read_text(encoding="utf-8"))
        actual_hash = content_hash(normalized_text)
        if source["content_hash"] != actual_hash:
            raise ValueError(
                f"Content hash mismatch for {source_path}: "
                f"expected {source['content_hash']!r}, calculated {actual_hash!r}"
            )

        validated_source = dict(source)
        validated_source["path"] = source_path
        validated_sources.append(validated_source)

    return validated_sources


def build_document_record(
    path: Path,
    repo_root: Path,
    ingested_at: str,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    source_path = relative_source_path(path, repo_root)
    source_type = SUPPORTED_SOURCE_TYPES[path.suffix.lower()]
    text = normalize_text(path.read_text(encoding="utf-8"))
    metadata = source_metadata or {}

    return {
        "document_id": stable_document_id(source_path),
        "source_name": metadata.get("source_name") or path.name,
        "source_type": source_type,
        "source_category": metadata.get("source_category") or "local_override",
        "source_path": source_path,
        "source_url": metadata.get("source_url"),
        "source_version": metadata.get("source_version"),
        "source_retrieved_at": metadata.get("retrieved_at"),
        "content_hash": content_hash(text),
        "text": text,
        "ingested_at": ingested_at,
    }


def load_documents(
    repo_root: str | Path | None = None,
    source_paths: Iterable[str | Path] | None = None,
    ingested_at: str | None = None,
    manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST_PATH,
) -> list[dict[str, object]]:
    """Load manifest sources by default, or explicit paths for compatible local use."""
    root = Path(repo_root) if repo_root is not None else project_root()
    timestamp = ingested_at or utc_timestamp()
    documents: list[dict[str, object]] = []

    if source_paths is None:
        source_entries = load_source_manifest(root, manifest_path)
        for source in source_entries:
            path = resolve_path(str(source["path"]), root)
            documents.append(build_document_record(path, root, timestamp, source))
        return documents

    for source_path in source_paths:
        path = resolve_path(source_path, root)
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SOURCE_TYPES:
            continue
        documents.append(build_document_record(path, root, timestamp))

    return documents


def write_documents(documents: Iterable[dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as jsonl_file:
        for document in documents:
            jsonl_file.write(json.dumps(document, ensure_ascii=False) + "\n")

    return output


def ingest_documents(
    repo_root: str | Path | None = None,
    source_paths: Iterable[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ingested_at: str | None = None,
    manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST_PATH,
) -> tuple[list[dict[str, object]], Path]:
    root = Path(repo_root) if repo_root is not None else project_root()
    output = resolve_path(output_path, root)
    documents = load_documents(root, source_paths, ingested_at, manifest_path)
    write_documents(documents, output)
    return documents, output


def main() -> None:
    documents, output_path = ingest_documents()
    print(f"Documents loaded: {len(documents)}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
