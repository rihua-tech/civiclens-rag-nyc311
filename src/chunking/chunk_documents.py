"""Split ingested documents into stable, section-aware local JSONL chunks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from src.ingestion.load_documents import content_hash, normalize_text


DEFAULT_INPUT_PATH = Path("data/processed/documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
CHUNKING_ALGORITHM_VERSION = "section-aware-heading-context-word-windows-v1"
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MARKDOWN_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input documents file not found: {input_path}")

    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_words(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    validate_chunk_settings(chunk_size, chunk_overlap)

    words = clean_whitespace(text).split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def split_markdown_sections(text: str) -> list[dict[str, object]]:
    """Split Markdown on ATX headings while ignoring heading-like text in code fences."""
    sections: list[dict[str, object]] = []
    heading_stack: list[str] = []
    current_title: str | None = None
    current_lines: list[str] = []
    active_fence: str | None = None

    def flush_section() -> None:
        section_text = normalize_text("\n".join(current_lines))
        if section_text:
            sections.append(
                {
                    "section_title": current_title,
                    "heading_path": list(heading_stack),
                    "text": section_text,
                }
            )

    for line in normalize_text(text).split("\n"):
        fence_match = MARKDOWN_FENCE_PATTERN.match(line)
        if fence_match:
            fence_marker = fence_match.group(1)
            if active_fence is None:
                active_fence = fence_marker
            elif fence_marker == active_fence:
                active_fence = None
            current_lines.append(line)
            continue

        heading_match = None if active_fence else MARKDOWN_HEADING_PATTERN.match(line)
        if not heading_match:
            current_lines.append(line)
            continue

        flush_section()
        current_lines = []
        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        current_title = title

    flush_section()
    return sections


def document_sections(document: dict) -> list[dict[str, object]]:
    text = str(document.get("text", ""))
    if document.get("source_type") == "markdown":
        return split_markdown_sections(text)

    normalized = normalize_text(text)
    if not normalized:
        return []
    return [{"section_title": None, "heading_path": [], "text": normalized}]


def stable_chunk_id(
    document_id: str,
    heading_path: list[str],
    section_index: int,
    section_chunk_index: int,
    chunk_text: str,
    chunking_config_hash: str,
) -> str:
    """Build a deterministic chunk ID independent of ingestion timestamps."""
    identity = json.dumps(
        {
            "document_id": document_id,
            "heading_path": heading_path,
            "section_index": section_index,
            "section_chunk_index": section_chunk_index,
            "content_hash": content_hash(chunk_text),
            "chunking_config_hash": chunking_config_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}_chunk_{digest}"


def build_chunking_config_hash(chunk_size: int, chunk_overlap: int) -> str:
    configuration = json.dumps(
        {
            "algorithm": CHUNKING_ALGORITHM_VERSION,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": chunk_overlap,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(configuration)


def build_chunk_record(
    document: dict,
    chunk_text: str,
    chunk_index: int,
    section_index: int,
    section_chunk_index: int,
    section_title: str | None,
    heading_path: list[str],
    chunking_config_hash: str,
) -> dict:
    document_id = document["document_id"]
    normalized_body = clean_whitespace(chunk_text)
    heading_context = " > ".join(heading_path)
    normalized_chunk_text = clean_whitespace(
        f"{heading_context}\n\n{normalized_body}" if heading_context else normalized_body
    )
    document_hash = document.get("content_hash") or content_hash(str(document.get("text", "")))

    return {
        "chunk_id": stable_chunk_id(
            document_id,
            heading_path,
            section_index,
            section_chunk_index,
            normalized_chunk_text,
            chunking_config_hash,
        ),
        "document_id": document_id,
        "chunk_text": normalized_chunk_text,
        "chunk_index": chunk_index,
        "section_title": section_title,
        "heading_path": heading_path,
        "source_name": document.get("source_name"),
        "source_type": document.get("source_type"),
        "source_category": document.get("source_category") or "local_override",
        "source_path": document.get("source_path"),
        "source_url": document.get("source_url"),
        "source_version": document.get("source_version"),
        "source_retrieved_at": document.get("source_retrieved_at"),
        "ingested_at": document.get("ingested_at"),
        "document_content_hash": document_hash,
        "chunking_config_hash": chunking_config_hash,
        "content_hash": content_hash(normalized_chunk_text),
        "word_count": len(normalized_chunk_text.split()),
    }


def chunk_documents(
    documents: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    validate_chunk_settings(chunk_size, chunk_overlap)
    chunks: list[dict] = []
    config_hash = build_chunking_config_hash(chunk_size, chunk_overlap)

    for document in documents:
        chunk_index = 0
        for section_index, section in enumerate(document_sections(document)):
            section_chunks = split_words(str(section["text"]), chunk_size, chunk_overlap)
            for section_chunk_index, chunk_text in enumerate(section_chunks):
                chunks.append(
                    build_chunk_record(
                        document=document,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        section_index=section_index,
                        section_chunk_index=section_chunk_index,
                        section_title=section["section_title"],
                        heading_path=list(section["heading_path"]),
                        chunking_config_hash=config_hash,
                    )
                )
                chunk_index += 1

    return chunks


def write_chunks(chunks: Iterable[dict], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as jsonl_file:
        for chunk in chunks:
            jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return output


def create_chunks(
    repo_root: str | Path | None = None,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[dict], Path]:
    root = Path(repo_root) if repo_root is not None else project_root()
    input_file = resolve_path(input_path, root)
    output_file = resolve_path(output_path, root)
    documents = load_jsonl(input_file)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    write_chunks(chunks, output_file)
    return chunks, output_file


def main() -> None:
    chunks, output_path = create_chunks()
    print(f"Chunks created: {len(chunks)}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
