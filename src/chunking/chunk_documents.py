"""Split ingested documents into local JSONL chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_PATH = Path("data/processed/documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80


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


def split_words(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

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


def build_chunk_record(document: dict, chunk_text: str, chunk_index: int) -> dict:
    token_count = len(chunk_text.split())
    document_id = document["document_id"]

    return {
        "chunk_id": f"{document_id}_chunk_{chunk_index:04d}",
        "document_id": document_id,
        "chunk_text": chunk_text,
        "chunk_index": chunk_index,
        "source_name": document["source_name"],
        "source_path": document["source_path"],
        "token_count": token_count,
    }


def chunk_documents(
    documents: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        document_chunks = split_words(document.get("text", ""), chunk_size, chunk_overlap)
        for chunk_index, chunk_text in enumerate(document_chunks):
            chunks.append(build_chunk_record(document, chunk_text, chunk_index))

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
