"""Application-controlled validation of provider citation IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.observability.latency import measure_latency


MODEL_DISPLAY_CITATION_PATTERN = re.compile(r"\s*\[\d+\]")
SOURCE_METADATA_KEYS = (
    "document_id",
    "source_type",
    "source_category",
    "source_url",
    "source_version",
    "source_retrieved_at",
    "section_title",
    "heading_path",
    "content_hash",
    "document_content_hash",
    "chunking_config_hash",
)


@dataclass(frozen=True)
class CitationValidationResult:
    valid_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    sources: tuple[dict[str, Any], ...]


def _unique_ids(citation_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            normalized
            for citation_id in citation_ids
            if (normalized := str(citation_id).strip())
        )
    )


def source_from_retrieved_chunk(
    chunk: dict[str, Any],
    citation_number: int,
) -> dict[str, Any]:
    """Rebuild authoritative provenance from application-owned retrieval data."""
    source: dict[str, Any] = {
        "source_name": str(chunk.get("source_name", "")),
        "source_path": str(chunk.get("source_path", "")),
        "chunk_id": str(chunk.get("chunk_id", "")),
        "citation_number": citation_number,
    }
    for key in SOURCE_METADATA_KEYS:
        value = chunk.get(key)
        if value not in (None, "", []):
            source[key] = value
    return source


def validate_citation_ids(
    citation_ids: Iterable[str],
    retrieved_chunks: Sequence[dict[str, Any]],
) -> CitationValidationResult:
    """Accept only retrieved chunk IDs and preserve provider ID order."""
    with measure_latency("citation_validation_ms"):
        retrieved_by_id = {
            str(chunk.get("chunk_id", "")): (position, chunk)
            for position, chunk in enumerate(retrieved_chunks, start=1)
            if str(chunk.get("chunk_id", "")).strip()
        }
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        sources: list[dict[str, Any]] = []
        for citation_id in _unique_ids(citation_ids):
            retrieved = retrieved_by_id.get(citation_id)
            if retrieved is None:
                invalid_ids.append(citation_id)
                continue
            position, chunk = retrieved
            valid_ids.append(citation_id)
            sources.append(source_from_retrieved_chunk(chunk, position))

        return CitationValidationResult(
            valid_ids=tuple(valid_ids),
            invalid_ids=tuple(invalid_ids),
            sources=tuple(sources),
        )


def add_validated_display_citations(
    answer: str,
    sources: Sequence[dict[str, Any]],
) -> str:
    """Discard model display numbers and rebuild them from validated IDs."""
    with measure_latency("citation_validation_ms"):
        cleaned_answer = MODEL_DISPLAY_CITATION_PATTERN.sub("", answer).strip()
        display_numbers = [
            f"[{int(source['citation_number'])}]"
            for source in sources
            if source.get("citation_number") is not None
        ]
        return " ".join((cleaned_answer, *display_numbers)).strip()

