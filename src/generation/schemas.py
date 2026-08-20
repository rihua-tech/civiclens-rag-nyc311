"""Application-owned schemas for grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


NO_ANSWER = "I do not have enough source context to answer that."


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class EvidenceItem:
    """Allow-listed retrieved evidence passed to an answer provider."""

    chunk_id: str
    chunk_text: str
    source_name: str
    source_path: str
    document_id: str = ""
    source_type: str = ""
    source_category: str = ""
    section_title: str = ""
    heading_path: tuple[str, ...] = ()
    retrieval_score: float = 0.0

    @classmethod
    def from_chunk(cls, chunk: dict[str, Any]) -> "EvidenceItem | None":
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        chunk_text = str(chunk.get("chunk_text", "")).strip()
        if not chunk_id or not chunk_text:
            return None

        score = 0.0
        for field in (
            "reranker_score",
            "fusion_score",
            "semantic_score",
            "lexical_score",
            "similarity_score",
        ):
            if chunk.get(field) is not None:
                score = float(chunk[field])
                break

        raw_heading_path = chunk.get("heading_path") or ()
        if isinstance(raw_heading_path, str):
            raw_heading_path = (raw_heading_path,)
        return cls(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_name=str(chunk.get("source_name", "")),
            source_path=str(chunk.get("source_path", "")),
            document_id=str(chunk.get("document_id", "")),
            source_type=str(chunk.get("source_type", "")),
            source_category=str(chunk.get("source_category", "")),
            section_title=str(chunk.get("section_title", "")),
            heading_path=tuple(str(item) for item in raw_heading_path),
            retrieval_score=score,
        )

    def provider_payload(self) -> dict[str, Any]:
        """Return only evidence and provenance needed by a remote provider."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "source_category": self.source_category,
            "section_title": self.section_title,
            "heading_path": list(self.heading_path),
        }


@dataclass(frozen=True)
class ProviderResult:
    """Provider-neutral answer text, stable citation IDs, and disposition."""

    answer: str
    citation_ids: tuple[str, ...]
    status: AnswerStatus
