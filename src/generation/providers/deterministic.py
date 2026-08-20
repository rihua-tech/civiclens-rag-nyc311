"""Deterministic context-only answer provider retained for local and CI use."""

from __future__ import annotations

import re
from typing import Sequence

from src.embeddings.embed_chunks import EMBEDDING_STOPWORDS, TOKEN_PATTERN
from src.generation.schemas import AnswerStatus, EvidenceItem, NO_ANSWER, ProviderResult


CODE_FENCE_PATTERN = re.compile(r"```[A-Za-z0-9_-]*|```")
MARKDOWN_HEADING_PATTERN = re.compile(r"(?:^|\s)#{1,6}\s+")
ARROW_SEPARATOR_PATTERN = re.compile(r"\s*(?:->|\u2192|\u2193)\s*")
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
LOW_VALUE_HEADINGS = {
    "architecture",
    "civiclens rag hybrid rag architecture",
    "design principle",
    "retrieval scope",
    "answer requirements",
    "local embedding storage flow",
    "local retrieval and cited answer flow",
    "local streamlit hybrid flow",
}
SECTION_PREFIXES = (
    "Design Principle ",
    "Retrieval Scope ",
    "Answer Requirements ",
    "Local Embedding Storage Flow ",
    "Local Retrieval and Cited Answer Flow ",
    "Local Streamlit Hybrid Flow ",
)
ARCHITECTURE_STEPS = (
    ("ingestion pipeline", "ingestion"),
    ("text cleaning + chunking", "text cleaning and chunking"),
    ("metadata tagging", "metadata tagging"),
    ("embedding generation", "embedding generation"),
    ("postgresql + pgvector", "PostgreSQL/pgvector storage"),
    ("retriever", "retrieval"),
    ("llm answer generator", "answer generation"),
    ("cited answer ui", "a cited answer UI"),
)
ANSWER_STOPWORDS = EMBEDDING_STOPWORDS | {
    "define",
    "definition",
    "does",
    "mean",
    "means",
}


def question_terms(question: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(question.lower())
        if token not in ANSWER_STOPWORDS
    }


def normalize_heading_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def normalize_markdown_for_answer(text: str) -> str:
    normalized_text = CODE_FENCE_PATTERN.sub(". ", text)
    normalized_text = MARKDOWN_HEADING_PATTERN.sub(". ", normalized_text)
    normalized_text = ARROW_SEPARATOR_PATTERN.sub(". ", normalized_text)
    return " ".join(normalized_text.split())


def strip_section_prefix(text: str) -> str:
    for prefix in SECTION_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text


def clean_answer_candidate(text: str) -> str:
    cleaned_text = LIST_MARKER_PATTERN.sub("", text.strip())
    cleaned_text = cleaned_text.replace("`", "")
    cleaned_text = cleaned_text.replace("|", " ")
    cleaned_text = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = strip_section_prefix(cleaned_text)
    return cleaned_text.strip(" -")


def ensure_sentence_ending(text: str) -> str:
    cleaned_text = text.strip()
    if not cleaned_text or cleaned_text[-1] in ".!?":
        return cleaned_text
    return f"{cleaned_text}."


def split_markdown_units(text: str) -> list[str]:
    compact_text = normalize_markdown_for_answer(text)
    if not compact_text:
        return []
    return [
        cleaned
        for unit in re.split(r"(?<=[.!?])\s+", compact_text)
        if (cleaned := clean_answer_candidate(unit))
    ]


def is_low_value_answer_candidate(sentence: str) -> bool:
    normalized_sentence = normalize_heading_text(sentence)
    if normalized_sentence in LOW_VALUE_HEADINGS:
        return True
    has_sentence_signal = re.search(
        r"\b(is|are|uses|used|should|must|include|includes|contain|contains|stored|remain|moves|runs|processes)\b",
        sentence.lower(),
    )
    return len(sentence.split()) < 6 and not has_sentence_signal


def split_sentences(text: str) -> list[str]:
    return [
        ensure_sentence_ending(sentence)
        for sentence in split_markdown_units(text)
        if not is_low_value_answer_candidate(sentence)
    ]


def is_question_like(sentence: str) -> bool:
    stripped_sentence = sentence.strip()
    return "?" in stripped_sentence or stripped_sentence.startswith("|")


def format_series(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def clean_source_part(source_part: str) -> str:
    return source_part.replace("README / Runbooks", "README/runbooks").strip(" .")


def architecture_summary_sentences(
    question: str,
    evidence: Sequence[EvidenceItem],
) -> list[tuple[str, int]]:
    terms = question_terms(question)
    if not {"architecture", "lakehouse"} & terms:
        return []

    for source_number, item in enumerate(evidence, start=1):
        units = split_markdown_units(item.chunk_text)
        normalized_units = [(unit, unit.lower()) for unit in units]
        source_unit = next(
            (
                unit
                for unit, lower_unit in normalized_units
                if "nyc 311 documentation" in lower_unit
                and "nyc 311 data dictionary" in lower_unit
            ),
            "",
        )
        pipeline_steps = [
            readable_step
            for step_key, readable_step in ARCHITECTURE_STEPS
            if any(step_key in lower_unit for _, lower_unit in normalized_units)
        ]
        design_sentence = next(
            (
                unit
                for unit, lower_unit in normalized_units
                if "structured metrics" in lower_unit
                or "documents and metadata" in lower_unit
            ),
            "",
        )
        if not source_unit or len(pipeline_steps) < 3:
            continue

        source_parts = [
            clean_source_part(part)
            for part in source_unit.split(" + ")
            if part.strip()
        ]
        selected = [
            (
                f"The architecture starts with {format_series(source_parts)}.",
                source_number,
            ),
            (
                f"It then moves through {format_series(pipeline_steps)}.",
                source_number,
            ),
        ]
        if design_sentence:
            selected.append((ensure_sentence_ending(design_sentence), source_number))
        return selected
    return []


def select_answer_sentences(
    question: str,
    evidence: Sequence[EvidenceItem],
    limit: int = 3,
) -> list[tuple[str, int]]:
    architecture_sentences = architecture_summary_sentences(question, evidence)
    if architecture_sentences:
        return architecture_sentences[:limit]

    terms = question_terms(question)
    scored_sentences: list[tuple[int, float, int, str]] = []
    for source_number, item in enumerate(evidence, start=1):
        for sentence in split_sentences(item.chunk_text):
            if is_question_like(sentence):
                continue
            sentence_terms = set(TOKEN_PATTERN.findall(sentence.lower()))
            overlap = len(terms & sentence_terms) if terms else 0
            if overlap:
                scored_sentences.append(
                    (overlap, item.retrieval_score, source_number, sentence)
                )

    scored_sentences.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[tuple[str, int]] = []
    seen_sentences: set[str] = set()
    for _, _, source_number, sentence in scored_sentences:
        normalized_sentence = sentence.lower()
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        selected.append((sentence, source_number))
        if len(selected) == limit:
            break
    return selected


def format_answer_text(selected_sentences: list[tuple[str, int]]) -> str:
    sentences = [ensure_sentence_ending(sentence) for sentence, _ in selected_sentences]
    if len(sentences) >= 3:
        return "\n".join(f"- {sentence}" for sentence in sentences)
    return " ".join(sentences)


class DeterministicAnswerProvider:
    provider_name = "local"
    model_name = "deterministic-context-extractor-v1"

    def generate(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> ProviderResult:
        if not evidence:
            return ProviderResult(NO_ANSWER, (), AnswerStatus.ABSTAINED)

        selected = select_answer_sentences(question, evidence)
        if not selected:
            return ProviderResult(
                NO_ANSWER,
                tuple(item.chunk_id for item in evidence),
                AnswerStatus.ABSTAINED,
            )

        citation_ids = tuple(
            dict.fromkeys(evidence[source_number - 1].chunk_id for _, source_number in selected)
        )
        return ProviderResult(
            format_answer_text(selected),
            citation_ids,
            AnswerStatus.ANSWERED,
        )

