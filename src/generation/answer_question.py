"""Generate a local context-only answer with source citations."""

from __future__ import annotations

import argparse
import re

from src.common.config import Settings
from src.embeddings.embed_chunks import EMBEDDING_STOPWORDS, TOKEN_PATTERN
from src.retrieval.retrieve_context import retrieve_context


NO_ANSWER = "I do not have enough source context to answer that."
DEFAULT_CONFIDENCE_NOTE = (
    "This answer is generated from retrieved local context only; verify source documents before making operational decisions."
)
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
    if not cleaned_text:
        return cleaned_text
    if cleaned_text[-1] in ".!?":
        return cleaned_text
    return f"{cleaned_text}."


def split_markdown_units(text: str) -> list[str]:
    compact_text = normalize_markdown_for_answer(text)
    if not compact_text:
        return []

    units: list[str] = []
    for unit in re.split(r"(?<=[.!?])\s+", compact_text):
        cleaned_unit = clean_answer_candidate(unit)
        if cleaned_unit:
            units.append(cleaned_unit)
    return units


def is_low_value_answer_candidate(sentence: str) -> bool:
    normalized_sentence = normalize_heading_text(sentence)
    if normalized_sentence in LOW_VALUE_HEADINGS:
        return True

    lower_sentence = sentence.lower()
    has_sentence_signal = re.search(
        r"\b(is|are|uses|used|should|must|include|includes|contain|contains|stored|remain|moves|runs|processes)\b",
        lower_sentence,
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


def unique_sources(retrieved_chunks: list[dict]) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    for chunk in retrieved_chunks:
        source_key = chunk["chunk_id"]
        if source_key in seen:
            continue
        seen.add(source_key)
        sources.append(
            {
                "source_name": chunk["source_name"],
                "source_path": chunk["source_path"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    return sources


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


def architecture_summary_sentences(question: str, retrieved_chunks: list[dict]) -> list[tuple[str, int]]:
    terms = question_terms(question)
    if not {"architecture", "lakehouse"} & terms:
        return []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        units = split_markdown_units(chunk["chunk_text"])
        normalized_units = [(unit, unit.lower()) for unit in units]

        source_unit = next(
            (
                unit
                for unit, lower_unit in normalized_units
                if "nyc 311 documentation" in lower_unit and "nyc 311 data dictionary" in lower_unit
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
                if "structured metrics" in lower_unit or "documents and metadata" in lower_unit
            ),
            "",
        )

        if not source_unit or len(pipeline_steps) < 3:
            continue

        source_parts = [clean_source_part(part) for part in source_unit.split(" + ") if part.strip()]
        selected_sentences = [
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
            selected_sentences.append((ensure_sentence_ending(design_sentence), source_number))
        return selected_sentences

    return []


def select_answer_sentences(question: str, retrieved_chunks: list[dict], limit: int = 3) -> list[tuple[str, int]]:
    architecture_sentences = architecture_summary_sentences(question, retrieved_chunks)
    if architecture_sentences:
        return architecture_sentences[:limit]

    terms = question_terms(question)
    scored_sentences: list[tuple[int, float, int, str]] = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        for sentence in split_sentences(chunk["chunk_text"]):
            if is_question_like(sentence):
                continue
            sentence_terms = set(TOKEN_PATTERN.findall(sentence.lower()))
            overlap = len(terms & sentence_terms) if terms else 0
            if overlap == 0:
                continue
            scored_sentences.append(
                (
                    overlap,
                    float(chunk.get("similarity_score", 0.0)),
                    source_number,
                    sentence,
                )
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


def format_cited_answer(selected_sentences: list[tuple[str, int]]) -> str:
    cited_sentences = [
        f"{ensure_sentence_ending(sentence)} [{source_number}]"
        for sentence, source_number in selected_sentences
    ]
    if len(cited_sentences) >= 3:
        return "\n".join(f"- {sentence}" for sentence in cited_sentences)
    return " ".join(cited_sentences)


def local_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "confidence_note": "No relevant source chunks were retrieved.",
            "retrieved_chunks": [],
        }

    selected_sentences = select_answer_sentences(question, retrieved_chunks)
    if not selected_sentences:
        return {
            "answer": NO_ANSWER,
            "sources": unique_sources(retrieved_chunks),
            "confidence_note": "Retrieved chunks did not contain enough direct overlap with the question.",
            "retrieved_chunks": retrieved_chunks,
        }

    return {
        "answer": format_cited_answer(selected_sentences),
        "sources": unique_sources(retrieved_chunks),
        "confidence_note": DEFAULT_CONFIDENCE_NOTE,
        "retrieved_chunks": retrieved_chunks,
    }


def openai_answer(question: str, retrieved_chunks: list[dict], settings: Settings) -> dict:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when USE_OPENAI_ANSWERS=true")

    from openai import OpenAI

    context = "\n\n".join(
        (
            f"Source {index}: {chunk['source_name']} | {chunk['source_path']} | {chunk['chunk_id']}\n"
            f"{chunk['chunk_text']}"
        )
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the provided context. Include bracketed source numbers. "
                    "If the context is insufficient, say so."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )

    return {
        "answer": response.choices[0].message.content or NO_ANSWER,
        "sources": unique_sources(retrieved_chunks),
        "confidence_note": DEFAULT_CONFIDENCE_NOTE,
        "retrieved_chunks": retrieved_chunks,
    }


def answer_question(
    question: str,
    top_k: int = 5,
    min_similarity: float = 0.05,
    settings: Settings | None = None,
) -> dict:
    active_settings = settings or Settings.from_env()
    retrieved_chunks = retrieve_context(
        question,
        top_k=top_k,
        min_similarity=min_similarity,
        settings=active_settings,
    )

    if active_settings.use_openai_answers and active_settings.openai_api_key:
        return openai_answer(question, retrieved_chunks, active_settings)
    return local_answer(question, retrieved_chunks)


def format_answer_response(response: dict) -> str:
    lines = ["Answer:", response["answer"], "", "Confidence:", response["confidence_note"], ""]
    lines.append("Sources:")
    if response["sources"]:
        for index, source in enumerate(response["sources"], start=1):
            lines.append(
                (
                    f"{index}. {source['source_name']} - "
                    f"{source['source_path']} - chunk {source['chunk_id']}"
                )
            )
    else:
        lines.append("None")

    lines.append("")
    lines.append(f"Retrieved chunks: {len(response['retrieved_chunks'])}")
    return "\n".join(lines)


def safe_console_text(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question with retrieved local context.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.05,
        help="Minimum similarity score for returned chunks",
    )
    args = parser.parse_args()

    response = answer_question(args.question, top_k=args.top_k, min_similarity=args.min_similarity)
    print(safe_console_text(format_answer_response(response)))


if __name__ == "__main__":
    main()
