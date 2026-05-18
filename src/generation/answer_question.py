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
ANSWER_GENERIC_TERMS = EMBEDDING_STOPWORDS | {
    "311",
    "common",
    "context",
    "data",
    "document",
    "documents",
    "does",
    "hand",
    "issues",
    "local",
    "mean",
    "means",
    "nyc",
    "off",
    "project",
    "source",
    "sources",
    "use",
    "used",
}


def question_terms(question: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(question.lower())
        if token not in EMBEDDING_STOPWORDS
    }


def answer_key_terms(question: str) -> set[str]:
    terms = question_terms(question)
    key_terms = terms - ANSWER_GENERIC_TERMS
    return key_terms or terms


def clean_candidate_text(text: str) -> str:
    cleaned_text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", " ", text)
    cleaned_text = cleaned_text.replace("```text", " ").replace("```", " ")
    cleaned_text = re.sub(r"#{1,6}\s*", " ", cleaned_text)
    cleaned_text = re.sub(r"`([^`]+)`", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip(" -|")
    return cleaned_text


def split_sentences(text: str) -> list[str]:
    prepared_text = " ".join(text.split())
    if not prepared_text:
        return []

    prepared_text = prepared_text.replace("```text", ". ").replace("```", ". ")
    prepared_text = re.sub(r"#{1,6}\s+", ". ", prepared_text)
    prepared_text = re.sub(r"\s+\|\s+", ". ", prepared_text)
    prepared_text = re.sub(r"\s+-\s+", ". ", prepared_text)
    prepared_text = re.sub(r"\s+\d+\.\s+", ". ", prepared_text)

    return [
        clean_candidate_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", prepared_text)
        if clean_candidate_text(sentence)
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


def select_answer_sentences(question: str, retrieved_chunks: list[dict], limit: int = 3) -> list[tuple[str, int]]:
    terms = answer_key_terms(question)
    scored_sentences: list[tuple[int, float, int, str]] = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        for sentence in split_sentences(chunk["chunk_text"]):
            if is_question_like(sentence):
                continue
            sentence_terms = set(TOKEN_PATTERN.findall(sentence.lower()))
            overlap = len(terms & sentence_terms) if terms else 0
            if overlap == 0:
                continue
            if len(sentence.split()) < 4:
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

    cited_sentences = [f"{sentence} [{source_number}]" for sentence, source_number in selected_sentences]
    return {
        "answer": " ".join(cited_sentences),
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
