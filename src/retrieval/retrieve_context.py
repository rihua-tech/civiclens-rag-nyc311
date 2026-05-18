"""Retrieve relevant chunks from PostgreSQL/pgvector for a user question."""

from __future__ import annotations

import argparse
from typing import Iterable

from src.common.config import Settings
from src.embeddings.embed_chunks import generate_embedding, vector_literal


DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.05


def validate_top_k(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    return top_k


def format_retrieval_rows(rows: Iterable[tuple]) -> list[dict]:
    results: list[dict] = []

    for rank, row in enumerate(rows, start=1):
        (
            chunk_id,
            document_id,
            chunk_text,
            source_name,
            source_path,
            similarity_score,
        ) = row
        results.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_text": chunk_text,
                "source_name": source_name,
                "source_path": source_path,
                "similarity_score": float(similarity_score),
                "rank": rank,
            }
        )

    return results


def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    settings: Settings | None = None,
) -> list[dict]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return []

    limit = validate_top_k(top_k)
    active_settings = settings or Settings.from_env()
    question_embedding = generate_embedding(cleaned_question, active_settings)
    question_vector = vector_literal(question_embedding)

    import psycopg

    with psycopg.connect(active_settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    chunk_text,
                    source_name,
                    source_path,
                    similarity_score
                FROM (
                    SELECT
                        chunk_id,
                        document_id,
                        chunk_text,
                        source_name,
                        source_path,
                        1 - (embedding <=> %s::vector) AS similarity_score
                    FROM chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                ) AS scored_chunks
                WHERE similarity_score >= %s
                ORDER BY similarity_score DESC
                """,
                (question_vector, question_vector, limit, min_similarity),
            )
            rows = cursor.fetchall()

    return format_retrieval_rows(rows)


def snippet(text: str, max_length: int = 240) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_length:
        return compact_text
    return compact_text[: max_length - 3].rstrip() + "..."


def safe_console_text(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def format_cli_results(question: str, results: list[dict]) -> str:
    lines = [f"Question: {question}"]
    if not results:
        lines.append("No relevant chunks found.")
        return "\n".join(lines)

    lines.append(f"Retrieved chunks: {len(results)}")
    for result in results:
        lines.append(
            (
                f"{result['rank']}. {result['source_name']} "
                f"({result['source_path']}) "
                f"score={result['similarity_score']:.4f} "
                f"chunk={result['chunk_id']}"
            )
        )
        lines.append(f"   {snippet(result['chunk_text'])}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant local RAG context.")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to return")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Minimum similarity score for returned chunks",
    )
    args = parser.parse_args()

    results = retrieve_context(args.question, top_k=args.top_k, min_similarity=args.min_similarity)
    print(safe_console_text(format_cli_results(args.question, results)))


if __name__ == "__main__":
    main()
