"""Semantic and PostgreSQL lexical retrieval with a stable result contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from src.common.config import Settings
from src.embeddings.embed_chunks import (
    fetch_embedding_profiles,
    validate_stored_embedding_profiles,
    vector_column_for_spec,
    vector_literal,
)
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider
from src.embeddings.providers.deterministic import EMBEDDING_STOPWORDS, TOKEN_PATTERN


DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.25
MAX_CANDIDATES = 100
MAX_LEXICAL_QUERY_TERMS = 12
RESULT_METADATA_FIELDS = (
    "chunk_id",
    "document_id",
    "chunk_text",
    "source_name",
    "source_type",
    "source_category",
    "source_path",
    "source_url",
    "source_version",
    "source_retrieved_at",
    "section_title",
    "heading_path",
    "word_count",
    "content_hash",
    "document_content_hash",
    "chunking_config_hash",
    "ingested_at",
)
DIAGNOSTIC_FIELDS = (
    "similarity_score",
    "semantic_score",
    "semantic_rank",
    "lexical_score",
    "lexical_rank",
    "fusion_score",
    "reranker_score",
    "pre_rerank_rank",
)
LEXICAL_STOPWORDS = EMBEDDING_STOPWORDS | {
    "about",
    "define",
    "definition",
    "does",
    "explain",
    "mean",
    "means",
    "tell",
}


def validate_top_k(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if top_k > MAX_CANDIDATES:
        raise ValueError(f"top_k must be less than or equal to {MAX_CANDIDATES}")
    return top_k


def validate_candidate_limit(candidate_limit: int, name: str) -> int:
    if candidate_limit <= 0:
        raise ValueError(f"{name} must be greater than 0")
    if candidate_limit > MAX_CANDIDATES:
        raise ValueError(f"{name} must be less than or equal to {MAX_CANDIDATES}")
    return candidate_limit


def result_from_row(
    row: tuple,
    *,
    retrieval_mode: str,
    semantic_rank: int | None = None,
    lexical_rank: int | None = None,
) -> dict:
    metadata_values = row[: len(RESULT_METADATA_FIELDS)]
    score = float(row[len(RESULT_METADATA_FIELDS)])
    result = dict(zip(RESULT_METADATA_FIELDS, metadata_values, strict=True))
    result["heading_path"] = list(result["heading_path"] or [])
    result.update({field: None for field in DIAGNOSTIC_FIELDS})
    if semantic_rank is not None:
        result["similarity_score"] = score
        result["semantic_score"] = score
        result["semantic_rank"] = semantic_rank
    if lexical_rank is not None:
        result["lexical_score"] = score
        result["lexical_rank"] = lexical_rank
    result["retrieval_mode"] = retrieval_mode
    result["rank"] = semantic_rank or lexical_rank
    return result


def format_retrieval_rows(rows: Iterable[tuple]) -> list[dict]:
    """Backward-compatible formatter for semantic pgvector rows."""
    return [
        result_from_row(row, retrieval_mode="semantic", semantic_rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]


def format_lexical_rows(rows: Iterable[tuple]) -> list[dict]:
    return [
        result_from_row(row, retrieval_mode="lexical", lexical_rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]


def lexical_query_text(question: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_PATTERN.findall(question.lower()):
        if token in LEXICAL_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) == MAX_LEXICAL_QUERY_TERMS:
            break
    return " ".join(terms) if terms else question.strip()


def retrieve_semantic_context(
    question: str,
    candidate_limit: int,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    settings: Settings | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[dict]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return []

    limit = validate_candidate_limit(candidate_limit, "semantic candidate count")
    active_settings = settings or Settings.from_env()
    active_provider = provider or create_embedding_provider(active_settings)
    active_spec = active_provider.spec
    vector_column = vector_column_for_spec(active_spec)

    import psycopg

    with psycopg.connect(active_settings.database_url) as connection:
        with connection.cursor() as cursor:
            profiles = fetch_embedding_profiles(cursor)
            validate_stored_embedding_profiles(profiles, active_spec)
            question_embedding = active_provider.embed(cleaned_question)
            question_vector = vector_literal(question_embedding)
            cursor.execute(
                f"""
                SELECT
                    chunk_id,
                    document_id,
                    chunk_text,
                    source_name,
                    source_type,
                    source_category,
                    source_path,
                    source_url,
                    source_version,
                    source_retrieved_at,
                    section_title,
                    heading_path,
                    word_count,
                    content_hash,
                    document_content_hash,
                    chunking_config_hash,
                    ingested_at,
                    semantic_score
                FROM (
                    SELECT
                        c.chunk_id,
                        c.document_id,
                        c.chunk_text,
                        c.source_name,
                        c.source_type,
                        c.source_category,
                        c.source_path,
                        c.source_url,
                        c.source_version,
                        c.source_retrieved_at,
                        c.section_title,
                        c.heading_path,
                        c.word_count,
                        c.content_hash,
                        c.document_content_hash,
                        c.chunking_config_hash,
                        c.ingested_at,
                        1 - (c.{vector_column} <=> %s::vector) AS semantic_score
                    FROM chunks AS c
                    INNER JOIN documents AS d ON d.document_id = c.document_id
                    WHERE c.{vector_column} IS NOT NULL
                      AND c.embedding_provider = %s
                      AND c.embedding_model = %s
                      AND c.embedding_dimension = %s
                      AND vector_dims(c.{vector_column}) = %s
                      AND c.content_hash IS NOT NULL
                      AND c.document_content_hash = d.content_hash
                      AND c.chunking_config_hash = d.chunking_config_hash
                    ORDER BY c.{vector_column} <=> %s::vector, c.chunk_id
                    LIMIT %s
                ) AS scored_chunks
                WHERE semantic_score >= %s
                ORDER BY semantic_score DESC, chunk_id
                """,
                (
                    question_vector,
                    active_spec.provider,
                    active_spec.model,
                    active_spec.dimension,
                    active_spec.dimension,
                    question_vector,
                    limit,
                    min_similarity,
                ),
            )
            rows = cursor.fetchall()

    return format_retrieval_rows(rows)


def retrieve_lexical_context(
    question: str,
    candidate_limit: int,
    settings: Settings | None = None,
) -> list[dict]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return []

    limit = validate_candidate_limit(candidate_limit, "lexical candidate count")
    active_settings = settings or Settings.from_env()
    query_text = lexical_query_text(cleaned_question)

    import psycopg

    with psycopg.connect(active_settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH lexical_query AS (
                    SELECT websearch_to_tsquery('english', %s) AS query
                )
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.chunk_text,
                    c.source_name,
                    c.source_type,
                    c.source_category,
                    c.source_path,
                    c.source_url,
                    c.source_version,
                    c.source_retrieved_at,
                    c.section_title,
                    c.heading_path,
                    c.word_count,
                    c.content_hash,
                    c.document_content_hash,
                    c.chunking_config_hash,
                    c.ingested_at,
                    ts_rank_cd(c.search_vector, lexical_query.query, 32) AS lexical_score
                FROM chunks AS c
                INNER JOIN documents AS d ON d.document_id = c.document_id
                CROSS JOIN lexical_query
                WHERE numnode(lexical_query.query) > 0
                  AND c.search_vector @@ lexical_query.query
                  AND c.content_hash IS NOT NULL
                  AND c.document_content_hash = d.content_hash
                  AND c.chunking_config_hash = d.chunking_config_hash
                ORDER BY lexical_score DESC, c.chunk_id
                LIMIT %s
                """,
                (query_text, limit),
            )
            rows = cursor.fetchall()

    return format_lexical_rows(rows)


def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    settings: Settings | None = None,
) -> list[dict]:
    active_settings = settings or Settings.from_env()
    limit = validate_top_k(top_k)
    from src.retrieval.hybrid_retriever import retrieve_with_mode

    return retrieve_with_mode(
        question,
        top_k=limit,
        min_similarity=min_similarity,
        settings=active_settings,
    )


def snippet(text: str, max_length: int = 240) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_length:
        return compact_text
    return compact_text[: max_length - 3].rstrip() + "..."


def result_display_score(result: dict) -> float | None:
    for field in (
        "reranker_score",
        "fusion_score",
        "semantic_score",
        "lexical_score",
        "similarity_score",
    ):
        value = result.get(field)
        if value is not None:
            return float(value)
    return None


def format_cli_results(question: str, results: list[dict]) -> str:
    lines = [f"Question: {question}"]
    if not results:
        lines.append("No relevant chunks found.")
        return "\n".join(lines)

    lines.append(f"Retrieved chunks: {len(results)}")
    for result in results:
        score = result_display_score(result)
        score_text = f"{score:.4f}" if score is not None else "n/a"
        lines.append(
            (
                f"{result['rank']}. {result['source_name']} "
                f"({result['source_path']}) "
                f"score={score_text} "
                f"chunk={result['chunk_id']}"
            )
        )
        lines.append(f"   {snippet(result['chunk_text'])}")

    return "\n".join(lines)


def safe_console_text(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant local RAG context.")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--mode", choices=("semantic", "hybrid"))
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.mode or args.rerank:
        settings = replace(
            settings,
            retrieval_mode=args.mode or settings.retrieval_mode,
            reranking_enabled=args.rerank or settings.reranking_enabled,
        )
    results = retrieve_context(
        args.question,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        settings=settings,
    )
    print(safe_console_text(format_cli_results(args.question, results)))


if __name__ == "__main__":
    main()
