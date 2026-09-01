"""Semantic and PostgreSQL lexical retrieval with a stable result contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
import re
from typing import Iterable

from src.common.config import PINECONE_VECTOR_STORE, Settings
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider
from src.embeddings.providers.deterministic import EMBEDDING_STOPWORDS, TOKEN_PATTERN
from src.vectorstores.base import VectorStore
from src.vectorstores.factory import create_vector_store
from src.vectorstores.models import (
    VectorIdentity,
    VectorMatch,
    VectorStoreConsistencyError,
)


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
    "achieve",
    "achieved",
    "define",
    "definition",
    "did",
    "does",
    "explain",
    "mean",
    "means",
    "percent",
    "percentage",
    "tell",
}
LEXICAL_TOKEN_ALIASES = {
    "retriever": "retrieval",
}
SCHEMA_FIELD_ALIASES = {
    "complaint_type": ("Complaint Type", "Problem"),
    "descriptor": ("Descriptor", "Problem Detail"),
    "closed_date": ("Closed Date",),
    "due_date": ("Due Date",),
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


def expand_schema_field_aliases(question: str) -> str:
    """Append corpus-supported display labels for schema-style field tokens."""

    cleaned_question = question.strip()
    additions: list[str] = []
    for field_name, aliases in SCHEMA_FIELD_ALIASES.items():
        field_pattern = rf"(?<![A-Za-z0-9_]){re.escape(field_name)}(?![A-Za-z0-9_])"
        if re.search(field_pattern, cleaned_question, flags=re.IGNORECASE) is None:
            continue
        for alias in aliases:
            if alias.casefold() not in cleaned_question.casefold() and alias not in additions:
                additions.append(alias)
    return " ".join((cleaned_question, *additions))


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
        token = LEXICAL_TOKEN_ALIASES.get(token, token)
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
    vector_store: VectorStore | None = None,
) -> list[dict]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return []

    limit = validate_candidate_limit(candidate_limit, "semantic candidate count")
    active_settings = settings or Settings.from_env()
    active_provider = provider or create_embedding_provider(active_settings)
    active_spec = active_provider.spec
    identities: tuple[VectorIdentity, ...] = ()
    if active_settings.vector_store_provider == PINECONE_VECTOR_STORE:
        from src.orchestration.readiness import load_current_corpus_identity

        identities = tuple(
            VectorIdentity(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                content_hash=item.content_hash,
                document_content_hash=item.document_content_hash,
                chunking_config_hash=item.chunking_config_hash,
            )
            for item in load_current_corpus_identity().chunks
        )
    active_store = vector_store or create_vector_store(
        active_settings,
        active_spec,
        identities,
    )
    question_embedding = active_provider.embed(cleaned_question)
    matches = active_store.query(
        question_embedding,
        candidate_limit=limit,
        min_similarity=min_similarity,
    )
    return hydrate_vector_matches(matches, active_settings)


def hydrate_vector_matches(
    matches: Iterable[VectorMatch],
    settings: Settings,
) -> list[dict]:
    """Hydrate vector IDs through authoritative current PostgreSQL metadata."""

    ordered_matches = list(matches)
    if not ordered_matches:
        return []
    match_ids = [match.identity.chunk_id for match in ordered_matches]
    if len(set(match_ids)) != len(match_ids):
        raise VectorStoreConsistencyError("Vector provider returned duplicate chunk IDs")

    import psycopg

    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
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
                    c.ingested_at
                FROM chunks AS c
                INNER JOIN documents AS d ON d.document_id = c.document_id
                WHERE c.chunk_id = ANY(%s)
                  AND c.content_hash IS NOT NULL
                  AND c.document_content_hash = d.content_hash
                  AND c.chunking_config_hash = d.chunking_config_hash
                """,
                (match_ids,),
            )
            rows = cursor.fetchall()

    if len({str(row[0]) for row in rows}) != len(rows):
        raise VectorStoreConsistencyError(
            "PostgreSQL returned duplicate vector-match metadata"
        )
    hydrated = {str(row[0]): row for row in rows}
    if set(hydrated) != set(match_ids):
        raise VectorStoreConsistencyError(
            "Vector matches are incomplete or stale in PostgreSQL"
        )

    results: list[dict] = []
    for match in ordered_matches:
        row = hydrated[match.identity.chunk_id]
        stored_identity = VectorIdentity(
            chunk_id=str(row[0]),
            document_id=str(row[1]),
            content_hash=str(row[13]),
            document_content_hash=str(row[14]),
            chunking_config_hash=str(row[15]),
        )
        if stored_identity != match.identity:
            raise VectorStoreConsistencyError(
                "Vector match metadata is incompatible with PostgreSQL"
            )
        results.append(
            result_from_row(
                (*row, match.score),
                retrieval_mode="semantic",
                semantic_rank=match.rank,
            )
        )
    return results


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
