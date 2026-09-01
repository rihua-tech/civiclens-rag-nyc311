import sys
from types import SimpleNamespace

from src.chunking.chunk_documents import chunk_documents
from src.common.config import Settings
from src.embeddings.providers import EmbeddingSpec
from src.embeddings.providers.deterministic import TOKEN_PATTERN
from src.ingestion.load_documents import load_documents
from src.retrieval.retrieve_context import (
    expand_schema_field_aliases,
    format_cli_results,
    format_retrieval_rows,
    lexical_query_text,
    retrieve_lexical_context,
    retrieve_semantic_context,
    validate_top_k,
)


def retrieval_row(score: float = 0.42) -> tuple:
    return (
        "chunk_1",
        "doc_1",
        "Chunk text about NYC 311 architecture.",
        "architecture.md",
        "markdown",
        "civiclens_project",
        "docs/architecture.md",
        "https://github.com/rihua-tech/civiclens-rag-nyc311/blob/main/docs/architecture.md",
        "Issue 8 curated corpus",
        "2026-08-17",
        "Architecture Boundary",
        ["CivicLens Runbook", "Architecture Boundary"],
        6,
        "sha256:chunk",
        "sha256:document",
        "sha256:chunking-config",
        "2026-08-17T00:00:00Z",
        score,
    )


def test_retriever_result_formatting_preserves_metadata():
    rows = [retrieval_row()]

    results = format_retrieval_rows(rows)

    assert results == [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "chunk_text": "Chunk text about NYC 311 architecture.",
            "source_name": "architecture.md",
            "source_type": "markdown",
            "source_category": "civiclens_project",
            "source_path": "docs/architecture.md",
            "source_url": "https://github.com/rihua-tech/civiclens-rag-nyc311/blob/main/docs/architecture.md",
            "source_version": "Issue 8 curated corpus",
            "source_retrieved_at": "2026-08-17",
            "section_title": "Architecture Boundary",
            "heading_path": ["CivicLens Runbook", "Architecture Boundary"],
            "word_count": 6,
            "content_hash": "sha256:chunk",
            "document_content_hash": "sha256:document",
            "chunking_config_hash": "sha256:chunking-config",
            "ingested_at": "2026-08-17T00:00:00Z",
            "similarity_score": 0.42,
            "semantic_score": 0.42,
            "semantic_rank": 1,
            "lexical_score": None,
            "lexical_rank": None,
            "fusion_score": None,
            "reranker_score": None,
            "pre_rerank_rank": None,
            "retrieval_mode": "semantic",
            "rank": 1,
        }
    ]


def test_top_k_validation_accepts_positive_values():
    assert validate_top_k(3) == 3


def test_top_k_validation_rejects_non_positive_values():
    try:
        validate_top_k(0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid top_k")


def test_cli_formatting_includes_similarity_and_source_metadata():
    output = format_cli_results(
        "What is the architecture?",
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "chunk_text": "Architecture context.",
                "source_name": "architecture.md",
                "source_path": "docs/architecture.md",
                "similarity_score": 0.31,
                "rank": 1,
            }
        ],
    )

    assert "score=0.3100" in output
    assert "architecture.md" in output
    assert "chunk_1" in output


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, parameters=None):
        self.calls.append((query, parameters))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.fake_cursor


class SequencedCursor(FakeCursor):
    def __init__(self, responses):
        super().__init__([])
        self.responses = list(responses)

    def fetchall(self):
        return self.responses.pop(0)


class FakeSemanticProvider:
    spec = EmbeddingSpec(
        "sentence_transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
        384,
    )

    def embed(self, text):
        assert text == "What does complaint type mean?"
        return [0.0] * 384

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


def test_postgresql_lexical_retrieval_is_parameterized_and_preserves_provenance(
    monkeypatch,
):
    cursor = FakeCursor([retrieval_row(0.75)])
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda _: FakeConnection(cursor)),
    )
    settings = Settings(
        database_url="postgresql://example",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
    )

    results = retrieve_lexical_context(
        "complaint_type",
        candidate_limit=10,
        settings=settings,
    )

    query, parameters = cursor.calls[0]
    assert "websearch_to_tsquery('english', %s)" in query
    assert "c.search_vector @@ lexical_query.query" in query
    assert "c.document_content_hash = d.content_hash" in query
    assert parameters == ("complaint_type", 10)
    assert results[0]["lexical_score"] == 0.75
    assert results[0]["lexical_rank"] == 1
    assert results[0]["source_category"] == "civiclens_project"
    assert results[0]["section_title"] == "Architecture Boundary"
    assert results[0]["heading_path"] == [
        "CivicLens Runbook",
        "Architecture Boundary",
    ]


def test_lexical_query_keeps_exact_identifier_and_removes_question_filler():
    assert lexical_query_text("What does complaint_type mean?") == "complaint_type"


def test_lexical_query_removes_metric_question_filler_that_blocks_evidence():
    question = "What percentage Recall@5 did the hybrid retriever achieve?"

    assert lexical_query_text(question) == "recall 5 hybrid retrieval"


def test_metric_query_terms_match_manifest_authorized_readme_chunk():
    question = "What percentage Recall@5 did the hybrid retriever achieve?"
    query_terms = set(lexical_query_text(question).split())
    chunks = chunk_documents(load_documents(ingested_at="metric-retrieval-test"))

    matching_chunks = [
        chunk
        for chunk in chunks
        if chunk["source_path"] == "README.md"
        and query_terms.issubset(
            set(TOKEN_PATTERN.findall(str(chunk["chunk_text"]).lower()))
        )
    ]

    assert any(
        chunk["section_title"] == "Proof at a Glance"
        and "0.8393" in chunk["chunk_text"]
        for chunk in matching_chunks
    )


def test_lexical_query_terms_are_bounded():
    query = " ".join(f"term{index}" for index in range(20))

    assert len(lexical_query_text(query).split()) == 12


def test_schema_field_alias_expansion_uses_field_guide_labels():
    expanded = expand_schema_field_aliases(
        "Compare complaint_type, descriptor, closed_date, and due_date."
    )

    assert expanded.startswith(
        "Compare complaint_type, descriptor, closed_date, and due_date."
    )
    assert "Complaint Type" in expanded
    assert "Problem" in expanded
    assert "Problem Detail" in expanded
    assert "Closed Date" in expanded
    assert "Due Date" in expanded


def test_schema_field_alias_expansion_does_not_change_ordinary_queries():
    question = "Which borough has the highest request count?"

    assert expand_schema_field_aliases(question) == question


def test_semantic_retrieval_validates_profile_dimension_and_uses_pgvector(
    monkeypatch,
):
    profile = (
        "sentence_transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
        384,
    )
    vector_row = (
        "chunk_1",
        "doc_1",
        "sha256:chunk",
        "sha256:document",
        "sha256:chunking-config",
        0.81,
    )
    cursor = SequencedCursor([[profile], [vector_row], [retrieval_row()[:-1]]])
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda _: FakeConnection(cursor)),
    )
    settings = Settings(
        database_url="postgresql://example",
        embedding_model=profile[1],
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
        embedding_provider=profile[0],
        embedding_dimension=profile[2],
    )

    results = retrieve_semantic_context(
        "What does complaint type mean?",
        candidate_limit=12,
        settings=settings,
        provider=FakeSemanticProvider(),
    )

    assert len(cursor.calls) == 3
    profile_query, _ = cursor.calls[0]
    semantic_query, parameters = cursor.calls[1]
    hydration_query, hydration_parameters = cursor.calls[2]
    assert "SELECT DISTINCT" in profile_query
    assert "c.semantic_embedding <=> %s::vector" in semantic_query
    assert "vector_dims(c.semantic_embedding) = %s" in semantic_query
    assert parameters[1:4] == profile
    assert parameters[4] == 384
    assert parameters[-2:] == (12, 0.25)
    assert "c.chunk_id = ANY(%s)" in hydration_query
    assert hydration_parameters == (["chunk_1"],)
    assert results[0]["semantic_score"] == 0.81
    assert results[0]["retrieval_mode"] == "semantic"
    assert results[0]["source_path"] == "docs/architecture.md"
