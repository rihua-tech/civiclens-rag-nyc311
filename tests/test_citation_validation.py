from dataclasses import dataclass

from src.common.config import Settings
from src.generation.answer_question import generate_answer_from_chunks
from src.generation.citation_validation import validate_citation_ids
from src.generation.schemas import AnswerStatus, EvidenceItem, NO_ANSWER, ProviderResult


def retrieved_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_a",
            "document_id": "doc_a",
            "chunk_text": "Complaint type is the broad category of the reported problem.",
            "source_name": "NYC 311 field guide",
            "source_path": "docs/knowledge/nyc311-service-request-fields.md",
            "source_type": "markdown",
            "source_category": "external_nyc311",
            "section_title": "Problem / Complaint Type",
            "heading_path": ["Field Definitions", "Problem / Complaint Type"],
        },
        {
            "chunk_id": "chunk_b",
            "document_id": "doc_b",
            "chunk_text": "Status describes the current state of a request.",
            "source_name": "NYC 311 field guide",
            "source_path": "docs/knowledge/nyc311-service-request-fields.md",
            "section_title": "Status",
            "heading_path": ["Field Definitions", "Status"],
        },
    ]


def local_settings() -> Settings:
    return Settings(
        database_url="postgresql://unused",
        embedding_model="local-deterministic-1536",
        use_openai_embeddings=False,
        use_openai_answers=False,
        openai_api_key="",
    )


@dataclass
class FakeProvider:
    result: ProviderResult
    provider_name: str = "openai"
    model_name: str = "fake-openai-model"

    def generate(
        self,
        _question: str,
        _evidence: list[EvidenceItem],
    ) -> ProviderResult:
        return self.result


def test_valid_citation_is_accepted_and_provenance_is_rebuilt():
    validation = validate_citation_ids(["chunk_a"], retrieved_chunks())

    assert validation.valid_ids == ("chunk_a",)
    assert validation.invalid_ids == ()
    assert validation.sources[0] == {
        "source_name": "NYC 311 field guide",
        "source_path": "docs/knowledge/nyc311-service-request-fields.md",
        "chunk_id": "chunk_a",
        "citation_number": 1,
        "document_id": "doc_a",
        "source_type": "markdown",
        "source_category": "external_nyc311",
        "section_title": "Problem / Complaint Type",
        "heading_path": ["Field Definitions", "Problem / Complaint Type"],
    }


def test_unknown_fabricated_and_duplicate_citations_are_filtered():
    validation = validate_citation_ids(
        ["chunk_a", "fake_chunk_999", "chunk_a", "unknown"],
        retrieved_chunks(),
    )

    assert validation.valid_ids == ("chunk_a",)
    assert validation.invalid_ids == ("fake_chunk_999", "unknown")
    assert len(validation.sources) == 1


def test_mixed_valid_and_invalid_ids_preserve_only_valid_sources():
    provider = FakeProvider(
        ProviderResult(
            answer="The field guide defines both fields. [999]",
            citation_ids=("chunk_b", "fake_chunk_999", "chunk_a"),
            status=AnswerStatus.ANSWERED,
        )
    )

    response = generate_answer_from_chunks(
        "What do complaint type and status mean?",
        retrieved_chunks(),
        settings=local_settings(),
        provider=provider,
    )

    assert response["answer_status"] == "answered"
    assert response["citation_ids"] == ["chunk_b", "chunk_a"]
    assert response["rejected_citation_ids"] == ["fake_chunk_999"]
    assert [source["chunk_id"] for source in response["sources"]] == [
        "chunk_b",
        "chunk_a",
    ]
    assert "[999]" not in response["answer"]
    assert response["answer"].endswith("[2] [1]")


def test_answered_result_with_zero_valid_citations_becomes_abstention():
    provider = FakeProvider(
        ProviderResult(
            answer="An unsupported answer.",
            citation_ids=("fake_chunk_999",),
            status=AnswerStatus.ANSWERED,
        )
    )

    response = generate_answer_from_chunks(
        "What does complaint type mean?",
        retrieved_chunks(),
        settings=local_settings(),
        provider=provider,
    )

    assert response["answer"] == NO_ANSWER
    assert response["answer_status"] == "abstained"
    assert response["sources"] == []
    assert response["citation_ids"] == []
    assert response["rejected_citation_ids"] == ["fake_chunk_999"]
    assert response["grounding_rejection_reason"] == "no_valid_citations"


def test_provider_abstention_remains_safe():
    provider = FakeProvider(
        ProviderResult("", (), AnswerStatus.ABSTAINED),
    )

    response = generate_answer_from_chunks(
        "What is unsupported?",
        retrieved_chunks(),
        settings=local_settings(),
        provider=provider,
    )

    assert response["answer"] == NO_ANSWER
    assert response["answer_status"] == "abstained"
    assert response["sources"] == []

