"""Orchestrate retrieval, answer providers, and application citation validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Sequence

from src.common.config import OPENAI_ANSWER_PROVIDER, Settings
from src.generation.citation_validation import (
    add_validated_display_citations,
    validate_citation_ids,
)
from src.generation.providers import (
    AnswerProvider,
    DeterministicAnswerProvider,
    build_answer_provider,
)
from src.generation.providers.base import AnswerProviderError
from src.generation.schemas import AnswerStatus, EvidenceItem, NO_ANSWER, ProviderResult
from src.observability.latency import measure_latency
from src.retrieval.retrieve_context import DEFAULT_MIN_SIMILARITY, retrieve_context


DEFAULT_CONFIDENCE_NOTE = (
    "This answer is generated from retrieved local context only; verify source "
    "documents before making operational decisions."
)
OPENAI_CONFIDENCE_NOTE = (
    "This answer was generated from retrieved context by the configured OpenAI "
    "provider and its citations were validated by CivicLens."
)


def _generate_with_timing(
    provider: AnswerProvider,
    question: str,
    evidence: Sequence[EvidenceItem],
) -> ProviderResult:
    with measure_latency("answer_generation_ms"):
        return provider.generate(question, evidence)


def build_evidence(retrieved_chunks: Sequence[dict[str, Any]]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for chunk in retrieved_chunks:
        item = EvidenceItem.from_chunk(chunk)
        if item is not None:
            evidence.append(item)
    return evidence


def unique_sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible source extraction using application-owned metadata."""
    citation_ids = [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks]
    return list(validate_citation_ids(citation_ids, retrieved_chunks).sources)


def _no_evidence_response(
    retrieved_chunks: list[dict[str, Any]],
    requested_provider: str,
    requested_model: str,
) -> dict[str, Any]:
    return {
        "answer": NO_ANSWER,
        "sources": [],
        "confidence_note": "No usable source chunks were retrieved.",
        "retrieved_chunks": retrieved_chunks,
        "answer_status": AnswerStatus.ABSTAINED.value,
        "answer_provider": requested_provider,
        "answer_model": requested_model,
        "citation_ids": [],
        "rejected_citation_ids": [],
        "provider_called": False,
        "fallback_used": False,
    }


def _ground_provider_result(
    result: ProviderResult,
    retrieved_chunks: list[dict[str, Any]],
    provider: AnswerProvider,
) -> dict[str, Any]:
    validation = validate_citation_ids(result.citation_ids, retrieved_chunks)
    provider_name = str(provider.provider_name)
    provider_model = str(provider.model_name)

    if result.status is AnswerStatus.ANSWERED and not validation.valid_ids:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "confidence_note": (
                "The provider answer was rejected because it contained no valid "
                "retrieved citation IDs."
            ),
            "retrieved_chunks": retrieved_chunks,
            "answer_status": AnswerStatus.ABSTAINED.value,
            "answer_provider": provider_name,
            "answer_model": provider_model,
            "citation_ids": [],
            "rejected_citation_ids": list(validation.invalid_ids),
            "provider_called": True,
            "fallback_used": False,
            "grounding_rejection_reason": "no_valid_citations",
        }

    if result.status is AnswerStatus.ABSTAINED:
        answer = NO_ANSWER
        confidence_note = "The answer provider abstained because evidence was insufficient."
    else:
        answer = add_validated_display_citations(result.answer, validation.sources)
        confidence_note = (
            OPENAI_CONFIDENCE_NOTE
            if provider_name == OPENAI_ANSWER_PROVIDER
            else DEFAULT_CONFIDENCE_NOTE
        )

    return {
        "answer": answer,
        "sources": list(validation.sources),
        "confidence_note": confidence_note,
        "retrieved_chunks": retrieved_chunks,
        "answer_status": result.status.value,
        "answer_provider": provider_name,
        "answer_model": provider_model,
        "citation_ids": list(validation.valid_ids),
        "rejected_citation_ids": list(validation.invalid_ids),
        "provider_called": True,
        "fallback_used": False,
    }


def _local_fallback(
    question: str,
    evidence: Sequence[EvidenceItem],
    retrieved_chunks: list[dict[str, Any]],
    fallback_from: str,
    fallback_reason: str,
) -> dict[str, Any]:
    local_provider = DeterministicAnswerProvider()
    local_result = _generate_with_timing(local_provider, question, evidence)
    response = _ground_provider_result(local_result, retrieved_chunks, local_provider)
    response.update(
        {
            "fallback_used": True,
            "fallback_from": fallback_from,
            "fallback_reason": fallback_reason,
        }
    )
    return response


def generate_answer_from_chunks(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    settings: Settings | None = None,
    provider: AnswerProvider | None = None,
) -> dict[str, Any]:
    """Generate and validate an answer from already-retrieved evidence."""
    active_settings = settings or Settings.from_env()
    evidence = build_evidence(retrieved_chunks)
    requested_provider = (
        str(provider.provider_name) if provider is not None else active_settings.answer_provider
    )
    requested_model = (
        str(provider.model_name) if provider is not None else active_settings.answer_model
    )
    if not evidence:
        return _no_evidence_response(
            retrieved_chunks,
            requested_provider,
            requested_model,
        )

    selected_provider = provider
    if selected_provider is None:
        try:
            selected_provider = build_answer_provider(active_settings)
        except AnswerProviderError as exc:
            return _local_fallback(
                question,
                evidence,
                retrieved_chunks,
                active_settings.answer_provider,
                exc.code,
            )
        except Exception:
            return _local_fallback(
                question,
                evidence,
                retrieved_chunks,
                active_settings.answer_provider,
                "provider_configuration",
            )

    try:
        result = _generate_with_timing(selected_provider, question, evidence)
        if not isinstance(result, ProviderResult):
            raise TypeError("provider returned a non-ProviderResult value")
        return _ground_provider_result(result, retrieved_chunks, selected_provider)
    except AnswerProviderError as exc:
        return _local_fallback(
            question,
            evidence,
            retrieved_chunks,
            str(selected_provider.provider_name),
            exc.code,
        )
    except Exception:
        return _local_fallback(
            question,
            evidence,
            retrieved_chunks,
            str(selected_provider.provider_name),
            "provider_failure",
        )


def local_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve the deterministic local answer API used by tests and evaluation."""
    provider = DeterministicAnswerProvider()
    evidence = build_evidence(retrieved_chunks)
    if not evidence:
        return _no_evidence_response(
            retrieved_chunks,
            provider.provider_name,
            provider.model_name,
        )
    return _ground_provider_result(
        _generate_with_timing(provider, question, evidence),
        retrieved_chunks,
        provider,
    )


def openai_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    """Backward-compatible opt-in entrypoint with controlled local fallback."""
    openai_settings = replace(
        settings,
        answer_provider=OPENAI_ANSWER_PROVIDER,
        use_openai_answers=True,
    )
    return generate_answer_from_chunks(
        question,
        retrieved_chunks,
        settings=openai_settings,
    )


def answer_question(
    question: str,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    settings: Settings | None = None,
    query_id: str | None = None,
) -> dict[str, Any]:
    active_settings = settings or Settings.from_env()
    with measure_latency("retrieval_ms"):
        retrieved_chunks = retrieve_context(
            question,
            top_k=top_k,
            min_similarity=min_similarity,
            settings=active_settings,
        )
    response = generate_answer_from_chunks(
        question,
        retrieved_chunks,
        settings=active_settings,
    )
    if query_id is not None:
        response["query_id"] = query_id
    return response


def format_answer_response(response: dict[str, Any]) -> str:
    lines = [
        "Answer:",
        response["answer"],
        "",
        "Confidence:",
        response["confidence_note"],
        "",
    ]
    lines.append("Sources:")
    if response["sources"]:
        for fallback_number, source in enumerate(response["sources"], start=1):
            citation_number = source.get("citation_number", fallback_number)
            lines.append(
                (
                    f"{citation_number}. {source['source_name']} - "
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
    parser = argparse.ArgumentParser(
        description="Answer a question with retrieved local context."
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.05,
        help="Minimum similarity score for returned chunks",
    )
    args = parser.parse_args()

    response = answer_question(
        args.question,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    print(safe_console_text(format_answer_response(response)))


if __name__ == "__main__":
    main()
