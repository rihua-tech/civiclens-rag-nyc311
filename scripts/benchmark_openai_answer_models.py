"""Isolated OpenAI answer-model benchmark for the CivicLens RAG contract.

This harness intentionally reuses production prompt, schema, provider error mapping,
citation validation, safe abstention, routing, retrieval settings, and evidence. It
adds the one model-specific Responses API argument needed for gpt-5.6-luna only in
the wrapper client used by this script.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import DefaultHttpxClient, OpenAI  # noqa: E402

from src.analytics.simple_analytics import (  # noqa: E402
    answer_analytics_question,
    looks_like_analytics_question,
)
from src.common.config import OPENAI_ANSWER_PROVIDER, Settings  # noqa: E402
from src.evaluation.evaluate_rag import (  # noqa: E402
    DEFAULT_EVALUATION_PATH,
    DEFAULT_TOP_K,
    OFFLINE_STRATEGY,
    EvaluationQuestion,
    StrategyDefinition,
    analytics_sources_are_sample_outputs,
    build_offline_retriever,
    evaluate_strategy_question,
    load_evaluation_questions,
    source_matches_hint,
)
from src.generation.answer_question import (  # noqa: E402
    NO_ANSWER,
    build_evidence,
    generate_answer_from_chunks,
)
from src.generation.providers.openai_provider import (  # noqa: E402
    APPLICATION_RULES,
    OpenAIAnswerProvider,
    OpenAIStructuredAnswer,
    build_provider_input,
)
from src.generation.schemas import EvidenceItem, ProviderResult  # noqa: E402
from src.ingestion.load_documents import load_documents  # noqa: E402
from src.chunking.chunk_documents import chunk_documents  # noqa: E402
from src.retrieval.retrieve_context import (  # noqa: E402
    DEFAULT_MIN_SIMILARITY,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
    / "openai_answer_model_benchmark.json"
)
BENCHMARK_MAX_RETRIES = 0
RENDER_AB_MODELS = ("gpt-4o-mini", "gpt-5.6-luna")
RENDER_AB_QUESTION_IDS = ("q010", "q019")
RENDER_AB_RUNS = 5
RENDER_AB_CALLS_PER_MODEL = len(RENDER_AB_QUESTION_IDS) * RENDER_AB_RUNS
RENDER_AB_TOTAL_CALLS = RENDER_AB_CALLS_PER_MODEL * len(RENDER_AB_MODELS)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    reasoning: dict[str, str] | None
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


MODEL_SPECS = (
    ModelSpec("gpt-4o-mini", None, 0.15, 0.075, 0.60),
    ModelSpec("gpt-4.1-nano", None, 0.10, 0.025, 0.40),
    ModelSpec("gpt-5.6-luna", {"effort": "none"}, 0.20, 0.02, 1.20),
)

MANUAL_QUESTION_IDS = ("q010", "q019", "q022", "q023")
PERCENTAGE_CASE_ID = "manual_percentage"
PERCENTAGE_QUESTION = "What percentage Recall@5 did the hybrid retriever achieve?"

MODEL_FAILURES = {
    "route mismatch",
    "expected citation is absent",
    "citation is invalid",
    "safe no-answer expectation failed",
    "unsupported answer detected",
    "real answer provider was not used successfully",
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


class AttemptTelemetry:
    def __init__(self) -> None:
        self.attempt_count = 0
        self.request_ids: list[str] = []
        self.status_codes: list[int] = []
        self.raw_error_type: str | None = None
        self.usage: dict[str, int | None] = {}
        self.response_request_id: str | None = None

    def on_request(self, _request: Any) -> None:
        self.attempt_count += 1

    def on_response(self, response: Any) -> None:
        self.status_codes.append(int(response.status_code))
        request_id = response.headers.get("x-request-id")
        if request_id:
            self.request_ids.append(str(request_id))

    def capture_response(self, response: Any) -> None:
        request_id = getattr(response, "_request_id", None)
        if request_id:
            self.response_request_id = str(request_id)
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        self.usage = {
            "input_tokens": optional_int(getattr(usage, "input_tokens", None)),
            "cached_input_tokens": optional_int(
                getattr(input_details, "cached_tokens", None)
            ),
            "output_tokens": optional_int(getattr(usage, "output_tokens", None)),
            "reasoning_tokens": optional_int(
                getattr(output_details, "reasoning_tokens", None)
            ),
            "total_tokens": optional_int(getattr(usage, "total_tokens", None)),
        }

    def capture_error(self, error: Exception) -> None:
        self.raw_error_type = type(error).__name__
        status_code = getattr(error, "status_code", None)
        error_code = getattr(error, "code", None)
        self.api_error_status_code = (
            int(status_code) if status_code is not None else None
        )
        self.api_error_code = str(error_code) if error_code is not None else None

    def as_dict(self) -> dict[str, Any]:
        request_id = self.response_request_id
        if request_id is None and self.request_ids:
            request_id = self.request_ids[-1]
        return {
            "attempt_count": self.attempt_count,
            "retry_detected": self.attempt_count > 1,
            "request_id": request_id,
            "response_request_ids": self.request_ids,
            "status_codes": self.status_codes,
            "raw_error_type": self.raw_error_type,
            "api_error_status_code": getattr(self, "api_error_status_code", None),
            "api_error_code": getattr(self, "api_error_code", None),
            "usage": self.usage,
        }


class InstrumentedResponses:
    def __init__(
        self,
        responses: Any,
        telemetry: AttemptTelemetry,
        reasoning: dict[str, str] | None,
    ) -> None:
        self._responses = responses
        self._telemetry = telemetry
        self._reasoning = reasoning

    def parse(self, **kwargs: Any) -> Any:
        if self._reasoning is not None:
            kwargs["reasoning"] = self._reasoning
        try:
            response = self._responses.parse(**kwargs)
        except Exception as exc:
            self._telemetry.capture_error(exc)
            raise
        self._telemetry.capture_response(response)
        return response


class InstrumentedClient:
    def __init__(
        self,
        client: OpenAI,
        telemetry: AttemptTelemetry,
        reasoning: dict[str, str] | None,
    ) -> None:
        self._client = client
        self.responses = InstrumentedResponses(
            client.responses,
            telemetry,
            reasoning,
        )

    def close(self) -> None:
        self._client.close()


class RecordingClientFactory:
    def __init__(self, reasoning: dict[str, str] | None) -> None:
        self.reasoning = reasoning
        self.created: list[tuple[InstrumentedClient, AttemptTelemetry]] = []

    def __call__(self, **kwargs: Any) -> InstrumentedClient:
        telemetry = AttemptTelemetry()
        http_client = DefaultHttpxClient(
            event_hooks={
                "request": [telemetry.on_request],
                "response": [telemetry.on_response],
            }
        )
        client = InstrumentedClient(
            OpenAI(**kwargs, http_client=http_client),
            telemetry,
            self.reasoning,
        )
        self.created.append((client, telemetry))
        return client

    def finish_latest(self) -> dict[str, Any]:
        if not self.created:
            return AttemptTelemetry().as_dict()
        client, telemetry = self.created[-1]
        result = telemetry.as_dict()
        client.close()
        return result


class RecordingProvider:
    provider_name = OPENAI_ANSWER_PROVIDER

    def __init__(self, settings: Settings, spec: ModelSpec) -> None:
        self.model_name = spec.model
        self._factory = RecordingClientFactory(spec.reasoning)
        self._delegate = OpenAIAnswerProvider(
            api_key=settings.openai_api_key,
            model_name=spec.model,
            timeout_seconds=settings.answer_timeout_seconds,
            max_retries=BENCHMARK_MAX_RETRIES,
            client_factory=self._factory,
        )
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> ProviderResult:
        started = time.perf_counter()
        error_type: str | None = None
        try:
            return self._delegate.generate(question, evidence)
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            telemetry = self._factory.finish_latest()
            telemetry.update(
                {
                    "generation_ms": round(latency_ms, 3),
                    "provider_error_type": error_type,
                }
            )
            self.calls.append(telemetry)


def evidence_manifest(
    question: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = build_evidence(chunks)
    payloads = [item.provider_payload() for item in evidence]
    provider_input = build_provider_input(question, evidence)
    return {
        "chunk_count": len(chunks),
        "usable_evidence_count": len(evidence),
        "ordered_chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in chunks],
        "evidence_sha256": sha256_json(payloads),
        "provider_input_sha256": "sha256:"
        + hashlib.sha256(provider_input.encode("utf-8")).hexdigest(),
        "provider_input_characters": len(provider_input),
    }


def is_analytics_routed(question: str) -> bool:
    response = answer_analytics_question(question)
    return response["mode"] == "analytics" or looks_like_analytics_question(question)


def precompute_evidence(
    questions: Sequence[EvaluationQuestion],
    top_k: int,
    min_similarity: float,
    *,
    include_percentage_case: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    retriever = build_offline_retriever(top_k, min_similarity)
    corpus_chunks = chunk_documents(
        load_documents(ingested_at="fixed-answer-model-benchmark")
    )
    questions_by_id = {question.question_id: question for question in questions}
    question_texts = {
        question.question_id: question.question
        for question in questions
        if not is_analytics_routed(question.question)
    }
    if include_percentage_case:
        question_texts[PERCENTAGE_CASE_ID] = PERCENTAGE_QUESTION
    chunks_by_id: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for question_id, question in question_texts.items():
        started = time.perf_counter()
        required_section_ids = (
            set(questions_by_id[question_id].relevant_ids)
            if question_id in questions_by_id
            else set()
        )
        required_chunks = [
            chunk
            for chunk in corpus_chunks
            if (
                f"{chunk.get('document_id', '')}::{chunk.get('section_title', '')}"
                in required_section_ids
            )
        ]
        if question_id == PERCENTAGE_CASE_ID:
            required_chunks = [
                chunk
                for chunk in corpus_chunks
                if "0.8393" in str(chunk.get("chunk_text", ""))
                and str(chunk.get("source_path", "")) == "README.md"
            ]
        matched_section_ids = {
            f"{chunk.get('document_id', '')}::{chunk.get('section_title', '')}"
            for chunk in required_chunks
        }
        missing_section_ids = required_section_ids - matched_section_ids

        retrieved_fillers = retriever(question, OFFLINE_STRATEGY)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for chunk in [*required_chunks, *retrieved_fillers, *corpus_chunks]:
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id or chunk_id in selected_ids:
                continue
            selected.append(dict(chunk))
            selected_ids.add(chunk_id)
            if len(selected) == top_k:
                break
        if len(selected) != top_k:
            raise RuntimeError(
                f"Fixed benchmark evidence for {question_id} has "
                f"{len(selected)} chunks; expected {top_k}."
            )
        chunks = selected
        elapsed_ms = (time.perf_counter() - started) * 1000
        chunks_by_id[question_id] = chunks
        manifests[question_id] = {
            "question": question,
            "evidence_preparation_ms": round(elapsed_ms, 3),
            "required_section_ids": sorted(required_section_ids),
            "missing_declared_section_ids": sorted(missing_section_ids),
            "required_chunk_count": len(required_chunks),
            **evidence_manifest(question, chunks),
        }
        print(
            f"EVIDENCE question_id={question_id} chunks={len(chunks)} "
            f"ms={elapsed_ms:.3f}",
            flush=True,
        )
    return chunks_by_id, manifests


def application_call(
    question: str,
    chunks: list[dict[str, Any]],
    settings: Settings,
    provider: RecordingProvider,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    call_count = len(provider.calls)
    response = generate_answer_from_chunks(
        question,
        chunks,
        settings=settings,
        provider=provider,
    )
    provider_call = provider.calls[-1] if len(provider.calls) > call_count else None
    return response, provider_call


def estimated_cost_usd(call: dict[str, Any] | None, spec: ModelSpec) -> float | None:
    if call is None:
        return None
    usage = call.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    cached_tokens = usage.get("cached_input_tokens") or 0
    uncached_tokens = max(int(input_tokens) - int(cached_tokens), 0)
    return (
        uncached_tokens * spec.input_usd_per_million
        + int(cached_tokens) * spec.cached_input_usd_per_million
        + int(output_tokens) * spec.output_usd_per_million
    ) / 1_000_000


def latency_record(
    model: str,
    question_id: str,
    question: str,
    run: int,
    response: dict[str, Any],
    provider_call: dict[str, Any] | None,
    evidence: dict[str, Any],
    spec: ModelSpec,
) -> dict[str, Any]:
    return {
        "model": model,
        "question_id": question_id,
        "question": question,
        "run": run,
        "evidence_sha256": evidence["evidence_sha256"],
        "ordered_chunk_ids": evidence["ordered_chunk_ids"],
        "answer": response.get("answer"),
        "answer_status": response.get("answer_status"),
        "citation_ids": response.get("citation_ids", []),
        "rejected_citation_ids": response.get("rejected_citation_ids", []),
        "fallback_used": response.get("fallback_used", False),
        "fallback_reason": response.get("fallback_reason"),
        "provider_call": provider_call,
        "estimated_cost_usd": estimated_cost_usd(provider_call, spec),
    }


def compact_render_ab_call(
    model: str,
    question_id: str,
    provider_call: dict[str, Any] | None,
    fallback_used: bool,
    *,
    provider_error_override: str | None = None,
) -> dict[str, Any]:
    call = provider_call or {}
    usage = call.get("usage") or {}
    status_codes = call.get("status_codes") or []
    provider_error = (
        provider_error_override
        or call.get("provider_error_type")
        or call.get("raw_error_type")
    )
    return {
        "model": model,
        "question_id": question_id,
        "generation_ms": call.get("generation_ms"),
        "attempt_count": call.get("attempt_count"),
        "http_status": status_codes[-1] if status_codes else None,
        "request_id": call.get("request_id"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "fallback_used": fallback_used,
        "provider_error": provider_error,
    }


def print_render_ab_result(event: str, payload: dict[str, Any]) -> None:
    print(
        f"{event} "
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def render_ab_failure_reason(
    response: dict[str, Any],
    provider_call: dict[str, Any] | None,
    compact_call: dict[str, Any],
) -> str | None:
    if provider_call is None:
        return "provider telemetry is missing"
    if (
        response.get("fallback_reason") == "malformed_provider_response"
        or compact_call["provider_error"] == "ProviderResponseError"
    ):
        return "malformed structured output"
    if compact_call["provider_error"] is not None:
        return "provider error occurred"
    if compact_call["fallback_used"]:
        return "provider fallback occurred"
    if (
        response.get("answer_provider") != OPENAI_ANSWER_PROVIDER
        or response.get("answer_status") not in {"answered", "abstained"}
    ):
        return "structured provider response is missing"
    if (
        compact_call["input_tokens"] is None
        or compact_call["output_tokens"] is None
    ):
        return "token usage is missing"
    if provider_call.get("status_codes") != [200]:
        return "HTTP status is not exactly 200"
    if compact_call["attempt_count"] != 1:
        return "HTTP attempt count is not exactly 1"
    return None


def validate_render_ab_evidence(
    chunks_by_id: dict[str, list[dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
) -> None:
    for question_id in RENDER_AB_QUESTION_IDS:
        chunks = chunks_by_id.get(question_id, [])
        manifest = manifests.get(question_id, {})
        if (
            len(chunks) != DEFAULT_TOP_K
            or manifest.get("chunk_count") != DEFAULT_TOP_K
            or manifest.get("usable_evidence_count") != DEFAULT_TOP_K
            or len(manifest.get("ordered_chunk_ids", [])) != DEFAULT_TOP_K
        ):
            raise RuntimeError(
                f"Render A/B evidence for {question_id} must contain exactly "
                f"{DEFAULT_TOP_K} full usable chunks."
            )


def run_render_ab_benchmark(
    settings: Settings,
    specs: Sequence[ModelSpec],
    providers: dict[str, RecordingProvider],
    questions_by_id: dict[str, EvaluationQuestion],
    chunks_by_id: dict[str, list[dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    if tuple(spec.model for spec in specs) != RENDER_AB_MODELS:
        raise RuntimeError("Render A/B mode requires exactly the two qualified models.")
    validate_render_ab_evidence(chunks_by_id, manifests)

    records: list[dict[str, Any]] = []
    preflight_models: set[str] = set()
    cases = [
        (question_id, questions_by_id[question_id].question)
        for question_id in RENDER_AB_QUESTION_IDS
    ]
    for run in range(1, RENDER_AB_RUNS + 1):
        for case_index, (question_id, question) in enumerate(cases):
            offset = (run + case_index - 1) % len(specs)
            ordered_specs = [*specs[offset:], *specs[:offset]]
            for spec in ordered_specs:
                provider = providers[spec.model]
                provider_call_count = len(provider.calls)
                try:
                    response, provider_call = application_call(
                        question,
                        chunks_by_id[question_id],
                        settings,
                        provider,
                    )
                except Exception as exc:
                    provider_call = (
                        provider.calls[-1]
                        if len(provider.calls) > provider_call_count
                        else None
                    )
                    compact_call = compact_render_ab_call(
                        spec.model,
                        question_id,
                        provider_call,
                        False,
                        provider_error_override=type(exc).__name__,
                    )
                    print_render_ab_result("RENDER_AB_CALL", compact_call)
                    checkpoint["status"] = "blocked"
                    checkpoint["block_reason"] = (
                        "Render A/B call failed with a provider error for "
                        f"model={spec.model} question_id={question_id}."
                    )
                    checkpoint["completed_at"] = utc_now()
                    write_checkpoint(output_path, checkpoint)
                    raise RuntimeError(checkpoint["block_reason"]) from None

                record = latency_record(
                    spec.model,
                    question_id,
                    question,
                    run,
                    response,
                    provider_call,
                    manifests[question_id],
                    spec,
                )
                records.append(record)
                checkpoint["latency_records"] = records
                compact_call = compact_render_ab_call(
                    spec.model,
                    question_id,
                    provider_call,
                    bool(response.get("fallback_used", False)),
                )
                print_render_ab_result("RENDER_AB_CALL", compact_call)
                failure_reason = render_ab_failure_reason(
                    response,
                    provider_call,
                    compact_call,
                )
                if failure_reason is not None:
                    checkpoint["status"] = "blocked"
                    checkpoint["block_reason"] = (
                        f"Render A/B validation failed for model={spec.model} "
                        f"question_id={question_id}: {failure_reason}"
                    )
                    checkpoint["completed_at"] = utc_now()
                    write_checkpoint(output_path, checkpoint)
                    raise RuntimeError(checkpoint["block_reason"])

                if spec.model not in preflight_models:
                    preflight_models.add(spec.model)
                    checkpoint["preflight_results"].append(
                        {
                            "model": spec.model,
                            "question_id": question_id,
                            "measured_call_number": len(records),
                            "success_criteria_passed": True,
                        }
                    )
                write_checkpoint(output_path, checkpoint)

    if len(records) != RENDER_AB_TOTAL_CALLS:
        raise RuntimeError(
            "Render A/B mode did not produce exactly "
            f"{RENDER_AB_TOTAL_CALLS} measured calls."
        )
    return records


def run_latency_benchmark(
    settings: Settings,
    specs: Sequence[ModelSpec],
    providers: dict[str, RecordingProvider],
    questions_by_id: dict[str, EvaluationQuestion],
    chunks_by_id: dict[str, list[dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
    runs: int,
    checkpoint: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    manual_cases = [
        (question_id, questions_by_id[question_id].question)
        for question_id in MANUAL_QUESTION_IDS
    ]
    manual_cases.insert(2, (PERCENTAGE_CASE_ID, PERCENTAGE_QUESTION))
    records: list[dict[str, Any]] = []
    for run in range(1, runs + 1):
        for case_index, (question_id, question) in enumerate(manual_cases):
            offset = (run + case_index - 1) % len(specs)
            ordered_specs = [*specs[offset:], *specs[:offset]]
            for spec in ordered_specs:
                response, provider_call = application_call(
                    question,
                    chunks_by_id[question_id],
                    settings,
                    providers[spec.model],
                )
                record = latency_record(
                    spec.model,
                    question_id,
                    question,
                    run,
                    response,
                    provider_call,
                    manifests[question_id],
                    spec,
                )
                records.append(record)
                checkpoint["latency_records"] = records
                write_checkpoint(output_path, checkpoint)
                latency = (
                    provider_call.get("generation_ms")
                    if provider_call is not None
                    else None
                )
                retry = (
                    provider_call.get("retry_detected")
                    if provider_call is not None
                    else None
                )
                print(
                    f"LATENCY model={spec.model} run={run} "
                    f"question_id={question_id} ms={latency} "
                    f"status={response.get('answer_status')} "
                    f"fallback={response.get('fallback_used', False)} retry={retry}",
                    flush=True,
                )
    return records


def run_preflight(
    settings: Settings,
    specs: Sequence[ModelSpec],
    providers: dict[str, RecordingProvider],
    question: EvaluationQuestion,
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        response, provider_call = application_call(
            question.question,
            chunks,
            settings,
            providers[spec.model],
        )
        usage = provider_call.get("usage", {}) if provider_call else {}
        status_codes = provider_call.get("status_codes", []) if provider_call else []
        structured_response_returned = bool(
            provider_call
            and provider_call.get("provider_error_type") is None
            and response.get("fallback_used") is False
            and response.get("answer_provider") == OPENAI_ANSWER_PROVIDER
            and response.get("answer_status") in {"answered", "abstained"}
        )
        success_criteria_passed = bool(
            structured_response_returned
            and provider_call.get("attempt_count") == 1
            and status_codes == [200]
            and usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
        )
        result = {
            "model": spec.model,
            "question_id": question.question_id,
            "evidence_sha256": manifest["evidence_sha256"],
            "ordered_chunk_ids": manifest["ordered_chunk_ids"],
            "answer_status": response.get("answer_status"),
            "fallback_used": response.get("fallback_used", False),
            "fallback_reason": response.get("fallback_reason"),
            "provider_call": provider_call,
            "structured_response_returned": structured_response_returned,
            "success_criteria_passed": success_criteria_passed,
        }
        results.append(result)
        print(
            f"PREFLIGHT model={spec.model} "
            f"ms={provider_call.get('generation_ms') if provider_call else None} "
            f"attempts={provider_call.get('attempt_count') if provider_call else None} "
            f"http_status={status_codes[-1] if status_codes else None} "
            f"api_error_code="
            f"{provider_call.get('api_error_code') if provider_call else None} "
            f"fallback={response.get('fallback_used', False)}",
            flush=True,
        )
    return results


def quality_failures(
    question: EvaluationQuestion,
    evaluation: dict[str, Any],
) -> list[str]:
    failures = [
        failure
        for failure in evaluation["failures"]
        if failure in MODEL_FAILURES
    ]
    if question.expected_behavior == "cited_answer":
        if evaluation["answer"] == NO_ANSWER:
            failures.append("expected a grounded answer, got safe no-answer")
    elif question.expected_behavior == "analytics_answer":
        if not analytics_sources_are_sample_outputs(evaluation["sources"]):
            failures.append("expected sample analytics output source")
        if not source_matches_hint(
            evaluation["sources"],
            question.expected_source_hint,
        ):
            failures.append("missing expected analytics source hint")
    return list(dict.fromkeys(failures))


def run_quality_evaluation(
    settings: Settings,
    specs: Sequence[ModelSpec],
    providers: dict[str, RecordingProvider],
    questions: Sequence[EvaluationQuestion],
    chunks_by_id: dict[str, list[dict[str, Any]]],
    top_k: int,
    checkpoint: dict[str, Any],
    output_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    strategy = OFFLINE_STRATEGY
    results_by_model: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        provider = providers[spec.model]
        application_responses: dict[str, dict[str, Any]] = {}

        def responder(
            question_text: str,
            chunks: list[dict[str, Any]],
        ) -> dict[str, Any]:
            response, _provider_call = application_call(
                question_text,
                chunks,
                settings,
                provider,
            )
            application_responses[question_text] = response
            return response

        def cached_retriever(
            question_text: str,
            _strategy: StrategyDefinition,
        ) -> list[dict[str, Any]]:
            question = next(
                item for item in questions if item.question == question_text
            )
            return [dict(chunk) for chunk in chunks_by_id[question.question_id]]

        model_results: list[dict[str, Any]] = []
        for question in questions:
            call_count = len(provider.calls)
            evaluation = evaluate_strategy_question(
                question,
                strategy,
                cached_retriever,
                top_k,
                application_responder=responder,
                answer_profile=OPENAI_ANSWER_PROVIDER,
            )
            provider_call = (
                provider.calls[-1] if len(provider.calls) > call_count else None
            )
            application_response = application_responses.get(question.question, {})
            model_failures = quality_failures(question, evaluation)
            result = {
                **evaluation,
                "model_quality_failures": model_failures,
                "model_quality_passed": not model_failures,
                "fallback_reason": application_response.get("fallback_reason"),
                "provider_call": provider_call,
                "estimated_cost_usd": estimated_cost_usd(provider_call, spec),
            }
            model_results.append(result)
            results_by_model[spec.model] = model_results
            checkpoint["quality_results"] = results_by_model
            write_checkpoint(output_path, checkpoint)
            print(
                f"QUALITY model={spec.model} question_id={question.question_id} "
                f"passed={not model_failures} fallback="
                f"{result['answer_fallback_used']} failures={len(model_failures)}",
                flush=True,
            )
    return results_by_model


def summarize_latency(
    records: Sequence[dict[str, Any]],
    specs: Sequence[ModelSpec],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        model_records = [record for record in records if record["model"] == spec.model]
        latencies = [
            float(record["provider_call"]["generation_ms"])
            for record in model_records
            if record["provider_call"] is not None
        ]
        costs = [
            float(record["estimated_cost_usd"])
            for record in model_records
            if record["estimated_cost_usd"] is not None
        ]
        output_tokens = [
            int(record["provider_call"]["usage"]["output_tokens"])
            for record in model_records
            if record["provider_call"] is not None
            and record["provider_call"].get("usage", {}).get("output_tokens")
            is not None
        ]
        summaries[spec.model] = {
            "call_count": len(model_records),
            "measured_call_count": len(latencies),
            "median_generation_ms": round(statistics.median(latencies), 3),
            "mean_generation_ms": round(statistics.fmean(latencies), 3),
            "minimum_generation_ms": round(min(latencies), 3),
            "maximum_generation_ms": round(max(latencies), 3),
            "mean_output_tokens": (
                round(statistics.fmean(output_tokens), 3) if output_tokens else None
            ),
            "fallback_count": sum(record["fallback_used"] for record in model_records),
            "error_count": sum(
                bool(record["provider_call"])
                and bool(record["provider_call"].get("provider_error_type"))
                for record in model_records
            ),
            "retry_count": sum(
                bool(record["provider_call"])
                and bool(record["provider_call"].get("retry_detected"))
                for record in model_records
            ),
            "mean_estimated_cost_usd": (
                statistics.fmean(costs) if costs else None
            ),
        }
    baseline_cost = summaries["gpt-4o-mini"]["mean_estimated_cost_usd"]
    for summary in summaries.values():
        cost = summary["mean_estimated_cost_usd"]
        summary["estimated_relative_cost"] = (
            round(cost / baseline_cost, 4) if cost and baseline_cost else None
        )
    return summaries


def summarize_render_ab(
    records: Sequence[dict[str, Any]],
    specs: Sequence[ModelSpec],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        model_records = [record for record in records if record["model"] == spec.model]
        if len(model_records) != RENDER_AB_CALLS_PER_MODEL:
            raise RuntimeError(
                f"Render A/B model {spec.model} must have exactly "
                f"{RENDER_AB_CALLS_PER_MODEL} measured calls."
            )
        provider_calls = [record["provider_call"] for record in model_records]
        latencies = [float(call["generation_ms"]) for call in provider_calls]
        input_tokens = [int(call["usage"]["input_tokens"]) for call in provider_calls]
        output_tokens = [
            int(call["usage"]["output_tokens"]) for call in provider_calls
        ]
        summary = {
            "model": spec.model,
            "call_count": len(model_records),
            "median_generation_ms": round(statistics.median(latencies), 3),
            "mean_generation_ms": round(statistics.fmean(latencies), 3),
            "minimum_generation_ms": round(min(latencies), 3),
            "maximum_generation_ms": round(max(latencies), 3),
            "retry_count": sum(
                bool(call.get("retry_detected")) or call.get("attempt_count") != 1
                for call in provider_calls
            ),
            "error_count": sum(
                bool(call.get("provider_error_type") or call.get("raw_error_type"))
                for call in provider_calls
            ),
            "fallback_count": sum(
                bool(record.get("fallback_used")) for record in model_records
            ),
            "average_input_tokens": round(statistics.fmean(input_tokens), 3),
            "average_output_tokens": round(statistics.fmean(output_tokens), 3),
        }
        summaries[spec.model] = summary
    return summaries


def summarize_quality(
    results_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for model, results in results_by_model.items():
        cited = [result for result in results if result["citation_expected"]]
        safe = [result for result in results if result["safe_no_answer_expected"]]
        analytics = [
            result
            for result in results
            if result["expected_answer_behavior"] == "analytics_answer"
        ]
        passed = [result for result in results if result["model_quality_passed"]]
        summaries[model] = {
            "case_count": len(results),
            "passed_case_count": len(passed),
            "quality_pass_rate": len(passed) / len(results),
            "answered_cases_passed": sum(
                result["model_quality_passed"] for result in cited
            ),
            "answered_case_count": len(cited),
            "citation_valid_cases": sum(result["citation_valid"] for result in cited),
            "citation_case_count": len(cited),
            "safe_abstention_cases": sum(
                result["safe_no_answer_correct"] for result in safe
            ),
            "safe_abstention_case_count": len(safe),
            "analytics_cases_passed": sum(
                result["model_quality_passed"] for result in analytics
            ),
            "analytics_case_count": len(analytics),
            "malformed_structured_output_count": sum(
                result.get("fallback_reason") == "malformed_provider_response"
                for result in results
            ),
            "fallback_count": sum(result["answer_fallback_used"] for result in results),
            "failure_count": len(results) - len(passed),
            "failed_cases": [
                {
                    "question_id": result["question_id"],
                    "failures": result["model_quality_failures"],
                    "fallback_reason": result.get("fallback_reason"),
                }
                for result in results
                if not result["model_quality_passed"]
            ],
        }
    return summaries


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the selected-model structured-response preflight and stop.",
    )
    parser.add_argument(
        "--render-ab",
        action="store_true",
        help=(
            "Run the fixed 20-call Render latency-only comparison for "
            "gpt-4o-mini and gpt-5.6-luna."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.model for spec in MODEL_SPECS],
        default=None,
    )
    return parser.parse_args(argv)


def resolve_model_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.render_ab:
        if args.preflight_only:
            raise ValueError("--render-ab cannot be combined with --preflight-only.")
        if args.runs != RENDER_AB_RUNS:
            raise ValueError(
                f"--render-ab requires exactly {RENDER_AB_RUNS} rounds per case."
            )
        if args.top_k != DEFAULT_TOP_K:
            raise ValueError(
                f"--render-ab requires exactly {DEFAULT_TOP_K} full chunks per case."
            )
        if args.min_similarity != DEFAULT_MIN_SIMILARITY:
            raise ValueError(
                "--render-ab requires the existing default retrieval threshold."
            )
        if args.models is not None and (
            len(args.models) != len(RENDER_AB_MODELS)
            or set(args.models) != set(RENDER_AB_MODELS)
        ):
            raise ValueError(
                "--render-ab allows exactly gpt-4o-mini and gpt-5.6-luna."
            )
        return RENDER_AB_MODELS

    return tuple(args.models or (spec.model for spec in MODEL_SPECS))


def main() -> int:
    args = parse_args()
    model_names = resolve_model_names(args)
    if not args.preflight_only and args.runs < 5:
        raise ValueError("The benchmark requires at least five runs per latency case.")

    settings = Settings.from_env()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this isolated benchmark.")
    specs = [spec for spec in MODEL_SPECS if spec.model in model_names]
    if not specs:
        raise RuntimeError("At least one model is required.")
    if "gpt-4o-mini" not in model_names:
        raise RuntimeError("The current gpt-4o-mini baseline must be included.")

    questions = load_evaluation_questions(DEFAULT_EVALUATION_PATH)
    questions_by_id = {question.question_id: question for question in questions}
    evidence_questions = (
        [questions_by_id[question_id] for question_id in RENDER_AB_QUESTION_IDS]
        if args.render_ab
        else questions
    )
    chunks_by_id, manifests = precompute_evidence(
        evidence_questions,
        args.top_k,
        args.min_similarity,
        include_percentage_case=not args.render_ab,
    )
    providers = {
        spec.model: RecordingProvider(settings, spec)
        for spec in specs
    }
    started_at = utc_now()
    output_path = args.output.resolve()
    checkpoint: dict[str, Any] = {
        "schema_version": "civiclens-openai-answer-model-benchmark-v1",
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
        "block_reason": None,
        "methodology": {
            "models": [asdict(spec) for spec in specs],
            "latency_runs_per_case": args.runs,
            "latency_case_ids": (
                list(RENDER_AB_QUESTION_IDS)
                if args.render_ab
                else [
                    "q010",
                    "q019",
                    PERCENTAGE_CASE_ID,
                    "q022",
                    "q023",
                ]
            ),
            "quality_dataset": str(
                DEFAULT_EVALUATION_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "quality_case_count": 0 if args.render_ab else len(questions),
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "retrieval_source": (
                "fixed checked-in full chunks: dataset-declared relevant sections "
                "plus deterministic top-k fillers"
            ),
            "retrieval_mode": OFFLINE_STRATEGY.retrieval_mode,
            "reranking_enabled": OFFLINE_STRATEGY.reranking_enabled,
            "answer_timeout_seconds": settings.answer_timeout_seconds,
            "benchmark_max_retries": BENCHMARK_MAX_RETRIES,
            "production_answer_max_retries": settings.answer_max_retries,
            "production_default_model": settings.answer_model,
            "provider_creates_client_per_generation": True,
            "identical_evidence_per_question_across_models": True,
            "evidence_truncated": False,
            "prompt_sha256": "sha256:"
            + hashlib.sha256(APPLICATION_RULES.encode("utf-8")).hexdigest(),
            "structured_output_schema": OpenAIStructuredAnswer.model_json_schema(),
            "responses_parse_supports_reasoning": (
                "reasoning"
                in inspect.signature(OpenAI(api_key="benchmark-inspection").responses.parse)
                .parameters
            ),
            "render_ab": args.render_ab,
            "preflight_in_first_measured_call": args.render_ab,
        },
        "evidence_manifests": manifests,
        "preflight_results": [],
        "latency_records": [],
        "quality_results": {},
        "latency_summary": {},
        "quality_summary": {},
    }
    write_checkpoint(output_path, checkpoint)

    if args.render_ab:
        latency_records = run_render_ab_benchmark(
            settings,
            specs,
            providers,
            questions_by_id,
            chunks_by_id,
            manifests,
            checkpoint,
            output_path,
        )
        latency_summary = summarize_render_ab(latency_records, specs)
        checkpoint["latency_summary"] = latency_summary
        checkpoint["status"] = "completed"
        checkpoint["completed_at"] = utc_now()
        write_checkpoint(output_path, checkpoint)
        for spec in specs:
            print_render_ab_result(
                "RENDER_AB_SUMMARY",
                latency_summary[spec.model],
            )
        print_render_ab_result(
            "RENDER_AB_COMPLETE",
            {"measured_call_count": len(latency_records)},
        )
        return 0

    preflight_question = questions_by_id["q010"]
    preflight_results = run_preflight(
        settings,
        specs,
        providers,
        preflight_question,
        chunks_by_id[preflight_question.question_id],
        manifests[preflight_question.question_id],
    )
    checkpoint["preflight_results"] = preflight_results
    failed_preflights = [
        result
        for result in preflight_results
        if not result["success_criteria_passed"]
    ]
    if failed_preflights:
        error_codes = sorted(
            {
                str(result["provider_call"].get("api_error_code"))
                for result in failed_preflights
                if result["provider_call"] is not None
                and result["provider_call"].get("api_error_code") is not None
            }
        )
        checkpoint["status"] = "blocked"
        checkpoint["block_reason"] = (
            "OpenAI provider preflight failed before benchmarking"
            + (f": {', '.join(error_codes)}" if error_codes else "")
        )
        checkpoint["completed_at"] = utc_now()
        write_checkpoint(output_path, checkpoint)
        print(
            f"BLOCKED reason={checkpoint['block_reason']} output={output_path}",
            flush=True,
        )
        return 2

    if args.preflight_only:
        checkpoint["status"] = "preflight_completed"
        checkpoint["completed_at"] = utc_now()
        write_checkpoint(output_path, checkpoint)
        print(f"PREFLIGHT_COMPLETE output={output_path}", flush=True)
        return 0

    latency_records = run_latency_benchmark(
        settings,
        specs,
        providers,
        questions_by_id,
        chunks_by_id,
        manifests,
        args.runs,
        checkpoint,
        output_path,
    )
    quality_results = run_quality_evaluation(
        settings,
        specs,
        providers,
        questions,
        chunks_by_id,
        args.top_k,
        checkpoint,
        output_path,
    )
    checkpoint["latency_summary"] = summarize_latency(latency_records, specs)
    checkpoint["quality_summary"] = summarize_quality(quality_results)
    checkpoint["status"] = "completed"
    checkpoint["completed_at"] = utc_now()
    write_checkpoint(output_path, checkpoint)
    print(f"COMPLETE output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
