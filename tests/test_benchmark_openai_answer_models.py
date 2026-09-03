from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import benchmark_openai_answer_models as benchmark


def successful_provider_call(call_number: int) -> dict[str, object]:
    return {
        "attempt_count": 1,
        "retry_detected": False,
        "request_id": f"req_{call_number}",
        "status_codes": [200],
        "raw_error_type": None,
        "provider_error_type": None,
        "generation_ms": float(call_number),
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 0,
        },
    }


def successful_response() -> dict[str, object]:
    return {
        "answer": "Grounded answer.",
        "answer_provider": "openai",
        "answer_status": "answered",
        "citation_ids": ["chunk_1"],
        "rejected_citation_ids": [],
        "fallback_used": False,
        "fallback_reason": None,
    }


def render_ab_inputs() -> tuple[dict[str, object], dict[str, object]]:
    chunks_by_id = {
        question_id: [
            {"chunk_id": f"{question_id}_chunk_{index}"}
            for index in range(benchmark.DEFAULT_TOP_K)
        ]
        for question_id in benchmark.RENDER_AB_QUESTION_IDS
    }
    manifests = {
        question_id: {
            "chunk_count": benchmark.DEFAULT_TOP_K,
            "usable_evidence_count": benchmark.DEFAULT_TOP_K,
            "ordered_chunk_ids": [
                chunk["chunk_id"] for chunk in chunks_by_id[question_id]
            ],
            "evidence_sha256": f"sha256:{question_id}",
        }
        for question_id in benchmark.RENDER_AB_QUESTION_IDS
    }
    return chunks_by_id, manifests


def test_render_ab_cli_resolves_only_qualified_models_and_fixed_shape() -> None:
    args = benchmark.parse_args(["--render-ab"])

    assert benchmark.resolve_model_names(args) == benchmark.RENDER_AB_MODELS
    assert benchmark.RENDER_AB_QUESTION_IDS == ("q010", "q019")
    assert benchmark.RENDER_AB_RUNS == 5
    assert benchmark.RENDER_AB_TOTAL_CALLS == 20
    assert benchmark.BENCHMARK_MAX_RETRIES == 0
    specs = {spec.model: spec for spec in benchmark.MODEL_SPECS}
    assert specs["gpt-4o-mini"].reasoning is None
    assert specs["gpt-5.6-luna"].reasoning == {"effort": "none"}

    invalid_args = benchmark.parse_args(
        [
            "--render-ab",
            "--models",
            "gpt-4o-mini",
            "gpt-4.1-nano",
            "gpt-5.6-luna",
        ]
    )
    with pytest.raises(ValueError, match="allows exactly"):
        benchmark.resolve_model_names(invalid_args)


def test_render_ab_runs_exactly_twenty_interleaved_measured_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str]] = []
    providers = {
        model: SimpleNamespace(model_name=model, calls=[])
        for model in benchmark.RENDER_AB_MODELS
    }
    question_text_by_id = {
        "q010": "What does complaint_type mean?",
        "q019": "What does descriptor mean?",
    }
    question_id_by_text = {
        question: question_id for question_id, question in question_text_by_id.items()
    }

    def fake_application_call(question, _chunks, _settings, provider):
        question_id = question_id_by_text[question]
        calls.append((question_id, provider.model_name))
        provider_call = successful_provider_call(len(calls))
        provider.calls.append(provider_call)
        return successful_response(), provider_call

    monkeypatch.setattr(benchmark, "application_call", fake_application_call)
    monkeypatch.setattr(benchmark, "write_checkpoint", lambda *_args: None)
    chunks_by_id, manifests = render_ab_inputs()
    questions_by_id = {
        question_id: SimpleNamespace(question=question)
        for question_id, question in question_text_by_id.items()
    }
    specs = tuple(
        spec
        for spec in benchmark.MODEL_SPECS
        if spec.model in benchmark.RENDER_AB_MODELS
    )
    checkpoint = {"latency_records": [], "preflight_results": []}

    records = benchmark.run_render_ab_benchmark(
        SimpleNamespace(),
        specs,
        providers,
        questions_by_id,
        chunks_by_id,
        manifests,
        checkpoint,
        tmp_path / "render-ab.json",
    )

    assert len(records) == 20
    assert [record["model"] for record in records].count("gpt-4o-mini") == 10
    assert [record["model"] for record in records].count("gpt-5.6-luna") == 10
    assert calls[:8] == [
        ("q010", "gpt-4o-mini"),
        ("q010", "gpt-5.6-luna"),
        ("q019", "gpt-5.6-luna"),
        ("q019", "gpt-4o-mini"),
        ("q010", "gpt-5.6-luna"),
        ("q010", "gpt-4o-mini"),
        ("q019", "gpt-4o-mini"),
        ("q019", "gpt-5.6-luna"),
    ]
    assert checkpoint["preflight_results"] == [
        {
            "model": "gpt-4o-mini",
            "question_id": "q010",
            "measured_call_number": 1,
            "success_criteria_passed": True,
        },
        {
            "model": "gpt-5.6-luna",
            "question_id": "q010",
            "measured_call_number": 2,
            "success_criteria_passed": True,
        },
    ]

    summaries = benchmark.summarize_render_ab(records, specs)
    assert summaries["gpt-4o-mini"]["call_count"] == 10
    assert summaries["gpt-5.6-luna"]["call_count"] == 10
    assert summaries["gpt-4o-mini"]["average_input_tokens"] == 100.0
    assert summaries["gpt-5.6-luna"]["average_output_tokens"] == 20.0

    output = capsys.readouterr().out
    assert output.count("RENDER_AB_CALL ") == 20
    assert "What does complaint_type mean?" not in output
    assert "What does descriptor mean?" not in output
    assert "retrieved_evidence_untrusted" not in output
    assert '"request_id":"req_1"' in output
    assert '"provider_error":null' in output


def test_render_ab_stops_after_first_invalid_measured_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    providers = {
        model: SimpleNamespace(model_name=model, calls=[])
        for model in benchmark.RENDER_AB_MODELS
    }
    call_count = 0

    def fake_application_call(_question, _chunks, _settings, provider):
        nonlocal call_count
        call_count += 1
        provider_call = successful_provider_call(call_count)
        provider_call["usage"]["input_tokens"] = None
        provider.calls.append(provider_call)
        return successful_response(), provider_call

    monkeypatch.setattr(benchmark, "application_call", fake_application_call)
    monkeypatch.setattr(benchmark, "write_checkpoint", lambda *_args: None)
    chunks_by_id, manifests = render_ab_inputs()
    questions_by_id = {
        "q010": SimpleNamespace(question="complaint_type"),
        "q019": SimpleNamespace(question="descriptor"),
    }
    specs = tuple(
        spec
        for spec in benchmark.MODEL_SPECS
        if spec.model in benchmark.RENDER_AB_MODELS
    )
    checkpoint = {"latency_records": [], "preflight_results": []}

    with pytest.raises(RuntimeError, match="token usage is missing"):
        benchmark.run_render_ab_benchmark(
            SimpleNamespace(),
            specs,
            providers,
            questions_by_id,
            chunks_by_id,
            manifests,
            checkpoint,
            tmp_path / "render-ab.json",
        )

    assert call_count == 1
    assert len(checkpoint["latency_records"]) == 1
    assert checkpoint["status"] == "blocked"


@pytest.mark.parametrize(
    ("response_updates", "call_updates", "expected"),
    [
        ({"fallback_reason": "malformed_provider_response"}, {}, "malformed"),
        ({}, {"provider_error_type": "ProviderUnavailableError"}, "provider error"),
        ({"fallback_used": True}, {}, "fallback"),
        ({}, {"usage": {"input_tokens": 100, "output_tokens": None}}, "token"),
        ({}, {"status_codes": [500]}, "HTTP status"),
        ({}, {"attempt_count": 2}, "attempt count"),
    ],
)
def test_render_ab_validation_rejects_unsafe_results(
    response_updates: dict[str, object],
    call_updates: dict[str, object],
    expected: str,
) -> None:
    response = successful_response()
    response.update(response_updates)
    provider_call = successful_provider_call(1)
    provider_call.update(deepcopy(call_updates))
    compact_call = benchmark.compact_render_ab_call(
        "gpt-4o-mini",
        "q010",
        provider_call,
        bool(response["fallback_used"]),
    )

    failure = benchmark.render_ab_failure_reason(
        response,
        provider_call,
        compact_call,
    )

    assert failure is not None
    assert expected in failure
