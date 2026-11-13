"""Markdown and JSON reporting for reproducible CivicLens evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_metric(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    if value is None:
        return "N/A (0 eligible)"
    return f"{float(value):.4f} (n={int(metric['denominator'])})"


def markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# CivicLens RAG Evaluation Results",
        "",
        f"- Evaluation profile: `{report['evaluation_profile']}`",
        f"- Dataset version: `{dataset['version']}`",
        f"- Evaluation timestamp: `{report['evaluation_timestamp']}`",
        f"- Questions: {dataset['question_count']}",
        f"- Retrieval relevance granularity: `{dataset['relevance_granularity']}`",
        f"- Result schema version: `{report['schema_version']}`",
        "",
        "## Interpretation Boundary",
        "",
        report["interpretation_boundary"],
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Recall@k | MRR | Expected-source retrieval | Routing | Citations present | Citations valid | Abstention recall | False abstention | Abstention integrity | Unsupported answers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    answer_evaluation = report.get("answer_evaluation")
    if answer_evaluation:
        lines[lines.index("## Strategy Comparison"):lines.index("## Strategy Comparison")] = [
            "## Optional Answer-Provider Evaluation",
            "",
            f"- Profile: `{answer_evaluation['profile']}`",
            f"- Provider: `{answer_evaluation['provider']}`",
            f"- Model: `{answer_evaluation['model']}`",
            "- Results are separate from the deterministic Issue 10 baseline.",
            "",
        ]

    for strategy in report["strategies"]:
        retrieval = strategy["retrieval_metrics"]
        application = strategy["application_metrics"]
        unsupported = application["unsupported_answer"]
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy["name"],
                    format_metric(retrieval["recall_at_k"]),
                    format_metric(retrieval["mrr"]),
                    format_metric(retrieval["expected_source_retrieval"]),
                    format_metric(application["routing_accuracy"]),
                    format_metric(application["citation_presence"]),
                    format_metric(application["citation_validity"]),
                    format_metric(application["abstention_recall"]),
                    format_metric(application["false_abstention_rate"]),
                    format_metric(application["abstention_integrity"]),
                    f"{unsupported['count']}/{unsupported['denominator']} ({unsupported['rate']:.4f})",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"Recall@{report['top_k']} is calculated per eligible question as "
            "`|relevant IDs intersect retrieved IDs at k| / |relevant IDs|`, then macro-averaged.",
            "MRR uses the reciprocal rank of the first relevant result and zero when none is retrieved, then macro-averages eligible questions.",
            "Questions without retrieval relevance labels are excluded from Recall@k, MRR, and expected-source denominators.",
            "Expected-source retrieval checks retrieved document IDs independently from section-level relevance.",
            "Application metrics are reported separately and prefer structured answer_status. Exact no-answer text is used only as a labeled legacy fallback when status is absent.",
            "Abstention recall uses expected-abstention questions; false-abstention rate uses answerable questions. Abstention integrity requires no sources or citation artifacts.",
            "Citation ID validity is application-owned but is not proof that the answer text is semantically supported; manual review remains separate.",
            "",
            "## Reproducibility Configuration",
            "",
        ]
    )

    for strategy in report["strategies"]:
        lines.append(f"### {strategy['name']}")
        lines.append("")
        for key, value in strategy["configuration"].items():
            lines.append(f"- `{key}`: `{value}`")
        provider = strategy["application_metrics"]["provider_and_fallback"]
        lines.append(f"- `answer_provider_counts`: `{provider['answer_provider_counts']}`")
        lines.append(f"- `provider_called_count`: `{provider['provider_called_count']}`")
        lines.append(f"- `fallback_count`: `{provider['fallback_count']}`")
        lines.append(f"- `fallback_rate`: `{provider['fallback_rate']}`")
        lines.append("")

    lines.extend(["## Failed Cases", ""])
    failed_cases = [
        failed_case
        for strategy in report["strategies"]
        for failed_case in strategy["failed_cases"]
    ]
    if not failed_cases:
        lines.append("No deterministic checks failed.")
    else:
        for failed_case in failed_cases:
            lines.extend(
                [
                    f"### {failed_case['strategy']} / {failed_case['question_id']}",
                    "",
                    f"- Question: {failed_case['question']}",
                    f"- Failures: {', '.join(failed_case['failures'])}",
                    f"- Expected route/behavior: `{failed_case['expected_route']}` / `{failed_case['expected_answer_behavior']}`",
                    f"- Actual route: `{failed_case['actual_route']}`",
                    f"- Retrieved relevance IDs: `{failed_case['retrieved_relevance_ids']}`",
                    f"- Retrieved source IDs: `{failed_case['retrieved_source_document_ids']}`",
                    "",
                ]
            )

    lines.extend(["## Known Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report: dict[str, Any],
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write disposable Markdown and JSON reports without touching the baseline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown_path = output_path / f"{stem}.md"
    json_path = output_path / f"{stem}.json"
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown_path, json_path
