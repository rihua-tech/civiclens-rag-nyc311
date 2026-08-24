"""Allowlisted CSV-backed analytics tool implementations."""

from __future__ import annotations

import csv
from pathlib import Path

from src.analytics.tools.base import (
    AgencyRequestVolumeInput,
    AgencyRequestVolumeRow,
    AnalyticsProvenance,
    AnalyticsTool,
    AnalyticsToolId,
    AnalyticsToolResult,
    BacklogSummaryInput,
    BacklogSummaryRow,
    BoroughRequestVolumeInput,
    BoroughRequestVolumeRow,
    ComplaintTypeRow,
    TopComplaintTypesInput,
)


SAMPLE_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "sample_outputs"
ALLOWED_SAMPLE_OUTPUT_FILES = frozenset(
    {
        "top_complaint_types.csv",
        "requests_by_borough.csv",
        "agency_request_volume.csv",
        "backlog_summary.csv",
    }
)


def load_sample_output(file_name: str) -> list[dict[str, str]]:
    """Read one fixed checked-in sample file without accepting arbitrary paths."""

    if file_name not in ALLOWED_SAMPLE_OUTPUT_FILES:
        raise ValueError("Unsupported sample analytics source.")

    path = SAMPLE_OUTPUT_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Sample analytics output not found: {file_name}")

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _parse_count(value: str) -> int:
    return int(value.replace(",", "").strip())


def _format_count(value: str | int) -> str:
    return f"{int(value):,}"


def _provenance(file_name: str) -> list[AnalyticsProvenance]:
    return [
        AnalyticsProvenance(
            source_name=file_name,
            source_path=f"data/sample_outputs/{file_name}",
        )
    ]


class _TopComplaintTypesTool(
    AnalyticsTool[TopComplaintTypesInput, ComplaintTypeRow]
):
    tool_id = AnalyticsToolId.TOP_COMPLAINT_TYPES
    tool_name = "Top complaint types"
    input_schema = TopComplaintTypesInput

    def _run(
        self,
        arguments: TopComplaintTypesInput,
    ) -> AnalyticsToolResult[ComplaintTypeRow]:
        rows = [
            ComplaintTypeRow(
                complaint_type=row["complaint_type"],
                request_count=_parse_count(row["request_count"]),
            )
            for row in load_sample_output("top_complaint_types.csv")
        ][: arguments.limit]
        summary = "; ".join(
            f"{row.complaint_type} ({_format_count(row.request_count)})"
            for row in rows[:5]
        )
        return AnalyticsToolResult[ComplaintTypeRow](
            tool_id=self.tool_id,
            tool_name=self.tool_name,
            summary=f"The top sample complaint types are: {summary}.",
            rows=rows,
            provenance=_provenance("top_complaint_types.csv"),
        )


class _BoroughRequestVolumeTool(
    AnalyticsTool[BoroughRequestVolumeInput, BoroughRequestVolumeRow]
):
    tool_id = AnalyticsToolId.BOROUGH_REQUEST_VOLUME
    tool_name = "Borough request volume"
    input_schema = BoroughRequestVolumeInput

    def _run(
        self,
        arguments: BoroughRequestVolumeInput,
    ) -> AnalyticsToolResult[BoroughRequestVolumeRow]:
        rows = sorted(
            (
                BoroughRequestVolumeRow(
                    borough=row["borough"],
                    request_count=_parse_count(row["request_count"]),
                )
                for row in load_sample_output("requests_by_borough.csv")
            ),
            key=lambda row: row.request_count,
            reverse=True,
        )[: arguments.limit]
        leader = rows[0]
        summary = "; ".join(
            f"{row.borough} ({_format_count(row.request_count)})" for row in rows[:5]
        )
        answer = (
            f"{leader.borough} has the highest sample complaint volume "
            f"with {_format_count(leader.request_count)} requests. "
            f"Borough totals: {summary}."
        )
        return AnalyticsToolResult[BoroughRequestVolumeRow](
            tool_id=self.tool_id,
            tool_name=self.tool_name,
            summary=answer,
            rows=rows,
            provenance=_provenance("requests_by_borough.csv"),
        )


class _AgencyRequestVolumeTool(
    AnalyticsTool[AgencyRequestVolumeInput, AgencyRequestVolumeRow]
):
    tool_id = AnalyticsToolId.AGENCY_REQUEST_VOLUME
    tool_name = "Agency request volume"
    input_schema = AgencyRequestVolumeInput

    def _run(
        self,
        arguments: AgencyRequestVolumeInput,
    ) -> AnalyticsToolResult[AgencyRequestVolumeRow]:
        rows = sorted(
            (
                AgencyRequestVolumeRow(
                    agency=row["agency"],
                    agency_name=row["agency_name"],
                    request_count=_parse_count(row["request_count"]),
                )
                for row in load_sample_output("agency_request_volume.csv")
            ),
            key=lambda row: row.request_count,
            reverse=True,
        )[: arguments.limit]
        summary = "; ".join(
            f"{row.agency} ({_format_count(row.request_count)})" for row in rows[:5]
        )
        return AnalyticsToolResult[AgencyRequestVolumeRow](
            tool_id=self.tool_id,
            tool_name=self.tool_name,
            summary=f"The agencies handling the most sample requests are: {summary}.",
            rows=rows,
            provenance=_provenance("agency_request_volume.csv"),
        )


class _BacklogSummaryTool(AnalyticsTool[BacklogSummaryInput, BacklogSummaryRow]):
    tool_id = AnalyticsToolId.BACKLOG_SUMMARY
    tool_name = "Backlog summary"
    input_schema = BacklogSummaryInput

    def _run(
        self,
        arguments: BacklogSummaryInput,
    ) -> AnalyticsToolResult[BacklogSummaryRow]:
        del arguments
        rows = [
            BacklogSummaryRow(
                status=row["status"],
                request_count=_parse_count(row["request_count"]),
            )
            for row in load_sample_output("backlog_summary.csv")
        ]
        counts = {row.status.lower(): row.request_count for row in rows}
        answer = (
            "The sample backlog summary shows "
            f"{_format_count(counts.get('open', 0))} open requests, "
            f"{_format_count(counts.get('in progress', 0))} in progress, "
            f"{_format_count(counts.get('overdue', 0))} overdue, and "
            f"{_format_count(counts.get('closed last 7 days', 0))} "
            "closed in the last 7 days."
        )
        return AnalyticsToolResult[BacklogSummaryRow](
            tool_id=self.tool_id,
            tool_name=self.tool_name,
            summary=answer,
            rows=rows,
            provenance=_provenance("backlog_summary.csv"),
        )


def build_analytics_tools() -> tuple[AnalyticsTool, ...]:
    """Build the complete fixed registry inventory."""

    return (
        _TopComplaintTypesTool(),
        _BoroughRequestVolumeTool(),
        _AgencyRequestVolumeTool(),
        _BacklogSummaryTool(),
    )
