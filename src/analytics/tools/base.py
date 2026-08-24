"""Typed contracts for fixed, read-only CivicLens analytics tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_ANALYTICS_RESULT_LIMIT = 10
MAX_ANALYTICS_RESULT_LIMIT = 10
MAX_ANALYTICS_TOOL_ID_LENGTH = 64
SAMPLE_DATA_DISCLAIMER = (
    "Sample analytics answer from checked-in CSV outputs only; "
    "not live NYC 311 data and not a production text-to-SQL result."
)


class AnalyticsToolId(StrEnum):
    """Stable identifiers for the complete Issue 16 tool allowlist."""

    TOP_COMPLAINT_TYPES = "top_complaint_types"
    BOROUGH_REQUEST_VOLUME = "borough_request_volume"
    AGENCY_REQUEST_VOLUME = "agency_request_volume"
    BACKLOG_SUMMARY = "backlog_summary"


class StrictToolModel(BaseModel):
    """Base model that rejects coercion, unknown fields, and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RankedAnalyticsInput(StrictToolModel):
    """Shared bounded input for ranked sample-output tools."""

    limit: int = Field(
        default=DEFAULT_ANALYTICS_RESULT_LIMIT,
        ge=1,
        le=MAX_ANALYTICS_RESULT_LIMIT,
    )


class TopComplaintTypesInput(RankedAnalyticsInput):
    """Arguments accepted by the top-complaint-types tool."""


class BoroughRequestVolumeInput(RankedAnalyticsInput):
    """Arguments accepted by the borough-volume tool."""


class AgencyRequestVolumeInput(RankedAnalyticsInput):
    """Arguments accepted by the agency-volume tool."""


class BacklogSummaryInput(StrictToolModel):
    """The fixed backlog summary accepts no user-selected parameters."""


class AnalyticsRow(StrictToolModel):
    """Marker base for typed tool-specific rows."""


class ComplaintTypeRow(AnalyticsRow):
    complaint_type: str
    request_count: int = Field(ge=0)


class BoroughRequestVolumeRow(AnalyticsRow):
    borough: str
    request_count: int = Field(ge=0)


class AgencyRequestVolumeRow(AnalyticsRow):
    agency: str
    agency_name: str
    request_count: int = Field(ge=0)


class BacklogSummaryRow(AnalyticsRow):
    status: str
    request_count: int = Field(ge=0)


class AnalyticsProvenance(StrictToolModel):
    """Application-owned provenance for an allowlisted sample source."""

    source_name: str
    source_path: str
    chunk_id: Literal["sample_output"] = "sample_output"
    source_type: Literal["checked_in_sample_csv"] = "checked_in_sample_csv"
    source_timestamp: datetime | None = None


RowT = TypeVar("RowT", bound=AnalyticsRow)
InputT = TypeVar("InputT", bound=StrictToolModel)


class AnalyticsToolResult(StrictToolModel, Generic[RowT]):
    """Provider-neutral structured result returned by every analytics tool."""

    tool_id: AnalyticsToolId
    tool_name: str
    summary: str
    rows: list[RowT] = Field(max_length=MAX_ANALYTICS_RESULT_LIMIT)
    provenance: list[AnalyticsProvenance] = Field(min_length=1)
    disclaimer: str = SAMPLE_DATA_DISCLAIMER


class UnknownAnalyticsToolError(ValueError):
    """Raised when a caller requests an ID outside the fixed registry."""


class InvalidAnalyticsToolArgumentsError(ValueError):
    """Raised when strict tool input validation fails."""


class AnalyticsTool(ABC, Generic[InputT, RowT]):
    """Common interface for fixed analytics tools.

    Tool implementations receive only their validated schema. Registry lookup
    and invocation are intentionally separate from natural-language routing.
    """

    tool_id: AnalyticsToolId
    tool_name: str
    input_schema: type[InputT]

    def invoke(
        self,
        arguments: Mapping[str, Any] | None = None,
    ) -> AnalyticsToolResult[RowT]:
        if arguments is None:
            payload: dict[str, Any] = {}
        elif isinstance(arguments, Mapping):
            payload = dict(arguments)
        else:
            raise InvalidAnalyticsToolArgumentsError(
                "Analytics tool arguments must be a mapping."
            )

        try:
            validated = self.input_schema.model_validate(payload, strict=True)
        except ValidationError as exc:
            raise InvalidAnalyticsToolArgumentsError(
                "Analytics tool arguments failed strict validation."
            ) from exc

        return self._run(validated)

    @abstractmethod
    def _run(self, arguments: InputT) -> AnalyticsToolResult[RowT]:
        """Execute one trusted, read-only operation with validated arguments."""
