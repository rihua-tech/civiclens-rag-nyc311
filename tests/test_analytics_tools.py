from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ValidationError

from src.analytics.simple_analytics import answer_analytics_question
from src.analytics.tools import (
    ANALYTICS_TOOL_REGISTRY,
    DEFAULT_ANALYTICS_RESULT_LIMIT,
    MAX_ANALYTICS_RESULT_LIMIT,
    SAMPLE_DATA_DISCLAIMER,
    AgencyRequestVolumeRow,
    AnalyticsToolId,
    AnalyticsToolResult,
    BacklogSummaryRow,
    BoroughRequestVolumeRow,
    ComplaintTypeRow,
    execute_analytics_tool,
    registered_tool_ids,
)


EXPECTED_TOOL_IDS = (
    "top_complaint_types",
    "borough_request_volume",
    "agency_request_volume",
    "backlog_summary",
)


def test_all_supported_tools_are_explicitly_and_uniquely_registered():
    assert registered_tool_ids() == EXPECTED_TOOL_IDS
    assert len(set(registered_tool_ids())) == 4
    assert isinstance(ANALYTICS_TOOL_REGISTRY, Mapping)
    assert {tool.tool_id.value for tool in ANALYTICS_TOOL_REGISTRY.values()} == set(
        EXPECTED_TOOL_IDS
    )
    assert all(
        issubclass(tool.input_schema, BaseModel)
        for tool in ANALYTICS_TOOL_REGISTRY.values()
    )
    with pytest.raises(TypeError):
        ANALYTICS_TOOL_REGISTRY[AnalyticsToolId.TOP_COMPLAINT_TYPES] = (
            ANALYTICS_TOOL_REGISTRY[AnalyticsToolId.TOP_COMPLAINT_TYPES]
        )


@pytest.mark.parametrize(
    ("tool_id", "arguments", "row_type", "source_path"),
    [
        (
            AnalyticsToolId.TOP_COMPLAINT_TYPES,
            {"limit": 3},
            ComplaintTypeRow,
            "data/sample_outputs/top_complaint_types.csv",
        ),
        (
            AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
            {"limit": 3},
            BoroughRequestVolumeRow,
            "data/sample_outputs/requests_by_borough.csv",
        ),
        (
            AnalyticsToolId.AGENCY_REQUEST_VOLUME,
            {"limit": 3},
            AgencyRequestVolumeRow,
            "data/sample_outputs/agency_request_volume.csv",
        ),
        (
            AnalyticsToolId.BACKLOG_SUMMARY,
            {},
            BacklogSummaryRow,
            "data/sample_outputs/backlog_summary.csv",
        ),
    ],
)
def test_valid_calls_return_typed_structured_rows_and_provenance(
    tool_id,
    arguments,
    row_type,
    source_path,
):
    result = execute_analytics_tool(tool_id, arguments)

    assert isinstance(result, AnalyticsToolResult)
    assert result.tool_id is tool_id
    assert result.tool_name
    assert result.summary
    assert result.rows
    assert all(isinstance(row, row_type) for row in result.rows)
    assert len(result.rows) <= MAX_ANALYTICS_RESULT_LIMIT
    assert result.provenance[0].source_path == source_path
    assert result.provenance[0].source_name == source_path.rsplit("/", 1)[-1]
    assert result.provenance[0].source_timestamp is None
    assert result.disclaimer == SAMPLE_DATA_DISCLAIMER


@pytest.mark.parametrize(
    "tool_id",
    [
        AnalyticsToolId.TOP_COMPLAINT_TYPES,
        AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
        AnalyticsToolId.AGENCY_REQUEST_VOLUME,
    ],
)
def test_ranked_tool_result_counts_are_bounded_by_validated_limit(tool_id):
    one = execute_analytics_tool(tool_id, {"limit": 1})
    maximum = execute_analytics_tool(
        tool_id,
        {"limit": MAX_ANALYTICS_RESULT_LIMIT},
    )
    default = execute_analytics_tool(tool_id)

    assert len(one.rows) == 1
    assert len(maximum.rows) <= MAX_ANALYTICS_RESULT_LIMIT
    assert len(default.rows) <= DEFAULT_ANALYTICS_RESULT_LIMIT


def test_tool_results_are_immutable():
    result = execute_analytics_tool(AnalyticsToolId.TOP_COMPLAINT_TYPES)

    with pytest.raises(ValidationError):
        result.summary = "changed"


@pytest.mark.parametrize(
    ("question", "expected_tool_id", "expected_arguments", "source_name"),
    [
        (
            "What are the top complaint types?",
            AnalyticsToolId.TOP_COMPLAINT_TYPES,
            {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
            "top_complaint_types.csv",
        ),
        (
            "Which borough has the highest complaint volume?",
            AnalyticsToolId.BOROUGH_REQUEST_VOLUME,
            {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
            "requests_by_borough.csv",
        ),
        (
            "Which agencies handle the most requests?",
            AnalyticsToolId.AGENCY_REQUEST_VOLUME,
            {"limit": DEFAULT_ANALYTICS_RESULT_LIMIT},
            "agency_request_volume.csv",
        ),
        (
            "What is the backlog summary?",
            AnalyticsToolId.BACKLOG_SUMMARY,
            {},
            "backlog_summary.csv",
        ),
    ],
)
def test_existing_analytics_router_executes_only_fixed_registered_tools(
    monkeypatch,
    question,
    expected_tool_id,
    expected_arguments,
    source_name,
):
    captured = {}
    real_execute = execute_analytics_tool

    def capture(tool_id, arguments):
        captured.update(tool_id=tool_id, arguments=arguments)
        return real_execute(tool_id, arguments)

    monkeypatch.setattr(
        "src.analytics.simple_analytics.execute_analytics_tool",
        capture,
    )

    response = answer_analytics_question(question)

    assert captured == {
        "tool_id": expected_tool_id,
        "arguments": expected_arguments,
    }
    assert response["mode"] == "analytics"
    assert response["sources"][0]["source_path"].endswith(source_name)
    assert isinstance(response["sample_rows"][0]["request_count"], str)


@pytest.mark.parametrize(
    ("question", "source_name"),
    [
        ("What are the top complaint types?", "top_complaint_types.csv"),
        ("Which borough has the highest complaint volume?", "requests_by_borough.csv"),
        ("Which agencies handle the most requests?", "agency_request_volume.csv"),
        ("What is the backlog summary?", "backlog_summary.csv"),
    ],
)
def test_all_existing_analytics_questions_preserve_the_application_contract(
    question,
    source_name,
):
    response = answer_analytics_question(question)

    assert response["mode"] == "analytics"
    assert response["answer"]
    assert response["sample_rows"]
    assert response["retrieved_chunks"] == []
    assert response["sources"] == [
        {
            "source_name": source_name,
            "source_path": f"data/sample_outputs/{source_name}",
            "chunk_id": "sample_output",
        }
    ]
    assert response["confidence_note"] == SAMPLE_DATA_DISCLAIMER
