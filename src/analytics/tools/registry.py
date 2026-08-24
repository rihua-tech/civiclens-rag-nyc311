"""Fixed registry and sole supported resolver for analytics tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from src.analytics.tools.base import (
    MAX_ANALYTICS_TOOL_ID_LENGTH,
    AnalyticsTool,
    AnalyticsToolId,
    AnalyticsToolResult,
    UnknownAnalyticsToolError,
)
from src.analytics.tools.definitions import build_analytics_tools


_TOOLS = build_analytics_tools()
if len({tool.tool_id for tool in _TOOLS}) != len(_TOOLS):  # pragma: no cover
    raise RuntimeError("Analytics tool IDs must be unique.")

ANALYTICS_TOOL_REGISTRY: Mapping[AnalyticsToolId, AnalyticsTool] = MappingProxyType(
    {tool.tool_id: tool for tool in _TOOLS}
)


def registered_tool_ids() -> tuple[str, ...]:
    """Return stable IDs in deterministic registry order."""

    return tuple(tool_id.value for tool_id in ANALYTICS_TOOL_REGISTRY)


def resolve_analytics_tool(tool_id: str | AnalyticsToolId) -> AnalyticsTool:
    """Resolve an exact allowlisted ID without dynamic lookup."""

    if isinstance(tool_id, AnalyticsToolId):
        normalized = tool_id
    elif isinstance(tool_id, str) and len(tool_id) <= MAX_ANALYTICS_TOOL_ID_LENGTH:
        try:
            normalized = AnalyticsToolId(tool_id)
        except ValueError as exc:
            raise UnknownAnalyticsToolError("Unknown analytics tool ID.") from exc
    else:
        raise UnknownAnalyticsToolError("Unknown analytics tool ID.")

    try:
        return ANALYTICS_TOOL_REGISTRY[normalized]
    except KeyError as exc:  # pragma: no cover - enum and registry stay in sync
        raise UnknownAnalyticsToolError("Unknown analytics tool ID.") from exc


def execute_analytics_tool(
    tool_id: str | AnalyticsToolId,
    arguments: Mapping[str, Any] | None = None,
) -> AnalyticsToolResult:
    """Validate and execute exactly one registered read-only analytics tool."""

    return resolve_analytics_tool(tool_id).invoke(arguments)

