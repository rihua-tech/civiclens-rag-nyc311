"""Request-scoped, structured backend latency instrumentation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
from time import perf_counter
from typing import Any, Literal, TypeVar


LatencyStage = Literal[
    "routing_ms",
    "db_connection_ms",
    "retrieval_ms",
    "answer_generation_ms",
    "citation_validation_ms",
]
Connection = TypeVar("Connection")
LATENCY_EVENT = "civiclens_backend_latency"
LATENCY_LOGGER = logging.getLogger("uvicorn.error")


@dataclass
class RequestLatency:
    """Accumulate elapsed stage time for one backend request."""

    clock: Callable[[], float] = perf_counter
    routing_ms: float = 0.0
    db_connection_ms: float = 0.0
    retrieval_ms: float = 0.0
    answer_generation_ms: float = 0.0
    citation_validation_ms: float = 0.0

    @contextmanager
    def measure(self, stage: LatencyStage) -> Iterator[None]:
        started = self.clock()
        try:
            yield
        finally:
            elapsed_ms = max(0.0, (self.clock() - started) * 1000.0)
            setattr(self, stage, getattr(self, stage) + elapsed_ms)

    def payload(
        self,
        *,
        route: str,
        outcome: str,
        total_ms: float,
        query_id: str | None,
    ) -> dict[str, Any]:
        return {
            "event": LATENCY_EVENT,
            "route": route,
            "outcome": outcome,
            "query_id": query_id,
            "routing_ms": round(self.routing_ms, 3),
            "db_connection_ms": round(self.db_connection_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "answer_generation_ms": round(self.answer_generation_ms, 3),
            "citation_validation_ms": round(self.citation_validation_ms, 3),
            "total_ms": round(max(0.0, total_ms), 3),
        }


_ACTIVE_REQUEST_LATENCY: ContextVar[RequestLatency | None] = ContextVar(
    "civiclens_request_latency",
    default=None,
)


@contextmanager
def capture_request_latency(timing: RequestLatency) -> Iterator[None]:
    token = _ACTIVE_REQUEST_LATENCY.set(timing)
    try:
        yield
    finally:
        _ACTIVE_REQUEST_LATENCY.reset(token)


@contextmanager
def measure_latency(stage: LatencyStage) -> Iterator[None]:
    timing = _ACTIVE_REQUEST_LATENCY.get()
    if timing is None:
        yield
        return
    with timing.measure(stage):
        yield


def connect_with_latency(
    connector: Callable[..., Connection],
    *args: Any,
    **kwargs: Any,
) -> Connection:
    """Measure connection acquisition without including query execution."""
    with measure_latency("db_connection_ms"):
        return connector(*args, **kwargs)


def emit_latency_event(
    timing: RequestLatency,
    *,
    route: str,
    outcome: str,
    total_ms: float,
    query_id: str | None,
) -> None:
    """Write one JSON event without allowing logging failures to affect answers."""
    payload = timing.payload(
        route=route,
        outcome=outcome,
        total_ms=total_ms,
        query_id=query_id,
    )
    try:
        LATENCY_LOGGER.info(
            json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )
    except Exception:  # pragma: no cover - logging must remain best effort
        return
