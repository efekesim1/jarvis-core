"""Run tracing.

Every agent run, model call and tool call becomes a span inside a trace, so a
failure in production can be read back afterwards instead of guessed at.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..logging import get_logger, redact

log = get_logger("tracing")


@dataclass(slots=True)
class Span:
    """One timed step inside a trace."""

    name: str
    kind: str = "generic"  # agent | model | tool | plan | evaluate | generic
    span_id: str = field(default_factory=lambda: f"sp_{uuid.uuid4().hex[:10]}")
    parent_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    status: str = "running"  # running | ok | error
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": redact(self.attributes),
        }


@dataclass(slots=True)
class Trace:
    """All the spans belonging to one user request."""

    name: str
    trace_id: str = field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    spans: list[Span] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "spans": [s.as_dict() for s in self.spans],
        }

    def failed_spans(self) -> list[Span]:
        return [s for s in self.spans if s.status == "error"]

    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.parent_id is None)


class Tracer:
    """Creates traces and records spans into them."""

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    def start_trace(self, name: str) -> Trace:
        trace = Trace(name=name)
        self._traces[trace.trace_id] = trace
        return trace

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces.values())

    @contextmanager
    def span(
        self,
        trace: Trace,
        name: str,
        *,
        kind: str = "generic",
        parent: Span | None = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Time a block of work and attach it to ``trace``."""
        span = Span(
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        trace.spans.append(span)
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if span.status == "running":
                span.status = "ok"
        finally:
            span.duration_ms = (time.perf_counter() - started) * 1000
            log.debug(
                "span %s (%s) %s in %.1fms", span.name, span.kind, span.status, span.duration_ms
            )
