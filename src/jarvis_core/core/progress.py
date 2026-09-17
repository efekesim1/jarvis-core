"""What a running request is doing, while it is still doing it.

A plan takes tens of seconds and spends money the whole time. Without this the
only honest thing an interface can show is a spinner, which says nothing about
whether the work is progressing, which agent is costing what, or whether
stopping now would waste anything.

The channel is a queue rather than a callback list, for one reason: the
producer is the orchestrator running agents concurrently and the consumer is an
HTTP response being written. Handing the producer a callback that writes to a
socket makes an agent's progress depend on the reader's back pressure. A
bounded queue that drops its oldest event instead keeps the run at full speed
and loses nothing that matters — the final answer never travels this way.

Nothing here is authoritative. Every number in an event is also in the run's
report, and the report is what gets stored. This is a window, not a record.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..logging import get_logger

log = get_logger("core.progress")


class ProgressKind(StrEnum):
    """The kinds of event a run can report.

    Deliberately small and closed. An interface can render every one of these,
    and a new kind is a change to both ends rather than a free-form string that
    one side silently ignores.
    """

    STARTED = "started"  # the run was accepted
    PLANNED = "planned"  # a plan exists; carries the steps
    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    TEXT = "text"  # a delta of the answer, as it is produced
    SYNTHESIZING = "synthesizing"  # the findings are being merged
    COST = "cost"  # spend so far
    WARNING = "warning"  # something went wrong that did not stop the run
    CANCELLED = "cancelled"
    DONE = "done"  # carries the final answer
    ERROR = "error"  # the run failed; carries why


@dataclass(slots=True, frozen=True)
class ProgressEvent:
    """One thing that happened, with the time it happened at."""

    kind: ProgressKind
    data: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "at": self.at.isoformat(), **self.data}


# Terminal kinds. Once one of these is emitted the run is over, and a consumer
# that sees one can stop reading without waiting for a timeout.
FINAL_KINDS = frozenset({ProgressKind.DONE, ProgressKind.ERROR, ProgressKind.CANCELLED})


class ProgressChannel:
    """A bounded, single-consumer stream of events from one run.

    ``emit`` never blocks and never raises: it is called from inside agent
    execution, where an exception would fail the run over a reporting problem.
    """

    def __init__(self, *, limit: int = 512) -> None:
        self._queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=limit)
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """How many events were discarded because the consumer fell behind.

        Reported rather than hidden: a non-zero count means the view was
        lossy, and a view that was lossy without saying so is worse than one
        that admits it.
        """
        return self._dropped

    @property
    def closed(self) -> bool:
        return self._closed

    def emit(self, kind: ProgressKind, /, **data: Any) -> None:
        """Record one event. Cheap, non-blocking, and safe to ignore."""
        if self._closed:
            return
        event = ProgressEvent(kind=kind, data=data)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest, not the newest: what just happened is more
            # useful than what happened four hundred events ago.
            try:
                self._queue.get_nowait()
                self._dropped += 1
                self._queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - racy only
                self._dropped += 1

    def close(self) -> None:
        """Say that nothing more will be emitted."""
        self._closed = True

    async def events(self, *, idle_timeout_s: float = 90.0) -> AsyncIterator[ProgressEvent]:
        """Yield events until a final one arrives, or the run goes quiet.

        The timeout exists because a crashed producer would otherwise hold a
        reader open forever. It is generous: a single research agent with a
        server-side search can legitimately take a minute to say anything.
        """
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=idle_timeout_s)
            except TimeoutError:
                log.warning("progress channel went quiet for %.0fs", idle_timeout_s)
                yield ProgressEvent(
                    kind=ProgressKind.ERROR,
                    data={"detail": f"the run stopped reporting for {idle_timeout_s:.0f}s"},
                )
                return
            yield event
            if event.kind in FINAL_KINDS:
                return


class NullProgress:
    """A channel that discards everything.

    So the orchestrator never has to check whether anyone is watching, and the
    CLI path stays exactly as fast as it was.
    """

    dropped = 0
    closed = False

    def emit(self, kind: ProgressKind, /, **data: Any) -> None:
        return

    def close(self) -> None:
        return


# What the orchestrator and JARVIS accept. Typed as the union rather than a
# Protocol because there are exactly two implementations and both are here.
Progress = ProgressChannel | NullProgress
