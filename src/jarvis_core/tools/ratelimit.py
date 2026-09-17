"""Rate limiting for tool calls.

A sliding window per tool, per agent and per task. Three scopes because they
catch different failures: a single runaway agent, a tool being hammered by
everyone at once, and one task that will not stop.

**The window can be shared across processes.** Without that, a limit of thirty
calls a minute is thirty *per process*: a night shift and a terminal session
each enforce it separately and the tool sees sixty. That is not a limit, it is
a limit-shaped number, and the whole point of the scope is to bound what the
outside world experiences.

Sharing it needs two things the in-memory version does not:

* **Wall-clock time.** ``time.monotonic()`` is right for one process — immune
  to the clock being adjusted — and meaningless between two, because each has
  its own zero. Shared windows are stamped in UTC.
* **One atomic step.** Checking and then recording leaves a gap two processes
  can both pass through. ``acquire()`` does both inside one immediate
  transaction, so the second one loses.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger
from ..mcp.models import RateLimit

log = get_logger("tools.ratelimit")


@dataclass(slots=True)
class LimitVerdict:
    allowed: bool
    reason: str = ""
    scope: str = ""
    retry_after_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "scope": self.scope,
            "retry_after_s": round(self.retry_after_s, 2),
        }


class RateLimiter:
    """Sliding-window limiter, safe across concurrent calls and, with a store,
    across processes."""

    def __init__(self, *, window_s: float = 60.0, store: Any = None) -> None:
        self.window_s = window_s
        self.store = store
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}
        self._task_counts: dict[str, int] = {}

    @property
    def shared(self) -> bool:
        """Whether this limiter is enforced across processes.

        Worth being able to ask. A caller that thinks a limit is global and is
        actually per-process is wrong by a factor of however many processes are
        running, and nothing else would tell it.
        """
        return self.store is not None

    def _prune(self, key: str, now: float) -> deque[float]:
        window = self._events.setdefault(key, deque())
        cutoff = now - self.window_s
        while window and window[0] < cutoff:
            window.popleft()
        return window

    def acquire(
        self,
        *,
        tool_id: str,
        agent_id: str,
        task_id: str | None,
        limit: RateLimit,
    ) -> LimitVerdict:
        """Check and record in one step, so two callers cannot both pass.

        This is what the gateway uses. ``check`` followed by ``record`` leaves
        a window between them, and with a shared store that window is wide
        enough for a second process to walk through — which is the whole thing
        the shared store was added to prevent.
        """
        if self.store is not None:
            return self.store.acquire(
                tool_id=tool_id,
                agent_id=agent_id,
                task_id=task_id,
                limit=limit,
                window_s=self.window_s,
            )
        with self._lock:
            verdict = self._check_locked(
                tool_id=tool_id, agent_id=agent_id, task_id=task_id, limit=limit
            )
            if verdict.allowed:
                self._record_locked(tool_id=tool_id, agent_id=agent_id, task_id=task_id)
            return verdict

    def check(
        self,
        *,
        tool_id: str,
        agent_id: str,
        task_id: str | None,
        limit: RateLimit,
    ) -> LimitVerdict:
        """Would this call be within every limit? Does not record it."""
        if self.store is not None:
            return self.store.check(
                tool_id=tool_id,
                agent_id=agent_id,
                task_id=task_id,
                limit=limit,
                window_s=self.window_s,
            )
        with self._lock:
            return self._check_locked(
                tool_id=tool_id, agent_id=agent_id, task_id=task_id, limit=limit
            )

    def _check_locked(
        self,
        *,
        tool_id: str,
        agent_id: str,
        task_id: str | None,
        limit: RateLimit,
    ) -> LimitVerdict:
        now = time.monotonic()
        tool_window = self._prune(f"tool:{tool_id}", now)
        if len(tool_window) >= limit.per_minute:
            return LimitVerdict(
                False,
                f"{tool_id} has been called {len(tool_window)} times in the last minute, "
                f"and its limit is {limit.per_minute}",
                "tool",
                max(0.0, self.window_s - (now - tool_window[0])),
            )

        agent_window = self._prune(f"agent:{agent_id}:{tool_id}", now)
        if len(agent_window) >= limit.per_agent_per_minute:
            return LimitVerdict(
                False,
                f"{agent_id} has called {tool_id} {len(agent_window)} times in the last "
                f"minute, and its limit is {limit.per_agent_per_minute}",
                "agent",
                max(0.0, self.window_s - (now - agent_window[0])),
            )

        if task_id is not None:
            used = self._task_counts.get(f"{task_id}:{tool_id}", 0)
            if used >= limit.per_task:
                return LimitVerdict(
                    False,
                    f"task {task_id} has called {tool_id} {used} times, "
                    f"and its limit is {limit.per_task}",
                    "task",
                )
        return LimitVerdict(True)

    def record(self, *, tool_id: str, agent_id: str, task_id: str | None) -> None:
        """Count a call that actually went out."""
        if self.store is not None:
            self.store.record(tool_id=tool_id, agent_id=agent_id, task_id=task_id)
            return
        with self._lock:
            self._record_locked(tool_id=tool_id, agent_id=agent_id, task_id=task_id)

    def _record_locked(self, *, tool_id: str, agent_id: str, task_id: str | None) -> None:
        now = time.monotonic()
        self._prune(f"tool:{tool_id}", now).append(now)
        self._prune(f"agent:{agent_id}:{tool_id}", now).append(now)
        if task_id is not None:
            key = f"{task_id}:{tool_id}"
            self._task_counts[key] = self._task_counts.get(key, 0) + 1

    def reset_task(self, task_id: str) -> None:
        if self.store is not None:
            self.store.reset_task(task_id)
            return
        with self._lock:
            for key in [k for k in self._task_counts if k.startswith(f"{task_id}:")]:
                del self._task_counts[key]

    def reset(self) -> None:
        if self.store is not None:
            self.store.reset()
        with self._lock:
            self._events.clear()
            self._task_counts.clear()
