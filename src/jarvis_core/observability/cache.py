"""The result cache and duplicate-work detector.

Two agents asked the same question is two bills for one answer. This module is
what stops that, and it does so at three levels:

* **In flight.** Two identical runs starting at the same moment share one
  model call; the second waits for the first rather than issuing its own.
* **Recently answered.** An identical run inside the time-to-live returns the
  stored result and costs nothing.
* **Reported.** Every hit is counted, so the saving is a number rather than a
  claim.

Deliberately in memory. A shared cache across processes is a Phase 4 concern
that arrives with the database; building it now would be a second storage
layer with no second process to use it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..logging import get_logger

log = get_logger("cache")

T = TypeVar("T")


def _normalize(value: Any) -> Any:
    """Reduce a value to something that hashes the same for equal meaning."""
    if isinstance(value, str):
        # Whitespace and case should not make two identical questions different.
        return " ".join(value.split()).strip().lower()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def work_key(*parts: Any) -> str:
    """A stable identity for a piece of work.

    Same agent, same goal, same inputs means the same key, whatever the
    whitespace, key order or capitalisation.
    """
    payload = json.dumps([_normalize(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class CacheStats:
    """What the cache saved."""

    hits: int = 0
    misses: int = 0
    coalesced: int = 0
    evictions: int = 0
    saved_usd: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses + self.coalesced

    @property
    def hit_rate(self) -> float:
        return (self.hits + self.coalesced) / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 3),
            "saved_usd": round(self.saved_usd, 6),
        }


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    stored_at: float
    cost_usd: float = 0.0
    hits: int = 0


class ResultCache(Generic[T]):
    """A bounded, expiring cache that also collapses concurrent duplicates."""

    def __init__(self, *, max_entries: int = 512, ttl_s: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.stats = CacheStats()
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._in_flight: dict[str, asyncio.Future[T]] = {}

    # -- plain lookups ----------------------------------------------------

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > self.ttl_s:
            del self._entries[key]
            self.stats.evictions += 1
            return None
        entry.hits += 1
        self._entries.move_to_end(key)
        return entry.value

    def put(self, key: str, value: T, *, cost_usd: float = 0.0) -> None:
        self._entries[key] = _Entry(value=value, stored_at=time.monotonic(), cost_usd=cost_usd)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def invalidate(self, key: str) -> bool:
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # -- the useful entry point -------------------------------------------

    async def run(
        self,
        key: str,
        producer: Callable[[], Awaitable[T]],
        *,
        cost_of: Callable[[T], float] | None = None,
        cacheable: Callable[[T], bool] | None = None,
    ) -> tuple[T, bool]:
        """Return a cached result, or produce one exactly once.

        Returns the value and whether it came from the cache. A second caller
        arriving while the first is still working waits for it rather than
        starting a duplicate run.
        """
        cached = self.get(key)
        if cached is not None:
            self.stats.hits += 1
            entry = self._entries.get(key)
            if entry is not None:
                self.stats.saved_usd += entry.cost_usd
            log.debug("cache hit %s", key[:12])
            return cached, True

        pending = self._in_flight.get(key)
        if pending is not None:
            self.stats.coalesced += 1
            log.debug("waiting on an identical run already in flight: %s", key[:12])
            value = await asyncio.shield(pending)
            entry = self._entries.get(key)
            if entry is not None:
                self.stats.saved_usd += entry.cost_usd
            return value, True

        self.stats.misses += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._in_flight[key] = future
        try:
            value = await producer()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            # Stop the exception from being reported as never retrieved when
            # nobody else was waiting on this key.
            future.exception()
            raise
        finally:
            self._in_flight.pop(key, None)

        if cacheable is None or cacheable(value):
            self.put(key, value, cost_usd=cost_of(value) if cost_of else 0.0)
        if not future.done():
            future.set_result(value)
        return value, False
