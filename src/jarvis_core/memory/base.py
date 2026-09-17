"""The memory interface.

Memory is not one bucket. Each layer has a different lifetime and a different
reason to exist, and nothing is written to memory just because it happened.

Storage arrives in Phase 4 (PostgreSQL with pgvector). This module defines the
shape so the rest of the system can already be written against it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class MemoryScope(StrEnum):
    """Which layer an item belongs to."""

    WORKING = "working"  # the current task only
    CONVERSATION = "conversation"
    USER = "user"  # what is true about the operator
    AGENT = "agent"  # what an agent learned about doing its job
    PROJECT = "project"
    KNOWLEDGE = "knowledge"
    DECISION = "decision"  # what was decided and why
    SYSTEM = "system"
    EPISODIC = "episodic"  # what happened, in order


@dataclass(slots=True)
class MemoryItem:
    """One remembered thing."""

    content: str
    scope: MemoryScope
    source: str
    item_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5
    importance: float = 0.5
    expiration: datetime | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expiration is None:
            return False
        return (now or datetime.now(UTC)) >= self.expiration


class MemoryStore(Protocol):
    """Where memory lives."""

    def write(self, item: MemoryItem) -> MemoryItem: ...
    def read(self, item_id: str) -> MemoryItem | None: ...
    def search(
        self, query: str, *, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemoryItem]: ...
    def forget(self, item_id: str) -> bool: ...


class InMemoryStore:
    """A keyword-matching store for development.

    Deliberately simple. It is replaced by pgvector semantic search, not
    extended.
    """

    def __init__(self, *, importance_floor: float = 0.3) -> None:
        self._items: dict[str, MemoryItem] = {}
        self.importance_floor = importance_floor

    def write(self, item: MemoryItem) -> MemoryItem:
        # Not everything that happens is worth remembering.
        if item.importance < self.importance_floor:
            return item
        self._items[item.item_id] = item
        return item

    def read(self, item_id: str) -> MemoryItem | None:
        item = self._items.get(item_id)
        if item is not None and item.is_expired():
            del self._items[item_id]
            return None
        return item

    def search(
        self, query: str, *, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemoryItem]:
        words = {w.lower() for w in query.split() if len(w) > 2}
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items.values():
            if item.is_expired() or (scope is not None and item.scope is not scope):
                continue
            lowered = item.content.lower()
            hits = sum(1 for w in words if w in lowered)
            if hits:
                scored.append((hits * item.importance, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def forget(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None
