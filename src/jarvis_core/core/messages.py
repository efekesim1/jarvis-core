"""Structured messages between agents.

Agents do not chat freely. They exchange typed envelopes, so every hand-off is
loggable, replayable and checkable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MessageType(StrEnum):
    REQUEST = "request"
    RESULT = "result"
    ERROR = "error"
    HANDOFF = "handoff"
    APPROVAL_REQUEST = "approval_request"
    STATUS = "status"


@dataclass(slots=True)
class AgentMessage:
    """One envelope passed between two parts of the system."""

    task_id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "payload": dict(self.payload),
            "confidence": self.confidence,
            "at": self.at.isoformat(),
        }


class MessageBus:
    """An in-process record of every message, for tracing and debugging."""

    def __init__(self, *, limit: int = 2000) -> None:
        self._limit = limit
        self._messages: list[AgentMessage] = []

    def publish(self, message: AgentMessage) -> AgentMessage:
        self._messages.append(message)
        if len(self._messages) > self._limit:
            del self._messages[0 : len(self._messages) - self._limit]
        return message

    def for_task(self, task_id: str) -> list[AgentMessage]:
        return [m for m in self._messages if m.task_id == task_id]

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)
