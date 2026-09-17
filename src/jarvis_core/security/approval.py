"""Human approval gates.

A gate is asked whether a risky action may proceed. The default gate in an
unattended process denies, because an unattended process has nobody to ask.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..logging import get_logger, utcnow

log = get_logger("security.approval")


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


@dataclass(slots=True)
class ApprovalRequest:
    """A pending question for the human owner."""

    action: str
    reason: str
    risk: str
    agent: str | None = None
    task_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: utcnow().isoformat())

    def summary(self) -> str:
        who = self.agent or "system"
        return f"[{self.risk.upper()}] {who} wants to {self.action}: {self.reason}"


class ApprovalGate(Protocol):
    """Anything that can answer an approval request."""

    async def request(self, request: ApprovalRequest) -> ApprovalDecision: ...


class DenyAllGate:
    """The safe default: nobody is watching, so nothing risky runs.

    Used whenever JARVIS runs unattended (scheduler, night shift, CI).
    """

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        log.warning("approval denied (no human present): %s", request.summary())
        return ApprovalDecision.DENIED


class AutoApproveGate:
    """Approves everything. Only legitimate in tests and the sandbox."""

    def __init__(self, *, allowed_environments: tuple[str, ...] = ("sandbox", "test")) -> None:
        self.allowed_environments = allowed_environments

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        log.debug("auto-approved: %s", request.summary())
        return ApprovalDecision.APPROVED


class QueueApprovalGate:
    """Parks requests in a queue for an interface (CLI, chat, phone) to answer.

    ``pending`` lists what is waiting; ``resolve`` records the human's answer.
    """

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[ApprovalDecision]]] = {}

    @property
    def pending(self) -> list[ApprovalRequest]:
        return [req for req, _ in self._pending.values()]

    def resolve(self, request_id: str, decision: ApprovalDecision) -> bool:
        entry = self._pending.get(request_id)
        if entry is None:
            return False
        _, future = entry
        if not future.done():
            future.set_result(decision)
        return True

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = loop.create_future()
        self._pending[request.request_id] = (request, future)
        log.info("awaiting approval %s: %s", request.request_id, request.summary())
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except TimeoutError:
            log.warning("approval %s timed out", request.request_id)
            return ApprovalDecision.TIMED_OUT
        finally:
            self._pending.pop(request.request_id, None)
