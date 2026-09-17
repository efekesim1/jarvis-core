"""The approval engine.

An approval is permission for one specific call, and nothing else. It is bound
to a request id and to a hash of the arguments, it expires, and it is consumed
when used. That combination is what stops the three ways an approval gets
abused: replaying an old one, moving one to a different tool, and getting one
approved for harmless arguments then calling with different ones.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from ..logging import AuditLog, get_logger
from ..mcp.models import ToolRequest, ToolRisk

log = get_logger("tools.approval")


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CONSUMED = "consumed"  # used for its one call, and now spent


def argument_fingerprint(arguments: dict[str, Any]) -> str:
    """A hash of the arguments an approval was granted for.

    Approving "delete the staging bucket" must not approve "delete the
    production bucket", so the arguments are part of what was approved.
    """
    payload = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def summarize_arguments(arguments: dict[str, Any], *, limit: int = 300) -> str:
    """A readable one-liner for the person deciding. Never a credential."""
    from ..logging import redact

    text = json.dumps(redact(arguments), ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass(slots=True)
class ToolApproval:
    """One decision about one call."""

    request_id: str
    tool_id: str
    agent_id: str
    risk: ToolRisk
    argument_hash: str
    reason: str = ""
    arguments_summary: str = ""
    task_id: str | None = None
    approval_id: str = field(default_factory=lambda: f"tapr_{uuid.uuid4().hex[:12]}")
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str = ""

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and datetime.now(UTC) >= self.expires_at

    def matches(self, request: ToolRequest) -> bool:
        """Whether this approval is for exactly this call.

        All four must line up. Any mismatch means the approval was granted for
        something else, whatever the caller believes.
        """
        return (
            self.request_id == request.request_id
            and self.tool_id == request.tool_id
            and self.agent_id == request.agent_id
            and self.argument_hash == argument_fingerprint(request.arguments)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "request_id": self.request_id,
            "tool_id": self.tool_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "risk": self.risk.value,
            "status": self.status.value,
            "reason": self.reason,
            "arguments_summary": self.arguments_summary,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
        }


class ApprovalEngine:
    """Holds pending approvals and answers whether a call may proceed."""

    def __init__(
        self,
        *,
        audit: AuditLog | None = None,
        ttl_s: float = 900.0,
        store: Any | None = None,
    ) -> None:
        self.audit = audit or AuditLog()
        self.ttl_s = ttl_s
        # With a store, approvals are visible to every process sharing it, and
        # a decision made in one terminal is honoured in another.
        self.store = store
        self._approvals: dict[str, ToolApproval] = {}
        self._by_request: dict[str, str] = {}

    def _persist(self, approval: ToolApproval) -> ToolApproval:
        if self.store is not None:
            self.store.save(approval)
        return approval

    def _reload(self, approval_id: str) -> ToolApproval | None:
        """Pick up a decision another process made."""
        if self.store is None:
            return self._approvals.get(approval_id)
        stored = self.store.get(approval_id)
        if stored is not None:
            self._approvals[stored.approval_id] = stored
            self._by_request[stored.request_id] = stored.approval_id
        return stored or self._approvals.get(approval_id)

    # -- raising a request ------------------------------------------------

    def require(self, request: ToolRequest, risk: ToolRisk, *, reason: str = "") -> ToolApproval:
        """Record that this call needs a human, and park it."""
        existing = self._by_request.get(request.request_id)
        if existing is not None:
            return self._approvals[existing]

        approval = ToolApproval(
            request_id=request.request_id,
            tool_id=request.tool_id,
            agent_id=request.agent_id,
            task_id=request.task_id,
            risk=risk,
            argument_hash=argument_fingerprint(request.arguments),
            reason=reason or f"{request.agent_id} wants to call {request.tool_id}",
            arguments_summary=summarize_arguments(request.arguments),
            expires_at=datetime.now(UTC) + timedelta(seconds=self.ttl_s),
        )
        self._approvals[approval.approval_id] = approval
        self._by_request[request.request_id] = approval.approval_id
        self._persist(approval)
        self.audit.emit(
            "tool.approval_required",
            agent=request.agent_id,
            tool=request.tool_id,
            task_id=request.task_id,
            result="pending",
            reason=approval.reason,
            metadata={
                "approval_id": approval.approval_id,
                "request_id": request.request_id,
                "risk": risk.value,
            },
        )
        log.info("approval %s required: %s", approval.approval_id, approval.reason)
        return approval

    # -- deciding ---------------------------------------------------------

    def approve(self, approval_id: str, *, by: str = "operator", note: str = "") -> ToolApproval:
        return self._decide(approval_id, ApprovalStatus.APPROVED, by=by, note=note)

    def reject(self, approval_id: str, *, by: str = "operator", note: str = "") -> ToolApproval:
        return self._decide(approval_id, ApprovalStatus.REJECTED, by=by, note=note)

    def cancel(self, approval_id: str, *, by: str = "system") -> ToolApproval:
        return self._decide(approval_id, ApprovalStatus.CANCELLED, by=by)

    def _decide(
        self, approval_id: str, status: ApprovalStatus, *, by: str, note: str = ""
    ) -> ToolApproval:
        approval = self._reload(approval_id)
        if approval is None:
            raise KeyError(f"no approval {approval_id!r}")
        if self.store is not None:
            ok, reason = self.store.decide(approval_id, status.value, by=by, note=note)
            if not ok:
                raise ValueError(reason)
            decided = self.store.get(approval_id)
            if decided is not None:
                self._approvals[approval_id] = decided
                approval = decided
            self.audit.emit(
                "tool.approved" if status is ApprovalStatus.APPROVED else "tool.rejected",
                actor=by,
                agent=approval.agent_id,
                tool=approval.tool_id,
                task_id=approval.task_id,
                result=status.value,
                reason=note,
                metadata={"approval_id": approval_id, "request_id": approval.request_id},
            )
            return approval
        if approval.status is not ApprovalStatus.PENDING:
            raise ValueError(
                f"approval {approval_id} is already {approval.status.value} and cannot be changed"
            )
        if approval.expired:
            approval.status = ApprovalStatus.EXPIRED
            raise ValueError(f"approval {approval_id} expired before it was decided")
        approval.status = status
        approval.decided_at = datetime.now(UTC)
        approval.decided_by = by
        approval.note = note
        self.audit.emit(
            "tool.approved" if status is ApprovalStatus.APPROVED else "tool.rejected",
            actor=by,
            agent=approval.agent_id,
            tool=approval.tool_id,
            task_id=approval.task_id,
            result=status.value,
            reason=note,
            metadata={"approval_id": approval_id, "request_id": approval.request_id},
        )
        return approval

    # -- using one --------------------------------------------------------

    def consume(self, request: ToolRequest) -> tuple[bool, str, ApprovalStatus | None, bool]:
        """Spend the approval for this call, if there is a valid one.

        Returns whether the call may proceed, why not if it may not, the
        status found, and whether the refusal was a security event rather than
        an ordinary "not approved yet". An approval is single use: consuming it
        marks it spent, so the same decision cannot authorise a second call.
        """
        approval_id = request.approval_id or self._by_request.get(request.request_id)
        if approval_id is None and self.store is not None:
            found = self.store.for_request(request.request_id)
            approval_id = found.approval_id if found else None
        if approval_id is None:
            return False, "no approval has been granted for this call", None, False

        approval = self._reload(approval_id)
        if approval is None:
            return False, "the approval referenced does not exist", None, False

        # Bound to this request, this tool, this agent, these arguments.
        if not approval.matches(request):
            self.audit.emit(
                "tool.security_blocked",
                agent=request.agent_id,
                tool=request.tool_id,
                task_id=request.task_id,
                result="blocked",
                reason="an approval was presented for a different call",
                metadata={"approval_id": approval_id, "request_id": request.request_id},
            )
            # Presenting an approval for a different tool, agent or set of
            # arguments is an attempted bypass, not a missing approval.
            return False, "this approval was granted for a different call", approval.status, True

        if approval.status is ApprovalStatus.CONSUMED:
            self.audit.emit(
                "tool.security_blocked",
                agent=request.agent_id,
                tool=request.tool_id,
                result="blocked",
                reason="an approval was replayed after being used",
                metadata={"approval_id": approval_id},
            )
            return False, "this approval has already been used", approval.status, True

        if approval.expired:
            approval.status = ApprovalStatus.EXPIRED
            return False, "the approval expired before it was used", ApprovalStatus.EXPIRED, False

        if approval.status is not ApprovalStatus.APPROVED:
            return (
                False,
                f"the approval is {approval.status.value}",
                approval.status,
                False,
            )

        if self.store is not None and not self.store.consume(approval.approval_id):
            # Another process spent it first. That is a replay, not a race to
            # be retried.
            approval.status = ApprovalStatus.CONSUMED
            return False, "this approval has already been used", ApprovalStatus.CONSUMED, True
        approval.status = ApprovalStatus.CONSUMED
        self._persist(approval)
        return True, "approved", ApprovalStatus.APPROVED, False

    # -- inspection -------------------------------------------------------

    def get(self, approval_id: str) -> ToolApproval | None:
        return self._reload(approval_id)

    def for_request(self, request_id: str) -> ToolApproval | None:
        if self.store is not None:
            stored = self.store.for_request(request_id)
            if stored is not None:
                self._approvals[stored.approval_id] = stored
                self._by_request[request_id] = stored.approval_id
                return stored
        approval_id = self._by_request.get(request_id)
        return self._approvals.get(approval_id) if approval_id else None

    @property
    def pending(self) -> list[ToolApproval]:
        self.expire_stale()
        if self.store is not None:
            return self.store.pending()
        return [a for a in self._approvals.values() if a.status is ApprovalStatus.PENDING]

    def all(self) -> list[ToolApproval]:
        if self.store is not None:
            return self.store.all()
        return sorted(self._approvals.values(), key=lambda a: a.created_at, reverse=True)

    def expire_stale(self) -> int:
        if self.store is not None:
            return self.store.expire_stale()
        expired = 0
        for approval in self._approvals.values():
            if approval.status is ApprovalStatus.PENDING and approval.expired:
                approval.status = ApprovalStatus.EXPIRED
                expired += 1
        return expired
