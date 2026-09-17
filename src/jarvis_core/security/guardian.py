"""The Security Guardian.

Every agent run and every tool call passes through here. The Guardian answers
one question: may this actor perform this action right now? It checks the
permissions the agent was granted, the risk of the action, the environment it
is running in, and whether a human needs to approve first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..logging import AuditLog, get_logger
from .approval import ApprovalDecision, ApprovalGate, ApprovalRequest, DenyAllGate
from .permissions import (
    ALWAYS_APPROVED,
    SANDBOX_ALLOWED,
    Permission,
    PermissionSet,
    RiskLevel,
    parse_risk,
)

log = get_logger("security.guardian")


class SecurityViolation(RuntimeError):
    """Raised when an action is refused outright."""


@dataclass(slots=True)
class GuardianVerdict:
    """The Guardian's answer about one proposed action."""

    allowed: bool
    reason: str
    required_approval: bool = False
    risk: RiskLevel = RiskLevel.LOW
    missing_permissions: frozenset[Permission] = field(default_factory=frozenset)

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise SecurityViolation(self.reason)


class SecurityGuardian:
    """Central authorization for agents, tools and model calls."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        audit: AuditLog | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.audit = audit or AuditLog()
        self.approval_gate: ApprovalGate = approval_gate or DenyAllGate()
        self._approval_floor = parse_risk(self.settings.require_approval_above, RiskLevel.MEDIUM)

    # -- policy -----------------------------------------------------------

    def needs_approval(self, risk: RiskLevel, required: PermissionSet) -> bool:
        """True when a human must see the action before it runs."""
        if required & ALWAYS_APPROVED:
            return True
        return risk.rank >= self._approval_floor.rank

    def check(
        self,
        *,
        action: str,
        granted: PermissionSet,
        required: PermissionSet = frozenset(),
        risk: RiskLevel | str = RiskLevel.LOW,
        agent: str | None = None,
        task_id: str | None = None,
    ) -> GuardianVerdict:
        """Decide on an action without asking a human yet.

        The synchronous half of the check. Call :meth:`authorize` to also run
        the approval step.
        """
        risk_level = parse_risk(risk)
        missing = frozenset(required) - frozenset(granted)

        if missing:
            names = ", ".join(sorted(p.value for p in missing))
            verdict = GuardianVerdict(
                allowed=False,
                reason=f"agent {agent or '?'} lacks permission(s): {names}",
                risk=risk_level,
                missing_permissions=missing,
            )
        elif self.settings.environment == "sandbox" and (
            unsafe := frozenset(required) - SANDBOX_ALLOWED
        ):
            names = ", ".join(sorted(p.value for p in unsafe))
            verdict = GuardianVerdict(
                allowed=False,
                reason=f"permission(s) {names} are not available in the sandbox environment",
                risk=risk_level,
                missing_permissions=unsafe,
            )
        else:
            verdict = GuardianVerdict(
                allowed=True,
                reason="permitted",
                required_approval=self.needs_approval(risk_level, frozenset(required)),
                risk=risk_level,
            )

        self.audit.emit(
            "guardian.check",
            subject=action,
            agent=agent,
            task_id=task_id,
            permission=",".join(sorted(p.value for p in required)) or None,
            result="allow" if verdict.allowed else "deny",
            reason=verdict.reason,
            metadata={"risk": risk_level.value, "needs_approval": verdict.required_approval},
        )
        return verdict

    async def authorize(
        self,
        *,
        action: str,
        granted: PermissionSet,
        required: PermissionSet = frozenset(),
        risk: RiskLevel | str = RiskLevel.LOW,
        agent: str | None = None,
        task_id: str | None = None,
        reason: str = "",
        details: dict[str, Any] | None = None,
    ) -> GuardianVerdict:
        """Full check, including the human approval round trip when needed."""
        verdict = self.check(
            action=action,
            granted=granted,
            required=required,
            risk=risk,
            agent=agent,
            task_id=task_id,
        )
        if not verdict.allowed or not verdict.required_approval:
            return verdict

        request = ApprovalRequest(
            action=action,
            reason=reason or f"{agent or 'system'} requested {action}",
            risk=verdict.risk.value,
            agent=agent,
            task_id=task_id,
            details=details or {},
        )
        decision = await self.approval_gate.request(request)
        approved = decision is ApprovalDecision.APPROVED
        self.audit.emit(
            "guardian.approval",
            subject=action,
            agent=agent,
            task_id=task_id,
            result=decision.value,
            reason=request.reason,
            metadata={"request_id": request.request_id, "risk": verdict.risk.value},
        )
        if approved:
            return GuardianVerdict(
                allowed=True,
                reason="approved by owner",
                required_approval=True,
                risk=verdict.risk,
            )
        return GuardianVerdict(
            allowed=False,
            reason=f"owner approval {decision.value} for {action}",
            required_approval=True,
            risk=verdict.risk,
        )
