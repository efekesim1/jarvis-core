"""The permission model.

Every agent declares the smallest set of permissions it needs. The Guardian
refuses any action that needs a permission the agent was not granted, so a
research agent can never spend money and a build agent can never deploy
without having asked for ``DEPLOY``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Permission(StrEnum):
    """Capabilities an agent may hold."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FINANCIAL = "financial"
    COMMUNICATION = "communication"
    DEPLOY = "deploy"
    ADMIN = "admin"


class RiskLevel(StrEnum):
    """How much damage an action could do if it goes wrong."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank <= other.rank


PermissionSet = frozenset[Permission]

# Permissions that always mean a human sees the action before it runs,
# whatever risk level the caller claimed.
ALWAYS_APPROVED: PermissionSet = frozenset(
    {Permission.FINANCIAL, Permission.DEPLOY, Permission.ADMIN}
)

# The permissions an agent may hold in a sandbox. Sandboxes never get
# production credentials, money, or deploy rights.
SANDBOX_ALLOWED: PermissionSet = frozenset(
    {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.NETWORK}
)


def parse_permissions(raw: Iterable[str] | None) -> PermissionSet:
    """Turn a list of permission names into a validated set.

    Raises ``ValueError`` on an unknown name so a typo in an agent definition
    fails loudly at load time rather than silently granting nothing.
    """
    if not raw:
        return frozenset()
    out: set[Permission] = set()
    for name in raw:
        try:
            out.add(Permission(str(name).strip().lower()))
        except ValueError as exc:
            valid = ", ".join(p.value for p in Permission)
            raise ValueError(f"unknown permission {name!r}; valid values are: {valid}") from exc
    return frozenset(out)


def parse_risk(raw: str | RiskLevel | None, default: RiskLevel = RiskLevel.LOW) -> RiskLevel:
    if raw is None:
        return default
    if isinstance(raw, RiskLevel):
        return raw
    try:
        return RiskLevel(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown risk level {raw!r}; use low, medium or high") from exc
