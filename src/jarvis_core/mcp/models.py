"""The things MCP deals in: servers, tools, invocations and results.

Data only. Nothing here connects to anything, decides anything, or trusts
anything; that is what makes it safe to pass these objects around.

Two rules are baked into the types rather than left to callers. A server
record never holds a credential, only a reference to where one lives. And a
tool result is marked untrusted from the moment it exists, because it came
from outside.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..security.permissions import Permission, PermissionSet, RiskLevel


class ToolRisk(StrEnum):
    """What a tool could do if it went wrong.

    Finer than the three-level ``RiskLevel`` used elsewhere, because "write"
    covers everything from filing an issue to deleting a repository.
    """

    READ_ONLY = "read_only"
    LOW_RISK_WRITE = "low_risk_write"
    HIGH_RISK_WRITE = "high_risk_write"
    DESTRUCTIVE = "destructive"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"

    @property
    def rank(self) -> int:
        return {
            "read_only": 0,
            "low_risk_write": 1,
            "high_risk_write": 2,
            "destructive": 3,
            "external_side_effect": 4,
        }[self.value]

    @property
    def needs_approval(self) -> bool:
        """Anything that could not be undone by trying again needs a human."""
        return self.rank >= ToolRisk.HIGH_RISK_WRITE.rank

    @property
    def is_write(self) -> bool:
        return self is not ToolRisk.READ_ONLY

    @property
    def guardian_risk(self) -> RiskLevel:
        """How the existing Guardian sees this."""
        if self is ToolRisk.READ_ONLY:
            return RiskLevel.LOW
        if self is ToolRisk.LOW_RISK_WRITE:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH


# The permission an agent must hold for each class of tool. A read tool needs
# READ; anything that changes the world needs WRITE on top of it.
RISK_PERMISSIONS: dict[ToolRisk, PermissionSet] = {
    ToolRisk.READ_ONLY: frozenset({Permission.READ}),
    ToolRisk.LOW_RISK_WRITE: frozenset({Permission.READ, Permission.WRITE}),
    ToolRisk.HIGH_RISK_WRITE: frozenset({Permission.READ, Permission.WRITE}),
    ToolRisk.DESTRUCTIVE: frozenset({Permission.READ, Permission.WRITE, Permission.ADMIN}),
    ToolRisk.EXTERNAL_SIDE_EFFECT: frozenset({Permission.READ, Permission.COMMUNICATION}),
}


class TrustLevel(StrEnum):
    """How much a server's word is worth."""

    UNTRUSTED = "untrusted"  # never gets write tools
    LIMITED = "limited"
    TRUSTED = "trusted"  # first-party, or reviewed

    @property
    def rank(self) -> int:
        return {"untrusted": 0, "limited": 1, "trusted": 2}[self.value]


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"

    @property
    def usable(self) -> bool:
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNKNOWN)


class TransportKind(StrEnum):
    IN_PROCESS = "in_process"  # the mock server and built-in adapters
    STDIO = "stdio"
    HTTP = "http"


@dataclass(slots=True)
class MCPServer:
    """A server record. Never holds a secret.

    ``auth_reference`` names an environment variable or secret-manager key.
    The value is resolved at the moment of use and never stored here, so a
    registry dump or an audit line cannot leak a credential.
    """

    server_id: str
    name: str
    transport: TransportKind = TransportKind.IN_PROCESS
    endpoint: str | None = None
    version: str = "1.0.0"
    auth_reference: str | None = None
    trust_level: TrustLevel = TrustLevel.LIMITED
    enabled: bool = True
    health: HealthStatus = HealthStatus.UNKNOWN
    allowed_tools: tuple[str, ...] = ()  # empty means every tool it publishes
    max_risk: ToolRisk = ToolRisk.READ_ONLY  # what this server may ever do
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def permits_tool(self, tool_name: str) -> bool:
        return not self.allowed_tools or tool_name in self.allowed_tools

    def as_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "name": self.name,
            "version": self.version,
            "transport": self.transport.value,
            "endpoint": self.endpoint,
            # The reference, deliberately, not the credential.
            "auth_reference": self.auth_reference,
            "trust_level": self.trust_level.value,
            "enabled": self.enabled,
            "health": self.health.value,
            "allowed_tools": list(self.allowed_tools),
            "max_risk": self.max_risk.value,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass(slots=True)
class RateLimit:
    """How often a tool may be called, and by whom."""

    per_minute: int = 60
    per_agent_per_minute: int = 30
    per_task: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_minute": self.per_minute,
            "per_agent_per_minute": self.per_agent_per_minute,
            "per_task": self.per_task,
        }


@dataclass(slots=True)
class ToolRetryPolicy:
    """How hard to try again. Writes do not get a second attempt by default."""

    max_attempts: int = 2
    backoff_s: float = 0.5
    idempotent: bool = True

    def attempts_for(self, risk: ToolRisk) -> int:
        """A non-idempotent write is never retried automatically.

        Retrying a call that may already have created something is how one
        issue becomes three.
        """
        if risk.is_write and not self.idempotent:
            return 1
        return max(1, min(self.max_attempts, 5))


@dataclass(slots=True)
class MCPTool:
    """A tool, as the system knows it rather than as its server described it.

    ``tool_id`` is namespaced by server, because two servers may both publish
    a tool called ``search`` and they are not the same tool.
    """

    tool_id: str
    name: str
    server_id: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    risk: ToolRisk = ToolRisk.HIGH_RISK_WRITE  # the safe default for the unknown
    capability: str = ""  # what agents are granted, e.g. "github.read"
    required_permissions: PermissionSet = field(default_factory=frozenset)
    allowed_agents: tuple[str, ...] = ()  # empty means policy decides
    timeout_s: float = 30.0
    retry_policy: ToolRetryPolicy = field(default_factory=ToolRetryPolicy)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    cacheable: bool | None = None  # None means "read tools only"
    cache_ttl_s: float = 300.0
    enabled: bool = True
    version: str = "1.0.0"
    schema_version: str = "1"
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.required_permissions:
            self.required_permissions = RISK_PERMISSIONS.get(
                self.risk, frozenset({Permission.READ, Permission.WRITE})
            )
        if not self.capability:
            self.capability = f"{self.server_id}.{self.name}"

    @property
    def available(self) -> bool:
        return self.enabled and self.unavailable_reason is None

    @property
    def is_cacheable(self) -> bool:
        """Read tools cache by default; writes never cache unless asked."""
        if self.cacheable is not None:
            return self.cacheable
        return self.risk is ToolRisk.READ_ONLY

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "server_id": self.server_id,
            "description": self.description,
            "capability": self.capability,
            "risk": self.risk.value,
            "required_permissions": sorted(p.value for p in self.required_permissions),
            "timeout_s": self.timeout_s,
            "rate_limit": self.rate_limit.as_dict(),
            "cacheable": self.is_cacheable,
            "enabled": self.enabled,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "version": self.version,
            "schema_version": self.schema_version,
            "input_schema": self.input_schema,
        }


@dataclass(slots=True, frozen=True)
class ToolRequest:
    """One invocation, immutable from the moment it is made.

    The request id follows the call through every stage, so a line in the
    audit log and a span in a trace refer to the same thing.
    """

    tool_id: str
    arguments: dict[str, Any]
    agent_id: str
    task_id: str | None = None
    actor: str = "operator"
    trace_id: str | None = None
    request_id: str = field(default_factory=lambda: f"treq_{uuid.uuid4().hex[:12]}")
    approval_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_id": self.tool_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "actor": self.actor,
            "trace_id": self.trace_id,
            "approval_id": self.approval_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class ToolResult:
    """What a tool gave back.

    ``untrusted`` is true on every result and is not settable to false by the
    server. Anything a tool returns came from outside the system and is
    evidence, never instruction.
    """

    request_id: str
    tool_id: str
    ok: bool
    content: Any = None
    text: str = ""
    error: Any = None  # a ToolFailure when ok is false
    duration_ms: float = 0.0
    attempts: int = 1
    from_cache: bool = False
    cost_usd: float = 0.0
    truncated: bool = False
    sanitized: bool = False
    warnings: list[str] = field(default_factory=list)
    untrusted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_id": self.tool_id,
            "ok": self.ok,
            "text": self.text,
            "content": self.content,
            "error": self.error.as_dict() if hasattr(self.error, "as_dict") else self.error,
            "duration_ms": round(self.duration_ms, 2),
            "attempts": self.attempts,
            "from_cache": self.from_cache,
            "cost_usd": round(self.cost_usd, 6),
            "truncated": self.truncated,
            "sanitized": self.sanitized,
            "warnings": list(self.warnings),
            "untrusted": self.untrusted,
        }
