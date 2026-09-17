"""The policy engine: who may call what.

Default deny, stated once and enforced everywhere. An agent is granted named
capabilities, not tools, so a tool can be swapped for a better one behind the
same capability without rewriting every agent that used it.

Nothing here executes anything. It answers one question and returns a reason
either way, because "denied" without a reason is unusable in an audit log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from ..mcp.errors import ToolErrorCode
from ..mcp.models import MCPServer, MCPTool, ToolRisk, TrustLevel
from ..security.permissions import PermissionSet

log = get_logger("tools.policy")


@dataclass(slots=True)
class AgentToolPolicy:
    """What one agent is allowed to reach.

    ``capabilities`` are the grant. A bare tool id in ``tools`` is an
    exception, used when one specific tool is wanted without its whole
    capability.
    """

    agent_id: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    tools: frozenset[str] = field(default_factory=frozenset)
    max_risk: ToolRisk = ToolRisk.READ_ONLY
    max_calls_per_task: int = 20

    def grants(self, tool: MCPTool) -> bool:
        if tool.tool_id in self.tools:
            return True
        if tool.capability in self.capabilities:
            return True
        # A capability may be granted as a prefix: "github.read" covers
        # "github.read.issues", but never "github.write".
        return any(tool.capability.startswith(f"{capability}.") for capability in self.capabilities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": sorted(self.capabilities),
            "tools": sorted(self.tools),
            "max_risk": self.max_risk.value,
            "max_calls_per_task": self.max_calls_per_task,
        }


@dataclass(slots=True)
class PolicyVerdict:
    """The answer, with the reason attached."""

    allowed: bool
    reason: str
    code: ToolErrorCode | None = None
    needs_approval: bool = False
    risk: ToolRisk = ToolRisk.READ_ONLY

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "code": self.code.value if self.code else None,
            "needs_approval": self.needs_approval,
            "risk": self.risk.value,
        }


class PolicyEngine:
    """Decides whether an agent may call a tool. Denies by default."""

    def __init__(
        self,
        *,
        policies: dict[str, AgentToolPolicy] | None = None,
        environment: str = "sandbox",
    ) -> None:
        self._policies: dict[str, AgentToolPolicy] = dict(policies or {})
        self.environment = environment

    # -- grants -----------------------------------------------------------

    def grant(self, policy: AgentToolPolicy) -> AgentToolPolicy:
        self._policies[policy.agent_id] = policy
        return policy

    def grant_capabilities(
        self,
        agent_id: str,
        capabilities: tuple[str, ...],
        *,
        max_risk: ToolRisk = ToolRisk.READ_ONLY,
        tools: tuple[str, ...] = (),
    ) -> AgentToolPolicy:
        return self.grant(
            AgentToolPolicy(
                agent_id=agent_id,
                capabilities=frozenset(capabilities),
                tools=frozenset(tools),
                max_risk=max_risk,
            )
        )

    def policy_for(self, agent_id: str) -> AgentToolPolicy | None:
        return self._policies.get(agent_id)

    def policies(self) -> list[AgentToolPolicy]:
        return sorted(self._policies.values(), key=lambda p: p.agent_id)

    def visible_tools(self, agent_id: str, tools: list[MCPTool]) -> list[MCPTool]:
        """What this agent is even allowed to know exists.

        Discovery is not authorisation: an agent enumerating tools sees only
        the ones it could call, so a listing cannot become a shopping list.
        """
        policy = self._policies.get(agent_id)
        if policy is None:
            return []
        return [
            tool
            for tool in tools
            if tool.available and policy.grants(tool) and tool.risk.rank <= policy.max_risk.rank
        ]

    # -- the decision -----------------------------------------------------

    def check(
        self,
        *,
        agent_id: str,
        tool: MCPTool,
        server: MCPServer,
        agent_permissions: PermissionSet,
    ) -> PolicyVerdict:
        """Decide one call. Every branch returns a reason."""
        policy = self._policies.get(agent_id)
        if policy is None:
            # Default deny. Being a registered agent grants nothing.
            return PolicyVerdict(
                False,
                f"agent {agent_id!r} has no tool policy, so it may call nothing",
                ToolErrorCode.PERMISSION_DENIED,
                risk=tool.risk,
            )

        if tool.allowed_agents and agent_id not in tool.allowed_agents:
            return PolicyVerdict(
                False,
                f"the tool {tool.tool_id} is restricted to other agents",
                ToolErrorCode.PERMISSION_DENIED,
                risk=tool.risk,
            )

        if not policy.grants(tool):
            return PolicyVerdict(
                False,
                f"agent {agent_id!r} is not granted the capability {tool.capability!r}",
                ToolErrorCode.PERMISSION_DENIED,
                risk=tool.risk,
            )

        if tool.risk.rank > policy.max_risk.rank:
            return PolicyVerdict(
                False,
                f"{tool.risk.value} exceeds the {policy.max_risk.value} ceiling for {agent_id!r}",
                ToolErrorCode.RISK_DENIED,
                risk=tool.risk,
            )

        # The agent's own permission set, from its definition, still applies.
        # A capability grant cannot conjure a permission the agent lacks.
        missing = frozenset(tool.required_permissions) - frozenset(agent_permissions)
        if missing:
            names = ", ".join(sorted(p.value for p in missing))
            return PolicyVerdict(
                False,
                f"agent {agent_id!r} lacks permission(s): {names}",
                ToolErrorCode.PERMISSION_DENIED,
                risk=tool.risk,
            )

        if tool.risk.rank > server.max_risk.rank:
            return PolicyVerdict(
                False,
                f"the server {server.server_id!r} may not serve {tool.risk.value} tools",
                ToolErrorCode.RISK_DENIED,
                risk=tool.risk,
            )

        # An untrusted server never gets to change anything.
        if server.trust_level is TrustLevel.UNTRUSTED and tool.risk.is_write:
            return PolicyVerdict(
                False,
                f"the server {server.server_id!r} is untrusted and may only serve reads",
                ToolErrorCode.RISK_DENIED,
                risk=tool.risk,
            )

        # The sandbox is not where production side effects happen.
        if self.environment == "sandbox" and tool.risk.rank >= ToolRisk.HIGH_RISK_WRITE.rank:
            return PolicyVerdict(
                False,
                f"{tool.risk.value} tools are not available in the sandbox environment",
                ToolErrorCode.RISK_DENIED,
                risk=tool.risk,
            )

        return PolicyVerdict(
            True,
            f"{agent_id} may call {tool.tool_id}",
            needs_approval=tool.risk.needs_approval,
            risk=tool.risk,
        )


def policies_from_registry(specs: Iterable[Any]) -> dict[str, AgentToolPolicy]:
    """Read every agent's grants from its own definition.

    One place to look: what an agent may reach is in the same file as its
    permissions and its prompt. An agent that declares no capabilities gets a
    policy granting nothing, which is not the same as having no policy but
    means the same thing at the gate.
    """
    policies: dict[str, AgentToolPolicy] = {}
    for spec in specs:
        try:
            max_risk = ToolRisk(spec.max_tool_risk)
        except ValueError:
            log.warning(
                "agent %s declares max_tool_risk %r, which is not a risk level; "
                "falling back to read_only",
                spec.id,
                spec.max_tool_risk,
            )
            max_risk = ToolRisk.READ_ONLY
        policies[spec.id] = AgentToolPolicy(
            agent_id=spec.id,
            capabilities=frozenset(spec.tool_capabilities),
            max_risk=max_risk,
        )
    return policies


def default_policies() -> dict[str, AgentToolPolicy]:
    """Grants for the agents on disk, read from their definitions."""
    from ..agents.registry import build_registry

    return policies_from_registry(build_registry().all())


@dataclass(slots=True)
class CapabilityCoverage:
    """Which of an agent's declared capabilities are actually reachable.

    Declaring a capability is a request, not a guarantee. An agent that asks
    for ``web.read`` when no web tool is registered can do nothing with it, and
    saying so plainly is better than letting it look ready.
    """

    agent_id: str
    backed: tuple[str, ...] = ()
    unbacked: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.unbacked

    @property
    def status(self) -> str:
        if not self.backed and not self.unbacked:
            return "no tools needed"
        if self.complete:
            return "ready"
        if self.backed:
            return "partial"
        return "no tools available"

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "backed": list(self.backed),
            "unbacked": list(self.unbacked),
        }


def capability_coverage(specs: Iterable[Any], tools: Iterable[Any]) -> list[CapabilityCoverage]:
    """Compare what each agent asks for against what is registered."""
    available = {tool.capability for tool in tools if tool.available}

    def is_backed(capability: str) -> bool:
        # A grant covers a capability and everything nested under it, so
        # "files.read" is backed by "files.read.read_file".
        return capability in available or any(
            existing.startswith(f"{capability}.") for existing in available
        )

    coverage: list[CapabilityCoverage] = []
    for spec in specs:
        backed = tuple(c for c in spec.tool_capabilities if is_backed(c))
        unbacked = tuple(c for c in spec.tool_capabilities if not is_backed(c))
        coverage.append(CapabilityCoverage(agent_id=spec.id, backed=backed, unbacked=unbacked))
    return coverage
