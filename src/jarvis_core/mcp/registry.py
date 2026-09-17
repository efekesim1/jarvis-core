"""The server registry and tool discovery.

Two rules run through this module.

**Default deny.** A server nobody registered does not exist. A disabled one is
gone. An unhealthy one is refused unless policy says otherwise. Nothing is
reachable by accident.

**Discovery is not authorisation.** A server may publish fifty tools; that says
nothing about which of them any agent may call. Discovery records what exists
and assigns each tool a risk, and the gateway decides the rest.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from ..logging import AuditLog, get_logger
from .errors import ToolError, ToolErrorCode
from .models import (
    HealthStatus,
    MCPServer,
    MCPTool,
    ToolRisk,
    TransportKind,
    TrustLevel,
)
from .transport import InProcessHandler, InProcessTransport, MCPTransport, StdioTransport

log = get_logger("mcp.registry")

# What a tool name suggests it does. Discovery uses this only as a starting
# point; an explicit classification always wins, and anything unrecognised
# lands on the safe side rather than being assumed harmless.
_RISK_HINTS: tuple[tuple[tuple[str, ...], ToolRisk], ...] = (
    (("delete", "destroy", "drop", "purge", "wipe", "remove_all"), ToolRisk.DESTRUCTIVE),
    (("deploy", "release", "publish", "rollout", "migrate"), ToolRisk.HIGH_RISK_WRITE),
    (
        ("pay", "transfer", "charge", "refund", "send_email", "send_message", "post"),
        ToolRisk.EXTERNAL_SIDE_EFFECT,
    ),
    (("create", "update", "write", "set", "add", "edit", "patch"), ToolRisk.LOW_RISK_WRITE),
    (
        ("get", "list", "read", "search", "find", "lookup", "fetch", "query", "echo", "status"),
        ToolRisk.READ_ONLY,
    ),
)


def classify_tool_risk(name: str, description: str = "") -> ToolRisk:
    """Guess what a tool does from its name. Errs towards caution.

    Matching is on whole word segments, not substrings: "widget" contains
    "get", and a tool called ``frobnicate_widget`` is not a read.

    A name that matches nothing is treated as a high-risk write, because the
    alternative is calling an unknown tool as though it were harmless.
    """
    segments = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
    for needles, risk in _RISK_HINTS:
        for needle in needles:
            # Multi-word hints ("remove_all", "send_email") are compared as a
            # sequence of segments; single words must match a segment exactly.
            parts = needle.split("_")
            if len(parts) == 1:
                if needle in segments:
                    return risk
            elif all(part in segments for part in parts):
                return risk
    return ToolRisk.HIGH_RISK_WRITE


def _valid_schema(schema: Any) -> bool:
    """A usable JSON Schema object, or not."""
    return isinstance(schema, dict) and schema.get("type", "object") == "object"


class MCPServerRegistry:
    """Every server the system knows about, and the transports to reach them."""

    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self.audit = audit or AuditLog()
        self._servers: dict[str, MCPServer] = {}
        self._transports: dict[str, MCPTransport] = {}

    # -- registration -----------------------------------------------------

    def register(self, server: MCPServer, *, transport: MCPTransport | None = None) -> MCPServer:
        self._servers[server.server_id] = server
        if transport is not None:
            self._transports[server.server_id] = transport
        self.audit.emit(
            "mcp.server_registered",
            subject=server.server_id,
            metadata={
                "transport": server.transport.value,
                "trust": server.trust_level.value,
                "max_risk": server.max_risk.value,
            },
        )
        return server

    def register_in_process(
        self,
        server_id: str,
        handler: InProcessHandler,
        *,
        name: str | None = None,
        trust_level: TrustLevel = TrustLevel.LIMITED,
        max_risk: ToolRisk = ToolRisk.READ_ONLY,
        allowed_tools: tuple[str, ...] = (),
    ) -> MCPServer:
        """Register a server that lives in this process."""
        server = MCPServer(
            server_id=server_id,
            name=name or server_id,
            transport=TransportKind.IN_PROCESS,
            trust_level=trust_level,
            max_risk=max_risk,
            allowed_tools=allowed_tools,
        )
        return self.register(server, transport=InProcessTransport(handler))

    def register_stdio(
        self,
        server_id: str,
        command: str,
        args: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> MCPServer:
        """Register a real MCP server run as a subprocess."""
        server = MCPServer(
            server_id=server_id,
            name=kwargs.pop("name", server_id),
            transport=TransportKind.STDIO,
            endpoint=command,
            **kwargs,
        )
        return self.register(server, transport=StdioTransport(command, args))

    # -- lookup -----------------------------------------------------------

    def get(self, server_id: str) -> MCPServer | None:
        return self._servers.get(server_id)

    def all(self) -> list[MCPServer]:
        return sorted(self._servers.values(), key=lambda s: s.server_id)

    def transport_for(self, server_id: str) -> MCPTransport | None:
        return self._transports.get(server_id)

    def require_usable(self, server_id: str, *, allow_degraded: bool = True) -> MCPServer:
        """Return a server, or say precisely why it cannot be used.

        Default deny: an unregistered server is not an error to work around,
        it is a server that does not exist.
        """
        server = self._servers.get(server_id)
        if server is None:
            raise ToolError(
                ToolErrorCode.SERVER_NOT_FOUND,
                f"no MCP server {server_id!r} is registered",
            )
        if not server.enabled:
            raise ToolError(ToolErrorCode.SERVER_DISABLED, f"the server {server_id!r} is disabled")
        if server.health is HealthStatus.DISABLED:
            raise ToolError(ToolErrorCode.SERVER_DISABLED, f"the server {server_id!r} is disabled")
        if server.health is HealthStatus.UNHEALTHY or (
            server.health is HealthStatus.DEGRADED and not allow_degraded
        ):
            raise ToolError(
                ToolErrorCode.SERVER_UNHEALTHY,
                f"the server {server_id!r} is {server.health.value}",
            )
        return server

    # -- lifecycle --------------------------------------------------------

    def enable(self, server_id: str, enabled: bool = True) -> MCPServer:
        server = self._servers[server_id]
        server.enabled = enabled
        server.updated_at = datetime.now(UTC)
        self.audit.emit(
            "mcp.server_disabled" if not enabled else "mcp.server_enabled",
            subject=server_id,
        )
        return server

    def set_health(self, server_id: str, health: HealthStatus) -> MCPServer:
        server = self._servers[server_id]
        server.health = health
        server.updated_at = datetime.now(UTC)
        return server

    async def check_health(self, server_id: str) -> HealthStatus:
        """Ask a server whether it is alive.

        Listing tools has no side effect, which is the point: a health check
        must never change anything in the world.
        """
        server = self._servers.get(server_id)
        if server is None:
            return HealthStatus.UNKNOWN
        if not server.enabled:
            return self.set_health(server_id, HealthStatus.DISABLED).health

        transport = self._transports.get(server_id)
        if transport is None:
            return self.set_health(server_id, HealthStatus.UNKNOWN).health
        try:
            if not transport.connected:
                await transport.connect(server)
            await transport.list_tools()
        except Exception as exc:
            log.warning("health check failed for %s: %s", server_id, exc)
            self.audit.emit(
                "mcp.server_failed", subject=server_id, result="unhealthy", reason=str(exc)[:200]
            )
            return self.set_health(server_id, HealthStatus.UNHEALTHY).health
        server.last_seen = datetime.now(UTC)
        self.audit.emit("mcp.server_connected", subject=server_id, result="healthy")
        return self.set_health(server_id, HealthStatus.HEALTHY).health

    async def check_all(self) -> dict[str, HealthStatus]:
        return {s.server_id: await self.check_health(s.server_id) for s in self.all()}


class ToolDiscovery:
    """Turns what a server publishes into tools the system can reason about.

    Nothing a server says is taken at face value. Names are namespaced so two
    servers cannot collide, risks are assigned by this side, and a tool whose
    schema does not parse is registered as unavailable rather than dropped, so
    it shows up in an inspection instead of silently vanishing.
    """

    def __init__(self, registry: MCPServerRegistry, *, audit: AuditLog | None = None) -> None:
        self.registry = registry
        self.audit = audit or registry.audit

    async def discover(
        self,
        server_id: str,
        *,
        risk_overrides: dict[str, ToolRisk] | None = None,
        capability_prefix: str | None = None,
    ) -> list[MCPTool]:
        server = self.registry.require_usable(server_id)
        transport = self.registry.transport_for(server_id)
        if transport is None:
            raise ToolError(
                ToolErrorCode.SERVER_NOT_FOUND,
                f"the server {server_id!r} has no transport configured",
            )
        if not transport.connected:
            await transport.connect(server)

        try:
            published = await transport.list_tools()
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                ToolErrorCode.MCP_PROTOCOL_ERROR, f"tool discovery failed: {exc}"
            ) from exc

        overrides = risk_overrides or {}
        prefix = capability_prefix or server_id
        tools: list[MCPTool] = []

        for entry in published:
            name = str(entry.get("name") or "").strip()
            if not name:
                log.warning("%s published a tool with no name; skipping", server_id)
                continue
            if not server.permits_tool(name):
                log.debug("%s.%s is not in the server's allowed list; skipping", server_id, name)
                continue

            risk = overrides.get(name) or classify_tool_risk(
                name, str(entry.get("description", ""))
            )
            # A server may never publish a tool riskier than it is trusted for.
            if risk.rank > server.max_risk.rank:
                log.info(
                    "%s.%s classified %s, above the server's %s ceiling; not registering",
                    server_id,
                    name,
                    risk.value,
                    server.max_risk.value,
                )
                self.audit.emit(
                    "mcp.tool_refused",
                    subject=f"{server_id}.{name}",
                    reason=f"{risk.value} exceeds the server ceiling of {server.max_risk.value}",
                )
                continue

            schema = entry.get("inputSchema") or entry.get("input_schema") or {}
            unavailable = None
            if not _valid_schema(schema):
                unavailable = "the server published a schema that is not a JSON Schema object"
                schema = {}
                log.warning("%s.%s: %s", server_id, name, unavailable)

            tool = MCPTool(
                tool_id=f"{server_id}.{name}",
                name=name,
                server_id=server_id,
                description=str(entry.get("description", ""))[:500],
                input_schema=schema,
                risk=risk,
                capability=f"{prefix}.{name}",
                unavailable_reason=unavailable,
            )
            tools.append(tool)
            self.audit.emit(
                "mcp.tool_discovered",
                subject=tool.tool_id,
                metadata={"risk": risk.value, "available": tool.available},
            )

        log.info("discovered %d tool(s) on %s", len(tools), server_id)
        return tools
