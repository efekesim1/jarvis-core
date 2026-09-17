"""The tool registry: one place that knows every callable tool.

Tools arrive from discovery on several servers, so two things matter here.
Identity is namespaced, because a ``search`` on one server is not a ``search``
on another. And registering a tool records that it exists, never that anyone
may call it.
"""

from __future__ import annotations

from typing import Any

from ..logging import AuditLog, get_logger
from ..mcp.errors import ToolError, ToolErrorCode
from ..mcp.models import MCPTool, ToolRisk

log = get_logger("tools.registry")


class ToolRegistry:
    """Every tool the system can see, indexed by id and by capability."""

    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self.audit = audit or AuditLog()
        self._tools: dict[str, MCPTool] = {}

    # -- registration -----------------------------------------------------

    def register(self, tool: MCPTool, *, replace: bool = False) -> MCPTool:
        existing = self._tools.get(tool.tool_id)
        if existing is not None and not replace:
            if existing.server_id != tool.server_id:
                # Should be impossible while ids carry the server, but a
                # collision here would silently reroute calls.
                raise ToolError(
                    ToolErrorCode.MCP_PROTOCOL_ERROR,
                    f"tool id {tool.tool_id!r} is claimed by two servers",
                )
            if existing.schema_version != tool.schema_version:
                log.warning(
                    "%s changed schema version %s -> %s; agents may need updating",
                    tool.tool_id,
                    existing.schema_version,
                    tool.schema_version,
                )
        self._tools[tool.tool_id] = tool
        return tool

    def register_many(self, tools: list[MCPTool], *, replace: bool = True) -> int:
        for tool in tools:
            self.register(tool, replace=replace)
        return len(tools)

    def remove_server_tools(self, server_id: str) -> int:
        """Drop every tool a server published, for a rediscovery."""
        doomed = [t for t in self._tools.values() if t.server_id == server_id]
        for tool in doomed:
            del self._tools[tool.tool_id]
        return len(doomed)

    # -- lookup -----------------------------------------------------------

    def get(self, tool_id: str) -> MCPTool | None:
        return self._tools.get(tool_id)

    def require(self, tool_id: str) -> MCPTool:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ToolError(
                ToolErrorCode.TOOL_NOT_FOUND,
                f"no tool {tool_id!r} is registered",
                details={"known": len(self._tools)},
            )
        if not tool.enabled:
            raise ToolError(ToolErrorCode.TOOL_DISABLED, f"the tool {tool_id!r} is disabled")
        if tool.unavailable_reason:
            raise ToolError(
                ToolErrorCode.TOOL_DISABLED,
                f"the tool {tool_id!r} is unavailable: {tool.unavailable_reason}",
            )
        return tool

    def all(self, *, available_only: bool = False) -> list[MCPTool]:
        tools = sorted(self._tools.values(), key=lambda t: t.tool_id)
        return [t for t in tools if t.available] if available_only else tools

    def by_server(self, server_id: str) -> list[MCPTool]:
        return [t for t in self.all() if t.server_id == server_id]

    def by_capability(self, capability: str) -> list[MCPTool]:
        return [t for t in self.all() if t.capability == capability]

    def by_risk(self, risk: ToolRisk) -> list[MCPTool]:
        return [t for t in self.all() if t.risk is risk]

    def capabilities(self) -> list[str]:
        return sorted({t.capability for t in self.all()})

    def enable(self, tool_id: str, enabled: bool = True) -> MCPTool:
        tool = self._tools[tool_id]
        tool.enabled = enabled
        self.audit.emit(
            "tool.enabled" if enabled else "tool.disabled", tool=tool_id, subject=tool_id
        )
        return tool

    def summary(self) -> dict[str, Any]:
        by_risk: dict[str, int] = {}
        for tool in self.all():
            by_risk[tool.risk.value] = by_risk.get(tool.risk.value, 0) + 1
        return {
            "tools": len(self._tools),
            "available": len(self.all(available_only=True)),
            "by_risk": dict(sorted(by_risk.items())),
            "capabilities": len(self.capabilities()),
        }

    def __len__(self) -> int:
        return len(self._tools)
