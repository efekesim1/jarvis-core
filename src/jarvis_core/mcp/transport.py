"""Transport: how bytes reach a server.

Kept behind an interface so the client never knows whether a tool runs in this
process, in a subprocess, or across the network. The in-process transport is
what the whole test suite runs on, which is why the suite needs no network.

Transports carry no policy. They connect, call, and report failures in the
system's own error taxonomy. Deciding whether a call is allowed happens well
before anything reaches here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from ..logging import get_logger
from .errors import ToolError, ToolErrorCode
from .models import MCPServer, TransportKind

log = get_logger("mcp.transport")


@runtime_checkable
class MCPTransport(Protocol):
    """Everything the client needs from a way of reaching a server."""

    kind: TransportKind

    async def connect(self, server: MCPServer) -> None: ...
    async def list_tools(self) -> list[dict[str, Any]]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...
    async def close(self) -> None: ...
    @property
    def connected(self) -> bool: ...


@runtime_checkable
class InProcessHandler(Protocol):
    """A server implemented in Python, in this process."""

    async def list_tools(self) -> list[dict[str, Any]]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class InProcessTransport:
    """Talks to a handler object directly.

    Used by the mock server and by built-in adapters such as the read-only
    GitHub integration, which speak an API rather than the MCP wire protocol
    but belong behind the same gateway.
    """

    kind = TransportKind.IN_PROCESS

    def __init__(self, handler: InProcessHandler) -> None:
        self._handler = handler
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, server: MCPServer) -> None:
        self._connected = True

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise ToolError(ToolErrorCode.MCP_PROTOCOL_ERROR, "transport is not connected")
        return await self._handler.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._connected:
            raise ToolError(ToolErrorCode.MCP_PROTOCOL_ERROR, "transport is not connected")
        return await self._handler.call_tool(name, arguments)

    async def close(self) -> None:
        self._connected = False


class StdioTransport:
    """A real MCP server run as a subprocess, over the official SDK.

    The SDK is an optional dependency: the core system and its whole test
    suite work without it, and this transport says so plainly rather than
    failing somewhere deeper.
    """

    kind = TransportKind.STDIO

    def __init__(self, command: str, args: tuple[str, ...] = ()) -> None:
        self.command = command
        self.args = args
        self._session: Any = None
        self._stack: Any = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self, server: MCPServer) -> None:
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ToolError(
                ToolErrorCode.NOT_IMPLEMENTED,
                "the official MCP SDK is not installed; run: pip install 'jarvis-core[mcp]'",
            ) from exc

        self._stack = AsyncExitStack()
        try:
            params = StdioServerParameters(command=self.command, args=list(self.args))
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except Exception as exc:
            await self.close()
            raise ToolError(
                ToolErrorCode.MCP_PROTOCOL_ERROR,
                f"could not start MCP server {server.server_id}: {exc}",
            ) from exc

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            raise ToolError(ToolErrorCode.MCP_PROTOCOL_ERROR, "transport is not connected")
        try:
            listing = await self._session.list_tools()
        except Exception as exc:
            raise ToolError(
                ToolErrorCode.MCP_PROTOCOL_ERROR, f"tool discovery failed: {exc}"
            ) from exc
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": getattr(tool, "inputSchema", {}) or {},
            }
            for tool in listing.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise ToolError(ToolErrorCode.MCP_PROTOCOL_ERROR, "transport is not connected")
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, f"the tool call failed: {exc}") from exc
        if getattr(result, "isError", False):
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, _text_of(result))
        return _text_of(result)

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # pragma: no cover - shutdown is best effort
                log.debug("error closing the transport: %s", exc)


def _text_of(result: Any) -> str:
    """Pull plain text out of an SDK result without importing its types."""
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


class HTTPTransport:
    """Placeholder for the streamable HTTP transport.

    Declared so the registry can describe an HTTP server, and deliberately not
    implemented: shipping a half-built network transport that nothing exercises
    would be worse than saying it is not here yet.
    """

    kind = TransportKind.HTTP

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    @property
    def connected(self) -> bool:
        return False

    async def connect(self, server: MCPServer) -> None:
        raise ToolError(
            ToolErrorCode.NOT_IMPLEMENTED,
            "the HTTP transport is not implemented; use stdio or an in-process adapter",
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        raise ToolError(ToolErrorCode.NOT_IMPLEMENTED, "the HTTP transport is not implemented")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise ToolError(ToolErrorCode.NOT_IMPLEMENTED, "the HTTP transport is not implemented")

    async def close(self) -> None:
        return None


async def call_with_timeout(
    transport: MCPTransport, name: str, arguments: dict[str, Any], timeout_s: float
) -> Any:
    """Call a tool, turning a hang into a normalised timeout."""
    try:
        return await asyncio.wait_for(transport.call_tool(name, arguments), timeout=timeout_s)
    except TimeoutError as exc:
        raise ToolError(
            ToolErrorCode.TIMEOUT, f"the tool did not answer within {timeout_s:.1f}s"
        ) from exc
