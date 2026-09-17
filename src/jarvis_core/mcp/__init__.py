"""Model Context Protocol connectivity.

MCP is how the system reaches anything outside itself. This package holds the
protocol side only: servers, transports, discovery and the shapes they deal
in. Deciding whether a call is allowed happens one layer up, in
``jarvis_core.tools``, and no agent ever imports from here.
"""

from .errors import RETRYABLE, ToolError, ToolErrorCode, ToolFailure
from .mock_server import MockMCPServer
from .models import (
    HealthStatus,
    MCPServer,
    MCPTool,
    RateLimit,
    ToolRequest,
    ToolResult,
    ToolRetryPolicy,
    ToolRisk,
    TransportKind,
    TrustLevel,
)
from .registry import MCPServerRegistry, ToolDiscovery, classify_tool_risk
from .transport import HTTPTransport, InProcessTransport, MCPTransport, StdioTransport

__all__ = [
    "HTTPTransport",
    "HealthStatus",
    "InProcessTransport",
    "MCPServer",
    "MCPServerRegistry",
    "MCPTool",
    "MCPTransport",
    "MockMCPServer",
    "RETRYABLE",
    "RateLimit",
    "StdioTransport",
    "ToolDiscovery",
    "ToolError",
    "ToolErrorCode",
    "ToolFailure",
    "ToolRequest",
    "ToolResult",
    "ToolRetryPolicy",
    "ToolRisk",
    "TransportKind",
    "TrustLevel",
    "classify_tool_risk",
]
