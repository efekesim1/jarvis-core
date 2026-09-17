"""Adapters for real external systems.

Each one is an in-process MCP server: it speaks whatever the service speaks,
publishes tools with schemas, and sits behind the gateway like anything else.
An agent cannot tell an adapter from a real MCP server, which is the point.

All of them are read-only. Writing to the world is a separate decision, agent
by agent and tool by tool, with an approval path already in place for it.
"""

from .files import WorkspaceFilesServer
from .github import GitHubReadOnlyServer
from .web import WebServer

__all__ = ["GitHubReadOnlyServer", "WebServer", "WorkspaceFilesServer"]
