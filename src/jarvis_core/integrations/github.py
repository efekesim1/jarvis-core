"""GitHub, read-only.

The first real integration, chosen because it is useful and because reading
cannot break anything. It speaks the GitHub REST API rather than the MCP wire
protocol, and sits behind the same in-process transport as everything else, so
the gateway cannot tell the difference and neither can an agent.

Nothing here writes. No issue is filed, no branch is pushed, no repository is
touched. Writing GitHub tools is a later decision with its own approval path,
and the architecture already supports it: a write tool would classify as a
write and stop at the approval gate like any other.

Without a token this falls back to a small offline fixture, so the test suite
and CI stay green with no credential.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..logging import get_logger
from ..mcp.errors import ToolError, ToolErrorCode

log = get_logger("integrations.github")

API_ROOT = "https://api.github.com"

# The environment variable holding the token. The value is read at the moment
# of use and never stored, logged, or put in a server record.
TOKEN_REFERENCE = "GITHUB_TOKEN"


class GitHubReadOnlyServer:
    """An in-process server publishing read-only GitHub tools."""

    def __init__(self, *, token_reference: str = TOKEN_REFERENCE, timeout_s: float = 20.0) -> None:
        self.token_reference = token_reference
        self.timeout_s = timeout_s

    # -- credentials ------------------------------------------------------

    @property
    def has_credentials(self) -> bool:
        return bool(os.environ.get(self.token_reference))

    def _headers(self) -> dict[str, str]:
        """Build request headers. The token never leaves this method."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "jarvis-core",
        }
        token = os.environ.get(self.token_reference)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # -- discovery --------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_repository",
                "description": "Reads a repository's public metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "maxLength": 100},
                        "repo": {"type": "string", "maxLength": 100},
                    },
                    "required": ["owner", "repo"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_issues",
                "description": "Lists open issues on a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "maxLength": 100},
                        "repo": {"type": "string", "maxLength": 100},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["owner", "repo"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "search_repositories",
                "description": "Searches public repositories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 256},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        ]

    # -- invocation -------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self.has_credentials:
            # No token is not a failure. The architecture is what this
            # integration proves, and a fixture proves it without a secret.
            log.info(
                "no %s is set; answering %s from the offline fixture", self.token_reference, name
            )
            return _fixture(name, arguments)

        handler = {
            "get_repository": self._get_repository,
            "list_issues": self._list_issues,
            "search_repositories": self._search_repositories,
        }.get(name)
        if handler is None:
            raise ToolError(ToolErrorCode.TOOL_NOT_FOUND, f"no GitHub tool named {name!r}")
        return await handler(arguments)

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            import httpx2 as httpx
        except ImportError:  # pragma: no cover - depends on the installed SDK
            try:
                import httpx
            except ImportError as exc:
                raise ToolError(
                    ToolErrorCode.NOT_IMPLEMENTED,
                    "no HTTP client is installed; the GitHub integration needs one",
                ) from exc

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{API_ROOT}{path}", headers=self._headers(), params=params or {}
                )
        except Exception as exc:
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, f"could not reach GitHub: {exc}") from exc

        if response.status_code in (401, 403):
            # Never echo the response body here: it can carry request details.
            raise ToolError(
                ToolErrorCode.AUTH_FAILURE,
                f"GitHub refused the request ({response.status_code}); check {self.token_reference}",
            )
        if response.status_code == 404:
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, "GitHub returned 404: not found")
        if response.status_code == 429:
            raise ToolError(ToolErrorCode.RATE_LIMITED, "GitHub rate limit reached")
        if response.status_code >= 400:
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, f"GitHub returned {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise ToolError(
                ToolErrorCode.INVALID_RESULT, "GitHub returned unparseable JSON"
            ) from exc

    async def _get_repository(self, arguments: dict[str, Any]) -> str:
        data = await self._request(f"/repos/{arguments['owner']}/{arguments['repo']}")
        return json.dumps(
            {
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "stars": data.get("stargazers_count"),
                "language": data.get("language"),
                "license": (data.get("license") or {}).get("spdx_id"),
                "pushed_at": data.get("pushed_at"),
                "open_issues": data.get("open_issues_count"),
                "archived": data.get("archived"),
            }
        )

    async def _list_issues(self, arguments: dict[str, Any]) -> str:
        data = await self._request(
            f"/repos/{arguments['owner']}/{arguments['repo']}/issues",
            {"state": "open", "per_page": min(int(arguments.get("limit", 10)), 50)},
        )
        return json.dumps(
            [
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "created_at": item.get("created_at"),
                }
                for item in data
                if isinstance(item, dict) and "pull_request" not in item
            ]
        )

    async def _search_repositories(self, arguments: dict[str, Any]) -> str:
        data = await self._request(
            "/search/repositories",
            {"q": arguments["query"], "per_page": min(int(arguments.get("limit", 10)), 25)},
        )
        return json.dumps(
            [
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "stars": item.get("stargazers_count"),
                    "license": (item.get("license") or {}).get("spdx_id"),
                }
                for item in data.get("items", [])
            ]
        )


def _fixture(name: str, arguments: dict[str, Any]) -> str:
    """A small offline answer, so the flow is exercisable with no token."""
    if name == "get_repository":
        return json.dumps(
            {
                "full_name": f"{arguments.get('owner')}/{arguments.get('repo')}",
                "description": "an offline fixture, returned because no GitHub token is set",
                "stars": 0,
                "language": None,
                "license": None,
                "pushed_at": None,
                "open_issues": 0,
                "archived": False,
                "_fixture": True,
            }
        )
    if name == "list_issues":
        return json.dumps([])
    if name == "search_repositories":
        return json.dumps([])
    raise ToolError(ToolErrorCode.TOOL_NOT_FOUND, f"no GitHub tool named {name!r}")
