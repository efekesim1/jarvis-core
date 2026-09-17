"""Reading files from a workspace, and nothing outside it.

The second real adapter. Reading is useful and cannot break anything, which is
why it comes before writing; a write tool would be a separate decision with an
approval path.

Everything here exists to stop one class of bug: an agent, or something an
agent read, talking its way into a path that is not the workspace. So every
path is resolved and then checked against the workspace root, symlinks
included, and anything that resolves outside is refused with the path it
resolved to rather than the path that was asked for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..logging import get_logger
from ..mcp.errors import ToolError, ToolErrorCode

log = get_logger("integrations.files")

# What may be read. A read tool that will hand back a private key is not a
# read tool worth having.
ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".sh",
        ".md",
        ".txt",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".html",
        ".css",
        ".sql",
        ".xml",
    }
)

# Never read, whatever the suffix says.
DENIED_NAMES: frozenset[str] = frozenset(
    {".env", ".env.local", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "credentials"}
)
DENIED_SUFFIXES: frozenset[str] = frozenset({".pem", ".key", ".p12", ".pfx", ".keystore"})
DENIED_PARTS: frozenset[str] = frozenset(
    {".git", ".ssh", ".aws", ".gnupg", "node_modules", ".venv"}
)

MAX_BYTES = 256 * 1024
MAX_ENTRIES = 200


class WorkspaceFilesServer:
    """An in-process server publishing read-only file tools.

    The workspace is the boundary. There is no argument, no configuration and
    no tool that moves it.
    """

    def __init__(self, workspace: Path | str, *, max_bytes: int = MAX_BYTES) -> None:
        self.workspace = Path(workspace).resolve()
        self.max_bytes = max_bytes

    # -- the boundary -----------------------------------------------------

    def _resolve(self, raw: str) -> Path:
        """Resolve a path and prove it is inside the workspace.

        Resolution happens first, so ``../..`` and a symlink pointing out are
        both caught by the same check rather than by a string test that a
        symlink would walk straight past.
        """
        cleaned = (raw or "").strip()
        if cleaned in ("", "."):
            candidate = self.workspace
        elif os.path.isabs(cleaned):
            # Resolved and checked like any other path; an absolute path inside
            # the workspace is fine, and one outside it is refused below.
            candidate = Path(cleaned)
        else:
            candidate = self.workspace / cleaned

        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # a symlink loop, for instance
            raise ToolError(
                ToolErrorCode.SECURITY_BLOCKED, f"that path could not be resolved: {exc}"
            ) from exc

        if resolved != self.workspace and self.workspace not in resolved.parents:
            log.warning("refused a path outside the workspace: %s -> %s", raw, resolved)
            raise ToolError(
                ToolErrorCode.SECURITY_BLOCKED,
                f"that path is outside the workspace (it resolves to {resolved})",
                details={"workspace": str(self.workspace)},
            )
        return resolved

    def _check_readable(self, path: Path) -> None:
        if any(part in DENIED_PARTS for part in path.parts):
            raise ToolError(ToolErrorCode.SECURITY_BLOCKED, "that directory is never readable")
        if path.name in DENIED_NAMES or path.suffix.lower() in DENIED_SUFFIXES:
            raise ToolError(
                ToolErrorCode.SECURITY_BLOCKED,
                f"{path.name} may hold a credential and is never readable",
            )
        if path.is_file() and path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ToolError(
                ToolErrorCode.SECURITY_BLOCKED,
                f"{path.suffix or 'files with no extension'} is not on the readable list",
            )

    # -- discovery --------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "read_file",
                "description": "Reads one text file from the workspace.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "maxLength": 400}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_directory",
                "description": "Lists what is in one workspace directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "maxLength": 400}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "search_files",
                "description": "Finds workspace files whose name matches a pattern.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "maxLength": 100},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        ]

    # -- invocation -------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        handler = {
            "read_file": self._read_file,
            "list_directory": self._list_directory,
            "search_files": self._search_files,
        }.get(name)
        if handler is None:
            raise ToolError(ToolErrorCode.TOOL_NOT_FOUND, f"no file tool named {name!r}")
        return handler(arguments)

    def _read_file(self, arguments: dict[str, Any]) -> str:
        path = self._resolve(str(arguments["path"]))
        self._check_readable(path)
        if not path.is_file():
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, "that is not a file")

        size = path.stat().st_size
        if size > self.max_bytes:
            raise ToolError(
                ToolErrorCode.RESULT_TOO_LARGE,
                f"that file is {size} bytes and the limit is {self.max_bytes}",
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(
                ToolErrorCode.INVALID_RESULT, "that file is not readable as text"
            ) from exc
        return json.dumps(
            {
                "path": str(path.relative_to(self.workspace)),
                "bytes": size,
                "content": text,
            }
        )

    def _list_directory(self, arguments: dict[str, Any]) -> str:
        path = self._resolve(str(arguments["path"]))
        if not path.is_dir():
            raise ToolError(ToolErrorCode.EXTERNAL_ERROR, "that is not a directory")

        entries: list[dict[str, Any]] = []
        for child in sorted(path.iterdir())[:MAX_ENTRIES]:
            if child.name in DENIED_NAMES or child.name in DENIED_PARTS:
                continue
            entries.append(
                {
                    "name": child.name,
                    "kind": "directory" if child.is_dir() else "file",
                    "bytes": child.stat().st_size if child.is_file() else None,
                    "readable": child.is_dir() or child.suffix.lower() in ALLOWED_SUFFIXES,
                }
            )
        return json.dumps(
            {"path": str(path.relative_to(self.workspace)) or ".", "entries": entries}
        )

    def _search_files(self, arguments: dict[str, Any]) -> str:
        pattern = str(arguments["pattern"])
        if pattern.startswith("/") or ".." in pattern:
            raise ToolError(
                ToolErrorCode.SECURITY_BLOCKED, "a search pattern may not leave the workspace"
            )
        limit = min(int(arguments.get("limit", 25)), 100)

        found: list[str] = []
        for match in self.workspace.rglob(pattern):
            if any(part in DENIED_PARTS for part in match.parts):
                continue
            if match.name in DENIED_NAMES or match.suffix.lower() in DENIED_SUFFIXES:
                continue
            if match.is_file() and match.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            found.append(str(match.relative_to(self.workspace)))
            if len(found) >= limit:
                break
        return json.dumps({"pattern": pattern, "matches": sorted(found)})
