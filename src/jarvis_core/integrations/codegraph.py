"""The code graph, behind the gateway.

`graphify` parses a folder into a knowledge graph with tree-sitter and answers
questions about it — where a symbol is defined, what reaches what, how two
things connect. It is deterministic, local, and asks no model anything for
code, which is what makes it worth having: an agent that can traverse a graph
does not have to grep, and a grep that misses is an agent that invents.

It is an *external* program, deliberately. Importing the package would pull
two dozen tree-sitter grammars into this process for a capability most runs
never use, and pin our dependency tree to its. Shelling out keeps both trees
separate and makes the absence of the tool an ordinary, reportable state
rather than an import error.

Four things this module holds to, and each is why the adapter exists at all
rather than an agent running `graphify` itself:

* **No command is ever built from a variable.** The executable is resolved
  once from `PATH`; every argument is a separate element of an argv list, and
  no shell is involved. A question containing `; rm -rf /` is one argument
  that `graphify` will fail to match, which is the correct outcome.
* **It reads, and only inside the workspace.** Three query subcommands are
  reachable — `query`, `explain`, `path`. The ones that write, install hooks,
  serve, or reach a model are not in the table, so there is no argument that
  gets to them.
* **It cannot run forever or return forever.** A timeout and an output cap,
  both enforced here rather than trusted to the child.
* **The result is data, not instruction.** It goes back as JSON and through
  the gateway's own sanitising like every other tool result. A comment in
  the operator's own code saying "ignore previous instructions" is a string in a graph,
  and it is treated as one.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..logging import get_logger
from ..mcp.errors import ToolError, ToolErrorCode

log = get_logger("integrations.codegraph")

# The executable this adapter drives. Looked up on PATH, never taken from a
# request, and never interpolated into a string.
EXECUTABLE = "graphify"

# Where `graphify` leaves the graph it built. Its absence is the difference
# between "not installed" and "installed but nothing indexed yet", and those
# need different sentences.
GRAPH_FILENAME = ".graphify/graph.json"

# One query is a graph traversal, not a crawl. Past this something is wrong
# with the graph rather than with the question.
TIMEOUT_S = 45.0

# What one answer may be. A graph query that returns a megabyte has answered
# the wrong question, and the context budget would drop most of it anyway.
MAX_OUTPUT_CHARS = 20_000

# The subcommands that read. Written out as a table rather than validated with
# a pattern: the set of things this adapter can invoke is the set of keys here,
# and adding one is an edit to this file, which is the point.
READ_COMMANDS: dict[str, tuple[str, ...]] = {
    "query": ("query",),
    "explain": ("explain",),
    "path": ("path",),
}

# A runner takes an argv list and a working directory and returns
# (exit code, stdout, stderr). Injectable so the tests exercise this module's
# own logic — the argv it builds, the caps it applies, the errors it maps —
# without needing `graphify` installed, and so the suite stays offline.
Runner = Callable[[list[str], Path], Awaitable[tuple[int, str, str]]]


async def _run(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run the child with no shell, and kill it if it overstays."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT_S)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ToolError(
            ToolErrorCode.TIMEOUT,
            f"the code graph did not answer within {TIMEOUT_S:.0f}s",
        ) from None
    return (
        process.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


class CodeGraphServer:
    """An in-process server publishing read-only code-graph tools."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        runner: Runner | None = None,
        executable: str | None = None,
    ) -> None:
        # Resolved once. Every child runs with this as its working directory,
        # and no request can change it.
        self.workspace = Path(workspace).resolve()
        self._runner = runner or _run
        # A test runner means the executable is not consulted at all; without
        # one it must be on PATH before anything is registered.
        self._executable = executable or (
            "<injected>" if runner else shutil.which(EXECUTABLE) or ""
        )

    # -- what is actually here --------------------------------------------

    @property
    def installed(self) -> bool:
        return bool(self._executable)

    @property
    def graph_path(self) -> Path:
        return self.workspace / GRAPH_FILENAME

    @property
    def indexed(self) -> bool:
        return self.graph_path.is_file()

    @property
    def available(self) -> bool:
        """Installed, pointed at a real directory, and something is indexed.

        All three, because a tool registered without them answers every call
        with the same error and an agent cannot tell a missing install from an
        empty index. When it is not available it is not registered, and the
        log says which of the three is missing.
        """
        return self.installed and self.workspace.is_dir() and self.indexed

    def why_unavailable(self) -> str | None:
        if not self.installed:
            return f"{EXECUTABLE} is not on PATH; install it with: pip install graphifyy"
        if not self.workspace.is_dir():
            return f"the workspace {self.workspace} does not exist"
        if not self.indexed:
            return (
                f"nothing is indexed yet; run `{EXECUTABLE} .` in {self.workspace} "
                f"to build {GRAPH_FILENAME}"
            )
        return None

    # -- discovery --------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "query",
                "description": (
                    "Asks the code graph a question about this workspace and returns "
                    "the nodes and edges that answer it. Traversal, not search: it "
                    "knows what calls what, not what matches a string."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "maxLength": 400},
                        "budget": {"type": "integer", "minimum": 100, "maximum": 8000},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "explain",
                "description": (
                    "Explains one symbol, file or concept in the graph: what it is, "
                    "what it touches, and where it is defined."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"node": {"type": "string", "maxLength": 200}},
                    "required": ["node"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "path",
                "description": (
                    "Traces the connection between two things in the graph, and says "
                    "plainly when there is none."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "maxLength": 200},
                        "target": {"type": "string", "maxLength": 200},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
            },
        ]

    # -- invocation -------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in READ_COMMANDS:
            raise ToolError(ToolErrorCode.TOOL_NOT_FOUND, f"no code graph tool named {name!r}")
        if not self.installed:
            raise ToolError(
                ToolErrorCode.NOT_IMPLEMENTED,
                self.why_unavailable() or "the code graph is not available",
            )

        argv = [self._executable, *READ_COMMANDS[name], *self._arguments_for(name, arguments)]
        code, out, err = await self._runner(argv, self.workspace)

        if code != 0:
            # The child's own words, capped. A stack trace from a subprocess is
            # not an answer, and an agent given one will quote it as evidence.
            detail = (err.strip() or out.strip() or "no output")[:400]
            raise ToolError(
                ToolErrorCode.EXTERNAL_ERROR,
                f"the code graph could not answer: {detail}",
            )

        text = out.strip()
        truncated = len(text) > MAX_OUTPUT_CHARS
        return json.dumps(
            {
                "tool": name,
                "workspace": self.workspace.name,
                "answer": text[:MAX_OUTPUT_CHARS],
                "truncated": truncated,
                # Said out loud because it is the whole reason to prefer this
                # over a model's recollection: nothing here was generated.
                "derived_from": "a tree-sitter parse of this workspace, not a model",
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _arguments_for(name: str, arguments: dict[str, Any]) -> list[str]:
        """Turn one validated request into argv elements.

        Each value becomes exactly one element. Nothing is joined, quoted or
        interpolated, so a value containing a space, a semicolon or a newline
        is a value — there is no shell for it to mean anything to.
        """
        if name == "query":
            argv = [str(arguments["question"])]
            budget = arguments.get("budget")
            if budget is not None:
                argv += ["--budget", str(int(budget))]
            return argv
        if name == "explain":
            return [str(arguments["node"])]
        return [str(arguments["source"]), str(arguments["target"])]

    def summary(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "indexed": self.indexed,
            "workspace": str(self.workspace),
            "graph": str(self.graph_path) if self.indexed else None,
            "unavailable_because": self.why_unavailable(),
        }
