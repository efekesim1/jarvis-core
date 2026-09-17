"""The workspace file tools, and the boundary they must not cross.

A read tool that can be talked into reading `/etc/passwd` or a private key is
worse than no read tool. Every test here is one way out of the workspace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fake_credentials import fake_anthropic_key

from jarvis_core.integrations.files import WorkspaceFilesServer
from jarvis_core.mcp.errors import ToolError, ToolErrorCode
from jarvis_core.mcp.models import ToolRequest, ToolRisk
from jarvis_core.tools.policy import AgentToolPolicy


@pytest.fixture
def workspace(tmp_path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "README.md").write_text("# a project\n")
    (tmp_path / ".env").write_text(
        f"ANTHROPIC_API_KEY={fake_anthropic_key('SHOULDNOTBEREADABLE')}\n"
    )
    (tmp_path / "server.pem").write_text("-----BEGIN " + "PRIVATE KEY-----\nnope\n")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    return tmp_path


@pytest.fixture
def files(workspace) -> WorkspaceFilesServer:
    return WorkspaceFilesServer(workspace)


async def call(files: WorkspaceFilesServer, name: str, **arguments):
    return json.loads(await files.call_tool(name, arguments))


# -- the happy path ---------------------------------------------------------


async def test_a_file_inside_the_workspace_reads(files):
    result = await call(files, "read_file", path="src/main.py")
    assert "print('hello')" in result["content"]
    assert result["path"] == "src/main.py"


async def test_a_directory_lists(files):
    result = await call(files, "list_directory", path=".")
    names = {entry["name"] for entry in result["entries"]}
    assert "README.md" in names
    assert "src" in names


async def test_search_finds_by_pattern(files):
    result = await call(files, "search_files", pattern="*.py")
    assert result["matches"] == ["src/main.py"]


# -- getting out of the workspace ------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "../../etc/passwd",
        "/etc/passwd",
        "src/../../../etc/hosts",
        "src/../../..",
        "./../..",
    ],
)
async def test_a_path_that_leaves_the_workspace_is_refused(files, path):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": path})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED


async def test_a_symlink_pointing_out_is_refused(files, workspace, tmp_path):
    """A string check on the path would walk straight past this."""
    outside = tmp_path.parent / "outside.md"
    outside.write_text("a secret elsewhere\n")
    link = workspace / "innocent.md"
    try:
        os.symlink(outside, link)
    except OSError:  # pragma: no cover - some filesystems disallow it
        pytest.skip("symlinks are not available here")

    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": "innocent.md"})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED
    assert "outside the workspace" in caught.value.message


async def test_a_search_pattern_cannot_escape(files):
    for pattern in ("../*", "/etc/*"):
        with pytest.raises(ToolError) as caught:
            await files.call_tool("search_files", {"pattern": pattern})
        assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED


# -- what must never be read -----------------------------------------------


async def test_the_env_file_is_never_readable(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": ".env"})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED
    assert "credential" in caught.value.message


async def test_a_private_key_is_never_readable(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": "server.pem"})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED


async def test_the_git_directory_is_never_readable(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": ".git/config"})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED


async def test_an_unlisted_extension_is_refused(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": "photo.png"})
    assert caught.value.code is ToolErrorCode.SECURITY_BLOCKED


async def test_a_listing_does_not_reveal_a_secret_file(files):
    result = await call(files, "list_directory", path=".")
    names = {entry["name"] for entry in result["entries"]}
    assert ".env" not in names


async def test_a_search_never_returns_a_secret(files):
    result = await call(files, "search_files", pattern="*")
    assert not any(".env" in match or ".pem" in match for match in result["matches"])


# -- limits ------------------------------------------------------------------


async def test_an_oversized_file_is_refused(workspace):
    big = workspace / "big.txt"
    big.write_text("x" * 5000)
    server = WorkspaceFilesServer(workspace, max_bytes=1000)
    with pytest.raises(ToolError) as caught:
        await server.call_tool("read_file", {"path": "big.txt"})
    assert caught.value.code is ToolErrorCode.RESULT_TOO_LARGE


async def test_a_directory_is_not_a_file(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("read_file", {"path": "src"})
    assert caught.value.code is ToolErrorCode.EXTERNAL_ERROR


async def test_an_unknown_tool_is_refused(files):
    with pytest.raises(ToolError) as caught:
        await files.call_tool("delete_everything", {})
    assert caught.value.code is ToolErrorCode.TOOL_NOT_FOUND


# -- through the gateway ----------------------------------------------------


async def test_the_file_tools_are_read_only(fabric):
    tools = fabric.tools.by_server("files")
    assert {t.name for t in tools} == {"read_file", "list_directory", "search_files"}
    assert all(t.risk is ToolRisk.READ_ONLY for t in tools)


async def test_no_file_write_tool_exists(fabric):
    """Writing files is a later decision with its own approval path."""
    names = {t.name for t in fabric.tools.by_server("files")}
    for forbidden in ("write_file", "delete_file", "move_file", "create_directory"):
        assert forbidden not in names


async def test_a_traversal_through_the_gateway_is_blocked_and_audited(
    fabric, reader_permissions, audit
):
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="reader",
            capabilities=frozenset({"files.read"}),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    result = await fabric.gateway.call(
        ToolRequest(
            tool_id="files.read_file",
            arguments={"path": "../../../etc/passwd"},
            agent_id="reader",
            task_id="tsk_test",
        ),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.SECURITY_BLOCKED
    assert any(e.action in ("tool.failed", "tool.denied") for e in audit.events)


async def test_an_agent_without_the_grant_cannot_read_files(fabric, reader_permissions):
    result = await fabric.gateway.call(
        ToolRequest(
            tool_id="files.read_file",
            arguments={"path": "README.md"},
            agent_id="stranger",
        ),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
