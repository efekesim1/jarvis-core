"""The command line: every command runs, and the two that spend say so first.

A CLI is thin, which is exactly why it is worth a test: nothing else catches a
command that calls a method the runtime does not have. Each command below is
invoked for real against the offline configuration, so a renamed attribute
fails here rather than in somebody's terminal.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from jarvis_core.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    [
        ["agents"],
        ["agents", "--capabilities"],
        ["models"],
        ["status"],
        ["status", "--json"],
        ["cost"],
        ["memory"],
        ["doctor"],
        ["verify"],
        ["plan", "research the market for AI note taking apps"],
    ],
)
def test_the_read_only_commands_run(command):
    result = runner.invoke(app, command)
    assert result.exit_code == 0, f"{command} failed:\n{result.output}\n{result.exception}"
    assert result.output.strip(), f"{command} printed nothing"


def test_plan_shows_the_waves_without_spending_anything():
    result = runner.invoke(app, ["plan", "research a market and then build something"])
    assert result.exit_code == 0
    assert "Execution waves" in result.output


def test_asking_with_no_credential_never_reports_success(monkeypatch):
    """Two honest outcomes, and no third.

    Either the router has nothing usable and the command refuses with a
    message saying what to set, or the SDK claims it might have a credential
    — it also resolves a login profile, so an absent environment variable is
    not proof — and the run fails. What must never happen is a zero exit and
    an answer that no model produced.
    """
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_ENABLED", "0")
    result = runner.invoke(app, ["ask", "anything", "--no-approve"])
    assert result.exit_code != 0, "a run with no credential reported success"
    assert (
        "No model provider is configured" in result.output
        or "failed" in result.output.lower()
        or "could not" in result.output.lower()
    ), result.output


def test_help_lists_what_the_tool_can_do():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "plan", "agents", "models", "status", "doctor", "serve"):
        assert command in result.output
