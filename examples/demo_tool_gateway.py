#!/usr/bin/env python3
"""The tool gateway, end to end, offline.

Shows the pipeline doing its job: a granted read goes through, an ungranted
agent gets nothing, a deploy stops for the operator, a hostile result is neutralised,
a leaked credential is redacted, and the same write cannot happen twice.

    python examples/demo_tool_gateway.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Forced, not `setdefault`. The container image bakes JARVIS_ENVIRONMENT=sandbox
# into its ENV, and the sandbox environment refuses high-risk writes by policy —
# which is the gate this demo exists to show being *approved*, so the demo could
# not run there at all. The CI job worked around it with `-e`, which left anyone
# running the file by hand inside the container with the same failure. The demo
# is the right place to state what it needs.
os.environ["JARVIS_ENVIRONMENT"] = "production"  # so writes are reachable
os.environ.setdefault("JARVIS_LOG_LEVEL", "ERROR")
# The demo must behave the same everywhere, so it uses the offline GitHub
# fixture rather than whatever credential the machine happens to carry.
os.environ.pop("GITHUB_TOKEN", None)

from jarvis_core.config import get_settings  # noqa: E402
from jarvis_core.logging import AuditLog, configure_logging  # noqa: E402
from jarvis_core.mcp.models import ToolRequest, ToolRisk  # noqa: E402
from jarvis_core.security.permissions import Permission  # noqa: E402
from jarvis_core.tools.policy import AgentToolPolicy  # noqa: E402
from jarvis_core.tools.setup import build_tool_fabric  # noqa: E402

READER = frozenset({Permission.READ, Permission.NETWORK})
WRITER = frozenset({Permission.READ, Permission.WRITE, Permission.NETWORK})


def show(label: str, result) -> None:
    if result.ok:
        body = result.text.replace("\n", " ")[:110]
        extra = []
        if result.from_cache:
            extra.append("from cache")
        if result.sanitized:
            extra.append("sanitised")
        if result.truncated:
            extra.append("truncated")
        note = f"  [{', '.join(extra)}]" if extra else ""
        print(f"  ALLOWED  {label:<34} {body}{note}")
    else:
        print(f"  BLOCKED  {label:<34} {result.error.code.value}: {result.error.message[:80]}")


async def main() -> int:
    configure_logging()
    audit = AuditLog()
    fabric = await build_tool_fabric(
        settings=get_settings(),
        audit=audit,
        include_mock=True,
        mock_max_risk=ToolRisk.DESTRUCTIVE,
        include_github=True,
    )
    gateway = fabric.gateway

    # A researcher: reads only, and only the reads it was granted.
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="researcher",
            capabilities=frozenset(
                {
                    "mock.echo",
                    "mock.lookup",
                    "mock.hostile",
                    "mock.leaky",
                    "mock.huge",
                    "github.read",
                }
            ),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    # An operator: may write, but high-risk work still stops for the operator.
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="operator",
            capabilities=frozenset({"mock.create_note", "mock.deploy"}),
            max_risk=ToolRisk.HIGH_RISK_WRITE,
        )
    )

    def request(tool: str, agent: str = "researcher", **arguments) -> ToolRequest:
        return ToolRequest(tool_id=tool, arguments=arguments, agent_id=agent, task_id="tsk_demo")

    print("\nWhat the gateway lets through, and what it does not\n")

    show(
        "a granted read",
        await gateway.call(request("mock.echo", message="hello"), agent_permissions=READER),
    )
    show(
        "the same read again",
        await gateway.call(request("mock.echo", message="hello"), agent_permissions=READER),
    )
    show(
        "an agent with no grants",
        await gateway.call(
            request("mock.echo", agent="stranger", message="hi"), agent_permissions=READER
        ),
    )
    show(
        "a tool outside the grant",
        await gateway.call(request("mock.delete_everything"), agent_permissions=READER),
    )
    show(
        "a bad argument type",
        await gateway.call(request("mock.lookup", record_id="seven"), agent_permissions=READER),
    )
    show(
        "a result full of instructions",
        await gateway.call(request("mock.hostile"), agent_permissions=READER),
    )
    show(
        "a result carrying a key",
        await gateway.call(request("mock.leaky"), agent_permissions=READER),
    )
    show(
        "a two-megabyte result", await gateway.call(request("mock.huge"), agent_permissions=READER)
    )
    show(
        "github, with no token",
        await gateway.call(
            request("github.get_repository", owner="anthropics", repo="anthropic-sdk-python"),
            agent_permissions=READER,
        ),
    )

    print()
    deploy = request("mock.deploy", agent="operator", target="production")
    show("a deploy, unapproved", await gateway.call(deploy, agent_permissions=WRITER))

    approval = fabric.approvals.pending[0]
    print(f"\n  Waiting on the operator: {approval.agent_id} wants {approval.tool_id}")
    print(f"  arguments: {approval.arguments_summary}")
    fabric.approvals.approve(approval.approval_id, by="operator", note="approved for the demo")
    print("  -> the operator approves\n")
    show("the same deploy, approved", await gateway.call(deploy, agent_permissions=WRITER))
    show("the approval, replayed", await gateway.call(deploy, agent_permissions=WRITER))

    print()
    note = request("mock.create_note", agent="operator", title="one note")
    show("a write", await gateway.call(note, agent_permissions=WRITER))
    show(
        "the same write again",
        await gateway.call(
            request("mock.create_note", agent="operator", title="one note"),
            agent_permissions=WRITER,
        ),
    )

    print(f"\n{gateway.summary()['gateway']}")
    dumped = "\n".join(event.to_json() for event in audit.events)

    problems: list[str] = []
    if fabric.mock.call_counts.get("delete_everything"):
        problems.append("a destructive tool ran")
    if fabric.mock.call_counts.get("deploy", 0) != 1:
        problems.append("the deploy did not run exactly once")
    if fabric.mock.call_counts.get("create_note", 0) != 1:
        problems.append("the write did not run exactly once")
    if "sk-ant-api03" in dumped:
        problems.append("a credential reached the audit log")
    if gateway.stats.security_blocks < 1:
        problems.append("no security block was recorded")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: every gate held, and nothing ran that should not have")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
