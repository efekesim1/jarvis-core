#!/usr/bin/env python3
"""The overnight loop that durable storage made possible.

    23:00  a night shift runs, hits something that needs the operator, and blocks
    07:00  the operator opens a different terminal, sees it, and decides
    07:01  the shift picks the decision up and finishes the work

Each step runs in a separate Python process against one database file, so this
is the real thing rather than three objects sharing memory.

    python examples/demo_overnight_approval.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")

# Each stage is a whole process. Nothing is shared but the database file.
SHIFT = """
import asyncio, sys
sys.path.insert(0, {src!r})
from jarvis_core.mcp.models import ToolRequest, ToolRisk
from jarvis_core.security.permissions import Permission
from jarvis_core.storage import open_storage
from jarvis_core.tools.policy import AgentToolPolicy
from jarvis_core.tools.setup import build_tool_fabric

async def main():
    storage = open_storage({db!r})
    fabric = await build_tool_fabric(
        include_mock=True, mock_max_risk=ToolRisk.DESTRUCTIVE,
        include_github=False, storage=storage,
    )
    fabric.policy.environment = "production"
    fabric.policy.grant(AgentToolPolicy(
        agent_id="deployment_agent",
        capabilities=frozenset({{"mock.deploy"}}),
        max_risk=ToolRisk.HIGH_RISK_WRITE,
    ))
    approval = storage.approvals.for_request({request_id!r}) if {resume} else None
    request = ToolRequest(
        tool_id="mock.deploy",
        arguments={{"target": "production"}},
        agent_id="deployment_agent",
        task_id="tsk_overnight",
        request_id=approval.request_id if approval else {request_id!r},
        approval_id=approval.approval_id if approval else None,
    )
    result = await fabric.gateway.call(
        request, agent_permissions=frozenset({{Permission.READ, Permission.WRITE}})
    )
    if result.ok:
        print("RESULT ok deployed=" + str(fabric.mock.call_counts.get("deploy", 0)))
    else:
        print("RESULT " + result.error.code.value)
    storage.close()

asyncio.run(main())
"""

OPERATOR = """
import sys
sys.path.insert(0, {src!r})
from jarvis_core.storage import open_storage

storage = open_storage({db!r})
pending = storage.approvals.pending()
for approval in pending:
    print("WAITING " + approval.agent_id + " wants " + approval.tool_id
          + " with " + approval.arguments_summary)
    ok, reason = storage.approvals.decide(
        approval.approval_id, "approved", by="operator", note="fine, ship it"
    )
    print("DECIDED " + ("approved" if ok else reason))
storage.close()
"""


def run(label: str, script: str, env: dict[str, str]) -> str:
    path = Path(env["_SCRIPT_DIR"]) / f"{label}.py"
    path.write_text(script)
    result = subprocess.run(  # noqa: S603 - a fixed script, no shell
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **{k: v for k, v in env.items() if not k.startswith("_")}},
    )
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(f"{label} failed")
    return result.stdout


def main() -> int:
    with tempfile.TemporaryDirectory() as work:
        db = str(Path(work) / "jarvis.db")
        env = {
            "JARVIS_ENVIRONMENT": "production",
            "JARVIS_LOG_LEVEL": "ERROR",
            "JARVIS_DATA_DIR": work,
            "_SCRIPT_DIR": work,
        }
        request_id = "treq_overnight_demo"

        print("\n23:00  the night shift runs\n")
        out = run(
            "shift",
            SHIFT.format(src=SRC, db=db, request_id=request_id, resume=False),
            env,
        )
        blocked = "approval_required" in out
        print(f"  {out.strip()}")
        print("  -> nothing was deployed; the shift blocked and moved on\n")

        print("07:00  the operator opens another terminal\n")
        out = run("decider", OPERATOR.format(src=SRC, db=db), env)
        for line in out.strip().splitlines():
            print(f"  {line}")
        decided = "DECIDED approved" in out
        print()

        print("07:01  the shift picks the decision up\n")
        out = run(
            "resume",
            SHIFT.format(src=SRC, db=db, request_id=request_id, resume=True),
            env,
        )
        print(f"  {out.strip()}")
        executed = "ok deployed=1" in out

        print("\n07:02  and the same approval, tried again\n")
        out = run(
            "replay",
            SHIFT.format(src=SRC, db=db, request_id=request_id, resume=True),
            env,
        )
        print(f"  {out.strip()}")
        replay_blocked = "security_blocked" in out

    problems = []
    if not blocked:
        problems.append("the shift did not block on approval")
    if not decided:
        problems.append("the decision was not recorded")
    if not executed:
        problems.append("the approved work did not run")
    if not replay_blocked:
        problems.append("a replayed approval was not blocked")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: blocked overnight, decided in the morning, ran once, replay refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
