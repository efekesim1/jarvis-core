#!/usr/bin/env python3
"""What the tool gateway costs.

Eleven checks run on every tool call — policy, risk, budget, rate limit,
approval, argument validation, duplicate detection, cache, execution, result
validation, redaction — and until now the honest answer to "what does that
cost" was "nobody has measured it".

Unknown is a bad answer for a number that is easy to get, and it is the kind
of unknown that turns into folklore: someone assumes it is free, someone else
assumes it is the bottleneck, and neither has run anything.

So this measures it against the same call made directly, with the same server
and the same arguments. It runs offline against the mock server, so what it
measures is the gateway's own work and not a network.

    python examples/benchmark_gateway.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JARVIS_ENVIRONMENT", "sandbox")
os.environ.setdefault("JARVIS_LOG_LEVEL", "ERROR")

from jarvis_core.mcp.mock_server import MockMCPServer  # noqa: E402
from jarvis_core.mcp.models import RateLimit, ToolRequest, ToolRisk  # noqa: E402
from jarvis_core.security.permissions import Permission  # noqa: E402
from jarvis_core.tools.policy import AgentToolPolicy  # noqa: E402
from jarvis_core.tools.setup import build_tool_fabric  # noqa: E402

RUNS = 200

# The echo tool's own limit is 30 a minute, which a benchmark would exhaust in
# its first second. Raising it here rather than in the tool definition: the
# limit is right for real use and wrong for measuring, and changing the shipped
# default to make a benchmark run would be the benchmark editing the system to
# suit itself.
BENCHMARK_LIMIT = RateLimit(per_minute=100_000, per_agent_per_minute=100_000, per_task=100_000)

# What the gateway may add per call before it is worth caring about. A tool
# call is a network round trip in every real case; a millisecond of local
# checking against that is not the thing to optimise. This is a regression
# bar, not a target.
BUDGET_MS = 2.0


async def main() -> int:
    fabric = await build_tool_fabric(include_mock=True)
    gateway = fabric.gateway
    server = MockMCPServer()

    # The echo tool: read-only, no side effects, and it does almost nothing —
    # which is what a benchmark of the *gateway* wants. A tool that did real
    # work would drown the number being measured.
    tool = next((t for t in fabric.tools.all() if t.tool_id == "mock.echo"), None)
    if tool is None:
        print("FAIL: the mock server did not register the echo tool")
        return 1

    arguments = {"message": "benchmark"}
    # An agent actually allowed to call this tool. Without one the benchmark
    # measures the policy check refusing, which is the cheapest path through
    # the gateway and the least interesting — the first version reported
    # 0.07ms and "201 denied", timing a door being shut.
    policy_agent = "benchmark_agent"
    gateway.policy.grant(
        AgentToolPolicy(
            agent_id=policy_agent,
            tools=frozenset({tool.tool_id}),
            max_risk=ToolRisk.READ_ONLY,
            max_calls_per_task=10_000,
        )
    )

    # Warm both paths: the first call through either pays for imports and
    # first-touch allocation, and timing that measures Python starting up.
    await server.call_tool(tool.name, arguments)
    # The permission the agent holds, separately from the policy grant. Both
    # are required — declaring a tool does not grant it — which is the point of
    # the split and also the reason the first version of this benchmark timed
    # nothing but refusals.
    permissions = frozenset({Permission.READ})
    tool.rate_limit = BENCHMARK_LIMIT
    await gateway.call(
        ToolRequest(tool_id=tool.tool_id, agent_id=policy_agent, arguments=arguments),
        agent_permissions=permissions,
    )

    direct: list[float] = []
    guarded: list[float] = []
    for index in range(RUNS):
        started = time.perf_counter()
        await server.call_tool(tool.name, arguments)
        direct.append((time.perf_counter() - started) * 1000)

        # A distinct argument each time, so the result cache and the duplicate
        # check do not turn this into a benchmark of a dictionary lookup.
        started = time.perf_counter()
        await gateway.call(
            ToolRequest(
                tool_id=tool.tool_id,
                agent_id=policy_agent,
                arguments={"message": f"benchmark {index}"},
            ),
            agent_permissions=permissions,
        )
        guarded.append((time.perf_counter() - started) * 1000)

    direct_median = statistics.median(direct)
    guarded_median = statistics.median(guarded)
    overhead = guarded_median - direct_median

    print(f"{'':22} {'median':>9} {'p95':>9}")
    for label, series in (("direct call", direct), ("through the gateway", guarded)):
        ordered = sorted(series)
        p95 = ordered[int(len(ordered) * 0.95)]
        print(f"{label:22} {statistics.median(series):>8.3f}ms {p95:>8.3f}ms")

    print()
    print(f"The gateway adds {overhead:.3f}ms per call, on median, over {RUNS} runs.")
    print(f"Its checks: {', '.join(gateway.stages())}" if hasattr(gateway, "stages") else "")
    print(
        "A real tool call is a network round trip. This is what the local\n"
        "checking costs against that, and it is the number that was missing."
    )

    stats = gateway.stats.as_dict()
    print(f"\n{stats['allowed']} allowed, {stats['denied']} denied during the run.")
    if stats["allowed"] < RUNS:
        print(
            f"\nFAIL: only {stats['allowed']} of {RUNS} calls got through, so this "
            "timed the gateway refusing rather than the gateway working"
        )
        return 1

    if overhead > BUDGET_MS:
        print(f"\nFAIL: {overhead:.3f}ms is over the {BUDGET_MS}ms regression bar")
        return 1
    print(f"\nPASS: {overhead:.3f}ms, inside the {BUDGET_MS}ms bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
