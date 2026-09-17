#!/usr/bin/env python3
"""Measure what the cost controls actually save.

Runs the same workload with the controls off and then on, and prints the
difference. No API key and no network: every model call is answered by the
mock provider, but the token counts and prices are the real ones from the
catalog, so the ratio is honest.

    python examples/benchmark_cost.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JARVIS_ENVIRONMENT", "sandbox")
os.environ.setdefault("JARVIS_LOG_LEVEL", "WARNING")

from jarvis_core.agents.registry import build_registry  # noqa: E402
from jarvis_core.config import get_settings  # noqa: E402
from jarvis_core.core.context import ContextBudget  # noqa: E402
from jarvis_core.core.runtime import JarvisRuntime  # noqa: E402
from jarvis_core.evaluation.engine import EvaluationEngine  # noqa: E402
from jarvis_core.logging import AuditLog, configure_logging  # noqa: E402
from jarvis_core.models.catalog import default_catalog  # noqa: E402
from jarvis_core.models.mock_provider import MockProvider  # noqa: E402
from jarvis_core.models.router import ModelRouter  # noqa: E402
from jarvis_core.security.approval import AutoApproveGate  # noqa: E402
from jarvis_core.security.guardian import SecurityGuardian  # noqa: E402

# A realistic day: some questions repeat, which is exactly where the money goes.
WORKLOAD = [
    "research an AI e-commerce business opportunity",
    "research the AI agent framework market",
    "research an AI e-commerce business opportunity",  # a repeat
    "what is MCP?",
    "research the AI agent framework market",  # a repeat
    "what is MCP?",  # a repeat
]

# A verbose agent. Long answers are what make context budgeting matter.
LONG_ANSWER = (
    "A detailed finding covering market size, the players already active, the "
    "regulatory picture, pricing patterns, and where the evidence is thin. "
) * 40


def responder(request):
    agent = request.metadata.get("agent", "unknown")
    return json.dumps(
        {
            "summary": f"[{agent}] {LONG_ANSWER}",
            "confidence": 0.82,
            "sources": [{"url": f"https://example.com/{agent}", "title": f"{agent} source"}],
        },
        ensure_ascii=False,
    )


def build(*, cache: bool, tight_context: bool) -> JarvisRuntime:
    settings = get_settings().model_copy()
    settings.cache_enabled = cache
    if not tight_context:
        # What the system looks like with no context budget: everything flows
        # downstream untouched.
        settings.max_finding_chars = 1_000_000
        settings.max_findings = 100
        settings.max_context_chars = 10_000_000
    settings.daily_budget_usd = 10_000  # the benchmark must not stop halfway

    audit = AuditLog()
    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))
    router = ModelRouter(
        providers={"mock": MockProvider(responder=responder)},
        catalog=catalog,
        settings=settings,
        audit=audit,
    )
    jarvis = JarvisRuntime(
        registry=build_registry(audit=audit, directory=ROOT / "agents"),
        router=router,
        guardian=SecurityGuardian(settings=settings, audit=audit, approval_gate=AutoApproveGate()),
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
        use_model_planner=False,
    )
    jarvis.context_budget = ContextBudget.from_settings(settings)
    return jarvis


async def measure(label: str, *, cache: bool, tight_context: bool) -> dict:
    jarvis = build(cache=cache, tight_context=tight_context)
    for request in WORKLOAD:
        await jarvis.handle(request)
    summary = jarvis.cost.summary()
    tokens = sum(r.input_tokens + r.output_tokens for r in jarvis.cost.records)
    return {
        "label": label,
        "model_calls": summary["calls"],
        "tokens": tokens,
        "cost_usd": summary["total_usd"],
        "cache": jarvis.orchestrator.cache.stats.as_dict(),
    }


def pct(before: float, after: float) -> str:
    if before <= 0:
        return "n/a"
    return f"{(1 - after / before) * 100:5.1f}%"


async def main() -> int:
    configure_logging()

    baseline = await measure("no controls", cache=False, tight_context=False)
    context_only = await measure("context budget only", cache=False, tight_context=True)
    full = await measure("context budget + cache", cache=True, tight_context=True)

    rows = [baseline, context_only, full]
    print(f"\nWorkload: {len(WORKLOAD)} requests, {len(set(WORKLOAD))} of them distinct\n")
    header = f"{'configuration':<26}{'calls':>7}{'tokens':>10}{'cost':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<26}{row['model_calls']:>7}{row['tokens']:>10}"
            f"{'$' + format(row['cost_usd'], '.5f'):>12}"
        )

    print("\nSaving against no controls")
    print(
        f"  context budget only     tokens {pct(baseline['tokens'], context_only['tokens'])}"
        f"   cost {pct(baseline['cost_usd'], context_only['cost_usd'])}"
    )
    print(
        f"  context budget + cache  tokens {pct(baseline['tokens'], full['tokens'])}"
        f"   cost {pct(baseline['cost_usd'], full['cost_usd'])}"
    )
    print(f"\ncache: {full['cache']}\n")

    # The benchmark is also a check: if the controls stop saving, this fails.
    if full["cost_usd"] >= baseline["cost_usd"]:
        print("FAIL: the controls did not reduce cost")
        return 1
    if full["model_calls"] >= baseline["model_calls"]:
        print("FAIL: the cache did not reduce the number of model calls")
        return 1
    print("PASS: the controls reduce both calls and cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
