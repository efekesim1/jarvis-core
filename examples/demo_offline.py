#!/usr/bin/env python3
"""Run the whole JARVIS pipeline with no API key and no network.

Every model call is answered by the mock provider, so this shows the plumbing
end to end - planning, parallel agents, evaluation, cost accounting and the
audit trail - without spending anything.

    python examples/demo_offline.py
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
from jarvis_core.core.runtime import JarvisRuntime  # noqa: E402
from jarvis_core.evaluation.engine import EvaluationEngine  # noqa: E402
from jarvis_core.logging import AuditLog, configure_logging  # noqa: E402
from jarvis_core.models.catalog import default_catalog  # noqa: E402
from jarvis_core.models.mock_provider import MockProvider  # noqa: E402
from jarvis_core.models.router import ModelRouter  # noqa: E402
from jarvis_core.security.approval import AutoApproveGate  # noqa: E402
from jarvis_core.security.guardian import SecurityGuardian  # noqa: E402

REQUEST = "ABD'de AI automation konusunda yeni bir business opportunity araştır"


def fake_answer(request) -> str:
    """Stand in for a real agent reply."""
    agent = request.metadata.get("agent", "unknown")
    return json.dumps(
        {
            "summary": (
                f"[{agent}] A worked finding for the request, with enough substance "
                "that the evaluation engine has something real to score: market "
                "context, a rough size, who is already active, and where the "
                "evidence is thin."
            ),
            "confidence": 0.82,
            "sources": [{"url": f"https://example.com/{agent}", "title": f"{agent} source"}],
        },
        ensure_ascii=False,
    )


def build_offline_jarvis() -> JarvisRuntime:
    settings = get_settings()
    audit = AuditLog()
    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))
    router = ModelRouter(
        providers={"mock": MockProvider(responder=fake_answer)},
        catalog=catalog,
        settings=settings,
        audit=audit,
    )
    return JarvisRuntime(
        registry=build_registry(audit=audit, directory=ROOT / "agents"),
        router=router,
        guardian=SecurityGuardian(settings=settings, audit=audit, approval_gate=AutoApproveGate()),
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
        use_model_planner=False,
    )


async def main() -> int:
    configure_logging("WARNING")
    jarvis = build_offline_jarvis()

    print(f"OPERATOR: {REQUEST}\n")
    plan = await jarvis.make_plan(REQUEST)
    print(f"JARVIS read this as: {plan.intent.value}")
    print(f"Plan ({plan.source}): {plan.rationale}")
    for index, wave in enumerate(plan.waves(), start=1):
        agents = ", ".join(s.agent_id for s in wave)
        print(f"  wave {index}: {agents}")

    print("\nrunning...\n")
    response = await jarvis.handle(REQUEST)

    print("=" * 72)
    print(response.answer)
    print("=" * 72)

    print("\nPer-agent results")
    for outcome in response.report.outcomes:
        score = f"{outcome.evaluation.overall:.2f}" if outcome.evaluation else "n/a"
        status = "ok" if outcome.ok else ("skipped" if outcome.skipped else "failed")
        print(
            f"  {outcome.step_id:<10} {outcome.agent_id:<22} {status:<8} "
            f"score={score}  cost=${outcome.result.cost_usd if outcome.result else 0:.5f}"
        )

    print(
        f"\ntotal cost ${response.cost_usd:.5f} | confidence {response.confidence:.2f} "
        f"| {len(response.sources)} source(s) | trace {response.trace_id}"
    )
    print(f"tasks: {jarvis.tasks.snapshot()}")
    print(f"audit entries: {len(jarvis.audit.events)}")
    for warning in response.warnings:
        print(f"warning: {warning}")
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
