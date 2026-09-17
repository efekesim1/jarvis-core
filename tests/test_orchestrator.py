"""Parallel execution, retry budgets and failure handling."""

from __future__ import annotations

import asyncio
import time

import pytest

from jarvis_core.agents.base import AgentContext, AgentResult
from jarvis_core.core.orchestrator import AgentOrchestrator
from jarvis_core.core.planner import Intent, Plan, PlanStep, RulePlanner
from jarvis_core.core.task import TaskEngine, TaskStatus
from jarvis_core.evaluation.engine import EvaluationEngine


@pytest.fixture
def orchestrator(registry, audit, settings, router):
    return AgentOrchestrator(
        registry=registry,
        tasks=TaskEngine(),
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
    )


class SlowRuntime:
    """A runtime that sleeps, so parallelism is observable."""

    name = "llm"

    def __init__(self, delay: float = 0.15) -> None:
        self.delay = delay
        self.started: list[float] = []

    async def run(self, spec, task, ctx: AgentContext) -> AgentResult:
        self.started.append(time.perf_counter())
        await asyncio.sleep(self.delay)
        return AgentResult(
            agent_id=spec.id,
            task_id=task.task_id,
            ok=True,
            text="a finding with enough words in it to score as reasonably complete " * 5,
            data={"summary": "x"},
            confidence=0.8,
            sources=[{"url": "https://example.com"}],
            model="claude-sonnet-5",
        )


class FailingRuntime:
    """Fails a fixed number of times, then succeeds."""

    name = "llm"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.attempts = 0

    async def run(self, spec, task, ctx: AgentContext) -> AgentResult:
        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("the provider fell over")
        return AgentResult(
            agent_id=spec.id,
            task_id=task.task_id,
            ok=True,
            text="recovered after a retry, with a long enough answer to score well " * 5,
            data={"summary": "ok"},
            confidence=0.8,
            model="claude-sonnet-5",
        )


async def test_independent_agents_actually_run_at_the_same_time(orchestrator, registry, context):
    runtime = SlowRuntime(delay=0.2)
    registry.register_runtime(runtime)
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="one"),
            PlanStep(step_id="b", agent_id="market_researcher", goal="two"),
            PlanStep(step_id="c", agent_id="trend_analyst", goal="three"),
        ],
    )
    started = time.perf_counter()
    report = await orchestrator.run_plan(plan, context)
    elapsed = time.perf_counter() - started

    assert len(report.successful) == 3
    # Three 0.2s agents run together take well under the 0.6s they would take in sequence.
    assert elapsed < 0.5


async def test_a_dependent_step_runs_after_its_dependency(orchestrator, registry, context):
    registry.register_runtime(SlowRuntime(delay=0.05))
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="gather"),
            PlanStep(
                step_id="b",
                agent_id="research_synthesizer",
                goal="merge",
                depends_on=["a"],
            ),
        ],
    )
    report = await orchestrator.run_plan(plan, context)
    first = report.outcome("a")
    second = report.outcome("b")
    assert first and second and first.ok and second.ok
    assert first.task.completed_at is not None
    assert second.task.started_at >= first.task.started_at


async def test_a_dependent_step_receives_its_dependency_findings(orchestrator, registry, context):
    registry.register_runtime(SlowRuntime(delay=0.01))
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="gather"),
            PlanStep(step_id="b", agent_id="research_synthesizer", goal="merge", depends_on=["a"]),
        ],
    )
    report = await orchestrator.run_plan(plan, context)
    downstream = report.outcome("b")
    assert downstream is not None
    assert downstream.task.inputs["findings"][0]["agent"] == "web_researcher"


async def test_a_transient_failure_is_retried(orchestrator, registry, context):
    runtime = FailingRuntime(fail_times=1)
    registry.register_runtime(runtime)
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[PlanStep(step_id="a", agent_id="web_researcher", goal="gather")],
    )
    report = await orchestrator.run_plan(plan, context)
    assert report.outcome("a").ok
    assert runtime.attempts == 2


async def test_retries_stop_at_the_agent_budget(orchestrator, registry, context):
    runtime = FailingRuntime(fail_times=99)
    registry.register_runtime(runtime)
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[PlanStep(step_id="a", agent_id="web_researcher", goal="gather")],
    )
    report = await orchestrator.run_plan(plan, context)
    outcome = report.outcome("a")
    assert not outcome.ok
    assert runtime.attempts == 2  # web_researcher declares max_attempts: 2
    assert outcome.task.status is TaskStatus.FAILED


async def test_a_step_whose_dependencies_all_failed_is_skipped(orchestrator, registry, context):
    registry.register_runtime(FailingRuntime(fail_times=99))
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="gather"),
            PlanStep(step_id="b", agent_id="research_synthesizer", goal="merge", depends_on=["a"]),
        ],
    )
    report = await orchestrator.run_plan(plan, context)
    downstream = report.outcome("b")
    assert downstream.skipped
    assert "no dependency produced a result" in downstream.error


async def test_an_unknown_agent_fails_its_step_without_crashing_the_run(
    orchestrator, registry, context
):
    registry.register_runtime(SlowRuntime(delay=0.01))
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="one"),
            PlanStep(step_id="b", agent_id="ghost_agent", goal="two"),
        ],
    )
    report = await orchestrator.run_plan(plan, context)
    assert report.outcome("a").ok
    assert not report.outcome("b").ok
    assert "unknown agent" in report.outcome("b").error


async def test_the_run_records_what_it_cost(orchestrator, registry, context):
    plan = RulePlanner(registry).plan("research the AI agent market")
    report = await orchestrator.run_plan(plan, context)
    assert report.cost_usd >= 0.0
    assert len(report.outcomes) == len(plan.steps)


async def test_parallelism_is_capped_by_the_configured_limit(
    registry, audit, settings, router, context, monkeypatch
):
    monkeypatch.setattr(settings, "max_parallel_agents", 2)
    orchestrator = AgentOrchestrator(
        registry=registry,
        tasks=TaskEngine(),
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
    )

    in_flight = 0
    peak = 0

    class CountingRuntime:
        name = "llm"

        async def run(self, spec, task, ctx):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return AgentResult(
                agent_id=spec.id, task_id=task.task_id, ok=True, text="x" * 400, confidence=0.8
            )

    registry.register_runtime(CountingRuntime())
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[PlanStep(step_id=f"s{i}", agent_id="web_researcher", goal="go") for i in range(5)],
    )
    await orchestrator.run_plan(plan, context)
    assert peak <= 2
