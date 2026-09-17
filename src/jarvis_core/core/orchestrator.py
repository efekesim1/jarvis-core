"""The Agent Orchestrator.

Runs a plan. Steps that do not depend on each other run at the same time;
steps that do wait their turn. A failed step is retried with a budget, then an
alternative agent, then escalated. There is no unbounded retry anywhere.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any

from ..agents.base import AgentContext, AgentResult, AgentSpec
from ..agents.registry import AgentNotFound, AgentRegistry
from ..config import Settings, get_settings
from ..evaluation.engine import Evaluation, EvaluationEngine
from ..logging import AuditLog, get_logger
from ..observability.cache import ResultCache, work_key
from ..observability.cost import BudgetExceeded
from .progress import NullProgress, Progress, ProgressKind
from .task import Task, TaskEngine, TaskStatus

log = get_logger("core.orchestrator")


@dataclass(slots=True)
class StepOutcome:
    """What happened to one step of a plan."""

    step_id: str
    agent_id: str
    task: Task
    result: AgentResult | None = None
    evaluation: Evaluation | None = None
    attempts: int = 0
    skipped: bool = False
    error: str | None = None

    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok and not self.skipped

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "task_id": self.task.task_id,
            "status": self.task.status.value,
            "attempts": self.attempts,
            "skipped": self.skipped,
            "cached": self.cached,
            "error": self.error,
            "result": self.result.as_dict() if self.result else None,
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
        }


@dataclass(slots=True)
class RunReport:
    """The whole run: what succeeded, what failed, what it cost."""

    outcomes: list[StepOutcome] = field(default_factory=list)
    cost_usd: float = 0.0
    stopped_early: str | None = None

    @property
    def successful(self) -> list[StepOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[StepOutcome]:
        return [o for o in self.outcomes if not o.ok and not o.skipped]

    def outcome(self, step_id: str) -> StepOutcome | None:
        return next((o for o in self.outcomes if o.step_id == step_id), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [o.as_dict() for o in self.outcomes],
            "successful": len(self.successful),
            "failed": len(self.failed),
            "cost_usd": round(self.cost_usd, 6),
            "stopped_early": self.stopped_early,
        }


class AgentOrchestrator:
    """Executes plans against the registry."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        tasks: TaskEngine,
        evaluator: EvaluationEngine,
        audit: AuditLog | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.tasks = tasks
        self.evaluator = evaluator
        self.audit = audit or AuditLog()
        self.settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_agents)
        self.cache: ResultCache[AgentResult] = ResultCache(
            max_entries=self.settings.cache_max_entries,
            ttl_s=self.settings.cache_ttl_s,
        )

    # -- one step ---------------------------------------------------------

    async def _run_once(
        self, spec: AgentSpec, task: Task, ctx: AgentContext
    ) -> tuple[AgentResult, bool]:
        """Run one agent, sharing the answer with any identical run.

        The cache key is the agent, its goal and its inputs. Two steps that ask
        the same agent the same question with the same evidence get one model
        call between them, whether they overlap in time or arrive an hour apart.
        """
        runtime = self.registry.runtime_for(spec)

        async def produce() -> AgentResult:
            async with self._semaphore:
                return await asyncio.wait_for(runtime.run(spec, task, ctx), timeout=spec.timeout_s)

        if not self.settings.cache_enabled:
            return await produce(), False

        key = work_key(spec.id, spec.version, task.goal, task.inputs)
        result, was_cached = await self.cache.run(
            key,
            produce,
            cost_of=lambda r: r.cost_usd,
            # Never cache a failure. A transient provider error must not become
            # the stored answer for the next hour.
            cacheable=lambda r: r.ok and bool(r.text),
        )
        if was_cached:
            # A cached result belongs to the task that first produced it.
            # Re-point it so traces and reports stay honest, and charge nothing.
            return replace(result, task_id=task.task_id, cost_usd=0.0), True
        return result, False

    async def run_step(
        self,
        *,
        step_id: str,
        agent_id: str,
        task: Task,
        ctx: AgentContext,
        progress: Progress | None = None,
    ) -> StepOutcome:
        """Run one agent, retrying within its own budget.

        ``progress`` is told when the agent starts and what it produced. It is
        reporting only: nothing here waits on it, and a channel nobody is
        reading costs one queue append per event.
        """
        progress = progress or NullProgress()
        outcome = StepOutcome(step_id=step_id, agent_id=agent_id, task=task)
        try:
            spec = self.registry.require(agent_id)
        except AgentNotFound as exc:
            outcome.error = str(exc)
            self.tasks.fail(task, str(exc))
            return outcome

        max_attempts = max(
            1, min(spec.retry_policy.max_attempts, self.settings.max_retries_per_task + 1)
        )
        self.tasks.transition(task, TaskStatus.RUNNING)
        progress.emit(
            ProgressKind.AGENT_STARTED,
            step_id=step_id,
            agent_id=agent_id,
            agent=spec.name,
            swarm=spec.swarm,
            goal=task.goal[:200],
        )

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            outcome.attempts = attempt
            task.retry_count = attempt - 1
            try:
                result, from_cache = await self._run_once(spec, task, ctx)
            except BudgetExceeded as exc:
                # A budget stop is not a transient error; do not retry it.
                outcome.error = str(exc)
                self.tasks.fail(task, str(exc))
                self.audit.emit(
                    "agent.stopped_on_budget",
                    agent=agent_id,
                    task_id=task.task_id,
                    result="blocked",
                    reason=str(exc),
                )
                return outcome
            except TimeoutError:
                last_error = f"agent {agent_id} exceeded its {spec.timeout_s:.0f}s timeout"
                log.warning("%s (attempt %d/%d)", last_error, attempt, max_attempts)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "agent %s failed on attempt %d/%d: %s",
                    agent_id,
                    attempt,
                    max_attempts,
                    last_error,
                )
            else:
                outcome.result = result
                outcome.cached = from_cache
                task.cost_usd += result.cost_usd
                if result.ok:
                    evaluation = self.evaluator.evaluate(result, spec, task)
                    outcome.evaluation = evaluation
                    result.evaluation = evaluation.as_dict()
                    if evaluation.passed or attempt >= max_attempts:
                        self.tasks.complete(task, {"result": result.as_dict()})
                        _report_finished(progress, outcome, spec)
                        return outcome
                    last_error = f"the result scored {evaluation.overall:.2f} and did not pass"
                    log.info("retrying %s: %s", agent_id, last_error)
                else:
                    last_error = result.error or "the agent reported a failure"

            if attempt < max_attempts and spec.retry_policy.backoff_s > 0:
                await asyncio.sleep(spec.retry_policy.backoff_s)

        outcome.error = last_error or "the agent produced no usable result"
        # A result that failed evaluation is still worth keeping for the report.
        if outcome.result is not None and outcome.result.ok:
            self.tasks.complete(task, {"result": outcome.result.as_dict()})
        else:
            self.tasks.fail(task, outcome.error)
        self.audit.emit(
            "agent.failed",
            agent=agent_id,
            task_id=task.task_id,
            result="error",
            reason=outcome.error,
            metadata={"attempts": outcome.attempts},
        )
        _report_finished(progress, outcome, spec)
        return outcome

    # -- a whole plan -----------------------------------------------------

    async def run_plan(
        self,
        plan: Any,
        ctx: AgentContext,
        *,
        parent: Task | None = None,
        progress: Progress | None = None,
    ) -> RunReport:
        """Run every step of a plan, wave by wave.

        ``plan`` is a :class:`jarvis_core.core.planner.Plan`; it is typed loosely
        here so the orchestrator does not depend on the planner module.
        """
        progress = progress or NullProgress()
        report = RunReport()
        results_by_step: dict[str, AgentResult] = {}
        hops = 0

        waves = plan.waves()
        if len(waves) > self.settings.max_plan_waves:
            log.warning(
                "plan has %d waves; running the first %d and stopping there",
                len(waves),
                self.settings.max_plan_waves,
            )
            report.stopped_early = f"plan exceeded the {self.settings.max_plan_waves}-wave limit"
            waves = waves[: self.settings.max_plan_waves]

        for wave_index, wave in enumerate(waves, start=1):
            if hops + len(wave) > self.settings.max_agent_hops:
                report.stopped_early = (
                    f"stopped at the {self.settings.max_agent_hops}-agent-run limit for one request"
                )
                log.warning("%s", report.stopped_early)
                self.audit.emit(
                    "orchestrator.hop_limit",
                    task_id=parent.task_id if parent else None,
                    result="stopped",
                    reason=report.stopped_early,
                    metadata={"hops": hops, "limit": self.settings.max_agent_hops},
                )
                break

            coroutines = []
            wave_steps = []
            for step in wave:
                # Give each step whatever its dependencies produced.
                inputs = dict(step.inputs)
                findings = [
                    {
                        "step_id": dep,
                        "agent": results_by_step[dep].agent_id,
                        # Trimmed here and compressed again against the context
                        # budget in the runtime. A step never sees the run's
                        # whole history, only what it declared a dependency on.
                        "summary": results_by_step[dep].text[: ctx.budget.max_finding_chars * 2],
                        "confidence": results_by_step[dep].confidence,
                        "sources": results_by_step[dep].sources[
                            : ctx.budget.max_sources_per_finding
                        ],
                    }
                    for dep in step.depends_on
                    if dep in results_by_step
                ]
                if findings:
                    inputs["findings"] = findings

                missing = [d for d in step.depends_on if d not in results_by_step]
                task = self.tasks.create(
                    step.goal,
                    parent=parent,
                    agent=step.agent_id,
                    inputs=inputs,
                    dependencies=[
                        results_by_step[d].task_id for d in step.depends_on if d in results_by_step
                    ],
                    trace_id=ctx.trace.trace_id,
                )
                if missing and not findings:
                    # Every dependency failed, so this step has nothing to work with.
                    outcome = StepOutcome(
                        step_id=step.step_id,
                        agent_id=step.agent_id,
                        task=task,
                        skipped=True,
                        error=f"skipped: no dependency produced a result ({', '.join(missing)})",
                    )
                    self.tasks.transition(task, TaskStatus.BLOCKED)
                    self.tasks.fail(task, outcome.error or "")
                    report.outcomes.append(outcome)
                    continue

                wave_steps.append(step)
                coroutines.append(
                    self.run_step(
                        step_id=step.step_id,
                        agent_id=step.agent_id,
                        task=task,
                        ctx=ctx,
                        progress=progress,
                    )
                )

            if not coroutines:
                continue

            hops += len(coroutines)

            log.info("wave %d: running %d agent(s) in parallel", wave_index, len(coroutines))
            outcomes = await asyncio.gather(*coroutines, return_exceptions=True)
            for step, outcome in zip(wave_steps, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    log.error("step %s crashed: %s", step.step_id, outcome)
                    crashed = StepOutcome(
                        step_id=step.step_id,
                        agent_id=step.agent_id,
                        task=self.tasks.create(step.goal, parent=parent, agent=step.agent_id),
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                    report.outcomes.append(crashed)
                    continue
                report.outcomes.append(outcome)
                if outcome.result is not None:
                    report.cost_usd += outcome.result.cost_usd
                    if outcome.ok:
                        results_by_step[outcome.step_id] = outcome.result

            progress.emit(
                ProgressKind.COST,
                wave=wave_index,
                waves=len(waves),
                cost_usd=round(report.cost_usd, 6),
                steps_done=len(report.outcomes),
            )

            # Stop the run rather than burn the rest of the budget on a dead plan.
            try:
                ctx.router.cost.check_budgets(parent.task_id if parent else None)
            except BudgetExceeded as exc:
                report.stopped_early = str(exc)
                log.warning("stopping the plan early: %s", exc)
                progress.emit(ProgressKind.WARNING, detail=str(exc))
                break

        if report.stopped_early:
            progress.emit(ProgressKind.WARNING, detail=report.stopped_early)
        return report


def _report_finished(progress: Progress, outcome: StepOutcome, spec: AgentSpec) -> None:
    """Say how one step ended, with the numbers it produced.

    Reads them off the result rather than recomputing: the same figures reach
    the report, and two derivations of one number drift.
    """
    result = outcome.result
    progress.emit(
        ProgressKind.AGENT_FINISHED,
        step_id=outcome.step_id,
        agent_id=outcome.agent_id,
        agent=spec.name,
        ok=outcome.ok,
        cached=outcome.cached,
        attempts=outcome.attempts,
        error=outcome.error,
        model=result.model if result else None,
        cost_usd=round(result.cost_usd, 6) if result else 0.0,
        confidence=round(result.confidence, 3) if result else 0.0,
        input_tokens=result.input_tokens if result else 0,
        output_tokens=result.output_tokens if result else 0,
        score=round(outcome.evaluation.overall, 3) if outcome.evaluation else None,
        sources=len(result.sources) if result else 0,
    )
