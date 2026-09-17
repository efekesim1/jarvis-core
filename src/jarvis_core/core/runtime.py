"""The runtime: the layer that turns a request into an answer.

Not a model. This is what works out what a request means, plans it, picks
agents, runs them in dependency order, judges what comes back, merges it into
one answer and reports what it cost.

    REQUEST -> PLAN -> AGENTS -> EVALUATION -> ANSWER

Everything here is generic. What an owner-specific build adds — a model of
the person, a goal tree, a skill library, a notice queue — arrives by
subclassing and answering the four hooks below, so there is exactly one
implementation of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.base import AgentContext
from ..agents.registry import AgentRegistry, build_registry
from ..config import Settings, get_settings
from ..evaluation.engine import EvaluationEngine
from ..logging import AuditLog, get_logger
from ..memory.base import MemoryScope
from ..memory.manager import MemoryManager, RetrievalBudget
from ..models.base import Message, ModelRequest
from ..models.router import ModelRouter, TaskKind, build_default_router
from ..observability.cost import BudgetExceeded, CostEngine
from ..observability.tracing import Tracer
from ..security.approval import ApprovalGate
from ..security.guardian import SecurityGuardian
from ..verification.verifier import AgentVerifier
from .context import ContextBudget
from .messages import MessageBus
from .orchestrator import AgentOrchestrator, RunReport
from .planner import Intent, ModelPlanner, Plan, RulePlanner, detect_intent
from .progress import NullProgress, Progress, ProgressKind
from .task import Task, TaskEngine, TaskStatus

log = get_logger("runtime")


@dataclass(slots=True)
class JarvisResponse:
    """What comes back to the operator."""

    request: str
    answer: str
    ok: bool
    intent: Intent
    plan: Plan
    report: RunReport
    task: Task
    trace_id: str
    cost_usd: float = 0.0
    confidence: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cached_steps: int = 0
    recalled: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "answer": self.answer,
            "ok": self.ok,
            "intent": self.intent.value,
            "task_id": self.task.task_id,
            "trace_id": self.trace_id,
            "cost_usd": round(self.cost_usd, 6),
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "warnings": list(self.warnings),
            "cached_steps": self.cached_steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "recalled": self.recalled,
            "plan": self.plan.as_dict(),
            "report": self.report.as_dict(),
        }


def name_of(registry: AgentRegistry, agent_id: str) -> str:
    """An agent's display name, or its id when it is not in the registry.

    A plan can name an agent that validation is about to reject, and a
    progress event that crashed on the way to reporting that would replace a
    clear refusal with a KeyError.
    """
    spec = registry.get(agent_id)
    return spec.name if spec is not None else agent_id


SYNTHESIS_SYSTEM = (
    "You are JARVIS. Merge the findings below into one direct answer. "
    "Lead with the answer. Say plainly where the findings disagree or where the evidence is "
    "thin. Do not invent anything that is not in the findings. Keep it tight."
)


class JarvisRuntime:
    """The orchestration and command layer."""

    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        router: ModelRouter | None = None,
        tasks: TaskEngine | None = None,
        guardian: SecurityGuardian | None = None,
        evaluator: EvaluationEngine | None = None,
        audit: AuditLog | None = None,
        tracer: Tracer | None = None,
        memory: MemoryManager | None = None,
        settings: Settings | None = None,
        use_model_planner: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.audit = audit or AuditLog()
        self.registry = registry if registry is not None else build_registry(audit=self.audit)
        self.router = router if router is not None else build_default_router(audit=self.audit)
        self.tasks = tasks or TaskEngine()
        self.guardian = guardian or SecurityGuardian(settings=self.settings, audit=self.audit)
        self.evaluator = evaluator or EvaluationEngine(router=self.router)
        self.tracer = tracer or Tracer()
        self.memory = memory or MemoryManager()
        # Whether each agent has been shown to work, as opposed to merely
        # existing. Built lazily by `verification`, since a sweep costs money.
        self._verifier: AgentVerifier | None = None
        # Attached by `attach_tools`; agents get it through the run context.
        self.tools: Any = None
        # Set by `build_jarvis` when a database is configured.
        self.storage: Any = None
        self.bus = MessageBus()
        # A bounded tail of what each agent did, so the system can look at
        # its own record without re-reading every trace.
        self._recorded_runs: list[dict[str, Any]] = []
        self.orchestrator = AgentOrchestrator(
            registry=self.registry,
            tasks=self.tasks,
            evaluator=self.evaluator,
            audit=self.audit,
            settings=self.settings,
        )
        self.context_budget = ContextBudget.from_settings(self.settings)
        self.retrieval_budget = RetrievalBudget()
        self.rule_planner = RulePlanner(self.registry)
        self.model_planner = ModelPlanner(self.registry, self.router) if use_model_planner else None

    # -- hooks for the subsystems a plain runtime does not have -----------
    #
    # `handle` needs four things that only exist once an owner-specific build
    # has been layered on: whether a goal id is real, which skills may be
    # loaded, where to record effort, and how the operator likes an answer
    # shaped. Each one is a method here so the pipeline stays one
    # implementation instead of two, and so a subclass adds behaviour by
    # answering a question rather than by copying `handle`.

    def _goal_exists(self, goal_id: str) -> bool:
        """Whether a goal id refers to something. No goal system: nothing does."""
        return False

    def _skill_registry(self) -> Any:
        """The skills an agent may load, or None when there are none."""
        return None

    def _record_goal_effort(self, goal_id: str, *, cost_usd: float) -> None:
        """Record effort against a goal. Effort, never progress."""

    def _operator_preferences(self) -> str:
        """How the operator wants answers shaped, as prompt text, or empty."""
        return ""

    # -- accessors --------------------------------------------------------

    @property
    def cost(self) -> CostEngine:
        return self.router.cost

    def attach_tools(self, fabric: Any) -> None:
        """Give this JARVIS a tool fabric.

        Without one, agents run with no tools at all rather than finding
        their own way to a server.
        """
        self.tools = fabric

    def set_approval_gate(self, gate: ApprovalGate) -> None:
        """Attach the interface that answers approval requests."""
        self.guardian.approval_gate = gate

    @property
    def verification(self) -> AgentVerifier:
        """The verifier, wired to this JARVIS and its storage.

        Lazy because building it loads whatever has been recorded, and most
        commands never ask.
        """
        if self._verifier is None:
            self._verifier = AgentVerifier(
                self.registry,
                self.router,
                evaluator=self.evaluator,
                audit=self.audit,
                store=self.storage.verification if self.storage is not None else None,
            )
        return self._verifier

    def status(self) -> dict[str, Any]:
        """A snapshot of everything this runtime can account for.

        A subclass that adds subsystems adds their summaries to this dict
        rather than replacing it, so a caller can read the generic keys
        without knowing which build it is talking to.
        """
        return {
            "environment": self.settings.environment,
            "agents": len(self.registry.all()),
            "swarms": self.registry.swarms(),
            "models": [s.id for s in self.router.usable_models()],
            "tasks": self.tasks.snapshot(),
            "cost": self.cost.summary(),
            "cache": self.orchestrator.cache.stats.as_dict(),
            "memory": self.memory.stats(),
            "verification": self.verification.summary(),
            "tools": self.tools.summary() if self.tools is not None else {"attached": False},
            "storage": self.storage.summary()
            if self.storage is not None
            else {
                "durable": False,
                "note": "in memory only; set the database path to make it durable",
            },
            "limits": {
                "max_parallel_agents": self.settings.max_parallel_agents,
                "max_agent_hops": self.settings.max_agent_hops,
                "max_plan_waves": self.settings.max_plan_waves,
                "max_context_chars": self.settings.max_context_chars,
            },
            "pending_agent_proposals": [p.spec.id for p in self.registry.proposals],
            "recent_audit": [e.action for e in self.audit.tail(10)],
        }

    # -- planning ---------------------------------------------------------

    async def make_plan(self, request: str) -> Plan:
        """Decide how to answer a request.

        A simple question skips the model planner entirely; paying a frontier
        model to plan a one-step lookup is waste.
        """
        intent = detect_intent(request)
        if intent is Intent.QUESTION or self.model_planner is None:
            return self.rule_planner.plan(request, intent=intent)
        return await self.model_planner.plan(request, intent=intent)

    # -- the main entry point ---------------------------------------------

    async def handle(
        self,
        request: str,
        *,
        goal_id: str | None = None,
        progress: Progress | None = None,
    ) -> JarvisResponse:
        """Take one request from the operator all the way to an answer.

        ``goal_id`` says which goal this work is meant to advance. It records
        the effort against that goal and nothing else — the work being done is
        not evidence that the goal moved, and only the operator's count of the measure
        says whether it did.

        ``progress`` receives what is happening while it happens, for an
        interface that has to show something during the thirty seconds this
        takes. It is a window on the same run, never a second source of truth:
        every figure it carries is also in the returned response.
        """
        progress = progress or NullProgress()
        trace = self.tracer.start_trace(request[:80])
        warnings: list[str] = []
        if goal_id is not None and not self._goal_exists(goal_id):
            warnings.append(f"no goal {goal_id}; the work is not attributed to anything")
            log.warning("request pointed at a goal that does not exist: %s", goal_id)
            goal_id = None
        root = self.tasks.create(request, priority=3, trace_id=trace.trace_id, goal_id=goal_id)
        self.audit.emit(
            "jarvis.request", actor="operator", subject=request[:200], task_id=root.task_id
        )

        ctx = AgentContext(
            router=self.router,
            guardian=self.guardian,
            audit=self.audit,
            tracer=self.tracer,
            trace=trace,
            bus=self.bus,
            budget=self.context_budget,
            gateway=self.tools.gateway if self.tools is not None else None,
            skills=self._skill_registry(),
        )

        # 0. Recall. What is already known changes what needs finding out.
        with self.tracer.span(trace, "recall", kind="memory") as span:
            recalled = self.memory.recall(request, budget=self.retrieval_budget)
            span.set(items=len(recalled))
        if recalled:
            log.info("recalled %d relevant memory item(s)", len(recalled))

        # 1. Plan.
        self.tasks.transition(root, TaskStatus.PLANNING)
        with self.tracer.span(trace, "plan", kind="plan"):
            plan = await self.make_plan(request)
        problems = plan.validate(self.registry)
        if problems:
            warnings.extend(problems)
            log.warning("plan problems: %s", "; ".join(problems))
            plan = self.rule_planner.plan(request, intent=plan.intent)
            if fallback_problems := plan.validate(self.registry):
                message = "; ".join(fallback_problems)
                self.tasks.fail(root, message)
                return JarvisResponse(
                    request=request,
                    answer=f"I could not build a workable plan: {message}",
                    ok=False,
                    intent=plan.intent,
                    plan=plan,
                    report=RunReport(),
                    task=root,
                    trace_id=trace.trace_id,
                    warnings=warnings,
                )

        progress.emit(
            ProgressKind.PLANNED,
            task_id=root.task_id,
            intent=plan.intent.value,
            source=plan.source,
            steps=[
                {
                    "step_id": step.step_id,
                    "agent_id": step.agent_id,
                    "agent": name_of(self.registry, step.agent_id),
                    "goal": step.goal[:200],
                    "depends_on": list(step.depends_on),
                }
                for step in plan.steps
            ],
        )

        # 2. Run the agents.
        self.tasks.transition(root, TaskStatus.RUNNING)
        try:
            report = await self.orchestrator.run_plan(plan, ctx, parent=root, progress=progress)
        except BudgetExceeded as exc:
            self.tasks.fail(root, str(exc))
            return JarvisResponse(
                request=request,
                answer=f"I stopped before starting: {exc}",
                ok=False,
                intent=plan.intent,
                plan=plan,
                report=RunReport(stopped_early=str(exc)),
                task=root,
                trace_id=trace.trace_id,
                warnings=[*warnings, str(exc)],
            )

        if report.stopped_early:
            warnings.append(report.stopped_early)
        for failure in report.failed:
            warnings.append(f"{failure.agent_id}: {failure.error}")

        for outcome in report.outcomes:
            if outcome.result is not None:
                self._recorded_runs.append(outcome.result.as_dict())
        if len(self._recorded_runs) > 500:
            del self._recorded_runs[: len(self._recorded_runs) - 500]

        # 3. Merge the findings into one answer.
        progress.emit(
            ProgressKind.SYNTHESIZING,
            findings=len(report.successful),
            agent=plan.synthesizer or "research_synthesizer",
        )
        with self.tracer.span(trace, "synthesis", kind="model"):
            answer, synth_cost, synth_usage = await self._synthesize(request, report, root, plan)

        total_cost = report.cost_usd + synth_cost
        # The same arithmetic as the cost, on the counts the cost came from.
        agent_input = sum(o.result.input_tokens for o in report.outcomes if o.result)
        agent_output = sum(o.result.output_tokens for o in report.outcomes if o.result)
        # The root task carries only what JARVIS itself spent; the child tasks carry
        # theirs, so a snapshot that sums every task does not double-count the run.
        root.cost_usd = synth_cost
        successes = report.successful
        confidence = (
            sum(o.result.confidence for o in successes if o.result) / len(successes)
            if successes
            else 0.0
        )
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for outcome in successes:
            for source in outcome.result.sources if outcome.result else []:
                key = str(source.get("url") or source.get("title") or "")
                if key and key not in seen:
                    seen.add(key)
                    sources.append(source)

        ok = bool(successes)
        if ok:
            self.tasks.complete(root, {"answer": answer, "cost_usd": total_cost})
        else:
            self.tasks.fail(root, "no agent produced a usable result")

        self._remember_run(request, answer, ok, confidence, total_cost, plan)
        # Working memory belongs to the task that made it.
        self.memory.forget_task(root.task_id)

        self.audit.emit(
            "jarvis.response",
            task_id=root.task_id,
            result="ok" if ok else "failed",
            metadata={
                "cost_usd": round(total_cost, 6),
                "steps": len(report.outcomes),
                "successful": len(successes),
                "trace_id": trace.trace_id,
            },
        )

        if goal_id is not None:
            # Effort, not progress. What this bought is a question about the
            # world, and the goal's measure is the only thing that answers it.
            self._record_goal_effort(goal_id, cost_usd=total_cost)

        return JarvisResponse(
            request=request,
            answer=answer,
            ok=ok,
            intent=plan.intent,
            plan=plan,
            report=report,
            task=root,
            trace_id=trace.trace_id,
            cost_usd=total_cost,
            confidence=confidence,
            sources=sources,
            warnings=warnings,
            cached_steps=sum(1 for o in report.outcomes if o.cached),
            recalled=[r.as_dict() for r in recalled],
            input_tokens=agent_input + synth_usage[0],
            output_tokens=agent_output + synth_usage[1],
        )

    def _remember_run(
        self,
        request: str,
        answer: str,
        ok: bool,
        confidence: float,
        cost_usd: float,
        plan: Plan,
    ) -> None:
        """Keep what will still matter tomorrow, and nothing else.

        An episode of every run, because knowing a question was already asked
        is cheap and useful. The answer itself only when it was good enough to
        be worth trusting later; a low-confidence answer remembered as fact is
        worse than not remembering it.
        """
        agents = ", ".join(sorted({s.agent_id for s in plan.steps}))
        self.memory.remember(
            f"Asked: {request} | outcome: {'answered' if ok else 'failed'} "
            f"| agents: {agents} | cost ${cost_usd:.4f}",
            scope=MemoryScope.EPISODIC,
            source="jarvis",
            importance=0.5 if ok else 0.6,  # a failure is worth more, not less
            confidence=0.9,
            metadata={"intent": plan.intent.value, "ok": ok},
        )
        if ok and confidence >= 0.6 and answer:
            self.memory.remember(
                f"On '{request}': {answer[:1200]}",
                scope=MemoryScope.KNOWLEDGE,
                source="jarvis",
                importance=min(1.0, 0.4 + confidence / 2),
                confidence=confidence,
                metadata={"intent": plan.intent.value},
            )

    async def _synthesize(
        self, request: str, report: RunReport, root: Task, plan: Plan | None = None
    ) -> tuple[str, float, tuple[int, int]]:
        """Merge agent findings into the answer the operator reads.

        Costs nothing in the common cases. A plan that ends in a synthesis agent has
        already merged the findings, and a single finding needs no merging at all;
        only a fan-out with no synthesis step pays for another model call.

        Returns the answer, what the merge cost, and the tokens that cost was
        derived from. The tokens travel with the money because a figure with
        no countable basis is the kind an interface ends up inventing.
        """
        successes = report.successful
        if not successes:
            reasons = "; ".join(f"{o.agent_id}: {o.error}" for o in report.outcomes if o.error)
            return (f"No agent produced a usable result. {reasons}".strip(), 0.0, (0, 0))
        if len(successes) == 1 and successes[0].result is not None:
            return successes[0].result.text, 0.0, (0, 0)

        # The plan's last step is the synthesizer. If it ran, its answer is the answer.
        if plan is not None and plan.steps:
            final_step = plan.steps[-1]
            if plan.synthesizer and final_step.agent_id == plan.synthesizer:
                outcome = report.outcome(final_step.step_id)
                if outcome is not None and outcome.ok and outcome.result is not None:
                    return outcome.result.text, 0.0, (0, 0)

        blocks = []
        for outcome in successes:
            result = outcome.result
            assert result is not None
            blocks.append(
                f"### {result.agent_id} (confidence {result.confidence:.2f})\n{result.text}"
            )
        prompt = f"## Request\n{request}\n\n## Findings\n" + "\n\n".join(blocks)

        # How the operator wants an answer shaped is part of the answer. A
        # runtime with nothing on file adds nothing, which is the usual case;
        # the private subclass renders what it has been told and no more.
        system = SYNTHESIS_SYSTEM
        if preferences := self._operator_preferences():
            system = f"{SYNTHESIS_SYSTEM}\n\n{preferences}"

        try:
            response = await self.router.complete(
                ModelRequest(
                    messages=[Message(role="user", content=prompt)],
                    system=system,
                    max_tokens=8000,
                ),
                kind=TaskKind.SYNTHESIS,
                agent="jarvis_synthesis",
                task_id=root.task_id,
            )
        except Exception as exc:
            log.warning("synthesis failed (%s); returning the findings unmerged", exc)
            return ("\n\n".join(blocks), 0.0, (0, 0))

        cost = self.cost.price(
            response.model, response.usage.input_tokens, response.usage.output_tokens
        )
        return response.text, cost, (response.usage.input_tokens, response.usage.output_tokens)


def build_runtime(
    *,
    agents_dir: Path | None = None,
    settings: Settings | None = None,
    approval_gate: ApprovalGate | None = None,
) -> JarvisRuntime:
    """Assemble a runtime wired to the real providers and the agents on disk.

    Storage is opened when one is configured, so tasks, memory, the audit
    trail and approvals are durable and shared with any other process
    pointing at the same database. With none configured everything stays in
    memory, which is fine for a single run and wrong for anything overnight.
    """
    from ..storage import open_storage

    settings = settings or get_settings()
    storage = open_storage(settings=settings)
    audit = AuditLog(sink=storage.audit if storage is not None else None)
    registry = build_registry(audit=audit, directory=agents_dir)
    router = build_default_router(settings=settings, audit=audit)
    guardian = SecurityGuardian(settings=settings, audit=audit, approval_gate=approval_gate)
    runtime = JarvisRuntime(
        registry=registry,
        router=router,
        tasks=TaskEngine(storage.tasks) if storage is not None else None,
        memory=MemoryManager(storage.memory) if storage is not None else None,
        guardian=guardian,
        audit=audit,
        settings=settings,
    )
    runtime.storage = storage
    return runtime


async def build_runtime_with_tools(
    *,
    agents_dir: Path | None = None,
    settings: Settings | None = None,
    approval_gate: ApprovalGate | None = None,
) -> JarvisRuntime:
    """A runtime with the tool fabric attached.

    Separate from `build_runtime` because discovery is asynchronous: a server
    is asked what it publishes rather than told.
    """
    from ..tools.setup import build_tool_fabric

    settings = settings or get_settings()
    runtime = build_runtime(agents_dir=agents_dir, settings=settings, approval_gate=approval_gate)
    runtime.attach_tools(
        await build_tool_fabric(
            settings=settings,
            audit=runtime.audit,
            cost=runtime.cost,
            tracer=runtime.tracer,
            router=runtime.router,
            workspace=settings.workspace_dir,
            storage=runtime.storage,
        )
    )
    return runtime
