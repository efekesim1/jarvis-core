"""Running an agent to find out whether it works.

The offline pass proves the plumbing: the definition loads, the prompt builds,
the runtime executes, the output parses, the evaluator scores it. It proves
nothing about the prompt, because the mock provider will happily satisfy any
prompt ever written.

The real pass is the one that matters, and it costs money, so it is opt-in and
bounded. Without a credential it is skipped — explicitly, with a reason, never
as a failure and never quietly as a pass. "No API key" is a fact about the
environment; reporting it as anything else is how a system ends up claiming
readiness it has not earned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext, AgentSpec
from ..agents.registry import AgentRegistry
from ..core.context import ContextBudget
from ..core.task import Task
from ..evaluation.engine import EvaluationEngine
from ..logging import AuditEvent, AuditLog, get_logger
from ..models.router import ModelRouter
from ..observability.tracing import Tracer
from ..security.approval import AutoApproveGate
from ..security.guardian import SecurityGuardian
from .claims import unsupported_claims
from .failures import FailureKind, classify, schema_gaps, summarise
from .preflight import PreflightResult, preflight
from .status import MIN_MEAN_SCORE, AgentVerification, VerificationRun

log = get_logger("verification")

# How far above the passing bar an agent has to be before one run counts as an
# answer. Inside this margin, a pass and a fail are the same evidence.
BORDERLINE_MARGIN = 0.1

# Refusals that say something about the environment rather than about the
# agent. A deploy agent in a sandbox that withholds DEPLOY is behaving exactly
# as designed; recording that as a defect trains everyone to ignore the failure
# column, which is the column that matters.
_ENVIRONMENTAL = (
    "not available in the sandbox",
    "approval denied",
    "permission",
    "no model provider is configured",
)


def _is_environmental(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _ENVIRONMENTAL)


def _label(kind: FailureKind | None) -> str:
    """The stored form of a classification. A run that succeeded stores nothing."""
    return kind.value if kind is not None else ""


@dataclass(slots=True)
class VerificationBudget:
    """What one verification sweep may spend, in four currencies.

    A sweep over 27 agents at three runs each is 81 model calls. Without a
    ceiling that is an unbounded bill triggered by a command that reads like a
    status check.

    Money is the obvious limit and the slowest to notice. A runaway retry loop
    reaches the call count first; an agent that will not stop generating
    reaches the token count first; a hung provider reaches the clock first.
    Whichever binds first stops the sweep, and the report says which one did.
    """

    max_usd: float = 2.0
    max_calls: int = 0  # 0 means no ceiling on the count
    max_tokens: int = 0
    timeout_s: float = 0.0
    repeats: int = 1
    max_agents: int = 0  # 0 means all
    # Only these agent ids, when set. This is how the second pass costs a
    # fraction of the first: verify all twenty-seven once, then re-run only the
    # ones that failed or scored near the bar, rather than paying for the
    # twenty that already answered well.
    only: frozenset[str] = frozenset()

    def covers(self, agent_id: str) -> bool:
        return not self.only or agent_id in self.only

    def exceeded(self, *, spent: float, calls: int, tokens: int, elapsed_s: float) -> str:
        """Which ceiling has been reached, named, or an empty string.

        Returns the name rather than a boolean so the audit record and the
        report can say *which* limit stopped the sweep. "Stopped at its
        budget" when it was actually the clock sends the next person to change
        the wrong number.
        """
        if spent >= self.max_usd:
            return f"cost ceiling ${self.max_usd:.2f}"
        if self.max_calls and calls >= self.max_calls:
            return f"call ceiling {self.max_calls}"
        if self.max_tokens and tokens >= self.max_tokens:
            return f"token ceiling {self.max_tokens:,}"
        if self.timeout_s and elapsed_s >= self.timeout_s:
            return f"time ceiling {self.timeout_s:.0f}s"
        return ""

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> VerificationBudget:
        """The configured governor, so the CLI and the API cannot drift apart."""
        base = {
            "max_usd": settings.real_validation_budget_usd,
            "max_calls": settings.real_run_max_calls,
            "max_tokens": settings.real_run_max_tokens,
            "timeout_s": settings.real_run_timeout_seconds,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


class AgentVerifier:
    """Runs agents against their declared smoke task and records what happened."""

    def __init__(
        self,
        registry: AgentRegistry,
        router: ModelRouter,
        *,
        evaluator: EvaluationEngine | None = None,
        guardian: SecurityGuardian | None = None,
        audit: AuditLog | None = None,
        store: Any = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.evaluator = evaluator or EvaluationEngine(router=router)
        # A verification run gets no gateway, so an agent has no way to reach a
        # tool, a file, the network or a credential. The only effect it can have
        # is one model call, and that is bounded by the sweep budget.
        #
        # With nothing to guard, an approval prompt guards nothing — and the
        # default deny gate would block every medium-risk agent from ever being
        # verified, which is how a system ends up unable to check the agents it
        # most needs to check. The gate is auto-approve here and nowhere else;
        # a test asserts the run context carries no gateway.
        self.guardian = guardian or SecurityGuardian(audit=audit, approval_gate=AutoApproveGate())
        self.audit = audit or AuditLog()
        self.store = store
        self.tracer = Tracer()
        self._records: dict[str, AgentVerification] = {}
        # Which ceiling ended the last sweep, if one did. Empty means it ran to
        # the end, which is a different report from "it stopped and we are not
        # saying why".
        self._stopped: str = ""
        # What the last real sweep's reachability probe found, if one ran.
        self.preflight: PreflightResult | None = None
        self._blocked_reason: str = ""
        if store is not None:
            for record in store.load():
                self._records[record.agent_id] = record

    # -- one agent --------------------------------------------------------

    async def verify_agent(
        self, spec: AgentSpec, *, real_model: bool, backed: bool = True
    ) -> VerificationRun | None:
        """Run one agent once. Returns ``None`` when there is nothing to run."""
        record = self._record_for(spec.id)
        record.capabilities_backed = backed
        record.has_smoke_task = bool(spec.smoke_task)
        record.real_blocked_reason = self._blocked_reason
        if not spec.smoke_task:
            return None

        task = Task(goal=spec.smoke_task, assigned_agent=spec.id, risk=spec.risk_level)
        trace = self.tracer.start_trace(f"verify {spec.id}")
        ctx = AgentContext(
            router=self.router,
            guardian=self.guardian,
            audit=self.audit,
            tracer=self.tracer,
            trace=trace,
            budget=ContextBudget(),
        )

        started = time.perf_counter()
        try:
            runtime = self.registry.runtime_for(spec)
            result = await runtime.run(spec, task, ctx)
        except Exception as exc:  # noqa: BLE001 - a crash is a result, not a stop
            message = str(exc)
            # The environment refusing an agent is not the agent failing. A
            # deploy agent in a sandbox that withholds DEPLOY is behaving
            # exactly as intended, and recording it as a defect would train
            # everyone to ignore the failure column.
            blocked = _is_environmental(message)
            run = VerificationRun(
                agent_id=spec.id,
                task=spec.smoke_task,
                real_model=real_model,
                ok=False,
                blocked=blocked,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=message if blocked else f"{type(exc).__name__}: {exc}",
            )
            run.failure = _label(classify(run, spec))
            record.runs.append(run)
            self._persist(record)
            log.warning("agent %s crashed during verification: %s", spec.id, exc)
            return run

        # The runtime catches a guardian refusal and reports it as a result
        # rather than raising, so the same check has to happen on this path.
        blocked = not result.ok and _is_environmental(result.error or "")

        evaluation = self.evaluator.evaluate(result, spec, task)
        claims = unsupported_claims(result.text, sources=len(result.sources)) if real_model else []
        missing = schema_gaps(result.data, spec)

        run = VerificationRun(
            agent_id=spec.id,
            task=spec.smoke_task,
            real_model=real_model,
            ok=result.ok,
            score=evaluation.overall,
            passed=evaluation.passed,
            confidence=result.confidence,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            model=result.model or "",
            sources=len(result.sources),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            schema_valid=not missing,
            missing_keys=missing,
            answer_chars=len(result.text or ""),
            truncated=result.stop_reason == "max_tokens",
            unsupported_claims=claims,
            blocked=blocked,
            error=result.error or "",
        )
        run.failure = _label(classify(run, spec))
        record.runs.append(run)
        self._persist(record)
        return run

    # -- a sweep ----------------------------------------------------------

    async def verify_all(
        self, *, real_model: bool = False, budget: VerificationBudget | None = None
    ) -> dict[str, AgentVerification]:
        """Verify every enabled agent, stopping if the budget runs out.

        ``real_model=True`` with no usable provider is not an error and not a
        silent downgrade: the sweep runs offline and every record says why it
        got no further.
        """
        budget = budget or VerificationBudget()
        self._stopped = ""
        if real_model and not self.router.usable_models():
            log.warning("no model provider is configured; verifying offline only")
            real_model = False

        # One cheap question before twenty-seven expensive ones. Without it a
        # dead credential or an empty account produces twenty-seven agent
        # failures, every one of them a lie about the agent: the prompt was
        # never read and the schema was never tested.
        if real_model:
            self.preflight = await preflight(self.router, audit=self.audit)
            if not self.preflight.ok:
                log.warning("real sweep not attempted: %s", self.preflight.render())
                self._stopped = f"preflight: {self.preflight.state.value}"
                self._blocked_reason = f"no real model run — {self.preflight.remedy}"
                real_model = False
            else:
                self._blocked_reason = ""

        # Offline means offline. Running the real router without a credential
        # would produce an authentication error and record it as an agent
        # failure, which is a lie about the agent.
        router = self.router if real_model else offline_router(self.router)
        original, self.router = self.router, router
        try:
            return await self._sweep(real_model=real_model, budget=budget)
        finally:
            self.router = original

    async def _sweep(
        self, *, real_model: bool, budget: VerificationBudget
    ) -> dict[str, AgentVerification]:

        backed = self._backed_capabilities()
        specs = self.registry.all()
        if budget.max_agents:
            specs = specs[: budget.max_agents]

        spent = 0.0
        paid_runs = 0
        tokens = 0
        started = time.perf_counter()
        for spec in specs:
            if not budget.covers(spec.id):
                continue
            for _ in range(max(1, budget.repeats) if real_model else 1):
                if real_model:
                    elapsed = time.perf_counter() - started
                    reached = budget.exceeded(
                        spent=spent, calls=paid_runs, tokens=tokens, elapsed_s=elapsed
                    )
                    # Stop before the call that would cross the ceiling, not
                    # after. A limit only noticed once exceeded is a report,
                    # not a limit — and the projection uses what this sweep has
                    # actually been costing rather than a guess written here.
                    projected = spent + (spent / paid_runs if paid_runs else 0.0)
                    if not reached and projected > budget.max_usd:
                        reached = f"cost ceiling ${budget.max_usd:.2f} on the next call"
                    if reached:
                        log.warning(
                            "verification stopped at %s after %d call(s), $%.4f, %d token(s); "
                            "%s and after were not run",
                            reached,
                            paid_runs,
                            spent,
                            tokens,
                            spec.id,
                        )
                        self._stopped = reached
                        self._note_budget_stop(spec.id, spent, budget.max_usd, reached)
                        return self._records
                run = await self.verify_agent(spec, real_model=real_model, backed=spec.id in backed)
                if run is not None and run.real_model:
                    spent += run.cost_usd
                    tokens += run.total_tokens
                    paid_runs += 1

        self.audit.record(
            AuditEvent(
                action="verification.sweep",
                actor="verifier",
                subject=f"{len(specs)} agent(s)",
                reason=f"{'real model' if real_model else 'offline only'}, ${spent:.4f}",
            )
        )
        return self._records

    # -- reading ----------------------------------------------------------

    def record(self, agent_id: str) -> AgentVerification | None:
        return self._records.get(agent_id)

    def unverified_targets(self) -> frozenset[str]:
        """Agents that have never reached a real model at all.

        What a budget-stopped sweep leaves behind. `retest_targets` covers
        agents that ran and came back short; this covers the ones that were
        never tried, which is a different set and the one a resumed sweep
        needs — without it, resuming re-runs the roster from the top and pays
        again for every agent that already answered before reaching the first
        that did not.

        An agent the environment refuses (no DEPLOY in a sandbox) is left out:
        it is not waiting on a budget, and putting it in a resume list would
        make every resumed sweep end on the same avoidable failure.
        """
        targets: set[str] = set()
        for spec in self.registry.all():
            record = self._records.get(spec.id)
            if record is None:
                targets.add(spec.id)
                continue
            if not record.real_runs and not record.blocked_runs:
                targets.add(spec.id)
        return frozenset(targets)

    def retest_targets(self) -> frozenset[str]:
        """Which agents a second real pass should spend money on.

        Two groups, and only these two. An agent that failed, because the point
        of a second pass is to find out whether the repair worked; and an agent
        that passed by a margin thinner than the noise between runs, because
        one run above the bar and one below it are the same evidence and the
        first one is not a verdict.

        Everything else has answered the question it was asked. Re-running it
        buys a number that is already known, at full price.
        """
        targets: set[str] = set()
        for record in self._records.values():
            real = record.real_runs
            if not real:
                continue
            if any(not run.usable or not run.passed for run in real):
                targets.add(record.agent_id)
                continue
            mean = record.mean_score
            if mean is not None and mean - MIN_MEAN_SCORE < BORDERLINE_MARGIN:
                targets.add(record.agent_id)
        return frozenset(targets)

    def records(self) -> dict[str, AgentVerification]:
        return dict(self._records)

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for record in self._records.values():
            status, _ = record.status()
            by_status[status.label] = by_status.get(status.label, 0) + 1
        for spec in self.registry.all():
            if spec.id not in self._records:
                by_status["defined"] = by_status.get("defined", 0) + 1
        return {
            "agents": len(self.registry.all()),
            "verified": len(self._records),
            "by_status": dict(sorted(by_status.items())),
            "real_runs": sum(len(r.real_runs) for r in self._records.values()),
            "blocked": sum(1 for r in self._records.values() if r.blocked_runs),
            "real_cost_usd": round(sum(r.real_cost for r in self._records.values()), 6),
            "real_tokens": sum(
                run.total_tokens for r in self._records.values() for run in r.real_runs
            ),
            "failures": summarise(
                [
                    FailureKind(run.failure)
                    for r in self._records.values()
                    for run in r.real_runs
                    if run.failure
                ]
            ),
            "retest": sorted(self.retest_targets()),
            "unverified": sorted(self.unverified_targets()),
            "stopped_at": self._stopped,
            "preflight": self.preflight.as_dict() if self.preflight else None,
            "unsupported_claims": sum(len(r.unsupported_claims) for r in self._records.values()),
        }

    # -- internals --------------------------------------------------------

    def _record_for(self, agent_id: str) -> AgentVerification:
        return self._records.setdefault(agent_id, AgentVerification(agent_id=agent_id))

    def _backed_capabilities(self) -> set[str]:
        """Agent ids whose declared capabilities are all provided by a tool.

        Without a fabric attached every agent counts as backed. The capability
        check is the fabric's job, and asserting a failure the verifier cannot
        actually see would be inventing one.
        """
        return {spec.id for spec in self.registry.all()}

    def _note_budget_stop(
        self, agent_id: str, spent: float, ceiling: float, reached: str = ""
    ) -> None:
        self.audit.record(
            AuditEvent(
                action="verification.budget_stop",
                actor="verifier",
                subject=agent_id,
                result=reached or "cost",
                reason=f"${spent:.4f} of ${ceiling:.2f} spent; stopped at the {reached}",
            )
        )

    def _persist(self, record: AgentVerification) -> None:
        if self.store is None:
            return
        try:
            self.store.save(record)
        except Exception as exc:  # noqa: BLE001 - storage must not break a sweep
            log.warning("could not persist verification for %s: %s", record.agent_id, exc)


def offline_router(reference: ModelRouter) -> ModelRouter:
    """A router that answers locally, for proving the plumbing.

    The mock reply is deliberately well-formed: it parses, it carries a
    confidence and a source, and it is long enough for the completeness check
    to have something to measure. That makes the offline pass a test of the
    system around the agent — definition, prompt assembly, runtime, parsing,
    evaluation — and not a test of the answer, which no local stand-in could
    give.

    An offline pass therefore means "nothing here is broken", never "this agent
    is good". The status ladder keeps those separate.
    """
    from dataclasses import replace

    from ..models.catalog import default_catalog
    from ..models.mock_provider import MockProvider, agent_reply

    provider = MockProvider(
        responder=lambda request: agent_reply(
            summary=(
                "A structurally complete answer produced without a model, so that "
                "verification can prove the agent's definition, prompt, runtime, "
                "parsing and evaluation all work. It says nothing about quality. " * 2
            ),
            confidence=0.8,
            sources=[{"url": "https://example.invalid/offline", "title": "Offline stand-in"}],
        )
    )
    # The real catalog, re-pointed at the mock provider, so routing decisions —
    # which tier for which work, and the step-down under budget pressure — are
    # exercised rather than bypassed. A single-model stand-in would verify
    # agents against a router that does not resemble the real one.
    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))

    return ModelRouter(
        providers={"mock": provider},
        catalog=catalog,
        settings=reference.settings,
        cost=reference.cost,
        audit=reference.audit,
    )
