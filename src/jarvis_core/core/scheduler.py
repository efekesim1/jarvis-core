"""The scheduler and the night shift.

The system must be able to work while the operator sleeps. It must not be able to run
away while the operator sleeps. Those two requirements are what this module is.

Every autonomous job carries its own limits: a budget, a deadline, a cap on how
many iterations it may take, and the agents it is allowed to use. A shift stops
the moment any limit is reached, and it always produces a report, including the
report that says it stopped early and why.

There is no open-ended loop anywhere in this file. That is deliberate.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from ..logging import AuditLog, get_logger
from ..observability.cost import BudgetExceeded
from ..security.permissions import RiskLevel

if TYPE_CHECKING:  # pragma: no cover
    from .jarvis import Jarvis

log = get_logger("scheduler")


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    # A job that reached something needing the operator. It is not a failure: the work
    # is waiting for a decision, and it goes into the morning report as such.
    BLOCKED_APPROVAL = "blocked_approval"


class StopReason(StrEnum):
    """Why a shift ended. Every shift has exactly one."""

    FINISHED = "finished"  # the queue emptied
    BUDGET = "budget"
    DEADLINE = "deadline"
    ITERATIONS = "iterations"
    SAFETY = "safety"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ScheduledJob:
    """One piece of autonomous work, and the limits it runs inside.

    A job with no budget or no deadline is not accepted; the constructor floors
    both. An overnight run with an unbounded job is how a bill happens.
    """

    request: str
    owner: str = "operator"
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    priority: int = 5
    budget_usd: float = 0.50
    timeout_s: float = 300.0
    max_iterations: int = 1
    allowed_agents: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    risk_level: RiskLevel = RiskLevel.LOW
    status: JobStatus = JobStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cost_usd: float = 0.0
    answer: str = ""
    error: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        # Limits are the point of this object, so they are enforced here rather
        # than trusted from the caller.
        self.budget_usd = max(0.01, min(self.budget_usd, 100.0))
        self.timeout_s = max(10.0, min(self.timeout_s, 3600.0))
        self.max_iterations = max(1, min(self.max_iterations, 5))

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None:
            return None
        return ((self.finished_at or datetime.now(UTC)) - self.started_at).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request": self.request,
            "owner": self.owner,
            "priority": self.priority,
            "status": self.status.value,
            "budget_usd": self.budget_usd,
            "cost_usd": round(self.cost_usd, 6),
            "timeout_s": self.timeout_s,
            "max_iterations": self.max_iterations,
            "risk_level": self.risk_level.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": round(self.duration_s, 2) if self.duration_s else None,
            "confidence": round(self.confidence, 3),
            "error": self.error,
        }


@dataclass(slots=True)
class ShiftReport:
    """What happened overnight, in the form the operator reads over coffee."""

    started_at: datetime
    finished_at: datetime | None = None
    stop_reason: StopReason = StopReason.FINISHED
    jobs: list[ScheduledJob] = field(default_factory=list)
    cost_usd: float = 0.0
    budget_usd: float = 0.0
    security_events: int = 0
    notes: list[str] = field(default_factory=list)
    pending_approvals: list[Any] = field(default_factory=list)

    @property
    def completed(self) -> list[ScheduledJob]:
        return [j for j in self.jobs if j.status is JobStatus.COMPLETED]

    @property
    def failed(self) -> list[ScheduledJob]:
        return [j for j in self.jobs if j.status is JobStatus.FAILED]

    @property
    def skipped(self) -> list[ScheduledJob]:
        return [j for j in self.jobs if j.status is JobStatus.SKIPPED]

    @property
    def blocked(self) -> list[ScheduledJob]:
        return [j for j in self.jobs if j.status is JobStatus.BLOCKED_APPROVAL]

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "stop_reason": self.stop_reason.value,
            "completed": len(self.completed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "blocked_on_approval": len(self.blocked),
            "pending_approvals": [a.as_dict() for a in self.pending_approvals],
            "cost_usd": round(self.cost_usd, 6),
            "budget_usd": self.budget_usd,
            "security_events": self.security_events,
            "notes": list(self.notes),
            "jobs": [j.as_dict() for j in self.jobs],
        }

    def render(self) -> str:
        """The morning report. Short, and honest about what did not happen."""
        duration = ""
        if self.finished_at:
            hours = (self.finished_at - self.started_at).total_seconds() / 3600
            duration = f" over {hours:.1f}h"

        lines = [
            "GOOD MORNING OPERATOR",
            "",
            f"Overnight{duration}: {len(self.completed)} of {len(self.jobs)} jobs completed.",
            f"Spent ${self.cost_usd:.4f} of a ${self.budget_usd:.2f} budget.",
        ]
        if self.stop_reason is not StopReason.FINISHED:
            lines.append(f"Stopped early: {self.stop_reason.value}.")
        if self.security_events:
            lines.append(f"Security events logged: {self.security_events}.")

        if self.completed:
            lines.append("")
            lines.append("What came back:")
            for job in sorted(self.completed, key=lambda j: j.confidence, reverse=True)[:5]:
                first_line = (
                    job.answer.strip().splitlines()[0] if job.answer.strip() else "(no answer)"
                )
                lines.append(f"  - {job.request}")
                lines.append(f"    {first_line[:160]}")
                lines.append(f"    confidence {job.confidence:.2f}, ${job.cost_usd:.4f}")

        if self.blocked:
            lines.append("")
            lines.append(f"Waiting for your approval ({len(self.blocked)}):")
            for job in self.blocked[:5]:
                lines.append(f"  - {job.request}")
                lines.append(f"    {job.error}")

        if self.pending_approvals:
            lines.append("")
            lines.append("Approvals to decide:")
            for approval in self.pending_approvals[:5]:
                lines.append(
                    f"  - [{approval.risk.value}] {approval.agent_id} wants {approval.tool_id}"
                )
                lines.append(f"    arguments: {approval.arguments_summary}")
                lines.append(f"    approve with: jarvis approvals approve {approval.approval_id}")

        if self.failed:
            lines.append("")
            lines.append("What failed:")
            for job in self.failed[:5]:
                lines.append(f"  - {job.request}: {job.error}")

        if self.skipped:
            lines.append("")
            lines.append(f"Not started ({len(self.skipped)}): the shift ended first.")

        for note in self.notes:
            lines.append(f"\nNote: {note}")
        return "\n".join(lines)


class Scheduler:
    """Runs a queue of jobs inside hard limits, then reports.

    The loop is bounded three ways at once: a wall-clock deadline, a total
    budget, and a finite queue. Any one of them ending the shift is normal and
    produces a report rather than an error.
    """

    def __init__(
        self,
        jarvis: Jarvis,
        *,
        settings: Settings | None = None,
        audit: AuditLog | None = None,
        max_parallel_jobs: int = 2,
    ) -> None:
        self.jarvis = jarvis
        self.settings = settings or get_settings()
        self.audit = audit or jarvis.audit
        # Two, not six. Each job runs its own waves of agents inside it, so
        # three concurrent jobs is closer to eighteen concurrent model calls —
        # enough to hit a provider rate limit and turn a quiet shift into a
        # night of retries. Raise it when the rate limits are known, not
        # because the machine has cores.
        self.max_parallel_jobs = max(1, max_parallel_jobs)
        self._queue: list[ScheduledJob] = []
        self._cancelled = False

    # -- the queue --------------------------------------------------------

    def submit(self, job: ScheduledJob) -> ScheduledJob:
        self._queue.append(job)
        self.audit.emit(
            "scheduler.submitted",
            actor=job.owner,
            subject=job.request[:200],
            metadata={"job_id": job.job_id, "budget_usd": job.budget_usd},
        )
        return job

    def submit_request(self, request: str, **kwargs: Any) -> ScheduledJob:
        return self.submit(ScheduledJob(request=request, **kwargs))

    @property
    def pending(self) -> list[ScheduledJob]:
        return [j for j in self._queue if j.status is JobStatus.PENDING]

    def cancel(self) -> None:
        """Stop the shift. Honoured whether or not one is running yet."""
        self._cancelled = True

    def resume(self) -> None:
        """Clear a previous cancel so the scheduler can be used again."""
        self._cancelled = False

    def _next_job(self) -> ScheduledJob | None:
        """The highest-priority job not yet claimed, and claim it.

        Claiming matters now that a wave takes several. The selector reads
        status, so without marking the job here the second call in a wave
        returns the same job again — which it did, and a two-job queue
        produced four completions.
        """
        pending = self.pending
        if not pending:
            return None
        job = min(pending, key=lambda j: (j.priority, j.request))
        job.status = JobStatus.RUNNING
        return job

    # -- running a shift --------------------------------------------------

    def _pending_approvals(self) -> list[Any]:
        """Approvals waiting on the operator, if a tool fabric is attached."""
        fabric = getattr(self.jarvis, "tools", None)
        if fabric is None:
            return []
        return list(fabric.approvals.pending)

    def _approvals_raised_by(self, job: ScheduledJob) -> list[Any]:
        return [
            a for a in self._pending_approvals() if a.task_id and a.created_at >= job.started_at
        ]

    async def _run_job(self, job: ScheduledJob) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        spent_before = self.jarvis.cost.summary()["total_usd"]
        try:
            response = await asyncio.wait_for(
                self.jarvis.handle(job.request), timeout=job.timeout_s
            )
        except TimeoutError:
            job.status = JobStatus.FAILED
            job.error = f"exceeded its {job.timeout_s:.0f}s timeout"
        except BudgetExceeded as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
        else:
            job.answer = response.answer
            job.confidence = response.confidence
            job.status = JobStatus.COMPLETED if response.ok else JobStatus.FAILED
            if not response.ok:
                job.error = "; ".join(response.warnings) or "no agent produced a usable result"
            # Work that reached something needing the operator is blocked, not failed.
            # The night shift never approves on his behalf.
            if self._approvals_raised_by(job):
                job.status = JobStatus.BLOCKED_APPROVAL
                job.error = "stopped: a tool call needs your approval"
        finally:
            job.finished_at = datetime.now(UTC)
            job.cost_usd = self.jarvis.cost.summary()["total_usd"] - spent_before

        self.audit.emit(
            "scheduler.job_finished",
            subject=job.request[:200],
            result=job.status.value,
            reason=job.error,
            metadata={"job_id": job.job_id, "cost_usd": round(job.cost_usd, 6)},
        )

    async def run_shift(
        self,
        *,
        budget_usd: float = 5.0,
        max_duration_s: float = 8 * 3600,
        max_jobs: int = 50,
    ) -> ShiftReport:
        """Work the queue until a limit is reached, then report.

        Nobody is watching, so the approval gate should be one that denies.
        That is the caller's choice, not this method's, but the shift records
        which gate was in place so the report is honest about it.

        ``max_jobs`` counts jobs, not waves. A limit that quietly meant twice
        as much work once concurrency was added would be a limit that changed
        meaning without anyone saying so.
        """
        started = datetime.now(UTC)
        deadline = started + timedelta(seconds=max_duration_s)
        report = ShiftReport(started_at=started, budget_usd=budget_usd)
        spent_at_start = self.jarvis.cost.summary()["total_usd"]
        audit_at_start = len(self.audit.events)
        # `_cancelled` is deliberately not reset: a cancel issued before the
        # shift starts is still a cancel.
        skipped_for_budget = False

        gate = type(self.jarvis.guardian.approval_gate).__name__
        if gate == "AutoApproveGate":
            report.notes.append(
                "the approval gate auto-approves; an unattended shift should deny instead"
            )
        self.audit.emit(
            "scheduler.shift_started",
            metadata={
                "budget_usd": budget_usd,
                "max_duration_s": max_duration_s,
                "queued": len(self.pending),
                "approval_gate": gate,
            },
        )

        started_jobs = 0
        while started_jobs < max_jobs:
            if self._cancelled:
                report.stop_reason = StopReason.CANCELLED
                break

            spent = self.jarvis.cost.summary()["total_usd"] - spent_at_start
            report.cost_usd = spent
            if spent >= budget_usd:
                report.stop_reason = StopReason.BUDGET
                report.notes.append(f"spent ${spent:.4f} of the ${budget_usd:.2f} shift budget")
                break
            if datetime.now(UTC) >= deadline:
                report.stop_reason = StopReason.DEADLINE
                break

            # A wave rather than one job, when more than one fits.
            #
            # The constraint that makes this safe is that the *whole wave* has
            # to fit in what is left of the budget. Running jobs concurrently
            # and checking the budget between them would be checking after the
            # spending: three jobs in flight can each be affordable alone and
            # together cost more than remains, and the overspend is discovered
            # when it is already paid for.
            #
            # So the wave is assembled first, against the budget, and only then
            # started. A job that does not fit waits for the next wave.
            wave: list[ScheduledJob] = []
            reserved = 0.0
            room = min(self.max_parallel_jobs, max_jobs - started_jobs)
            while len(wave) < room:
                job = self._next_job()
                if job is None:
                    break
                if spent + reserved + job.budget_usd > budget_usd:
                    job.status = JobStatus.SKIPPED
                    job.error = "not enough shift budget left for this job"
                    report.jobs.append(job)
                    skipped_for_budget = True
                    continue
                wave.append(job)
                reserved += job.budget_usd

            if not wave:
                report.stop_reason = (
                    StopReason.BUDGET if skipped_for_budget else StopReason.FINISHED
                )
                if skipped_for_budget:
                    report.notes.append(
                        "some jobs never started: their own budget did not fit in what was left"
                    )
                break

            report.jobs.extend(wave)
            started_jobs += len(wave)
            if len(wave) == 1:
                await self._run_job(wave[0])
            else:
                # `return_exceptions` so one job crashing does not cancel the
                # others mid-flight. `_run_job` records its own failures; this
                # is for the case where it cannot, and losing two good jobs
                # because a third raised would be the shift punishing itself.
                await asyncio.gather(*(self._run_job(job) for job in wave), return_exceptions=True)

            for job in wave:
                log.info("job: %s -> %s ($%.4f)", job.request[:60], job.status.value, job.cost_usd)

        if started_jobs >= max_jobs and report.stop_reason is StopReason.FINISHED:
            report.stop_reason = StopReason.ITERATIONS

        # Anything still pending never started.
        for job in self.pending:
            job.status = JobStatus.SKIPPED
            job.error = "the shift ended before this job started"
            report.jobs.append(job)

        report.finished_at = datetime.now(UTC)
        report.cost_usd = self.jarvis.cost.summary()["total_usd"] - spent_at_start
        report.pending_approvals = self._pending_approvals()
        report.security_events = sum(
            1
            for event in self.audit.events[audit_at_start:]
            if event.action.startswith("security.") or event.action == "guardian.approval"
        )
        self.audit.emit(
            "scheduler.shift_finished",
            result=report.stop_reason.value,
            metadata={
                "completed": len(report.completed),
                "failed": len(report.failed),
                "skipped": len(report.skipped),
                "blocked_on_approval": len(report.blocked),
                "cost_usd": round(report.cost_usd, 6),
            },
        )
        return report
