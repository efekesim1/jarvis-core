"""The agent reality matrix: five rungs, twenty-seven agents, nothing asserted.

Every cell is computed from recorded runs. There is no method here that takes a
status as an argument, and that is the point: the moment a status can be
written down, it stops being a measurement and becomes a claim.

Three cell values, and the difference between two of them is the whole reason
this file exists:

* ``PASS`` — the evidence for this rung exists.
* ``FAIL`` — it was attempted and the evidence came back short.
* ``BLOCKED`` — it was never attempted, because the environment cannot attempt
  it. No API key is the usual reason.

A system that reports BLOCKED as FAIL looks broken when it is merely
unconfigured. A system that reports BLOCKED as PASS is lying. Keeping the third
value is what lets ``jarvis doctor`` refuse to go green.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .status import AgentStatus, AgentVerification


class Cell(StrEnum):
    """What is known about one rung for one agent."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"

    @property
    def proven(self) -> bool:
        return self is Cell.PASS


# The rungs, in the order they must be climbed. DEFINED is not here: it is what
# a file gets for loading, and a matrix column that is always PASS measures
# nothing.
RUNGS: tuple[tuple[str, AgentStatus], ...] = (
    ("offline", AgentStatus.OFFLINE_TESTED),
    ("real", AgentStatus.REAL_MODEL_TESTED),
    ("evaluated", AgentStatus.EVALUATED),
    ("production", AgentStatus.PRODUCTION_READY),
)


@dataclass(slots=True)
class AgentRow:
    """One agent's line in the matrix, and the numbers behind it."""

    agent_id: str
    status: AgentStatus
    reason: str
    cells: dict[str, Cell]
    quality: float | None
    cost_usd: float
    latency_ms: float | None
    tokens: int
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_id,
            "status": self.status.label,
            "reason": self.reason,
            **{name: self.cells[name].value for name, _ in RUNGS},
            "quality": None if self.quality is None else round(self.quality, 3),
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "tokens": self.tokens,
            "failures": list(self.failures),
        }


def _cell(record: AgentVerification | None, rung: AgentStatus, status: AgentStatus) -> Cell:
    """Whether this agent reached this rung, and if not, whether it could have.

    The distinction is entirely about whether an attempt was possible. An agent
    that has no real run *and* no credential to make one has not failed at
    anything; an agent that made three real runs and scored 0.3 has.
    """
    if status >= rung:
        return Cell.PASS
    if record is None:
        return Cell.BLOCKED
    if rung >= AgentStatus.REAL_MODEL_TESTED and not record.real_runs:
        # Nothing was ever sent to a real model, so nothing above the offline
        # rung was tested. That is a fact about the environment.
        return Cell.BLOCKED
    if rung is AgentStatus.OFFLINE_TESTED and not record.offline_runs:
        # Either it was never run, or every run was refused by the system on
        # purpose — a deploy agent in a sandbox that withholds DEPLOY is the
        # standing example. Neither is the agent failing at anything.
        return Cell.BLOCKED
    if rung >= AgentStatus.REAL_MODEL_TESTED and record.only_environmental_failures:
        # Every real attempt died for a reason outside the agent: the account
        # ran out of credit, the key was rejected, a rate limit. The call never
        # reached a prompt, so nothing about this agent was tested.
        #
        # The whole file exists for the FAIL/BLOCKED distinction and this is the
        # case it got wrong. Six research agents showed FAIL after the $5 of
        # credit ran out mid-sweep — reading as six defective agents when what
        # had actually happened was a billing event the taxonomy had already
        # named `environmental`. Consulting it here is the fix.
        return Cell.BLOCKED
    return Cell.FAIL


def build(
    records: dict[str, AgentVerification], agent_ids: list[str] | None = None
) -> list[AgentRow]:
    """One row per agent, including agents that were never run at all.

    An agent missing from the records is the most important row in the table —
    leaving it out would make the matrix describe the agents that happened to be
    verified rather than the roster.
    """
    ids = agent_ids if agent_ids is not None else sorted(records)
    rows: list[AgentRow] = []
    for agent_id in ids:
        record = records.get(agent_id)
        if record is None:
            status, reason = AgentStatus.DEFINED, "never run"
        else:
            status, reason = record.status()
        real = record.real_runs if record else []
        usable = [run for run in real if run.usable]
        rows.append(
            AgentRow(
                agent_id=agent_id,
                status=status,
                reason=reason,
                cells={name: _cell(record, rung, status) for name, rung in RUNGS},
                quality=record.mean_score if record else None,
                cost_usd=record.real_cost if record else 0.0,
                latency_ms=(
                    sum(run.latency_ms for run in usable) / len(usable) if usable else None
                ),
                tokens=sum(run.total_tokens for run in real),
                failures=sorted({run.failure for run in real if run.failure}),
            )
        )
    return rows


def totals(rows: list[AgentRow]) -> dict[str, Any]:
    """The bottom of the report: how many of each rung, and what it all cost."""
    counts = {name: sum(1 for row in rows if row.cells[name].proven) for name, _ in RUNGS}
    blocked = {name: sum(1 for row in rows if row.cells[name] is Cell.BLOCKED) for name, _ in RUNGS}
    qualities = [row.quality for row in rows if row.quality is not None]
    return {
        "agents": len(rows),
        "passed": counts,
        "blocked": blocked,
        "total_cost_usd": round(sum(row.cost_usd for row in rows), 6),
        "total_tokens": sum(row.tokens for row in rows),
        "mean_quality": round(sum(qualities) / len(qualities), 3) if qualities else None,
        # Agents where something actually went wrong: a recorded failure kind,
        # or an offline rung that was attempted and came back short.
        #
        # Deliberately *not* "any FAIL cell". PRODUCTION_READY needs three runs,
        # so after a deliberate one-run sweep every healthy agent carries a FAIL
        # on that rung — and listing all sixteen as "failed" told the exact
        # opposite of what the sweep had found. A bar that needs evidence nobody
        # has spent yet is not a defect in the agent.
        "failed_agents": sorted(
            row.agent_id
            for row in rows
            if row.cells["offline"] is Cell.FAIL
            or (row.failures and row.cells["real"] is Cell.FAIL)
        ),
        # Kept apart from failures on purpose: these agents were never reached,
        # so naming them alongside a real defect buries the one line worth
        # acting on.
        "blocked_by_environment": sorted(
            row.agent_id for row in rows if row.failures and row.cells["real"] is Cell.BLOCKED
        ),
        "short_of_production": sorted(
            row.agent_id
            for row in rows
            if row.cells["evaluated"].proven and not row.cells["production"].proven
        ),
    }


def render(rows: list[AgentRow]) -> str:
    """The matrix as text, for a terminal and for a document that outlives it."""
    header = ("AGENT", "OFFLINE", "REAL", "EVALUATED", "PRODUCTION", "QUALITY", "COST", "LATENCY")
    body: list[tuple[str, ...]] = []
    for row in rows:
        body.append(
            (
                row.agent_id,
                row.cells["offline"].value,
                row.cells["real"].value,
                row.cells["evaluated"].value,
                row.cells["production"].value,
                "—" if row.quality is None else f"{row.quality:.2f}",
                "—" if not row.cost_usd else f"${row.cost_usd:.4f}",
                "—" if row.latency_ms is None else f"{row.latency_ms:.0f}ms",
            )
        )
    widths = [
        max(len(header[i]), *(len(line[i]) for line in body)) if body else len(header[i])
        for i in range(len(header))
    ]
    out = ["  ".join(name.ljust(widths[i]) for i, name in enumerate(header))]
    out.append("  ".join("-" * width for width in widths))
    out.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) for line in body)

    summary = totals(rows)
    out.append("")
    for name, _ in RUNGS:
        passed, blocked = summary["passed"][name], summary["blocked"][name]
        note = f", {blocked} blocked" if blocked else ""
        out.append(f"{name:<11} {passed}/{summary['agents']}{note}")
    out.append("")
    out.append(
        f"{summary['total_tokens']:,} tokens, ${summary['total_cost_usd']:.4f} spent on real calls"
    )
    if summary["failed_agents"]:
        out.append(f"failed: {', '.join(summary['failed_agents'])}")
    if held := summary["blocked_by_environment"]:
        out.append(f"{len(held)} never reached the model (account, not agent): {', '.join(held)}")
    if short := summary["short_of_production"]:
        out.append(f"{len(short)} evaluated, short of production: more runs needed, not a defect")
    return "\n".join(out)
