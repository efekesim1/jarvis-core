"""What each part of the system has actually been shown to do.

`make check` going green is one fact, and it is smaller than it looks. It means
the offline suite passed — which for a migration chain or a permission table is
the whole question, and for an agent's answer quality is almost none of it.
Reporting both as "tested" is how a system ends up describing itself as ready.

So each subsystem carries two declarations that cannot be quietly satisfied:

* the test modules that cover it, which a test asserts actually exist, and
* whether proving it works *needs a model at all*.

From those two, plus what the verification store holds, the level is computed:

* ``PROVEN`` — its behaviour is decided by code, and code is what the offline
  suite exercises. A rate limiter either counts correctly or it does not; no
  model is going to change the answer.
* ``PARTIAL`` — the plumbing is proven and the thing it exists for is not. The
  agent runtime provably assembles a prompt, routes it, parses the reply and
  scores it. Whether the reply was any good is not in that sentence.
* ``BLOCKED`` — it needs a real model and there is no credential, so it has not
  been attempted. Distinct from failing, and reported that way.
* ``UNVERIFIED`` — nothing covers it.

A subsystem with no declaration is UNVERIFIED, and a test asserts every package
in the tree has one. That is the part that keeps this honest: adding code
without tests moves a line to UNVERIFIED rather than leaving the report alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Readiness(StrEnum):
    """How far one subsystem has been shown to work."""

    PROVEN = "PROVEN"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class Subsystem:
    """One part of the system, and what would have to be true to trust it."""

    package: str
    concern: str
    tests: tuple[str, ...] = ()
    # True when the subsystem's *purpose* needs a model, not merely when it can
    # call one. The tool gateway calls models on behalf of agents and still
    # belongs here as False: what it exists to do is refuse things, and a
    # refusal is decided by code.
    needs_model: bool = False
    note: str = ""

    def level(self, *, real_evidence: bool) -> Readiness:
        if not self.tests:
            return Readiness.UNVERIFIED
        if not self.needs_model:
            return Readiness.PROVEN
        return Readiness.PARTIAL if real_evidence else Readiness.BLOCKED


# The roster. Ordered by the layering in CLAUDE.md so the report reads the way
# the system is built, top to bottom.
SUBSYSTEMS: tuple[Subsystem, ...] = (
    Subsystem(
        "config",
        "settings, budgets and limits",
        ("test_jarvis.py", "test_cost_enforcement.py"),
        note="every ceiling in the system is read from here",
    ),
    Subsystem("logging", "the audit trail", ("test_security.py",)),
    Subsystem("text", "term extraction, English and Turkish", ("test_twin.py", "test_skills.py")),
    Subsystem(
        "security",
        "permissions, risk, sensitivity, secrets, approval",
        ("test_security.py", "test_tool_gateway.py"),
        note="single-use approvals, default deny, secret redaction",
    ),
    Subsystem(
        "memory",
        "what is remembered, and what is refused",
        ("test_memory.py",),
        note="retrieval is lexical; see KNOWN_ISSUES",
    ),
    Subsystem(
        "twin",
        "what is known about the operator, and what may never be derived",
        ("test_twin.py",),
        note="protected categories, contested traits, the constitution",
    ),
    Subsystem("goals", "the goal tree and counted progress", ("test_goals.py",)),
    Subsystem(
        "storage",
        "the database, its migrations and its stores",
        ("test_storage.py",),
        note="v1 through the chain to the current schema",
    ),
    Subsystem(
        "models",
        "providers, catalog, router, token estimation",
        ("test_models.py", "test_cost_enforcement.py"),
        needs_model=True,
        note="routing logic is proven offline; whether the chosen model is the "
        "better answer is not",
    ),
    Subsystem(
        "observability",
        "cost accounting, cache, tracing",
        ("test_cost_enforcement.py", "test_cache.py"),
    ),
    Subsystem(
        "mcp", "servers, transports, discovery, error taxonomy", ("test_tool_integration.py",)
    ),
    Subsystem(
        "integrations",
        "the web, GitHub and file adapters",
        ("test_web_tool.py", "test_files_tool.py", "test_tool_integration.py"),
        note="file paths are resolved then checked, so a symlink out is caught",
    ),
    Subsystem(
        "tools",
        "the gateway: policy, approval, validation, limits",
        ("test_tool_gateway.py", "test_tool_integration.py"),
        note="eleven gates, each with a test that asserts the refusal",
    ),
    Subsystem(
        "agents",
        "definitions, registry, runtime",
        ("test_agents.py", "test_agent_roster.py"),
        needs_model=True,
        note="the definitions are audited for free; the answers are not",
    ),
    Subsystem(
        "core",
        "context, tasks, planner, orchestrator, scheduler, JARVIS",
        (
            "test_context.py",
            "test_task_engine.py",
            "test_planner.py",
            "test_orchestrator.py",
            "test_scheduler.py",
            "test_jarvis.py",
        ),
        needs_model=True,
        note="injection defence and context budgets are proven; plan quality is not",
    ),
    Subsystem(
        "evaluation",
        "scoring an answer",
        ("test_evaluation.py",),
        needs_model=True,
        note="the scoring is deterministic; whether it agrees with a human is untested",
    ),
    Subsystem(
        "skills",
        "loading, and measuring whether they help",
        ("test_skills.py", "test_skill_evaluation.py"),
        needs_model=True,
        note="what loads is proven; whether it helps needs two arms and a model",
    ),
    Subsystem(
        "knowledge",
        "acquisition, ending at a proposal",
        ("test_knowledge.py",),
        needs_model=True,
    ),
    Subsystem(
        "experiments",
        "the sandbox and the lab",
        ("test_experiments.py",),
        note="the sandbox holds no money, deploy or admin rights, by test",
    ),
    Subsystem(
        "improvement",
        "observe, analyse, propose — and no apply",
        ("test_improvement.py",),
        note="a test asserts there is no `apply` method, by name",
    ),
    Subsystem(
        "business", "opportunities, and the numbers it will not invent", ("test_business.py",)
    ),
    Subsystem("events", "what is worth interrupting for", ("test_events.py",)),
    Subsystem("briefing", "the brief and the executive view", ("test_briefing.py",)),
    Subsystem("api", "the HTTP interface", ("test_api.py",)),
    Subsystem(
        "chat",
        "two modes, and what the paid one will not start without",
        ("test_chat.py",),
        note="a confirmation bound to one message, single-use, expiring",
    ),
    Subsystem(
        "verification",
        "telling an agent that exists from one that works",
        ("test_verification.py",),
        note="this file's own honesty is tested here",
    ),
)


@dataclass(slots=True)
class ReadinessReport:
    """Every subsystem's level, and the evidence behind each one."""

    rows: list[tuple[Subsystem, Readiness]] = field(default_factory=list)
    real_runs: int = 0

    def by_level(self, level: Readiness) -> list[Subsystem]:
        return [system for system, got in self.rows if got is level]

    @property
    def ready(self) -> bool:
        """Whether anything here can be called production ready.

        It cannot, while a single subsystem is BLOCKED or UNVERIFIED. This is
        the property that refuses to go green, and it is deliberately harsh:
        a report that says "mostly ready" is read as "ready".
        """
        return not self.by_level(Readiness.BLOCKED) and not self.by_level(Readiness.UNVERIFIED)

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for _, level in self.rows:
            counts[level.value] = counts.get(level.value, 0) + 1
        return {
            "subsystems": len(self.rows),
            "by_level": counts,
            "real_runs": self.real_runs,
            "ready": self.ready,
            "rows": [
                {
                    "package": system.package,
                    "concern": system.concern,
                    "level": level.value,
                    "tests": list(system.tests),
                    "note": system.note,
                }
                for system, level in self.rows
            ],
        }


def assess(*, real_runs: int = 0) -> ReadinessReport:
    """The readiness of every subsystem, given what real evidence exists.

    ``real_runs`` is the count of real-model verification runs on record. It is
    the only input that can lift a model-dependent subsystem off BLOCKED, and
    it comes from the verification store rather than from a flag.
    """
    real = real_runs > 0
    return ReadinessReport(
        rows=[(system, system.level(real_evidence=real)) for system in SUBSYSTEMS],
        real_runs=real_runs,
    )


def declared_packages() -> frozenset[str]:
    return frozenset(system.package for system in SUBSYSTEMS)


def source_packages(root: Path) -> frozenset[str]:
    """Every module and package directly under ``src/jarvis_core``.

    Used by a test to assert the roster above has not fallen behind the tree.
    A subsystem nobody declared would otherwise be silently absent from every
    readiness report, which is the failure mode this whole file exists to stop.
    """
    names: set[str] = set()
    for entry in root.iterdir():
        if entry.name.startswith(("_", ".")):
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.suffix == ".py" and entry.stem != "cli":
            names.add(entry.stem)
    return frozenset(names)
