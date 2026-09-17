"""Checking an agent definition without spending anything.

Running an agent against a real model is the only way to know whether its
answers are good. It is not the only way to know whether it is *broken*.

A definition can be internally contradictory in ways that guarantee failure
before a single token is spent: a policy that requires sources from an agent
whose output shape has nowhere to put them, a prompt that tells the model to
search the web for an agent the gateway will not let near a network, a retry
policy that promises an alternative model to an agent allowed exactly one.

Every check here is deterministic, needs no credential, and describes a
failure that would actually happen. That is the bar: a lint that reports style
preferences trains people to ignore lints, and the next thing it says is the
thing that mattered.

What this cannot tell you is whether an agent is any good. `jarvis verify
--real` is still the only thing that can, and nothing here pretends otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..agents.base import AgentSpec
from ..logging import get_logger
from ..security.permissions import Permission
from ..text import terms

log = get_logger("verification.audit")


class Severity(StrEnum):
    """How much a finding matters."""

    BREAKS = "breaks"  # this agent cannot succeed as defined
    DEGRADES = "degrades"  # it will run, and worse than it should
    NOTE = "note"  # worth knowing, not worth blocking


@dataclass(slots=True)
class Finding:
    """One thing wrong with a definition, and what to do about it."""

    agent_id: str
    check: str
    severity: Severity
    what: str
    fix: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "check": self.check,
            "severity": self.severity.value,
            "what": self.what,
            "fix": self.fix,
        }


# Which permission each capability prefix needs at the gateway. An agent whose
# prompt tells it to search the web, holding `web.read` but not NETWORK, fails
# every call — and the failure looks like a model problem.
CAPABILITY_PERMISSION: dict[str, Permission] = {
    "web.": Permission.NETWORK,
    "github.": Permission.NETWORK,
    "files.": Permission.READ,
}

# Phrases in a prompt that describe reaching the outside world, and the
# capability that makes them possible.
PROMPT_CAPABILITY_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsearch (?:the )?web\b|\bweb search\b|\bsearch online\b", re.I), "web."),
    (re.compile(r"\bread (?:the )?repositor|\bgithub\b", re.I), "github."),
    (re.compile(r"\bread (?:the )?file|\bthe workspace\b", re.I), "files."),
)


def audit_agent(spec: AgentSpec, *, known_models: frozenset[str] | None = None) -> list[Finding]:
    """Everything wrong with one definition that can be known for free."""
    findings: list[Finding] = []

    def add(check: str, severity: Severity, what: str, fix: str) -> None:
        findings.append(Finding(spec.id, check, severity, what, fix))

    schema_keys = {str(key).lower() for key in spec.output_schema}
    policy = spec.evaluation_policy

    # -- the output shape the runtime actually reads ----------------------

    # `AgentResult.text` comes from `parsed["summary"]`, falling back to the
    # raw response. Without `summary` in the schema the model has no reason to
    # produce one, and every downstream reader gets a JSON blob as prose.
    if spec.output_schema and "summary" not in schema_keys:
        add(
            "output_summary",
            Severity.BREAKS,
            "the output schema has no 'summary', so the runtime falls back to the raw "
            "JSON as this agent's answer, and synthesis reads that",
            "add a 'summary' key describing the answer in plain language",
        )

    # `confidence` is read the same way and defaults to 0.6 when absent, which
    # means the evaluator scores every run of this agent identically.
    if spec.output_schema and "confidence" not in schema_keys:
        add(
            "output_confidence",
            Severity.DEGRADES,
            "no 'confidence' in the output schema, so every run reports the 0.6 "
            "default and the evaluator cannot tell a sure answer from a guess",
            "add 'confidence: 0.0' to the schema",
        )

    # -- the policy against the shape -------------------------------------

    if policy.require_sources and spec.output_schema and "sources" not in schema_keys:
        add(
            "sources_impossible",
            Severity.BREAKS,
            "require_sources is on but the output schema has nowhere to put sources, "
            "so every run fails the hard source check",
            "add a 'sources' list to the schema, or turn require_sources off",
        )

    if "source_quality" in policy.criteria and not policy.require_sources:
        add(
            "sources_optional",
            Severity.DEGRADES,
            "source_quality is scored but sources are not required, so an answer "
            "with none is judged on a criterion it cannot fail",
            "set require_sources: true, or drop source_quality from the criteria",
        )

    # -- reaching the world ------------------------------------------------

    for prefix, permission in CAPABILITY_PERMISSION.items():
        wants = any(cap.startswith(prefix) for cap in spec.tool_capabilities)
        if wants and permission not in spec.permissions:
            add(
                "permission_missing",
                Severity.BREAKS,
                f"declares {prefix}* capabilities but not the {permission.value} "
                "permission, so the gateway refuses every one of those calls",
                f"add '{permission.value}' to permissions",
            )

    for pattern, prefix in PROMPT_CAPABILITY_HINTS:
        if pattern.search(spec.system_prompt) and not any(
            cap.startswith(prefix) for cap in spec.tool_capabilities
        ):
            add(
                "prompt_promises_a_tool",
                Severity.DEGRADES,
                f"the prompt describes work needing {prefix}* and the agent has no such "
                "capability, so it will try and be refused",
                f"add a {prefix}* capability, or stop telling it to do that",
            )

    # -- retries and models ------------------------------------------------

    # An empty `allowed_models` means "no restriction", not "one model": the
    # router picks from the whole catalogue. Only an explicit list of one
    # leaves a retry with nowhere else to go. The first version of this check
    # read empty as one and reported four agents that were fine.
    if spec.retry_policy.alternative_model and len(spec.allowed_models) == 1:
        add(
            "no_alternative_model",
            Severity.DEGRADES,
            f"the retry policy promises a different model on failure and only "
            f"{spec.allowed_models[0]} is allowed, so a retry repeats the call that "
            "just failed",
            "list a second model in allowed_models, or set alternative_model: false",
        )

    if known_models is not None:
        unknown = [m for m in spec.allowed_models if m not in known_models]
        if unknown:
            add(
                "unknown_model",
                Severity.BREAKS,
                f"allows {', '.join(unknown)}, which the catalogue does not have, so "
                "the router cannot route to it",
                "use a model from config/models.json, or add it there",
            )

    # -- the prompt itself -------------------------------------------------

    if not spec.system_prompt.strip():
        add(
            "no_prompt",
            Severity.BREAKS,
            "no system prompt, so this agent is a name and a schema",
            "write one, or delete the agent",
        )
    elif len(spec.system_prompt) < 200:
        add(
            "thin_prompt",
            Severity.DEGRADES,
            f"the prompt is {len(spec.system_prompt)} characters, which is a job title "
            "rather than instructions",
            "say what good output looks like and what to refuse",
        )
    elif not re.search(r"\b(never|do not|don't|rules|avoid|refuse)\b", spec.system_prompt, re.I):
        add(
            "no_boundaries",
            Severity.NOTE,
            "the prompt says what to do and never what not to, and the failure modes "
            "are what a prompt is actually for",
            "add the two or three things this agent must not do",
        )

    # -- the smoke task ----------------------------------------------------

    if not spec.smoke_task.strip():
        add(
            "no_smoke_task",
            Severity.BREAKS,
            "no smoke task, so nothing can verify this agent works",
            "add a real task it can do, phrased the way the operator would ask",
        )
    else:
        declared = terms(f"{spec.name} {spec.role} {' '.join(spec.capabilities)}")
        asked = terms(spec.smoke_task)
        if declared and asked and not (declared & asked):
            add(
                "smoke_task_unrelated",
                Severity.NOTE,
                "the smoke task shares no words with what this agent says it does, so "
                "a pass proves the plumbing and not the agent",
                "phrase the task in the terms the agent declares, or widen the capabilities",
            )

    # -- budget ------------------------------------------------------------

    # A rough floor: every schema key needs room for a real answer, and a
    # truncated JSON object fails to parse rather than arriving short.
    #
    # A server tool shares this budget, and that is the part the first version
    # of this check missed. `market_researcher` declares web_search and a
    # nine-key schema at max_tokens 8000; the old floor asked for 4,500, so the
    # definition passed — and then the real run pulled 96 sources, returned 124
    # characters of text, and arrived with all nine keys missing. The search
    # blocks and the model's commentary between them come out of the same
    # ceiling as the answer, so a searching agent needs room for both.
    #
    # The multiplier is a heuristic fitted to that one case, not a measurement,
    # and it is deliberately stated as such: it flags an agent that would have
    # truncated, which is worth a DEGRADES finding even when the exact number
    # is arguable.
    keys = len(spec.output_schema)
    needed = max(1000, keys * 500)
    if spec.server_tools:
        needed = max(needed, keys * 1000 + 2000)
    if spec.max_tokens < needed:
        searching = (
            " and it declares a server tool that shares that budget" if spec.server_tools else ""
        )
        add(
            "output_budget",
            Severity.DEGRADES,
            f"max_tokens is {spec.max_tokens} for a {keys}-key schema{searching}; "
            "a truncated JSON object fails to parse rather than arriving short",
            f"raise max_tokens to about {needed}",
        )

    return findings


@dataclass(slots=True)
class AuditReport:
    """Every finding across the roster, and what it adds up to."""

    findings: list[Finding] = field(default_factory=list)
    agents: int = 0

    @property
    def breaking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BREAKS]

    @property
    def degrading(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.DEGRADES]

    @property
    def notes(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.NOTE]

    @property
    def clean_agents(self) -> int:
        touched = {f.agent_id for f in self.findings if f.severity is not Severity.NOTE}
        return self.agents - len(touched)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agents": self.agents,
            "clean": self.clean_agents,
            "breaking": len(self.breaking),
            "degrading": len(self.degrading),
            "notes": len(self.notes),
            "findings": [f.as_dict() for f in self.findings],
        }

    def render(self) -> str:
        if not self.findings:
            return (
                f"All {self.agents} definitions are internally consistent.\n"
                "This says nothing about answer quality — `jarvis verify --real` is "
                "the only thing that can."
            )
        lines = [
            f"{self.clean_agents} of {self.agents} definitions are clean. "
            f"{len(self.breaking)} would break, {len(self.degrading)} would degrade."
        ]
        for label, group in (
            ("Would break", self.breaking),
            ("Would degrade", self.degrading),
            ("Worth knowing", self.notes),
        ):
            if not group:
                continue
            lines.append(f"\n{label}:")
            for finding in group:
                lines.append(f"  {finding.agent_id}: {finding.what}")
                lines.append(f"    fix: {finding.fix}")
        return "\n".join(lines)


def audit_roster(
    specs: list[AgentSpec], *, known_models: frozenset[str] | None = None
) -> AuditReport:
    """Audit every definition. Free, offline, and deterministic."""
    report = AuditReport(agents=len(specs))
    for spec in specs:
        report.findings.extend(audit_agent(spec, known_models=known_models))
    log.info(
        "audited %d agent(s): %d breaking, %d degrading",
        len(specs),
        len(report.breaking),
        len(report.degrading),
    )
    return report
