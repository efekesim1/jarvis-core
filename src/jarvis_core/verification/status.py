"""The difference between an agent that exists and an agent that works.

Twenty-seven YAML files load, pass their capability check and appear in
`jarvis agents`. Six of them have ever produced an answer from a real model.
Nothing in the system said so, and every report about "the agent roster" was
therefore a claim about files rather than about behaviour.

This module fixes that by refusing to let a status be asserted. Every level is
*derived* from recorded runs, and each one requires evidence the level below it
could not have produced:

* ``DEFINED`` — the file loads and its declared capabilities are backed. This
  is what a YAML file gets for free, and it says nothing about quality.
* ``OFFLINE_TESTED`` — it ran end to end against the mock provider without
  crashing. Proves the plumbing, not the prompt.
* ``REAL_MODEL_TESTED`` — it ran against a real model and produced parseable,
  schema-shaped output. This is the level no amount of offline testing can
  reach, and without a credential it is never claimed.
* ``EVALUATED`` — that output cleared the evaluator's bar, sources included
  where the agent's own policy requires them.
* ``PRODUCTION_READY`` — it did that repeatedly, with scores that did not
  swing, and made no claim it could not support.

A missing credential produces ``OFFLINE_TESTED`` and a reason saying so. It
never produces a higher level and never produces a failure: not having an API
key is a fact about the environment, not a defect in the agent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any


class AgentStatus(IntEnum):
    """How far an agent has been proven, ordered so comparisons work."""

    DEFINED = 0
    OFFLINE_TESTED = 1
    REAL_MODEL_TESTED = 2
    EVALUATED = 3
    PRODUCTION_READY = 4

    @property
    def label(self) -> str:
        return self.name.lower()


# What ``PRODUCTION_READY`` costs, stated here rather than inline so that
# arguing about the bar is a conversation about four numbers.
MIN_RUNS_FOR_PRODUCTION = 3
MIN_MEAN_SCORE = 0.65
MAX_SCORE_SPREAD = 0.2  # an agent whose score swings is not dependable
MAX_FAILURE_RATE = 0.0  # in three runs, one failure is a third of them

# Below this many characters there is no answer to grade, whatever the
# evaluator would have scored it.
MIN_ANSWER_CHARS = 40


@dataclass(slots=True)
class VerificationRun:
    """One recorded attempt at making an agent do its job."""

    agent_id: str
    task: str
    real_model: bool
    ok: bool
    score: float = 0.0
    passed: bool = False
    confidence: float = 0.0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    sources: int = 0
    # What it consumed, as the provider counted it. Recorded next to the cost
    # so the bill can be checked against the price list rather than believed.
    input_tokens: int = 0
    output_tokens: int = 0
    # Whether the answer carried every key the agent's own output_schema
    # promised, and which ones it did not. A run that parses but omits half its
    # schema is a schema failure, not a low score.
    schema_valid: bool = True
    missing_keys: list[str] = field(default_factory=list)
    # How much answer there was to grade at all. An empty answer that parses is
    # a prompt that did not land, and scoring it would report the evaluator's
    # opinion of nothing.
    answer_chars: int = 0
    # Whether the model was cut off rather than finishing. Kept separate from
    # `schema_valid`, because a truncated answer fails the schema for a reason
    # that has nothing to do with the schema.
    truncated: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    error: str = ""
    # Which of the nine failure kinds this was, classified from the fields
    # above by ``failures.classify`` when the run was recorded. Empty means the
    # run succeeded. Kept as a string so a stored record survives the enum
    # gaining a member.
    failure: str = ""
    # The environment refused to run it, rather than the agent failing. A
    # deploy agent cannot be verified in a sandbox that withholds DEPLOY, and
    # recording that as a defect would be blaming the agent for a rule the
    # system is enforcing on purpose.
    blocked: bool = False
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usable(self) -> bool:
        """Whether this run produced an answer worth taking seriously.

        Three separate things have to be true, and ``ok`` is only the first:
        the call returned, the answer had the shape the agent's own definition
        promised, and there was enough of it to read. A run that returns an
        empty object is not a success with a low score; it is not an answer.
        """
        return self.ok and self.schema_valid and self.answered

    @property
    def answered(self) -> bool:
        """Whether there was enough of an answer to be worth grading.

        The bar is deliberately low. This separates "produced nothing" from
        "produced something bad"; deciding which side of *good* an answer falls
        on is the evaluator's job, not this property's.
        """
        return self.answer_chars >= MIN_ANSWER_CHARS

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task": self.task[:120],
            "real_model": self.real_model,
            "ok": self.ok,
            "score": round(self.score, 3),
            "passed": self.passed,
            "confidence": round(self.confidence, 3),
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "sources": self.sources,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "schema_valid": self.schema_valid,
            "missing_keys": list(self.missing_keys),
            "answer_chars": self.answer_chars,
            "truncated": self.truncated,
            "unsupported_claims": list(self.unsupported_claims),
            "error": self.error,
            "failure": self.failure,
            "blocked": self.blocked,
            "at": self.at.isoformat(),
        }


@dataclass(slots=True)
class AgentVerification:
    """Everything known about whether one agent works, and what it rests on."""

    agent_id: str
    runs: list[VerificationRun] = field(default_factory=list)
    capabilities_backed: bool = True
    has_smoke_task: bool = True
    # Why no real run was attempted, when something outside this agent stopped
    # it. Set from the sweep's reachability probe. Without it every agent reads
    # "set ANTHROPIC_API_KEY" even when the key is set and correct and the
    # account simply has no credit — which sends the reader to fix the one
    # thing that is already right.
    real_blocked_reason: str = ""

    @property
    def real_runs(self) -> list[VerificationRun]:
        return [run for run in self.runs if run.real_model and not run.blocked]

    @property
    def offline_runs(self) -> list[VerificationRun]:
        return [run for run in self.runs if not run.real_model and not run.blocked]

    @property
    def blocked_runs(self) -> list[VerificationRun]:
        return [run for run in self.runs if run.blocked]

    @property
    def real_cost(self) -> float:
        """What verification actually cost.

        Offline runs go through a mock provider that still prices its usage, so
        summing every run reports a bill nobody paid. Only real runs spend
        money, and a cost report that overstates is as useless as one that
        understates.
        """
        return sum(run.cost_usd for run in self.runs if run.real_model)

    @property
    def mean_score(self) -> float | None:
        scores = [run.score for run in self.real_runs if run.usable]
        return statistics.mean(scores) if scores else None

    @property
    def score_spread(self) -> float | None:
        """How much the score moved between runs of the same task.

        Two runs is enough to see a swing; one is not enough to see anything,
        and returning 0.0 for a single run would read as "perfectly stable".
        """
        scores = [run.score for run in self.real_runs if run.usable]
        return max(scores) - min(scores) if len(scores) >= 2 else None

    @property
    def failure_rate(self) -> float:
        real = self.real_runs
        return sum(1 for run in real if not run.usable) / len(real) if real else 0.0

    @property
    def failures(self) -> list[str]:
        """The failure kinds recorded against real runs, newest last."""
        return [run.failure for run in self.real_runs if run.failure]

    @property
    def only_environmental_failures(self) -> bool:
        """Whether every real attempt died for a reason outside this agent.

        Billing, authentication and rate limits say something about the account,
        not the prompt: the call never reached the model, so nothing about the
        agent was tested. Treating those as a defect is the same mistake as
        reporting BLOCKED as FAIL, in the other direction.

        False when there are no real runs at all — that is a different state,
        and `_cell` already has a branch for it.
        """
        from .failures import FailureKind

        real = self.real_runs
        if not real:
            return False
        if any(run.usable for run in real):
            return False
        kinds = [run.failure for run in real]
        if not all(kinds):
            return False
        return all(FailureKind(kind).is_environmental for kind in kinds)

    @property
    def unsupported_claims(self) -> list[str]:
        return [claim for run in self.real_runs if run.usable for claim in run.unsupported_claims]

    def status(self) -> tuple[AgentStatus, str]:
        """The level this agent has actually reached, and why not the next one.

        The reason matters more than the level. "REAL_MODEL_TESTED — no
        credential, so nothing above this was attempted" and "REAL_MODEL_TESTED
        — scored 0.41, below the 0.65 bar" are the same word and completely
        different situations.
        """
        if not self.capabilities_backed:
            return AgentStatus.DEFINED, "it declares a capability no tool provides"
        if not self.has_smoke_task:
            return AgentStatus.DEFINED, "no smoke task declared, so nothing can be run"
        if not self.runs:
            return AgentStatus.DEFINED, "never run"

        if all(run.blocked for run in self.runs):
            reason = self.runs[-1].error
            return AgentStatus.DEFINED, f"cannot be run in this environment: {reason}"

        if not any(run.ok for run in self.offline_runs) and not self.real_runs:
            failed = next((run for run in self.offline_runs if run.error), None)
            return (
                AgentStatus.DEFINED,
                f"offline run failed: {failed.error if failed else 'unknown'}",
            )

        real = self.real_runs
        if not real:
            return (
                AgentStatus.OFFLINE_TESTED,
                self.real_blocked_reason
                or "no real model run — set ANTHROPIC_API_KEY and run `jarvis verify --real`",
            )

        successful = [run for run in real if run.usable]
        if not successful:
            last = real[-1]
            if self.only_environmental_failures:
                return AgentStatus.OFFLINE_TESTED, (
                    f"no real run reached the model: {last.failure or 'environment'} — "
                    "this says nothing about the agent"
                )
            return AgentStatus.OFFLINE_TESTED, (
                f"no real run produced a usable answer: {_why_unusable(last)}"
            )

        if not any(run.passed for run in successful):
            best = max(run.score for run in successful)
            return (
                AgentStatus.REAL_MODEL_TESTED,
                f"best real score {best:.2f} did not pass the evaluator",
            )

        if claims := self.unsupported_claims:
            return (
                AgentStatus.EVALUATED,
                f"made {len(claims)} claim(s) it could not support: {claims[0][:60]}",
            )
        if len(successful) < MIN_RUNS_FOR_PRODUCTION:
            return (
                AgentStatus.EVALUATED,
                f"{len(successful)} successful real run(s); "
                f"{MIN_RUNS_FOR_PRODUCTION} are needed to call it repeatable",
            )
        if self.failure_rate > MAX_FAILURE_RATE:
            return AgentStatus.EVALUATED, f"{self.failure_rate:.0%} of real runs failed"

        mean = self.mean_score or 0.0
        if mean < MIN_MEAN_SCORE:
            return AgentStatus.EVALUATED, f"mean score {mean:.2f} is below the {MIN_MEAN_SCORE} bar"

        spread = self.score_spread
        if spread is not None and spread > MAX_SCORE_SPREAD:
            return (
                AgentStatus.EVALUATED,
                f"score swung by {spread:.2f} between runs; an agent that is "
                "sometimes good is not dependable",
            )

        return AgentStatus.PRODUCTION_READY, f"{len(successful)} real runs, mean {mean:.2f}"

    def as_dict(self) -> dict[str, Any]:
        status, reason = self.status()
        return {
            "agent_id": self.agent_id,
            "status": status.label,
            "reason": reason,
            "runs": len(self.runs),
            "real_runs": len(self.real_runs),
            "mean_score": None if self.mean_score is None else round(self.mean_score, 3),
            "score_spread": None if self.score_spread is None else round(self.score_spread, 3),
            "failure_rate": round(self.failure_rate, 3),
            "failures": self.failures,
            "total_tokens": sum(run.total_tokens for run in self.real_runs),
            "unsupported_claims": len(self.unsupported_claims),
            "blocked_runs": len(self.blocked_runs),
            "real_cost_usd": round(self.real_cost, 6),
        }


def _why_unusable(run: VerificationRun) -> str:
    """One sentence saying which of the three bars this run missed.

    ``usable`` is a conjunction, so a bare "failed" leaves the reader to guess
    which conjunct broke — and the three imply completely different repairs.
    """
    if not run.ok:
        return run.error or "the run did not complete"
    if not run.schema_valid:
        missing = ", ".join(run.missing_keys[:4]) or "keys"
        return f"the answer omitted {missing} from its own output schema"
    if not run.answered:
        return f"the answer was {run.answer_chars} characters, too little to grade"
    return "unknown"
