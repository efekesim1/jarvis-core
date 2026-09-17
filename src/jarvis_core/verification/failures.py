"""Naming what went wrong, so the same failure is not fixed twice.

A sweep over twenty-seven agents produces a column of red. "Nine failed" is
almost no information: nine schema failures are one bug in the prompt template,
while nine different kinds are nine separate pieces of work. So every failed run
is classified, from what was actually recorded about it, into one of nine kinds
that each imply a different repair:

* ``OUTPUT_TRUNCATED`` — it ran out of output budget mid-answer. Raise
  ``max_tokens`` or narrow the schema; do not touch the prompt.
* ``SCHEMA_FAILURE`` — it answered *completely*, and the answer did not have the
  shape its own definition asked for. Fix the schema or the prompt.
* ``SOURCE_FAILURE`` — its policy requires citations and it cited nothing.
* ``PROMPT_FAILURE`` — it produced nothing usable to grade. The instruction did
  not land.
* ``EVALUATION_FAILURE`` — it ran, parsed and cited, and the answer was not good
  enough. This is the only kind that is about quality.
* ``MODEL_FAILURE`` — the provider refused, errored, or was unavailable.
* ``TOOL_FAILURE`` — a tool call it depended on failed.
* ``PERMISSION_FAILURE`` — the system refused it on purpose. Not a defect.
* ``TIMEOUT`` — it ran out of time.
* ``UNKNOWN`` — none of the above matched, which is itself worth seeing rather
  than hiding inside a bucket that sounds specific.

Classification reads the run, never re-runs it. A taxonomy that costs a model
call to apply would be a taxonomy nobody applies.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..agents.base import AgentSpec
    from .status import VerificationRun


class FailureKind(StrEnum):
    """Why one run did not succeed."""

    OUTPUT_TRUNCATED = "output_truncated"
    SCHEMA_FAILURE = "schema_failure"
    TOOL_FAILURE = "tool_failure"
    PROMPT_FAILURE = "prompt_failure"
    MODEL_FAILURE = "model_failure"
    EVALUATION_FAILURE = "evaluation_failure"
    SOURCE_FAILURE = "source_failure"
    PERMISSION_FAILURE = "permission_failure"
    TIMEOUT = "timeout"
    # Three that say nothing about the agent and everything about the account
    # it was run from. Kept apart from MODEL_FAILURE because that reads as "the
    # model could not do it", and a dead key is not the model's opinion.
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"

    @property
    def is_environmental(self) -> bool:
        """Whether this says something about the environment, not the agent.

        These four must never be counted against an agent's record. An empty
        account produces twenty-seven of them, and reading that as twenty-seven
        broken agents is the exact mistake this taxonomy exists to prevent.
        """
        return self in (
            FailureKind.AUTHENTICATION,
            FailureKind.BILLING,
            FailureKind.RATE_LIMIT,
            FailureKind.PERMISSION_FAILURE,
        )


# Substrings that identify a kind from an error message. Ordered by how
# specific they are: "approval denied" is a permission story even though the
# word "denied" could belong to several.
_MARKERS: tuple[tuple[FailureKind, tuple[str, ...]], ...] = (
    # Account-level faults first, and read from the provider's own wording via
    # the models layer, so that one dead credential never reads as an agent
    # defect. `classify_fault` lives beside the router because the router needs
    # the same reading to decide whether another model is worth trying.
    (FailureKind.BILLING, ("credit balance", "billing", "plans & billing", "insufficient funds")),
    (
        FailureKind.AUTHENTICATION,
        ("authentication_error", "api key is invalid", "invalid x-api-key", "unauthorized"),
    ),
    (FailureKind.RATE_LIMIT, ("rate_limit", "rate limit", "too many requests", "overloaded")),
    (FailureKind.TIMEOUT, ("timeout", "timed out", "deadline exceeded")),
    (
        FailureKind.PERMISSION_FAILURE,
        (
            "approval denied",
            "approval required",
            "permission",
            "not available in the sandbox",
            "refused by policy",
            "denied by policy",
        ),
    ),
    (
        FailureKind.TOOL_FAILURE,
        ("tool call", "tool failed", "gateway", "no such tool", "transport"),
    ),
    (
        FailureKind.MODEL_FAILURE,
        (
            "declined the request",
            "no model provider",
            "no usable model",
            "authentication",
            "unauthorized",
            "rate limit",
            "overloaded",
            "api error",
            "connection",
        ),
    ),
)


def schema_gaps(data: dict[str, Any] | None, spec: AgentSpec) -> list[str]:
    """Keys the agent's own ``output_schema`` promised and the answer omitted.

    Private keys (leading underscore) are the definition talking to itself and
    are not part of what the model was asked to produce.
    """
    if not spec.output_schema:
        return []
    required = [str(key) for key in spec.output_schema if not str(key).startswith("_")]
    present = data or {}
    return [key for key in required if key not in present]


def classify(run: VerificationRun, spec: AgentSpec | None = None) -> FailureKind | None:
    """What kind of failure this run was, or ``None`` when it was not one.

    The order is the argument. A crash outranks a bad score, because an agent
    that never answered cannot be said to have answered badly; and among the
    ways of not answering, the ones the system caused (permission, timeout)
    outrank the ones the agent caused, so that a deliberate refusal is never
    filed as a defect in the agent.
    """
    error = (run.error or "").lower()
    if error:
        for kind, markers in _MARKERS:
            if any(marker in error for marker in markers):
                return kind

    if not run.ok:
        return FailureKind.UNKNOWN if error else FailureKind.MODEL_FAILURE

    # It answered. Now the answer is what is on trial, and the checks run from
    # the most mechanical to the most subjective: a missing key is a fact, a low
    # score is a judgement.
    # Before the schema check, because a cut-off answer has missing keys *as a
    # symptom*. Reporting it as a schema failure sends the next person to edit
    # the schema when the ceiling is what ran out — and `market_researcher`
    # taught this the expensive way: 96 sources, 124 characters of text, all
    # nine schema keys absent, filed as a schema defect.
    if run.truncated:
        return FailureKind.OUTPUT_TRUNCATED

    if not run.schema_valid:
        return FailureKind.SCHEMA_FAILURE

    # Whether there is an answer at all comes before whether it was cited: an
    # agent cannot fail to cite an answer it never gave, and reporting an empty
    # reply as a missing citation would send the repair to the wrong place.
    if not run.answered:
        return FailureKind.PROMPT_FAILURE

    if spec is not None and spec.evaluation_policy.require_sources and run.sources == 0:
        return FailureKind.SOURCE_FAILURE

    if not run.passed:
        return FailureKind.EVALUATION_FAILURE

    return None


def summarise(kinds: list[FailureKind]) -> dict[str, int]:
    """How many of each kind, in the enum's own order so reports do not shuffle."""
    counts = {kind.value: 0 for kind in FailureKind}
    for kind in kinds:
        counts[kind.value] += 1
    return {name: count for name, count in counts.items() if count}
