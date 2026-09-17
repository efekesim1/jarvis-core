"""The Evaluation Engine.

An agent saying "done" proves nothing. Every result is scored before it counts,
so weak agents show up in the numbers instead of in a bad decision months later.

Scoring runs in two passes. The cheap pass is deterministic and free: it checks
structure, sources, confidence, cost and latency. The expensive pass asks a
model to judge quality, and is only used where it earns its cost.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..logging import get_logger
from ..models.base import Message, ModelRequest
from ..models.router import ModelRouter, TaskKind

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checking
    from ..agents.base import AgentResult, AgentSpec
    from ..core.task import Task

log = get_logger("evaluation")

# What each criterion is worth in the overall score.
WEIGHTS: dict[str, float] = {
    "correctness": 0.25,
    "relevance": 0.20,
    "completeness": 0.20,
    "source_quality": 0.15,
    "security": 0.10,
    "cost": 0.05,
    "latency": 0.05,
}


@dataclass(slots=True)
class Score:
    """One criterion, scored from 0 to 1."""

    criterion: str
    value: float
    note: str = ""

    def __post_init__(self) -> None:
        self.value = min(1.0, max(0.0, float(self.value)))


@dataclass(slots=True)
class Evaluation:
    """The verdict on one agent result."""

    scores: list[Score] = field(default_factory=list)
    overall: float = 0.0
    passed: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    business_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "passed": self.passed,
            "confidence": round(self.confidence, 3),
            "scores": {s.criterion: round(s.value, 3) for s in self.scores},
            "notes": list(self.notes),
            "business_value": self.business_value,
        }


class EvaluationEngine:
    """Scores agent results."""

    def __init__(
        self,
        *,
        router: ModelRouter | None = None,
        pass_threshold: float = 0.55,
        expensive_task_usd: float = 0.50,
        slow_task_ms: float = 60_000.0,
    ) -> None:
        self.router = router
        self.pass_threshold = pass_threshold
        self.expensive_task_usd = expensive_task_usd
        self.slow_task_ms = slow_task_ms

    # -- the free pass ----------------------------------------------------

    def evaluate(
        self, result: AgentResult, spec: AgentSpec, task: Task | None = None
    ) -> Evaluation:
        """Score a result without spending anything."""
        scores: list[Score] = []
        notes: list[str] = []
        policy = spec.evaluation_policy

        if not result.ok:
            notes.append(result.error or "the agent reported a failure")
            evaluation = Evaluation(
                scores=[Score("correctness", 0.0, "the run failed")],
                overall=0.0,
                passed=False,
                confidence=0.0,
                notes=notes,
            )
            return evaluation

        # Correctness: did it produce the shape it was asked for?
        text = (result.text or "").strip()
        has_structure = bool(result.data)
        correctness = 0.9 if has_structure and text else 0.5 if text else 0.1
        if not has_structure:
            notes.append("the reply was not valid JSON, so only the raw text is usable")
        scores.append(Score("correctness", correctness))

        # Relevance: does the answer share vocabulary with the goal?
        relevance = 0.6
        if task is not None and text:
            goal_words = {w.lower() for w in task.goal.split() if len(w) > 4}
            if goal_words:
                text_lower = text.lower()
                hits = sum(1 for w in goal_words if w in text_lower)
                relevance = min(1.0, 0.4 + 0.6 * hits / len(goal_words))
        scores.append(Score("relevance", relevance))

        # Completeness: length as a rough proxy, plus schema coverage.
        completeness = min(1.0, len(text) / 400) if text else 0.0
        if spec.output_schema:
            required = [k for k in spec.output_schema if not k.startswith("_")]
            if required:
                present = sum(1 for k in required if k in result.data)
                completeness = (completeness + present / len(required)) / 2
        scores.append(Score("completeness", completeness))

        # Source quality. A missing citation from an agent that must cite is a
        # hard failure, not a deduction: an unsourced research claim is not a
        # weaker answer, it is an unusable one.
        source_count = len(result.sources)
        missing_required_sources = policy.require_sources and source_count == 0
        if missing_required_sources:
            source_score = 0.0
            notes.append("this agent must cite sources and cited none")
        elif source_count == 0:
            source_score = 0.5
        else:
            source_score = min(1.0, 0.5 + 0.1 * source_count)
        scores.append(Score("source_quality", source_score, f"{source_count} source(s)"))

        # Security: does the text look like it is leaking a credential?
        security = 1.0
        lowered = text.lower()
        for marker in ("sk-ant-", "sk-proj-", "-----begin private key", "aws_secret_access_key"):
            if marker in lowered:
                security = 0.0
                notes.append("the output appears to contain a credential and was flagged")
                break
        scores.append(Score("security", security))

        # Cost and latency, scored against what "normal" looks like.
        cost_score = (
            1.0
            if result.cost_usd <= 0
            else max(0.0, 1.0 - result.cost_usd / max(self.expensive_task_usd, 1e-9))
        )
        scores.append(Score("cost", cost_score, f"${result.cost_usd:.4f}"))
        latency_score = max(0.0, 1.0 - result.latency_ms / max(self.slow_task_ms, 1.0))
        scores.append(Score("latency", latency_score, f"{result.latency_ms:.0f}ms"))

        total_weight = sum(WEIGHTS.get(s.criterion, 0.0) for s in scores)
        overall = (
            sum(s.value * WEIGHTS.get(s.criterion, 0.0) for s in scores) / total_weight
            if total_weight
            else 0.0
        )

        # A confident agent with a weak score should not be trusted; blend the two.
        confidence = (result.confidence + overall) / 2
        passed = (
            overall >= self.pass_threshold
            and confidence >= policy.min_confidence
            and security > 0.0
            and not missing_required_sources
        )
        if not passed and overall < self.pass_threshold:
            notes.append(f"score {overall:.2f} is below the {self.pass_threshold:.2f} bar")

        return Evaluation(
            scores=scores,
            overall=overall,
            passed=passed,
            confidence=confidence,
            notes=notes,
        )

    # -- the paid pass ----------------------------------------------------

    async def judge(
        self, result: AgentResult, spec: AgentSpec, task: Task, *, base: Evaluation | None = None
    ) -> Evaluation:
        """Ask a model to judge quality. Only worth it for high-value work."""
        evaluation = base or self.evaluate(result, spec, task)
        if self.router is None:
            return evaluation
        prompt = (
            "You are grading another agent's work. Reply with a single JSON object holding "
            '"correctness", "relevance", "completeness" and "business_value", each a number '
            'from 0 to 1, plus "note" (one short sentence).\n\n'
            f"## Goal\n{task.goal}\n\n## Agent output\n{result.text[:6000]}"
        )
        try:
            response = await self.router.complete(
                ModelRequest(
                    messages=[Message(role="user", content=prompt)],
                    max_tokens=1000,
                ),
                kind=TaskKind.REVIEW,
                agent="evaluation_engine",
                task_id=task.task_id,
            )
        except Exception as exc:
            log.warning("model judging failed; keeping the free score: %s", exc)
            evaluation.notes.append("model judging was unavailable")
            return evaluation

        from ..agents.base import extract_json  # local import avoids a cycle at module load

        parsed = extract_json(response.text) or {}
        judged: list[Score] = []
        for criterion in ("correctness", "relevance", "completeness"):
            raw = parsed.get(criterion)
            if raw is None:
                continue
            try:
                judged.append(Score(criterion, float(raw), "model judged"))
            except (TypeError, ValueError):
                continue
        if judged:
            # Average the free score and the judged score for each criterion.
            by_name = {s.criterion: s for s in evaluation.scores}
            for score in judged:
                existing = by_name.get(score.criterion)
                by_name[score.criterion] = Score(
                    score.criterion,
                    (existing.value + score.value) / 2 if existing else score.value,
                    "free check and model judge combined",
                )
            evaluation.scores = list(by_name.values())
            total_weight = sum(WEIGHTS.get(s.criterion, 0.0) for s in evaluation.scores)
            evaluation.overall = (
                sum(s.value * WEIGHTS.get(s.criterion, 0.0) for s in evaluation.scores)
                / total_weight
                if total_weight
                else evaluation.overall
            )
            evaluation.passed = evaluation.overall >= self.pass_threshold
        business_value = parsed.get("business_value")
        if business_value is not None:
            with suppress(TypeError, ValueError):
                evaluation.business_value = min(1.0, max(0.0, float(business_value)))
        if note := parsed.get("note"):
            evaluation.notes.append(str(note))
        return evaluation
