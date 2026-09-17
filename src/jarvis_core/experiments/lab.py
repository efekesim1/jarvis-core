"""The experiment lab.

An experiment answers one question: is the candidate better than what we have?
Not "does it run" and not "is it interesting". Better, measured, against the
thing it would replace.

The lab never adopts anything. It produces a verdict and evidence, and a person
decides. That boundary is the same one the knowledge pipeline draws, for the
same reason: the expensive mistake is not running bad code, it is keeping it.
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..logging import AuditLog, get_logger
from .sandbox import Sandbox, SandboxLimits, SandboxResult

log = get_logger("experiments.lab")


class Verdict(StrEnum):
    """What an experiment concluded."""

    BETTER = "better"
    EQUAL = "equal"
    WORSE = "worse"
    UNSAFE = "unsafe"  # it did something it should not have
    FAILED = "failed"  # it did not run, or its tests did not pass
    INCONCLUSIVE = "inconclusive"  # nothing to compare against

    @property
    def adoptable(self) -> bool:
        """Whether this verdict could lead to adoption, if a human agrees."""
        return self is Verdict.BETTER


@dataclass(slots=True)
class Measurement:
    """One timed run of one thing."""

    label: str
    ok: bool
    duration_s: float
    output: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "duration_s": round(self.duration_s, 4),
            "reason": self.reason,
        }


@dataclass(slots=True)
class Experiment:
    """A candidate, what it would replace, and how to tell them apart."""

    name: str
    candidate_code: str
    baseline_code: str | None = None
    test_code: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    runs: int = 3
    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:10]}")

    def __post_init__(self) -> None:
        # One run measures noise. Twenty measures patience.
        self.runs = max(1, min(self.runs, 10))


@dataclass(slots=True)
class ExperimentReport:
    """What the lab found, and what it refuses to conclude."""

    experiment_id: str
    name: str
    verdict: Verdict
    reason: str = ""
    candidate: list[Measurement] = field(default_factory=list)
    baseline: list[Measurement] = field(default_factory=list)
    tests: SandboxResult | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _median(measurements: list[Measurement]) -> float | None:
        good = [m.duration_s for m in measurements if m.ok]
        return statistics.median(good) if good else None

    @property
    def candidate_median_s(self) -> float | None:
        return self._median(self.candidate)

    @property
    def baseline_median_s(self) -> float | None:
        return self._median(self.baseline)

    @property
    def speedup(self) -> float | None:
        """How many times faster the candidate is. None when uncomparable."""
        candidate, baseline = self.candidate_median_s, self.baseline_median_s
        if not candidate or not baseline:
            return None
        return baseline / candidate

    @property
    def separated(self) -> bool:
        """Whether the two sets of timings actually pull apart.

        The within-arm spread underestimates how much a *median* can move: with
        three runs an arm can be internally tight and still land entirely above
        or below the other by luck. That is what kept the identical-code test
        failing intermittently even after the noise floor was measured and the
        arms interleaved.

        Overlapping ranges mean the difference is inside the noise, whatever the
        medians say. Non-overlapping ranges mean every candidate run beat every
        baseline run, which three coin flips do not produce. It is a cruder test
        than a proper statistic and it has the property that matters: identical
        code cannot pass it.
        """
        candidate = [m.duration_s for m in self.candidate if m.ok]
        baseline = [m.duration_s for m in self.baseline if m.ok]
        if len(candidate) < 2 or len(baseline) < 2:
            # Not enough to say. Fall back to the threshold alone rather than
            # claiming separation from a single measurement each.
            return True

        if max(candidate) < min(baseline):
            gap = min(baseline) - max(candidate)
            fastest = max(candidate)
        elif max(baseline) < min(candidate):
            gap = min(candidate) - max(baseline)
            fastest = max(baseline)
        else:
            return False

        # Non-overlapping is not enough on its own: with three runs each, two
        # identical arms fall entirely apart by luck about one time in twenty,
        # and one time in twenty is often enough to keep a test flaking. So the
        # gap between them also has to be larger than the wobble inside them.
        spread = max((max(arm) - min(arm)) for arm in (candidate, baseline) if len(arm) >= 2)
        return gap > spread and gap / fastest > 0.05

    @property
    def noise_ratio(self) -> float:
        """How much the same code varied against itself, run to run.

        This is the machine's own contribution, measured rather than assumed.
        A build agent, a busy CI runner and a laptop on battery all have very
        different noise floors, and a fixed threshold is right on none of them:
        the lab once called two runs of *identical* code a 1.22x improvement.

        Reported as the widest spread seen on either side, so a candidate has
        to beat the noise that was actually present, not the noise someone
        expected when the constant was written.
        """
        spreads = [1.0]
        for series in (self.baseline, self.candidate):
            timings = [m.duration_s for m in series if m.ok and m.duration_s > 0]
            if len(timings) >= 2:
                spreads.append(max(timings) / min(timings))
        return max(spreads)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "verdict": self.verdict.value,
            "adoptable": self.verdict.adoptable,
            "reason": self.reason,
            "candidate_median_s": self.candidate_median_s,
            "baseline_median_s": self.baseline_median_s,
            "speedup": round(self.speedup, 3) if self.speedup else None,
            "noise_ratio": round(self.noise_ratio, 3),
            "separated": self.separated,
            "tests": self.tests.as_dict() if self.tests else None,
            "runs": {
                "candidate": [m.as_dict() for m in self.candidate],
                "baseline": [m.as_dict() for m in self.baseline],
            },
            "notes": list(self.notes),
            "started_at": self.started_at.isoformat(),
        }

    def render(self) -> str:
        """The summary a person reads before deciding."""
        lines = [
            f"EXPERIMENT  {self.name}",
            f"Verdict:    {self.verdict.value.upper()}",
            f"Why:        {self.reason}",
        ]
        if self.speedup:
            lines.append(
                f"Speed:      {self.speedup:.2f}x "
                f"({self.baseline_median_s:.4f}s -> {self.candidate_median_s:.4f}s, median)"
            )
        if self.tests is not None:
            lines.append(f"Tests:      {'passed' if self.tests.ok else 'failed'}")
        for note in self.notes:
            lines.append(f"Note:       {note}")
        lines.append("")
        lines.append("Nothing has been adopted. A better verdict is a proposal, not a decision.")
        return "\n".join(lines)


class ExperimentLab:
    """Runs candidates against baselines in the sandbox, and reports."""

    # The floor: below this a difference is not worth acting on even on a
    # perfectly quiet machine. The real bar is this *or* the noise measured
    # during the run, whichever is higher — see ``ExperimentReport.noise_ratio``.
    MEANINGFUL_SPEEDUP = 1.15

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        audit: AuditLog | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.sandbox = sandbox or Sandbox(limits)
        self.audit = audit or AuditLog()
        self.reports: list[ExperimentReport] = []

    # -- one run ----------------------------------------------------------

    def _measure(self, label: str, code: str, files: dict[str, str]) -> Measurement:
        started = time.perf_counter()
        result = self.sandbox.run_code(code, files=files)
        return Measurement(
            label=label,
            ok=result.ok,
            # The sandbox's own timing, which excludes process startup where
            # it can. Falls back to wall clock.
            duration_s=result.duration_s or (time.perf_counter() - started),
            output=result.stdout,
            reason=result.reason,
        )

    # -- the experiment ---------------------------------------------------

    def run(self, experiment: Experiment) -> ExperimentReport:
        """Run one experiment. Never raises; a failure is a verdict."""
        report = ExperimentReport(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            verdict=Verdict.INCONCLUSIVE,
        )
        self.audit.emit(
            "experiment.started",
            subject=experiment.name[:200],
            metadata={"experiment_id": experiment.experiment_id, "runs": experiment.runs},
        )

        # 1. Does it run at all? A candidate that cannot start is not slow, it
        #    is broken, and timing it would be dishonest.
        first = self._measure("candidate", experiment.candidate_code, experiment.files)
        report.candidate.append(first)
        if not first.ok:
            report.verdict = Verdict.FAILED
            report.reason = f"the candidate did not run: {first.reason}"
            return self._finish(report)

        # 2. Do its own tests pass? A faster wrong answer is not an improvement.
        if experiment.test_code:
            files = {**experiment.files, "candidate.py": experiment.candidate_code}
            report.tests = self.sandbox.run_code(experiment.test_code, files=files)
            if not report.tests.ok:
                report.verdict = Verdict.FAILED
                report.reason = f"the candidate's tests failed: {report.tests.reason}"
                return self._finish(report)

        # 3. Measure both sides, interleaved.
        #
        # The first version ran every candidate and then every baseline, and
        # that hid a whole class of error: a machine that gets busy partway
        # through lands entirely on the second arm and reads as a real
        # difference. The within-arm spread does not catch it, because each arm
        # is internally consistent — it is the *drift between* them that lies.
        #
        # Interleaving means a slow patch is shared. It is the same fix the
        # skill A/B needed, and it should have been here first.
        remaining = max(0, experiment.runs - 1)

        if experiment.baseline_code is None:
            report.verdict = Verdict.INCONCLUSIVE
            report.reason = "there is nothing to compare this against"
            report.notes.append(
                "a candidate with no baseline can be judged on correctness, not on being better"
            )
            return self._finish(report)

        report.baseline.append(
            self._measure("baseline", experiment.baseline_code, experiment.files)
        )
        for _ in range(remaining):
            report.candidate.append(
                self._measure("candidate", experiment.candidate_code, experiment.files)
            )
            report.baseline.append(
                self._measure("baseline", experiment.baseline_code, experiment.files)
            )

        if report.baseline_median_s is None:
            report.verdict = Verdict.INCONCLUSIVE
            report.reason = "the baseline itself did not run, so there is no comparison"
            return self._finish(report)

        # 4. Compare against the larger of the fixed floor and the noise this
        #    run actually saw. Being stricter than necessary costs a proposal
        #    nobody makes; being looser costs a wrong one that the operator acts on.
        speedup = report.speedup or 1.0
        threshold = max(self.MEANINGFUL_SPEEDUP, report.noise_ratio)
        noisy = report.noise_ratio > self.MEANINGFUL_SPEEDUP
        measured = f" (the machine itself varied by {report.noise_ratio:.2f}x)" if noisy else ""

        separated = report.separated
        if speedup >= threshold and separated:
            report.verdict = Verdict.BETTER
            report.reason = f"{speedup:.2f}x faster than what we have, on median{measured}"
        elif speedup <= 1 / threshold and separated:
            report.verdict = Verdict.WORSE
            report.reason = f"{1 / speedup:.2f}x slower than what we have, on median{measured}"
        elif not separated and (speedup >= threshold or speedup <= 1 / threshold):
            report.verdict = Verdict.EQUAL
            report.reason = (
                f"the medians differ by {speedup:.2f}x but the runs overlap, so the "
                "difference is inside the noise. Slower runs of the faster side were "
                "slower than faster runs of the other"
            )
        else:
            report.verdict = Verdict.EQUAL
            report.reason = (
                f"within noise of what we have ({speedup:.2f}x against a "
                f"{threshold:.2f}x bar{measured}); no reason to change on speed alone"
            )
        return self._finish(report)

    def _finish(self, report: ExperimentReport) -> ExperimentReport:
        self.reports.append(report)
        self.audit.emit(
            "experiment.finished",
            subject=report.name[:200],
            result=report.verdict.value,
            reason=report.reason,
            metadata={
                "experiment_id": report.experiment_id,
                "speedup": round(report.speedup, 3) if report.speedup else None,
                # Stated explicitly: a verdict is not an adoption.
                "adopted": False,
            },
        )
        log.info("%s: %s (%s)", report.name, report.verdict.value, report.reason)
        return report

    def summary(self) -> dict[str, Any]:
        by_verdict: dict[str, int] = {}
        for report in self.reports:
            by_verdict[report.verdict.value] = by_verdict.get(report.verdict.value, 0) + 1
        return {
            "experiments": len(self.reports),
            "by_verdict": dict(sorted(by_verdict.items())),
            "adoptable": sum(1 for r in self.reports if r.verdict.adoptable),
            "sandbox": self.sandbox.describe(),
        }
