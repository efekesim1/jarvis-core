#!/usr/bin/env python3
"""The sandbox and the experiment lab: running code nobody has read.

Shows the sandbox holding under a timeout, a memory limit and a credential
scrub, then the lab measuring a real candidate against what it would replace
and refusing a faster answer that is wrong.

Nothing is adopted. That is the point: a run that passes is evidence, not
consent, and there is no path from this file to anything in production.

    python examples/demo_sandbox.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JARVIS_ENVIRONMENT", "sandbox")
os.environ.setdefault("JARVIS_LOG_LEVEL", "ERROR")

from jarvis_core.experiments import (  # noqa: E402
    Experiment,
    ExperimentLab,
    Sandbox,
    SandboxLimits,
    Verdict,
)
from jarvis_core.logging import AuditLog, configure_logging  # noqa: E402

# A real optimisation with a real difference: a membership test against a list
# is O(n) and against a set is O(1), so the pair below differ by a factor of
# hundreds rather than of two.
#
# Two was what this used to be — `sum(range(n))` against a loop — and it was
# not enough. Once the lab started requiring the runs to actually pull apart
# (rather than trusting a threshold on the medians), a 2x difference stopped
# separating reliably on a machine running the rest of the test suite, and this
# demo failed intermittently in CI. The lab was right and the example was thin.
FAST = (
    "def dupes(n):\n"
    "    xs = set(range(n))\n"
    "    return sum(1 for x in range(n) if x in xs)\n\n\n"
    "print(dupes(4000))\n"
)
SLOW = (
    "def dupes(n):\n"
    "    xs = list(range(n))\n"
    "    return sum(1 for x in range(n) if x in xs)\n\n\n"
    "print(dupes(4000))\n"
)
TESTS = "from candidate import dupes\nassert dupes(10) == 10\nprint('ok')\n"


def line(label: str, detail: str) -> None:
    print(f"  {label:<34} {detail}")


def main() -> int:
    configure_logging()
    audit = AuditLog()
    sandbox = Sandbox(SandboxLimits(timeout_s=5.0, cpu_s=5, memory_mb=128))
    lab = ExperimentLab(sandbox=sandbox, audit=audit)

    print("\nWhat the sandbox stops\n")
    line("working code", sandbox.run_code("print(6 * 7)").stdout.strip())
    line(
        "a credential lookup",
        sandbox.run_code("import os; print('ANTHROPIC_API_KEY' in os.environ)").stdout.strip(),
    )
    quick = Sandbox(SandboxLimits(timeout_s=0.5, cpu_s=5))
    line("an endless loop", quick.run_code("while True:\n    pass").reason)
    small = Sandbox(SandboxLimits(timeout_s=10.0, memory_mb=64, cpu_s=5))
    line(
        "a memory grab",
        small.run_code("x = bytearray(400*1024*1024)\nprint('got it')").reason or "stopped",
    )
    line("a write outside it", sandbox.run_code("pass", files={"../out.py": "x"}).reason)

    described = sandbox.describe()
    print(f"\n  This sandbox is a {described['kind']}, not a security boundary.")
    for limitation in described["does_not_isolate"]:
        print(f"    - it does not isolate {limitation}")

    print("\n\nWhat the lab measures\n")
    faster = lab.run(
        Experiment(
            name="builtin sum against a hand-written loop",
            candidate_code=FAST,
            baseline_code=SLOW,
            test_code=TESTS,
            runs=3,
        )
    )
    print(faster.render())

    wrong = lab.run(
        Experiment(
            name="a faster wrong answer",
            candidate_code="def dupes(n):\n    return 0\n\n\nprint(dupes(10))\n",
            baseline_code=SLOW,
            test_code=TESTS,
            runs=2,
        )
    )
    print()
    print(wrong.render())

    problems: list[str] = []
    if faster.verdict is not Verdict.BETTER:
        problems.append("the faster candidate was not judged better")
    if wrong.verdict is not Verdict.FAILED:
        problems.append("a wrong answer was not caught by its own tests")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: the sandbox held, the lab measured, and nothing was adopted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
