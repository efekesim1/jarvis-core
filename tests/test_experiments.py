"""The sandbox and the experiment lab.

Two questions. Does the sandbox actually hold, and does the lab tell the truth
about what it measured.
"""

from __future__ import annotations

import pytest
from fake_credentials import fake_anthropic_key

from jarvis_core.experiments import (
    Experiment,
    ExperimentLab,
    Sandbox,
    SandboxLimits,
    Verdict,
)


@pytest.fixture
def sandbox() -> Sandbox:
    return Sandbox(SandboxLimits(timeout_s=10.0, cpu_s=5, memory_mb=256))


@pytest.fixture
def lab(sandbox, audit) -> ExperimentLab:
    return ExperimentLab(sandbox=sandbox, audit=audit)


# -- the sandbox runs things ------------------------------------------------


def test_working_code_runs(sandbox):
    result = sandbox.run_code("print(6 * 7)")
    assert result.ok
    assert result.stdout.strip() == "42"
    assert result.duration_s > 0


def test_a_non_zero_exit_is_a_failure_not_a_crash(sandbox):
    result = sandbox.run_code("raise SystemExit(3)")
    assert not result.ok
    assert result.exit_code == 3
    assert "exited with code 3" in result.reason


def test_an_exception_is_reported_with_its_traceback(sandbox):
    result = sandbox.run_code("raise ValueError('deliberate')")
    assert not result.ok
    assert "ValueError" in result.stderr


def test_empty_code_is_refused_without_running_anything(sandbox):
    result = sandbox.run_code("   ")
    assert not result.ok
    assert "no code" in result.reason


def test_oversized_code_is_refused(sandbox):
    result = sandbox.run_code("x = 1\n" * 200_000)
    assert not result.ok
    assert "larger than" in result.reason


def test_extra_files_are_available_to_the_code(sandbox):
    result = sandbox.run_code(
        "from helper import greeting\nprint(greeting())",
        files={"helper.py": "def greeting():\n    return 'hello from a file'\n"},
    )
    assert result.ok
    assert "hello from a file" in result.stdout


# -- the sandbox holds ------------------------------------------------------


def test_no_credential_reaches_the_sandbox(sandbox, monkeypatch):
    """An allowlist, so a secret nobody predicted still does not get through."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_anthropic_key("SHOULDNOTBEVISIBLE"))
    monkeypatch.setenv("SOME_FUTURE_SECRET", "also-not-visible")
    result = sandbox.run_code("import os; print(sorted(os.environ))")
    assert result.ok
    assert "ANTHROPIC_API_KEY" not in result.stdout
    assert "SOME_FUTURE_SECRET" not in result.stdout


def test_the_sandbox_says_it_is_a_sandbox(sandbox):
    result = sandbox.run_code("import os; print(os.environ.get('JARVIS_SANDBOX'))")
    assert result.stdout.strip() == "1"


def test_code_that_never_finishes_is_stopped(sandbox):
    quick = Sandbox(SandboxLimits(timeout_s=0.5, cpu_s=5))
    result = quick.run_code("while True:\n    pass")
    assert not result.ok
    assert result.timed_out
    assert "did not finish" in result.reason


def test_code_that_eats_memory_is_stopped(sandbox):
    small = Sandbox(SandboxLimits(timeout_s=15.0, memory_mb=64, cpu_s=10))
    result = small.run_code("x = bytearray(500 * 1024 * 1024)\nprint('allocated')")
    assert not result.ok
    assert "allocated" not in result.stdout


def test_forking_is_refused_when_the_limit_can_bind(sandbox):
    """The process limit holds for an unprivileged user, and not for root.

    Root carries CAP_SYS_RESOURCE, so RLIMIT_NPROC does not bind it. The
    sandbox says so rather than reporting a limit it cannot enforce.
    """
    from jarvis_core.experiments.sandbox import privileged

    result = sandbox.run_code(
        "import os\n"
        "try:\n"
        "    os.fork()\n"
        "    print('forked')\n"
        "except OSError:\n"
        "    print('refused')\n"
    )
    if privileged():
        assert sandbox.describe()["privileged"] is True
        assert any("root" in line for line in sandbox.describe()["does_not_isolate"]), (
            "running as root must be declared, not hidden"
        )
    else:
        assert "forked" not in result.stdout


def test_a_process_spawned_before_a_timeout_does_not_survive_it(sandbox):
    """Killing the child alone would leave its children running."""
    import os
    import time as clock

    # The one place the process ceiling is lifted: this test must create a
    # descendant before it can prove the timeout kills the whole group.
    quick = Sandbox(SandboxLimits(timeout_s=0.6, cpu_s=10, max_processes=None))
    marker = "operator-sandbox-orphan-marker"
    result = quick.run_code(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(30)  # {marker}\"])\n"
        "print('spawned', flush=True)\n"
        "time.sleep(30)\n"
    )
    assert result.timed_out
    clock.sleep(0.4)
    # Nothing carrying the marker should still be running. The grep pipeline
    # itself carries the marker in its own command line, so exclude it.
    survivors = os.popen(f"ps -eo pid,args | grep '{marker}' | grep -v grep || true").read().strip()
    assert not survivors, f"a process outlived its sandbox: {survivors}"


def test_output_is_truncated_rather_than_filling_a_disk(sandbox):
    small = Sandbox(SandboxLimits(timeout_s=15.0, max_output_chars=500, cpu_s=10))
    result = small.run_code("print('x' * 100000)")
    assert result.truncated
    assert len(result.stdout) < 2000


def test_a_file_written_outside_the_sandbox_is_refused(sandbox):
    result = sandbox.run_code("print('never runs')", files={"../escape.py": "bad"})
    assert not result.ok
    assert "outside the sandbox" in result.reason


def test_the_sandbox_is_honest_about_what_it_does_not_isolate(sandbox):
    described = sandbox.describe()
    assert described["kind"] == "subprocess"
    assert any("network" in line for line in described["does_not_isolate"])
    assert any("security boundary" in line for line in described["does_not_isolate"])
    assert "container" in described["real_isolation"]


def test_each_run_gets_its_own_directory(sandbox):
    first = sandbox.run_code("open('marker.txt', 'w').write('here')\nprint('wrote')")
    assert first.ok
    second = sandbox.run_code("import os; print('marker.txt' in os.listdir('.'))")
    assert second.stdout.strip() == "False"


# -- the lab measures -------------------------------------------------------


# Timing tests use sleeps rather than CPU work. What is under test is the
# lab's judgement, not Python's arithmetic, and a sleep takes the same time on
# a loaded machine as on a quiet one — so these assert the decision logic
# without the whole suite's own load deciding the outcome.
SLOW = "import time; time.sleep(0.20); print('slow')"
FAST = "import time; time.sleep(0.05); print('fast')"


def test_a_faster_candidate_is_called_better(lab):
    report = lab.run(
        Experiment(name="four times faster", candidate_code=FAST, baseline_code=SLOW, runs=3)
    )
    assert report.verdict is Verdict.BETTER
    assert report.speedup > 1.15
    assert report.separated
    assert report.verdict.adoptable


def test_a_real_speedup_in_real_code_is_still_recognised(lab):
    """The sleeps prove the logic; this proves it fires on actual work.

    Allowed to come back EQUAL under heavy load, because that is the honest
    answer when the machine is noisier than the difference. What it must never
    do is call the slower side better.
    """
    report = lab.run(
        Experiment(
            name="builtin sum against a loop",
            candidate_code="print(sum(range(300000)))",
            baseline_code="t = 0\nfor i in range(300000):\n    t += i\nprint(t)",
            runs=3,
        )
    )
    assert report.verdict in (Verdict.BETTER, Verdict.EQUAL)
    assert report.speedup > 1.0


def test_a_slower_candidate_is_called_worse(lab):
    report = lab.run(
        Experiment(name="four times slower", candidate_code=SLOW, baseline_code=FAST, runs=3)
    )
    # Under load — the whole suite running in parallel, say — the measured
    # noise floor can grow past the real difference, and INCONCLUSIVE is then
    # the honest answer rather than a bug. What must never happen is the
    # slower candidate being called better, so that is asserted unconditionally
    # and the sharper claim only where the machine was quiet enough to make it.
    assert report.verdict is not Verdict.BETTER
    assert not report.verdict.adoptable
    if report.noise_ratio < 1.5:
        assert report.verdict is Verdict.WORSE, (
            f"a quiet machine ({report.noise_ratio:.2f}x noise) should have called "
            f"this worse: {report.reason}"
        )


def test_an_equal_candidate_is_not_called_better(lab):
    """A three percent difference is the machine, not the code."""
    same = "print(sum(range(200000)))"
    report = lab.run(
        Experiment(name="the same thing twice", candidate_code=same, baseline_code=same, runs=3)
    )
    assert report.verdict is Verdict.EQUAL
    assert not report.verdict.adoptable


def test_identical_code_is_never_called_better_however_noisy_the_machine(lab):
    """The bug this guards: a fixed threshold assumes a known noise floor.

    On a loaded machine two runs of the *same* code differed by 1.22x, which
    cleared the 1.15x bar and produced a verdict of BETTER — the lab
    recommending a change that was not a change. The bar is now the measured
    spread of the runs themselves, so noise cannot clear it by definition.

    Repeated because the failure was intermittent; once is not evidence.

    It came back twice more. First because the arms ran consecutively, so a
    machine that got busy partway through landed entirely on one side — the
    within-arm spread cannot catch that, since each arm stays internally
    consistent and it is the drift between them that lies. The runs are
    interleaved now.

    Then again, more rarely, because with three runs an arm can be internally
    tight and still land wholly above the other by luck. The fix for that is
    `separated`: overlapping ranges mean the difference is inside the noise
    whatever the medians say, and identical code cannot produce non-overlapping
    ranges.
    """
    same = "print(sum(range(150000)))"
    for attempt in range(8):
        report = lab.run(
            Experiment(name=f"identical {attempt}", candidate_code=same, baseline_code=same, runs=3)
        )
        assert report.verdict is not Verdict.BETTER, (
            f"identical code judged better on attempt {attempt}: {report.reason}"
        )


def test_the_noise_floor_is_measured_rather_than_assumed(lab):
    same = "print(sum(range(150000)))"
    report = lab.run(Experiment(name="noise", candidate_code=same, baseline_code=same, runs=3))
    assert report.noise_ratio >= 1.0
    assert report.as_dict()["noise_ratio"] == pytest.approx(report.noise_ratio, abs=0.001)


def test_a_candidate_that_does_not_run_is_not_timed(lab):
    report = lab.run(
        Experiment(
            name="broken",
            candidate_code="this is not python",
            baseline_code="print(1)",
            runs=3,
        )
    )
    assert report.verdict is Verdict.FAILED
    assert report.baseline == []  # nothing was measured against it


def test_a_candidate_whose_tests_fail_is_not_measured(lab):
    """A faster wrong answer is not an improvement."""
    report = lab.run(
        Experiment(
            name="fast and wrong",
            candidate_code="def total(n):\n    return 0\nprint(total(10))",
            test_code="from candidate import total\nassert total(10) == 45",
            baseline_code="print(sum(range(10)))",
            runs=2,
        )
    )
    assert report.verdict is Verdict.FAILED
    assert "tests failed" in report.reason
    assert report.baseline == []


def test_a_candidate_whose_tests_pass_is_measured(lab):
    report = lab.run(
        Experiment(
            name="fast and right",
            candidate_code="def total(n):\n    return sum(range(n))\nprint(total(100000))",
            test_code="from candidate import total\nassert total(10) == 45\nprint('ok')",
            baseline_code="print(sum(range(100000)))",
            runs=2,
        )
    )
    assert report.tests is not None and report.tests.ok
    assert report.verdict in (Verdict.BETTER, Verdict.EQUAL, Verdict.WORSE)


def test_a_candidate_with_no_baseline_is_inconclusive(lab):
    report = lab.run(Experiment(name="alone", candidate_code="print('fine')", runs=2))
    assert report.verdict is Verdict.INCONCLUSIVE
    assert not report.verdict.adoptable
    assert any("baseline" in note for note in report.notes)


def test_a_broken_baseline_does_not_make_a_candidate_look_good(lab):
    report = lab.run(
        Experiment(
            name="unfair",
            candidate_code="print('fine')",
            baseline_code="raise SystemExit(1)",
            runs=2,
        )
    )
    assert report.verdict is Verdict.INCONCLUSIVE


def test_the_run_count_is_bounded(lab):
    assert Experiment(name="x", candidate_code="pass", runs=500).runs <= 10
    assert Experiment(name="x", candidate_code="pass", runs=0).runs >= 1


# -- nothing is adopted -----------------------------------------------------


def test_a_better_verdict_is_a_proposal_not_a_decision(lab):
    report = lab.run(
        Experiment(
            name="faster",
            candidate_code="print(sum(range(300000)))",
            baseline_code="t = 0\nfor i in range(300000):\n    t += i\nprint(t)",
            runs=2,
        )
    )
    assert "Nothing has been adopted" in report.render()


def test_every_experiment_is_audited_as_not_adopted(lab, audit):
    lab.run(Experiment(name="x", candidate_code="print(1)", runs=1))
    finished = [e for e in audit.events if e.action == "experiment.finished"]
    assert finished
    assert finished[0].metadata["adopted"] is False


def test_the_summary_reports_what_was_found(lab):
    lab.run(Experiment(name="a", candidate_code="print(1)", runs=1))
    lab.run(Experiment(name="b", candidate_code="raise SystemExit(1)", runs=1))
    summary = lab.summary()
    assert summary["experiments"] == 2
    assert summary["by_verdict"]["failed"] == 1
    assert "does_not_isolate" in summary["sandbox"]


def test_the_arms_are_interleaved_so_a_slow_patch_is_shared(lab):
    """Consecutive arms let a busy moment land entirely on one side.

    Asserted structurally rather than by timing, because reproducing the drift
    reliably would mean loading the machine from inside the test suite.
    """
    import inspect

    from jarvis_core.experiments import lab as module

    source = inspect.getsource(module.ExperimentLab.run)
    candidate_first = source.index('self._measure("candidate"')
    baseline_first = source.index('self._measure("baseline"')
    last_candidate = source.rindex('self._measure("candidate"')
    # A candidate measurement must appear after the first baseline one: that is
    # only true if the loop alternates rather than running one arm then the other.
    assert last_candidate > baseline_first > candidate_first


def test_overlapping_runs_mean_the_difference_is_inside_the_noise(lab):
    """Medians can differ while every run of one side overlaps the other.

    With three runs per arm that happens by luck often enough to matter, and it
    is exactly the case a threshold on the medians cannot see.
    """
    same = "print(sum(range(120000)))"
    report = lab.run(Experiment(name="overlap", candidate_code=same, baseline_code=same, runs=3))
    assert not report.separated or report.verdict is not Verdict.BETTER


def test_a_genuinely_faster_candidate_still_separates(lab):
    report = lab.run(
        Experiment(name="clearly faster", candidate_code=FAST, baseline_code=SLOW, runs=3)
    )
    assert report.separated
    assert report.verdict is Verdict.BETTER


# -- the process ceiling says whether it is on ------------------------------
#
# A CI fix made `max_processes=0` mean "skip the limit", which inverts what zero
# means for a ceiling: before, `setrlimit(NPROC, (0, 0))` denied every fork.
# Anyone writing `max_processes=0` to lock a sandbox down would have got the
# opposite, and the only thing saying otherwise was a comment.


def test_zero_processes_still_means_zero_not_unlimited():
    """The semantic that was inverted. `None` is how you turn a ceiling off."""
    assert SandboxLimits(max_processes=0).max_processes == 0
    assert SandboxLimits(max_processes=None).max_processes is None
    assert SandboxLimits().max_processes == 1


def test_a_sandbox_says_when_its_process_ceiling_is_off():
    """Surfaced like `privileged`, rather than left to be read off a call site."""
    guarded = Sandbox(SandboxLimits()).describe()
    assert guarded["process_ceiling_enforced"] is True
    assert "process count" in guarded["isolates"]

    lifted = Sandbox(SandboxLimits(max_processes=None)).describe()
    assert lifted["process_ceiling_enforced"] is False
    assert "process count" not in lifted["isolates"]


def test_the_default_sandbox_enforces_a_process_ceiling():
    """The failure mode this limit exists for is a fork bomb."""
    assert Sandbox().limits.max_processes == 1
