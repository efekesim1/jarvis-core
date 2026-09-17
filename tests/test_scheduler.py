"""The night shift: it must work unattended, and it must stop."""

from __future__ import annotations

import asyncio

import pytest

from jarvis_core.core.scheduler import (
    JobStatus,
    ScheduledJob,
    Scheduler,
    StopReason,
)
from jarvis_core.security.approval import DenyAllGate


@pytest.fixture
def scheduler(jarvis, audit):
    return Scheduler(jarvis, audit=audit)


# -- job limits -------------------------------------------------------------


def test_a_job_cannot_be_created_without_bounds():
    job = ScheduledJob(request="x", budget_usd=0.0, timeout_s=0.0, max_iterations=0)
    assert job.budget_usd >= 0.01
    assert job.timeout_s >= 10.0
    assert job.max_iterations >= 1


def test_a_job_cannot_ask_for_an_unlimited_budget():
    job = ScheduledJob(request="x", budget_usd=10_000.0, max_iterations=999)
    assert job.budget_usd <= 100.0
    assert job.max_iterations <= 5


def test_a_job_carries_everything_needed_to_audit_it():
    job = ScheduledJob(request="research something", owner="operator")
    payload = job.as_dict()
    for field in ("job_id", "owner", "priority", "budget_usd", "timeout_s", "risk_level", "status"):
        assert field in payload


# -- running a shift --------------------------------------------------------


async def test_a_shift_works_the_queue_and_reports(scheduler):
    scheduler.submit_request("what is MCP?")
    scheduler.submit_request("what is pgvector?")
    report = await scheduler.run_shift(budget_usd=5.0)

    assert report.stop_reason is StopReason.FINISHED
    assert len(report.completed) == 2
    assert report.cost_usd > 0
    assert report.finished_at is not None


async def test_higher_priority_jobs_run_first(scheduler):
    scheduler.submit_request("low priority question about MCP", priority=9)
    scheduler.submit_request("urgent question about pgvector", priority=1)
    report = await scheduler.run_shift(budget_usd=5.0)
    assert report.jobs[0].request.startswith("urgent")


async def test_a_shift_stops_when_the_budget_is_gone(scheduler):
    for index in range(20):
        scheduler.submit_request(f"distinct question number {index} about markets")
    report = await scheduler.run_shift(budget_usd=0.01)

    assert report.stop_reason is StopReason.BUDGET
    assert report.skipped
    # It stopped rather than working the whole queue.
    assert len(report.completed) < 20


async def test_a_job_too_expensive_for_the_remaining_budget_is_skipped_not_cut_off(
    scheduler,
):
    scheduler.submit_request("a cheap question about MCP", budget_usd=0.01)
    scheduler.submit_request("an expensive one about markets", budget_usd=50.0)
    report = await scheduler.run_shift(budget_usd=0.5)

    expensive = next(j for j in report.jobs if "expensive" in j.request)
    assert expensive.status is JobStatus.SKIPPED
    assert "budget" in (expensive.error or "")


async def test_a_shift_stops_at_its_deadline(scheduler):
    for index in range(10):
        scheduler.submit_request(f"question {index} about markets")
    report = await scheduler.run_shift(budget_usd=100.0, max_duration_s=0.0)
    assert report.stop_reason is StopReason.DEADLINE


async def test_a_shift_stops_at_its_job_limit(scheduler):
    for index in range(10):
        scheduler.submit_request(f"question {index} about markets")
    report = await scheduler.run_shift(budget_usd=100.0, max_jobs=2)
    assert report.stop_reason is StopReason.ITERATIONS
    assert len(report.completed) == 2


async def test_a_shift_can_be_cancelled(scheduler):
    scheduler.submit_request("question about MCP")
    scheduler.submit_request("another question about markets")
    scheduler.cancel()
    report = await scheduler.run_shift(budget_usd=5.0)
    assert report.stop_reason is StopReason.CANCELLED
    assert not report.completed


async def test_an_empty_queue_finishes_immediately(scheduler):
    report = await scheduler.run_shift(budget_usd=5.0)
    assert report.stop_reason is StopReason.FINISHED
    assert report.jobs == []


# -- failure handling -------------------------------------------------------


async def test_one_failing_job_does_not_end_the_shift(scheduler, jarvis, monkeypatch):
    calls = {"n": 0}
    original = jarvis.handle

    async def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the provider fell over")
        return await original(request)

    monkeypatch.setattr(jarvis, "handle", flaky)
    scheduler.submit_request("first question about MCP")
    scheduler.submit_request("second question about markets")
    report = await scheduler.run_shift(budget_usd=5.0)

    assert len(report.failed) == 1
    assert len(report.completed) == 1
    assert report.stop_reason is StopReason.FINISHED


async def test_a_job_that_runs_too_long_is_cut_off(scheduler, jarvis, monkeypatch):
    async def slow(request):
        await asyncio.sleep(5)

    monkeypatch.setattr(jarvis, "handle", slow)
    scheduler.submit(ScheduledJob(request="a slow one", timeout_s=10.0))
    # The floor is 10s, so shorten it directly to keep the test quick.
    scheduler.pending[0].timeout_s = 0.05
    report = await scheduler.run_shift(budget_usd=5.0)

    assert len(report.failed) == 1
    assert "timeout" in report.failed[0].error


# -- safety -----------------------------------------------------------------


async def test_an_unattended_shift_notes_a_gate_that_approves_everything(scheduler):
    scheduler.submit_request("question about MCP")
    report = await scheduler.run_shift(budget_usd=5.0)
    # The test fixture uses an auto-approving gate, which is wrong overnight.
    assert any("deny" in note for note in report.notes)


async def test_a_denying_gate_draws_no_complaint(scheduler, jarvis):
    jarvis.set_approval_gate(DenyAllGate())
    scheduler.submit_request("question about MCP")
    report = await scheduler.run_shift(budget_usd=5.0)
    assert not any("deny" in note for note in report.notes)


async def test_the_whole_shift_is_audited(scheduler, audit):
    scheduler.submit_request("question about MCP")
    await scheduler.run_shift(budget_usd=5.0)
    actions = [e.action for e in audit.events]
    assert "scheduler.submitted" in actions
    assert "scheduler.shift_started" in actions
    assert "scheduler.job_finished" in actions
    assert "scheduler.shift_finished" in actions


# -- the morning report -----------------------------------------------------


async def test_the_morning_report_reads_as_a_briefing(scheduler):
    scheduler.submit_request("what is MCP?")
    report = await scheduler.run_shift(budget_usd=5.0)
    text = report.render()

    assert "GOOD MORNING OPERATOR" in text
    assert "jobs completed" in text
    assert "$" in text


async def test_the_report_says_when_the_shift_stopped_early(scheduler):
    for index in range(10):
        scheduler.submit_request(f"question {index} about markets")
    report = await scheduler.run_shift(budget_usd=0.01)
    assert "Stopped early" in report.render()


async def test_the_report_lists_failures(scheduler, jarvis, monkeypatch):
    async def always_fails(request):
        raise RuntimeError("nothing works today")

    monkeypatch.setattr(jarvis, "handle", always_fails)
    scheduler.submit_request("a doomed question")
    report = await scheduler.run_shift(budget_usd=5.0)
    assert "What failed" in report.render()


async def test_the_report_serializes_for_storage(scheduler):
    scheduler.submit_request("what is MCP?")
    report = await scheduler.run_shift(budget_usd=5.0)
    payload = report.as_dict()
    assert payload["stop_reason"] == "finished"
    assert payload["jobs"]


# -- running more than one job at a time ------------------------------------


async def test_a_wave_has_to_fit_the_budget_before_it_starts(scheduler):
    """Checking the budget between concurrent jobs is checking after the spend.

    Three jobs in flight can each be affordable alone and together cost more
    than remains, and the overspend is discovered when it is already paid for.
    So the wave is assembled against the budget and only then started.
    """
    for index in range(2):
        scheduler.submit(ScheduledJob(request=f"question {index} about markets", budget_usd=0.6))
    report = await scheduler.run_shift(budget_usd=1.0)

    # One job at 0.6 fits in 1.0. Two do not, so the second is skipped before
    # anything starts rather than started and cut off halfway.
    assert len(report.completed) == 1
    assert len(report.skipped) == 1
    assert "not enough shift budget" in (report.skipped[0].error or "")


async def test_max_jobs_counts_jobs_not_waves(scheduler):
    """A limit that quietly meant twice as much work once concurrency arrived
    would be a limit that changed meaning without anyone saying so."""
    for index in range(8):
        scheduler.submit_request(f"question {index} about markets")
    report = await scheduler.run_shift(budget_usd=50.0, max_jobs=3)

    assert len([j for j in report.jobs if j.status is not JobStatus.SKIPPED]) == 3
    assert report.stop_reason is StopReason.ITERATIONS


async def test_a_job_is_claimed_when_it_is_picked(scheduler):
    """The selector reads status, so without claiming, the second call in a
    wave returns the same job again — a two-job queue produced four
    completions."""
    scheduler.submit_request("first question about MCP")
    scheduler.submit_request("second question about markets")
    report = await scheduler.run_shift(budget_usd=5.0)

    assert len(report.completed) == 2
    assert len({job.job_id for job in report.jobs}) == len(report.jobs)


async def test_one_job_crashing_does_not_cancel_the_others(scheduler, jarvis, monkeypatch):
    """Losing two good jobs because a third raised is the shift punishing itself."""
    original = jarvis.handle
    seen: list[str] = []

    async def flaky(request, **kwargs):
        seen.append(request)
        if "poison" in request:
            raise RuntimeError("the provider fell over")
        return await original(request, **kwargs)

    monkeypatch.setattr(jarvis, "handle", flaky)
    scheduler.submit(ScheduledJob(request="a poison question about MCP", priority=1))
    scheduler.submit(ScheduledJob(request="a fine question about markets", priority=2))
    report = await scheduler.run_shift(budget_usd=5.0)

    assert len(report.failed) == 1
    assert len(report.completed) == 1


async def test_the_default_parallelism_is_conservative(scheduler):
    """Each job runs its own waves of agents, so three concurrent jobs is
    closer to eighteen concurrent model calls."""
    assert scheduler.max_parallel_jobs <= 2
