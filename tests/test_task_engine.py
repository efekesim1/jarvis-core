"""The task lifecycle."""

from __future__ import annotations

import pytest

from jarvis_core.core.task import InvalidTransition, Task, TaskEngine, TaskStatus
from jarvis_core.security.permissions import RiskLevel


def test_a_new_task_starts_queued():
    task = Task(goal="find something")
    assert task.status is TaskStatus.QUEUED
    assert task.task_id.startswith("tsk_")
    assert not task.is_terminal


def test_a_legal_transition_records_timestamps():
    task = Task(goal="x")
    task.transition(TaskStatus.RUNNING)
    assert task.started_at is not None
    task.transition(TaskStatus.COMPLETED)
    assert task.completed_at is not None
    assert task.is_terminal
    assert task.duration_s is not None


def test_an_illegal_transition_is_refused():
    task = Task(goal="x")
    task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.COMPLETED)
    with pytest.raises(InvalidTransition):
        task.transition(TaskStatus.RUNNING)


def test_transitioning_to_the_same_status_is_a_no_op():
    task = Task(goal="x")
    task.transition(TaskStatus.QUEUED)
    assert task.status is TaskStatus.QUEUED


def test_the_engine_tracks_parents_and_children():
    engine = TaskEngine()
    parent = engine.create("research a market", priority=2)
    child = engine.create("search the web", parent=parent, agent="web_researcher")
    assert child.parent_task_id == parent.task_id
    assert [t.task_id for t in engine.children(parent)] == [child.task_id]


def test_dependencies_are_only_met_once_they_complete():
    engine = TaskEngine()
    first = engine.create("gather")
    second = engine.create("analyze", dependencies=[first.task_id])
    assert not engine.dependencies_met(second)
    engine.transition(first, TaskStatus.RUNNING)
    engine.complete(first, {"result": "done"})
    assert engine.dependencies_met(second)


def test_a_missing_dependency_never_counts_as_met():
    engine = TaskEngine()
    task = engine.create("analyze", dependencies=["tsk_does_not_exist"])
    assert not engine.dependencies_met(task)


def test_failure_records_the_error():
    engine = TaskEngine()
    task = engine.create("x")
    engine.transition(task, TaskStatus.RUNNING)
    engine.fail(task, "the provider timed out")
    assert task.status is TaskStatus.FAILED
    assert task.error == "the provider timed out"


def test_the_snapshot_counts_by_status_and_sums_cost():
    engine = TaskEngine()
    done = engine.create("a")
    engine.transition(done, TaskStatus.RUNNING)
    engine.add_cost(done, 0.25)
    engine.complete(done, {})
    engine.create("b")
    snapshot = engine.snapshot()
    assert snapshot["total"] == 2
    assert snapshot["by_status"]["completed"] == 1
    assert snapshot["by_status"]["queued"] == 1
    assert snapshot["total_cost_usd"] == pytest.approx(0.25)


def test_a_task_serializes_for_storage():
    task = Task(goal="x", risk=RiskLevel.HIGH)
    payload = task.as_dict()
    assert payload["risk"] == "high"
    assert payload["status"] == "queued"
