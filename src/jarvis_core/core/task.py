"""The Task Engine.

Every request from the operator becomes a Task. A task carries its own goal, state,
dependencies, cost and risk, so the system can pause, retry, escalate or
report on any piece of work without special cases.

Storage is behind an interface. Today it is in memory; PostgreSQL slots in
later without the engine's callers changing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from ..logging import get_logger
from ..security.permissions import RiskLevel

log = get_logger("core.task")


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    REVIEW = "review"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

# Which transitions are legal. Anything else is a bug and is rejected.
_ALLOWED: dict[TaskStatus, frozenset[TaskStatus]] = {
    # A queued task can be rejected before it ever starts: an unknown agent, a
    # dead dependency, or a budget stop.
    TaskStatus.QUEUED: frozenset(
        {
            TaskStatus.PLANNING,
            TaskStatus.RUNNING,
            TaskStatus.BLOCKED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.WAITING,
            TaskStatus.REVIEW,
            TaskStatus.APPROVAL_REQUIRED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.WAITING: frozenset(
        {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.BLOCKED: frozenset(
        {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.REVIEW: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.APPROVAL_REQUIRED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset({TaskStatus.QUEUED, TaskStatus.RUNNING}),  # a retry
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    """A status change that the lifecycle does not allow."""


@dataclass(slots=True)
class Task:
    """One unit of work."""

    goal: str
    task_id: str = field(default_factory=lambda: f"tsk_{uuid.uuid4().hex[:12]}")
    parent_task_id: str | None = None
    priority: int = 5  # 1 is most urgent, 9 least
    status: TaskStatus = TaskStatus.QUEUED
    assigned_agent: str | None = None
    dependencies: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    cost_usd: float = 0.0
    risk: RiskLevel = RiskLevel.LOW
    error: str | None = None
    trace_id: str | None = None
    # Which Goal this work advances, when it advances one. Not to be confused
    # with `goal` above, which is the sentence describing the work itself —
    # the names collide because both were right in their own module and
    # renaming the older one would touch every caller.
    goal_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    def transition(self, status: TaskStatus) -> None:
        """Move to a new status, refusing an illegal move."""
        if status == self.status:
            return
        if status not in _ALLOWED[self.status]:
            raise InvalidTransition(
                f"task {self.task_id} cannot move from {self.status.value} to {status.value}"
            )
        self.status = status
        now = datetime.now(UTC)
        if status is TaskStatus.RUNNING and self.started_at is None:
            self.started_at = now
        if status in TERMINAL_STATUSES:
            self.completed_at = now

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "goal": self.goal,
            "priority": self.priority,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "dependencies": list(self.dependencies),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "cost_usd": round(self.cost_usd, 6),
            "risk": self.risk.value,
            "error": self.error,
            "trace_id": self.trace_id,
        }


class TaskStore(Protocol):
    """Where tasks live. Swapped for PostgreSQL in a later phase."""

    def add(self, task: Task) -> Task: ...
    def get(self, task_id: str) -> Task | None: ...
    def update(self, task: Task) -> Task: ...
    def list(
        self, *, status: TaskStatus | None = None, parent: str | None = None
    ) -> list[Task]: ...


class InMemoryTaskStore:
    """The development store. Everything is lost when the process exits."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def list(self, *, status: TaskStatus | None = None, parent: str | None = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status is status]
        if parent is not None:
            tasks = [t for t in tasks if t.parent_task_id == parent]
        return sorted(tasks, key=lambda t: (t.priority, t.created_at))


class TaskEngine:
    """Creates tasks, moves them through their lifecycle, and reports on them."""

    def __init__(self, store: TaskStore | None = None) -> None:
        self.store: TaskStore = store or InMemoryTaskStore()

    def create(
        self,
        goal: str,
        *,
        parent: Task | str | None = None,
        agent: str | None = None,
        inputs: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        priority: int = 5,
        risk: RiskLevel = RiskLevel.LOW,
        trace_id: str | None = None,
        goal_id: str | None = None,
    ) -> Task:
        parent_id = parent.task_id if isinstance(parent, Task) else parent
        task = Task(
            goal=goal,
            goal_id=goal_id,
            parent_task_id=parent_id,
            assigned_agent=agent,
            inputs=dict(inputs or {}),
            dependencies=list(dependencies or []),
            priority=priority,
            risk=risk,
            trace_id=trace_id,
        )
        self.store.add(task)
        log.debug("created task %s: %s", task.task_id, goal)
        return task

    def get(self, task_id: str) -> Task | None:
        return self.store.get(task_id)

    def transition(self, task: Task, status: TaskStatus, *, error: str | None = None) -> Task:
        task.transition(status)
        if error is not None:
            task.error = error
        return self.store.update(task)

    def complete(self, task: Task, outputs: dict[str, Any]) -> Task:
        task.outputs.update(outputs)
        return self.transition(task, TaskStatus.COMPLETED)

    def fail(self, task: Task, error: str) -> Task:
        return self.transition(task, TaskStatus.FAILED, error=error)

    def children(self, task: Task | str) -> list[Task]:
        parent_id = task.task_id if isinstance(task, Task) else task
        return self.store.list(parent=parent_id)

    def dependencies_met(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep = self.store.get(dep_id)
            if dep is None or dep.status is not TaskStatus.COMPLETED:
                return False
        return True

    def add_cost(self, task: Task, cost_usd: float) -> Task:
        task.cost_usd += cost_usd
        return self.store.update(task)

    def snapshot(self) -> dict[str, Any]:
        tasks = self.store.list()
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return {
            "total": len(tasks),
            "by_status": counts,
            "total_cost_usd": round(sum(t.cost_usd for t in tasks), 6),
        }
