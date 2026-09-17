"""The command layer: tasks, planning, orchestration and JARVIS itself.

Exports are resolved lazily. ``agents`` depends on ``core.task`` and
``core.messages``, while ``core.jarvis`` depends on ``agents``; importing
everything eagerly here would close that loop at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - names re-exported for type checkers
    # ruff: noqa: F401
    from .jarvis import Jarvis, JarvisResponse, build_jarvis
    from .messages import AgentMessage, MessageBus, MessageType
    from .orchestrator import AgentOrchestrator, RunReport, StepOutcome
    from .planner import Intent, ModelPlanner, Plan, PlanStep, RulePlanner, detect_intent
    from .task import (
        InMemoryTaskStore,
        InvalidTransition,
        Task,
        TaskEngine,
        TaskStatus,
        TaskStore,
    )

_EXPORTS: dict[str, str] = {
    "AgentMessage": "messages",
    "MessageBus": "messages",
    "MessageType": "messages",
    "InMemoryTaskStore": "task",
    "InvalidTransition": "task",
    "Task": "task",
    "TaskEngine": "task",
    "TaskStatus": "task",
    "TaskStore": "task",
    "Intent": "planner",
    "ModelPlanner": "planner",
    "Plan": "planner",
    "PlanStep": "planner",
    "RulePlanner": "planner",
    "detect_intent": "planner",
    "AgentOrchestrator": "orchestrator",
    "RunReport": "orchestrator",
    "StepOutcome": "orchestrator",
    "Jarvis": "jarvis",
    "JarvisResponse": "jarvis",
    "build_jarvis": "jarvis",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
