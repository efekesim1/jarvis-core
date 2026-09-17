"""Model abstraction: providers, catalog, and the router that chooses between them.

Exports are resolved lazily. ``observability.cost`` needs ``models.catalog`` for
pricing, while ``models.router`` needs ``observability.cost`` for budgets;
importing the router eagerly here would close that loop depending on which
module a caller happened to import first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - names re-exported for type checkers
    # ruff: noqa: F401
    from .base import (
        Message,
        ModelError,
        ModelProvider,
        ModelRefusal,
        ModelRequest,
        ModelResponse,
        ModelUnavailable,
        Usage,
    )
    from .catalog import ModelCatalog, ModelSpec, default_catalog
    from .mock_provider import MockProvider
    from .router import ModelRouter, RoutingDecision, TaskKind, build_default_router

_EXPORTS: dict[str, str] = {
    "Message": "base",
    "ModelError": "base",
    "ModelProvider": "base",
    "ModelRefusal": "base",
    "ModelRequest": "base",
    "ModelResponse": "base",
    "ModelUnavailable": "base",
    "Usage": "base",
    "ModelCatalog": "catalog",
    "ModelSpec": "catalog",
    "default_catalog": "catalog",
    "MockProvider": "mock_provider",
    "ModelRouter": "router",
    "RoutingDecision": "router",
    "TaskKind": "router",
    "build_default_router": "router",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
