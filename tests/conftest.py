"""Shared test fixtures.

Every test runs offline: no network, no API key, no cost. The MockProvider
stands in for every real model, which is what makes the suite runnable
anywhere, including CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_core.agents.base import AgentContext
from jarvis_core.agents.registry import AgentRegistry, build_registry
from jarvis_core.config import get_settings, reset_settings_cache
from jarvis_core.core.runtime import JarvisRuntime
from jarvis_core.core.task import TaskEngine
from jarvis_core.evaluation.engine import EvaluationEngine
from jarvis_core.logging import AuditLog
from jarvis_core.models.catalog import default_catalog
from jarvis_core.models.mock_provider import MockProvider, agent_reply
from jarvis_core.models.router import ModelRouter
from jarvis_core.observability.cost import CostEngine
from jarvis_core.observability.tracing import Tracer
from jarvis_core.security.approval import AutoApproveGate
from jarvis_core.security.guardian import SecurityGuardian

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Point every test at a throwaway directory and a sandbox environment."""
    monkeypatch.setenv("JARVIS_ENVIRONMENT", "sandbox")
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Same reasoning: a machine that happens to hold an NVIDIA key would give
    # every offline test a second usable provider it was never written against.
    for vendor in ("NVIDIA", "XAI", "GEMINI"):
        monkeypatch.delenv(f"{vendor}_API_KEY", raising=False)
        monkeypatch.delenv(f"{vendor}_MODEL", raising=False)
    # The local model tier needs no credential, so deleting keys does not
    # switch it off: a developer with Ollama running would otherwise hand every
    # offline test a live provider, a second catalogue and a real HTTP call to
    # localhost. Tests that want it construct it with an injected transport.
    monkeypatch.setenv("OLLAMA_ENABLED", "0")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    # The suite never reaches the network. A token the machine happens to hold
    # would turn an offline test into a live API call.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Nor any ambient storage or workspace configuration: a test must see the
    # world its fixtures built, not the one the shell happened to export.
    # Set empty rather than deleted: a developer's .env would otherwise supply
    # a value, and a test must see the world its fixtures built.
    monkeypatch.setenv("JARVIS_DATABASE_PATH", "")
    monkeypatch.setenv("JARVIS_WORKSPACE_DIR", "")
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(path=tmp_path / "audit.jsonl")


# Re-exported so tests keep importing it from here. It lives in the mock
# provider because verification needs the same helper at runtime, and two
# copies would drift into looking like an agent regression.
agent_reply = agent_reply


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(
        responder=lambda request: agent_reply(
            summary=(
                "Detailed finding covering the requested topic with enough substance "
                "to look like real work, including specifics and context so the "
                "completeness check has something to measure against. " * 3
            ),
            confidence=0.85,
            sources=[{"url": "https://example.com/a", "title": "Example source"}],
        )
    )


@pytest.fixture
def cost(settings) -> CostEngine:
    return CostEngine(settings=settings, catalog=default_catalog())


@pytest.fixture
def router(provider, cost, audit, settings) -> ModelRouter:
    return ModelRouter(
        providers={"mock": provider},
        catalog=_mock_catalog(),
        cost=cost,
        settings=settings,
        audit=audit,
    )


def _mock_catalog():
    """The real catalog, re-pointed at the mock provider."""
    from dataclasses import replace

    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))
    return catalog


@pytest.fixture
def registry(audit) -> AgentRegistry:
    return build_registry(audit=audit, directory=AGENTS_DIR)


@pytest.fixture
def guardian(settings, audit) -> SecurityGuardian:
    return SecurityGuardian(settings=settings, audit=audit, approval_gate=AutoApproveGate())


@pytest.fixture
def tracer() -> Tracer:
    return Tracer()


@pytest.fixture
def context(router, guardian, audit, tracer) -> AgentContext:
    return AgentContext(
        router=router,
        guardian=guardian,
        audit=audit,
        tracer=tracer,
        trace=tracer.start_trace("test"),
    )


@pytest.fixture
def jarvis(registry, router, guardian, audit, settings) -> JarvisRuntime:
    return JarvisRuntime(
        registry=registry,
        router=router,
        tasks=TaskEngine(),
        guardian=guardian,
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
        use_model_planner=False,
    )


# --- the tool fabric -------------------------------------------------------


@pytest.fixture
async def fabric(settings, audit, cost, tracer, router):
    """A fabric with the mock server registered and every risk reachable.

    Tests need the destructive tools to exist so the gates guarding them can
    be exercised. A running system registers the mock at read-only, or not at
    all.
    """
    from jarvis_core.mcp.models import ToolRisk
    from jarvis_core.tools.setup import build_tool_fabric

    return await build_tool_fabric(
        settings=settings,
        audit=audit,
        cost=cost,
        tracer=tracer,
        include_mock=True,
        mock_max_risk=ToolRisk.EXTERNAL_SIDE_EFFECT,
        include_github=True,
        # The same servers a running system registers, so a test cannot pass
        # by being given a smaller world than production has.
        router=router,
        workspace=AGENTS_DIR.parent,
    )


@pytest.fixture
def gateway(fabric):
    return fabric.gateway


@pytest.fixture
def reader_permissions():
    """The permission set a read-only agent holds."""
    from jarvis_core.security.permissions import Permission

    return frozenset({Permission.READ, Permission.NETWORK})


@pytest.fixture
def writer_permissions():
    from jarvis_core.security.permissions import Permission

    return frozenset({Permission.READ, Permission.WRITE, Permission.NETWORK, Permission.ADMIN})
