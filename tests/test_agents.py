"""Agent definitions, the registry, and the LLM runtime."""

from __future__ import annotations

import pytest
import yaml

from jarvis_core.agents.base import AgentSpec, LLMAgentRuntime, RetryPolicy, extract_json
from jarvis_core.agents.registry import AgentNotFound, AgentProposal, AgentRegistry, DuplicateAgent
from jarvis_core.core.task import Task
from jarvis_core.security.permissions import Permission, RiskLevel


def test_every_shipped_agent_definition_loads(registry):
    ids = {s.id for s in registry.all()}
    assert "web_researcher" in ids
    assert len(ids) >= 3


def test_the_first_agent_is_configured_as_specified(registry):
    spec = registry.require("web_researcher")
    assert spec.swarm == "research"
    assert "web_search" in spec.tools
    assert spec.permissions == frozenset({Permission.READ, Permission.NETWORK})
    assert spec.risk_level is RiskLevel.LOW
    assert spec.evaluation_policy.require_sources
    assert spec.system_prompt


def test_no_shipped_agent_holds_more_permission_than_it_needs(registry):
    for spec in registry.all():
        assert Permission.ADMIN not in spec.permissions, spec.id
        assert Permission.FINANCIAL not in spec.permissions, spec.id


def test_an_agent_id_must_be_well_formed():
    with pytest.raises(ValueError, match="must be lowercase"):
        AgentSpec.from_dict({"id": "Bad Id!", "name": "n", "role": "r", "swarm": "research"})


def test_a_missing_required_field_fails_loudly():
    with pytest.raises(ValueError, match="missing"):
        AgentSpec.from_dict({"id": "some_agent", "name": "n"})


def test_infinite_retry_is_rejected():
    with pytest.raises(ValueError, match="capped at 5"):
        RetryPolicy.from_dict({"max_attempts": 99})
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy.from_dict({"max_attempts": 0})


def test_registering_the_same_id_twice_is_refused(registry):
    spec = registry.require("web_researcher")
    with pytest.raises(DuplicateAgent):
        registry.register(spec)


def test_an_unknown_agent_lists_what_is_available(registry):
    with pytest.raises(AgentNotFound, match="registered agents"):
        registry.require("no_such_agent")


def test_agents_can_be_found_by_capability(registry):
    found = registry.find_by_capability("research")
    assert any(s.id == "web_researcher" for s in found)


def test_a_generated_agent_does_not_run_until_it_is_approved(registry):
    spec = AgentSpec.from_dict(
        {"id": "generated_agent", "name": "Generated", "role": "test", "swarm": "research"}
    )
    registry.propose(AgentProposal(spec=spec, origin="system_builder", rationale="a gap"))
    assert registry.get("generated_agent") is None
    assert [p.spec.id for p in registry.proposals] == ["generated_agent"]

    registry.approve("generated_agent")
    assert registry.get("generated_agent") is not None


def test_a_rejected_proposal_is_dropped(registry):
    spec = AgentSpec.from_dict(
        {"id": "rejected_agent", "name": "R", "role": "test", "swarm": "research"}
    )
    registry.propose(AgentProposal(spec=spec, origin="system_builder"))
    registry.reject("rejected_agent", reason="not needed")
    assert registry.get("rejected_agent") is None
    assert registry.proposals == []


def test_a_broken_definition_file_is_skipped_not_fatal(tmp_path, audit):
    (tmp_path / "good.yaml").write_text(
        yaml.safe_dump({"id": "good_agent", "name": "G", "role": "r", "swarm": "research"})
    )
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"name": "no id"}))
    registry = AgentRegistry(audit=audit)
    assert registry.load_dir(tmp_path) == 1
    assert registry.get("good_agent") is not None


def test_json_is_extracted_from_a_fenced_reply():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_is_extracted_from_prose():
    assert extract_json('Here you go: {"a": 1} - hope that helps') == {"a": 1}


def test_extracting_json_from_plain_prose_returns_nothing():
    assert extract_json("no json at all here") is None


async def test_an_agent_run_returns_a_structured_result(registry, context):
    spec = registry.require("web_researcher")
    task = Task(goal="find out what MCP is")
    result = await LLMAgentRuntime().run(spec, task, context)
    assert result.ok
    assert result.agent_id == "web_researcher"
    assert result.text
    assert 0.0 <= result.confidence <= 1.0
    assert result.cost_usd >= 0.0
    assert result.model


async def test_an_agent_run_is_traced_and_messaged(registry, context):
    spec = registry.require("web_researcher")
    task = Task(goal="find out what MCP is")
    await LLMAgentRuntime().run(spec, task, context)
    assert any(s.name == "web_researcher" for s in context.trace.spans)
    assert context.bus.for_task(task.task_id)


async def test_an_agent_denied_by_the_guardian_does_not_call_a_model(registry, context, router):
    from jarvis_core.security.approval import DenyAllGate

    context.guardian.approval_gate = DenyAllGate()
    spec = registry.require("web_researcher")
    # Force approval by raising the agent's risk.
    spec = AgentSpec.from_dict(
        {
            "id": "risky_agent",
            "name": "Risky",
            "role": "r",
            "swarm": "research",
            "permissions": ["read"],
            "risk_level": "high",
        }
    )
    before = len(router.cost.records)
    result = await LLMAgentRuntime().run(spec, Task(goal="do something risky"), context)
    assert not result.ok
    assert len(router.cost.records) == before
