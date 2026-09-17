"""The twenty-seven agents, and whether they are real.

Writing a YAML file is not building an agent. These tests ask the questions
that separate the two: does every one load, is each held to least privilege,
does anything claim a capability the system cannot honour, and can the planner
actually staff a request from the roster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_core.core.planner import Intent, RulePlanner, detect_intent
from jarvis_core.mcp.models import ToolRisk
from jarvis_core.security.permissions import Permission
from jarvis_core.tools.policy import capability_coverage, policies_from_registry

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

EXPECTED = {
    "research": {
        "web_researcher",
        "market_researcher",
        "news_researcher",
        "trend_analyst",
        "competitor_analyst",
        "social_researcher",
        "data_collector",
        "source_verifier",
        "research_synthesizer",
    },
    "business": {
        "business_strategist",
        "ecommerce_agent",
        "product_hunter",
        "sales_agent",
        "lead_generation_agent",
        "marketing_agent",
        "finance_agent",
        "pricing_agent",
        "business_analyst",
    },
    "build": {
        "software_architect",
        "claude_developer",
        "codex_developer",
        "github_agent",
        "qa_agent",
        "security_agent",
        "api_agent",
        "automation_agent",
        "deployment_agent",
    },
}


# -- the roster -------------------------------------------------------------


def test_all_twenty_seven_agents_load(registry):
    assert len(registry.all()) == 27


@pytest.mark.parametrize("swarm", sorted(EXPECTED))
def test_each_swarm_is_complete(registry, swarm):
    assert {s.id for s in registry.by_swarm(swarm)} == EXPECTED[swarm]


def test_every_agent_has_what_it_needs_to_run(registry):
    """A definition missing any of these is a file, not an agent."""
    for spec in registry.all():
        assert spec.system_prompt, f"{spec.id} has no prompt"
        assert spec.output_schema, f"{spec.id} has no output schema"
        assert spec.capabilities, f"{spec.id} declares no capabilities"
        assert spec.allowed_models is not None
        assert spec.timeout_s > 0, f"{spec.id} has no timeout"
        assert 1 <= spec.retry_policy.max_attempts <= 5, f"{spec.id} retries unboundedly"
        assert spec.evaluation_policy.criteria, f"{spec.id} has no evaluation criteria"


def test_agent_ids_are_unique(registry):
    ids = [s.id for s in registry.all()]
    assert len(ids) == len(set(ids))


def test_each_agent_lives_in_its_declared_swarm():
    for path in AGENTS_DIR.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert f"swarm: {path.parent.name}" in text, f"{path} is in the wrong directory"


# -- least privilege --------------------------------------------------------


def test_no_agent_can_spend_money(registry):
    for spec in registry.all():
        assert Permission.FINANCIAL not in spec.permissions, spec.id


def test_no_agent_holds_admin(registry):
    for spec in registry.all():
        assert Permission.ADMIN not in spec.permissions, spec.id


def test_only_the_deployment_agent_may_deploy(registry):
    holders = {s.id for s in registry.all() if Permission.DEPLOY in s.permissions}
    assert holders == {"deployment_agent"}


def test_the_finance_agent_models_money_and_cannot_move_it(registry):
    spec = registry.require("finance_agent")
    assert Permission.FINANCIAL not in spec.permissions
    assert spec.permissions == frozenset({Permission.READ})


def test_agents_that_reach_the_network_say_so(registry):
    """Declaring web search without NETWORK loses the tool, so it is a bug."""
    for spec in registry.all():
        if "web_search" in spec.tools or "web_fetch" in spec.tools:
            assert Permission.NETWORK in spec.permissions, spec.id


def test_agents_without_network_declare_no_network_tools(registry):
    for spec in registry.all():
        if Permission.NETWORK not in spec.permissions:
            assert not set(spec.tools) & {"web_search", "web_fetch"}, spec.id


def test_no_agent_may_write_through_a_tool(registry):
    """Nothing shipped writes to the world. That stays a per-agent decision."""
    for spec in registry.all():
        assert spec.max_tool_risk == "read_only", spec.id


def test_only_the_github_agent_and_researchers_reach_github(registry):
    holders = {
        s.id for s in registry.all() if any(c.startswith("github") for c in s.tool_capabilities)
    }
    assert "github_agent" in holders
    # Nothing in the business swarm has any business with a repository.
    assert not holders & EXPECTED["business"]


# -- capabilities are real --------------------------------------------------


async def test_no_agent_claims_a_capability_the_system_cannot_honour(fabric, registry):
    """Declaring a capability with no tool behind it is a false promise."""
    unbacked = {
        c.agent_id: c.unbacked
        for c in capability_coverage(registry.all(), fabric.tools.all())
        if c.unbacked
    }
    assert not unbacked, f"agents claiming capabilities nothing backs: {unbacked}"


async def test_every_agent_has_a_policy_derived_from_its_own_definition(registry):
    policies = policies_from_registry(registry.all())
    assert len(policies) == 27
    for spec in registry.all():
        policy = policies[spec.id]
        assert policy.capabilities == frozenset(spec.tool_capabilities)
        assert policy.max_risk is ToolRisk.READ_ONLY


async def test_an_agent_reaches_exactly_what_its_definition_grants(fabric, registry):
    fabric.policy._policies.update(policies_from_registry(registry.all()))
    visible = {t.capability for t in fabric.gateway.available_tools("github_agent")}
    assert visible
    assert all(c.startswith("github.read") for c in visible)

    # And one that asks for nothing reaches nothing.
    assert fabric.gateway.available_tools("business_strategist") == []


# -- the planner can staff a request ---------------------------------------


def test_a_business_request_draws_on_the_business_swarm(registry):
    plan = RulePlanner(registry).plan("research an AI e-commerce business opportunity")
    assert plan.intent is Intent.BUSINESS
    assert plan.steps[-1].agent_id in EXPECTED["business"]
    assert plan.validate(registry) == []


def test_a_build_request_draws_on_the_build_swarm(registry):
    plan = RulePlanner(registry).plan("review the security of this repository architecture")
    assert plan.intent is Intent.BUILD
    used = {s.agent_id for s in plan.steps}
    assert used & EXPECTED["build"]
    assert plan.validate(registry) == []


def test_a_research_request_draws_on_the_research_swarm(registry):
    plan = RulePlanner(registry).plan("find the latest news about AI agent frameworks")
    assert plan.intent is Intent.RESEARCH
    assert {s.agent_id for s in plan.steps} & EXPECTED["research"]


def test_a_plan_never_runs_more_than_three_gatherers(registry):
    """A fourth gatherer rarely finds anything and always costs a call."""
    for request in (
        "research an AI e-commerce business opportunity",
        "find the latest news about AI agent frameworks",
        "review this repository's architecture and security",
    ):
        plan = RulePlanner(registry).plan(request)
        assert len(plan.waves()[0]) <= 3, request


def test_a_short_question_still_gets_one_agent(registry):
    for question in ("what is MCP?", "MCP nedir?", "pgvector ne kadar hızlı?"):
        plan = RulePlanner(registry).plan(question)
        assert len(plan.steps) == 1, question


def test_turkish_questions_are_recognised_despite_word_order():
    """Turkish puts the question word last, English puts it first."""
    assert detect_intent("MCP nedir?") is Intent.QUESTION
    assert detect_intent("what is MCP?") is Intent.QUESTION
    assert detect_intent("bu nasıl çalışıyor?") is Intent.QUESTION


def test_a_long_turkish_request_is_not_mistaken_for_a_question():
    intent = detect_intent(
        "nasıl bir e-ticaret işi kurabilirim, pazarı ve rakipleri detaylı araştır"
    )
    assert intent is Intent.BUSINESS


def test_every_agent_can_be_planned_for(registry):
    """Each agent is reachable by some request, or it is dead weight."""
    planner = RulePlanner(registry)
    reachable: set[str] = set()
    for request in (
        "research an AI e-commerce business opportunity",
        "find the latest news about AI agent frameworks",
        "review this repository's architecture and security",
        "what is MCP?",
    ):
        reachable |= {s.agent_id for s in planner.plan(request).steps}
    # The rule planner staffs the common shapes; the rest are reachable
    # through the model planner, which sees the whole catalogue.
    assert len(reachable) >= 8
    catalogue = registry.catalog_for_planner()
    for spec in registry.all():
        assert spec.id in catalogue, f"{spec.id} is invisible to the model planner"


# -- the definitions say what they do ---------------------------------------


def test_agents_that_must_cite_sources_say_so(registry):
    """A research agent that can invent a finding is worse than none."""
    for agent_id in ("web_researcher", "news_researcher", "market_researcher"):
        assert registry.require(agent_id).evaluation_policy.require_sources, agent_id


def test_the_deployment_agent_never_retries_itself(registry):
    spec = registry.require("deployment_agent")
    assert spec.retry_policy.max_attempts == 1
    assert spec.risk_level.value == "high"


def test_the_independent_reviewer_is_not_pinned_to_one_model(registry):
    """Its whole value is being a different model from the author."""
    assert registry.require("codex_developer").allowed_models == ()
