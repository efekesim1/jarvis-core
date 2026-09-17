"""Intent detection, plan shape, and the dependency graph."""

from __future__ import annotations

from jarvis_core.core.planner import Intent, Plan, PlanStep, RulePlanner, detect_intent


def test_a_business_request_is_recognised():
    assert detect_intent("find a new business opportunity in AI automation") is Intent.BUSINESS
    assert detect_intent("ABD'de AI automation için yeni bir iş fırsatı araştır") is Intent.BUSINESS


def test_a_build_request_is_recognised():
    assert detect_intent("implement a GitHub integration and deploy it") is Intent.BUILD


def test_a_short_factual_question_stays_a_question():
    assert detect_intent("what is MCP?") is Intent.QUESTION
    assert detect_intent("nedir bu pgvector?") is Intent.QUESTION


def test_a_simple_question_gets_one_agent_not_a_swarm(registry):
    plan = RulePlanner(registry).plan("what is MCP?")
    assert len(plan.steps) == 1


def test_a_business_request_fans_out_then_merges(registry):
    plan = RulePlanner(registry).plan("research an AI e-commerce business opportunity")
    assert plan.intent is Intent.BUSINESS
    assert len(plan.steps) > 1
    assert plan.steps[-1].step_id == "synthesis"
    assert plan.steps[-1].depends_on


def test_a_plan_only_names_registered_agents(registry):
    plan = RulePlanner(registry).plan("research the AI agent market")
    assert plan.validate(registry) == []


def test_independent_steps_land_in_the_same_wave(registry):
    plan = RulePlanner(registry).plan("research the AI agent market and its competitors")
    waves = plan.waves()
    assert len(waves[0]) >= 2
    # Every step appears exactly once across the waves.
    flat = [s.step_id for wave in waves for s in wave]
    assert sorted(flat) == sorted(s.step_id for s in plan.steps)


def test_a_dependent_step_waits_for_its_wave(registry):
    plan = RulePlanner(registry).plan("research an AI e-commerce business opportunity")
    waves = plan.waves()
    positions = {s.step_id: i for i, wave in enumerate(waves) for s in wave}
    for step in plan.steps:
        for dep in step.depends_on:
            assert positions[dep] < positions[step.step_id]


def test_a_cycle_is_caught_by_validation(registry):
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="x", depends_on=["b"]),
            PlanStep(step_id="b", agent_id="web_researcher", goal="x", depends_on=["a"]),
        ],
    )
    problems = plan.validate(registry)
    assert any("cycle" in p for p in problems)


def test_an_unknown_agent_is_caught_by_validation(registry):
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[PlanStep(step_id="a", agent_id="nonexistent_agent", goal="x")],
    )
    assert any("unknown agent" in p for p in plan.validate(registry))


def test_a_duplicate_step_id_is_caught(registry):
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="x"),
            PlanStep(step_id="a", agent_id="web_researcher", goal="y"),
        ],
    )
    assert any("duplicate" in p for p in plan.validate(registry))


def test_waves_terminate_even_on_a_cyclic_plan(registry):
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="x", depends_on=["b"]),
            PlanStep(step_id="b", agent_id="web_researcher", goal="x", depends_on=["a"]),
        ],
    )
    waves = plan.waves()
    assert sum(len(w) for w in waves) == 2


# -- staffing the right specialist ------------------------------------------


def test_the_rule_planner_can_reach_every_agent(registry):
    """Nineteen of twenty-seven were reachable only through the model planner.

    A pricing question got the market researcher, because that is who was on
    the fixed list — the pricing agent existed and was never staffed. This
    walks one question per agent's declared work and asserts nothing is
    stranded.
    """
    planner = RulePlanner(registry)
    questions = [
        "set a price for this and defend the number",
        "what is the runway and margin on this revenue",
        "find who to talk to for lead generation",
        "design the api schemas and integration",
        "write the deployment and rollback plan",
        "report what the community and forums are saying about this",
        "gather the pricing tiers of the three cloud providers in one shape",
        "check whether these findings are supported by their sources",
        "turn this repeated manual job into one that runs itself",
        "find the way into this web form that takes a filename",
        "try to break this before a user does",
        "decide the shape of this system before we write it",
        "report what happened recently in agent tooling with dates",
        "separate a real shift from noise and say where it is heading",
        "find products worth selling or building",
        "turn interest into a conversation and a close",
        "work out how people find out this exists",
        "decide where to play and what to refuse",
        "test whether an online product will sell without holding stock",
        "read this repository and report what is in it",
        "write the code for a function that parses a duration string",
        "merge these different market sizes into one answer",
        "give me a second opinion on this algorithm and its complexity",
        "what is the competitive landscape here and where are they weak",
    ]
    staffed: set[str] = set()
    for question in questions:
        staffed.update(step.agent_id for step in planner.plan(question).steps)

    stranded = {spec.id for spec in registry.all()} - staffed
    assert not stranded, f"no question reaches these agents: {sorted(stranded)}"


def test_a_specialist_beats_the_default_on_a_question(registry):
    """One agent, but the right one — the specialist costs the same."""
    plan = RulePlanner(registry).plan("what is the runway on this revenue")
    assert [s.agent_id for s in plan.steps] == ["finance_agent"]


def test_the_swarm_is_a_preference_not_a_filter(registry):
    """Filtering by swarm lost the pricing agent on a pricing question.

    The intent detector read "set a price and defend the number" as research,
    and the pricing agent lives in business, so it was never considered.
    Intent is a guess and a guess must not hide a specialist that named the
    work exactly.
    """
    plan = RulePlanner(registry).plan("set a price for this AI tool and defend the number")
    assert plan.steps[0].agent_id == "pricing_agent"


def test_one_shared_word_is_not_enough_to_staff_an_agent(registry):
    """ "analysis" appears across half the roster; staffing on it costs a call."""
    planner = RulePlanner(registry)
    assert planner._matching("do some analysis") == []


def test_a_request_nobody_declared_still_gets_the_usual_trio(registry):
    """The fallback is what stops an unmatched request staffing nobody."""
    plan = RulePlanner(registry).plan(
        "please look into the thing we discussed and come back with something"
    )
    assert len(plan.steps) >= 2


def test_the_plan_says_which_agents_matched_directly(registry):
    plan = RulePlanner(registry).plan("design the api schemas and integration for this")
    assert "matched the request directly" in plan.rationale


def test_capability_matching_ignores_the_system_prompt(registry):
    """Prompts are long and full of incidental words. Only declarations match.

    Same rule as skill keywords: an agent says what it is for, and that is what
    gets matched.
    """
    planner = RulePlanner(registry)
    spec = registry.require("qa_agent")
    assert spec.system_prompt
    # A distinctive phrase from inside the prompt should not, on its own, pull
    # the agent in — only its declared capabilities do that.
    matched = planner._matching("operator personal operating system inside")
    assert "qa_agent" not in matched
