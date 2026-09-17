"""Telling an agent that exists from an agent that works.

Twenty-seven YAML files load, pass their capability check and appear in
`jarvis agents`. Six had ever produced an answer from a real model, and nothing
in the system said so — every report about "the agent roster" was a claim about
files.

The rule these tests protect is that a status cannot be asserted. Every level is
derived from recorded runs, and each requires evidence the level below could not
have produced. In particular: no credential must never become a claim of
readiness, and must never become a claim of failure either.
"""

from __future__ import annotations

import pytest

from jarvis_core.agents.registry import build_registry
from jarvis_core.verification import (
    MIN_RUNS_FOR_PRODUCTION,
    AgentStatus,
    AgentVerification,
    AgentVerifier,
    VerificationBudget,
    VerificationRun,
    unsupported_claims,
)


def run(**overrides) -> VerificationRun:
    defaults = {
        "agent_id": "a",
        "task": "do the thing",
        "real_model": True,
        "ok": True,
        "score": 0.8,
        "passed": True,
        # A run only counts once it produced an answer with the shape its own
        # definition promised. These tests are about the ladder, so the default
        # run is one that cleared that bar; `test_verification_evidence.py`
        # covers what happens when it does not.
        "answer_chars": 500,
    }
    defaults.update(overrides)
    return VerificationRun(**defaults)


def record(*runs: VerificationRun, **kwargs) -> AgentVerification:
    return AgentVerification(agent_id="a", runs=list(runs), **kwargs)


# -- the ladder -------------------------------------------------------------


def test_a_file_that_loads_is_only_defined():
    assert record().status()[0] is AgentStatus.DEFINED


def test_an_agent_with_no_smoke_task_cannot_get_past_defined():
    """Nothing can be run, so nothing can be shown. Same rule as skill keywords."""
    status, why = record(run(), has_smoke_task=False).status()
    assert status is AgentStatus.DEFINED
    assert "no smoke task" in why


def test_an_unbacked_capability_holds_it_at_defined():
    status, why = record(run(), capabilities_backed=False).status()
    assert status is AgentStatus.DEFINED
    assert "no tool provides" in why


def test_an_offline_run_proves_the_plumbing_and_no_more():
    status, why = record(run(real_model=False)).status()
    assert status is AgentStatus.OFFLINE_TESTED
    assert "no real model run" in why


def test_a_real_run_that_does_not_pass_stops_at_real_model_tested():
    status, why = record(run(score=0.4, passed=False)).status()
    assert status is AgentStatus.REAL_MODEL_TESTED
    assert "did not pass" in why


def test_one_passing_run_is_evaluated_not_production_ready():
    status, why = record(run()).status()
    assert status is AgentStatus.EVALUATED
    assert str(MIN_RUNS_FOR_PRODUCTION) in why


def test_enough_consistent_passing_runs_reach_production_ready():
    status, _ = record(*[run(score=0.8) for _ in range(MIN_RUNS_FOR_PRODUCTION)]).status()
    assert status is AgentStatus.PRODUCTION_READY


def test_an_agent_that_is_sometimes_good_is_not_dependable():
    """A score that swings is the difference between working and getting lucky."""
    status, why = record(run(score=0.95), run(score=0.95), run(score=0.5)).status()
    assert status is AgentStatus.EVALUATED
    assert "swung" in why


def test_a_single_failure_among_real_runs_blocks_production_ready():
    status, why = record(run(), run(), run(), run(ok=False, error="timeout")).status()
    assert status is AgentStatus.EVALUATED
    assert "failed" in why


def test_a_good_mean_below_the_bar_blocks_production_ready():
    status, why = record(*[run(score=0.6) for _ in range(4)]).status()
    assert status is AgentStatus.EVALUATED
    assert "below the" in why


def test_an_uncited_claim_blocks_production_ready():
    status, why = record(
        *[run() for _ in range(4)],
        run(unsupported_claims=["a percentage, uncited: it grew 40%"]),
    ).status()
    assert status is AgentStatus.EVALUATED
    assert "could not support" in why


# -- the environment is not the agent ---------------------------------------


def test_no_credential_is_never_reported_as_a_failure():
    """It is a fact about the environment. Reporting it otherwise is a lie."""
    status, why = record(run(real_model=False)).status()
    assert status is AgentStatus.OFFLINE_TESTED
    assert "failed" not in why
    assert "ANTHROPIC_API_KEY" in why


def test_an_agent_the_environment_refuses_is_not_marked_broken():
    """A deploy agent in a sandbox that withholds DEPLOY is working as designed."""
    status, why = record(
        run(
            real_model=False,
            ok=False,
            blocked=True,
            error="permission(s) deploy are not available in the sandbox environment",
        )
    ).status()
    assert status is AgentStatus.DEFINED
    assert "cannot be run in this environment" in why


def test_a_blocked_run_does_not_count_as_a_real_run():
    held = record(run(blocked=True, ok=False))
    assert held.real_runs == []
    assert len(held.blocked_runs) == 1


# -- cost -------------------------------------------------------------------


def test_offline_runs_do_not_report_a_bill_nobody_paid():
    """The mock provider prices its usage, so summing every run overstates."""
    held = record(run(real_model=False, cost_usd=0.02), run(real_model=True, cost_usd=0.05))
    assert held.real_cost == pytest.approx(0.05)


# -- uncited claims ---------------------------------------------------------


def test_a_hard_number_with_no_source_is_flagged():
    found = unsupported_claims("The market grew 34% last year.", sources=0)
    assert len(found) == 1
    assert "percentage" in found[0]


def test_the_same_number_with_a_source_is_not_flagged():
    assert unsupported_claims("The market grew 34% last year.", sources=2) == []


def test_a_hedged_number_is_not_a_claim():
    assert unsupported_claims("Roughly 30%, by my estimate, unverified.", sources=0) == []


def test_prose_with_no_checkable_claim_is_not_flagged():
    assert unsupported_claims("It is a strong product with good reviews.", sources=0) == []


@pytest.mark.parametrize(
    "text",
    [
        "It costs $4,500 per seat.",
        "There are 2.3 million users.",
        "The change landed in 2024.",
        "It requires version 3.11 or later.",
    ],
)
def test_several_shapes_of_checkable_claim_are_caught(text):
    assert unsupported_claims(text, sources=0)


# -- running it -------------------------------------------------------------


async def test_an_offline_sweep_verifies_the_whole_roster(router, audit):
    verifier = AgentVerifier(build_registry(audit=audit), router, audit=audit)
    records = await verifier.verify_all(real_model=False)

    assert len(records) == len(verifier.registry.all())
    summary = verifier.summary()
    assert summary["real_runs"] == 0
    assert summary["real_cost_usd"] == 0.0


async def test_an_offline_sweep_claims_nothing_above_offline_tested(router, audit):
    verifier = AgentVerifier(build_registry(audit=audit), router, audit=audit)
    records = await verifier.verify_all(real_model=False)
    for held in records.values():
        assert held.status()[0] <= AgentStatus.OFFLINE_TESTED


async def test_asking_for_real_without_a_provider_falls_back_and_says_so(router, audit, caplog):
    """It must not claim a real run, and must not report a failure either."""
    from jarvis_core.models.router import ModelRouter

    empty = ModelRouter(settings=router.settings, cost=router.cost, audit=audit)
    verifier = AgentVerifier(build_registry(audit=audit), empty, audit=audit)
    records = await verifier.verify_all(real_model=True, budget=VerificationBudget(max_agents=2))
    for held in records.values():
        assert held.real_runs == []


async def test_a_verification_run_gets_no_gateway(router, audit):
    """The approval gate is auto-approve here, so this is what makes it safe.

    With no gateway an agent cannot reach a tool, a file, the network or a
    credential. The only effect a verification run can have is one model call.
    """
    import inspect

    from jarvis_core.verification import verifier as module

    source = inspect.getsource(module.AgentVerifier.verify_agent)
    assert "gateway" not in source, (
        "a verification run must never be given a gateway; the auto-approve "
        "gate is only safe because there is nothing to reach"
    )


async def test_a_sweep_stops_at_its_budget(router, audit, monkeypatch):
    verifier = AgentVerifier(build_registry(audit=audit), router, audit=audit)

    # Force the real-model path with a router that answers, and a budget that
    # the first run exhausts.
    monkeypatch.setattr(verifier, "_backed_capabilities", lambda: set())
    records = await verifier.verify_all(
        real_model=True, budget=VerificationBudget(max_usd=0.0, repeats=1, max_agents=5)
    )
    assert len(records) < len(verifier.registry.all())


async def test_verification_survives_a_restart(router, audit, tmp_path):
    from jarvis_core.storage.setup import open_storage

    path = tmp_path / "verify.db"
    first = open_storage(path)
    registry = build_registry(audit=audit)
    verifier = AgentVerifier(registry, router, audit=audit, store=first.verification)
    await verifier.verify_agent(registry.all()[0], real_model=False)
    first.close()

    second = open_storage(path)
    reopened = AgentVerifier(registry, router, audit=audit, store=second.verification)
    assert reopened.records()
    assert next(iter(reopened.records().values())).runs
    second.close()


# -- the roster itself ------------------------------------------------------


def test_every_agent_declares_a_real_task_it_can_do():
    """Without one the agent is unverifiable, and unverifiable is not shippable."""
    for spec in build_registry().all():
        assert spec.smoke_task, f"{spec.id} has no smoke task"
        assert len(spec.smoke_task) > 40, f"{spec.id} has a smoke task too vague to judge"


# -- what can be known without spending anything ----------------------------


def spec_from(**overrides):
    """A minimal valid definition, for asserting one check at a time."""
    from jarvis_core.agents.base import AgentSpec, EvaluationPolicy, RetryPolicy
    from jarvis_core.security.permissions import Permission

    defaults = {
        "id": "a",
        "name": "An Agent",
        "role": "does a thing",
        "swarm": "research",
        "capabilities": ("research", "thing"),
        "permissions": frozenset({Permission.READ}),
        "system_prompt": "You do the thing. " * 20 + " Never invent a source.",
        "smoke_task": "Do the research thing that this agent declares it can do.",
        "output_schema": {"summary": "...", "confidence": 0.0},
        "evaluation_policy": EvaluationPolicy(),
        "retry_policy": RetryPolicy(),
        "max_tokens": 8000,
    }
    defaults.update(overrides)
    return AgentSpec(**defaults)


def checks_for(spec) -> set[str]:
    from jarvis_core.verification import audit_agent

    return {finding.check for finding in audit_agent(spec)}


def test_a_consistent_definition_has_nothing_to_report():
    assert checks_for(spec_from()) == set()


def test_requiring_sources_with_nowhere_to_put_them_breaks_every_run():
    """The hard source check fails, so the agent can never pass. Eight agents
    shipped like this, and nothing said so until the audit existed."""
    from jarvis_core.agents.base import EvaluationPolicy
    from jarvis_core.verification import Severity, audit_agent

    spec = spec_from(
        evaluation_policy=EvaluationPolicy(require_sources=True),
        output_schema={"summary": "...", "confidence": 0.0},
    )
    finding = next(f for f in audit_agent(spec) if f.check == "sources_impossible")
    assert finding.severity is Severity.BREAKS


def test_a_schema_with_sources_and_a_policy_that_wants_them_is_fine():
    from jarvis_core.agents.base import EvaluationPolicy

    spec = spec_from(
        evaluation_policy=EvaluationPolicy(require_sources=True),
        output_schema={"summary": "...", "sources": [], "confidence": 0.0},
    )
    assert "sources_impossible" not in checks_for(spec)


def test_a_capability_without_its_permission_is_refused_at_the_gateway():
    """Eight agents declared web or github capabilities without NETWORK."""
    from jarvis_core.verification import Severity, audit_agent

    spec = spec_from(tool_capabilities=("web.read",))
    finding = next(f for f in audit_agent(spec) if f.check == "permission_missing")
    assert finding.severity is Severity.BREAKS
    assert "network" in finding.fix


def test_a_schema_with_no_summary_hands_raw_json_to_synthesis():
    assert "output_summary" in checks_for(spec_from(output_schema={"findings": []}))


def test_a_schema_with_no_confidence_scores_every_run_the_same():
    assert "output_confidence" in checks_for(spec_from(output_schema={"summary": "..."}))


def test_an_empty_model_list_means_no_restriction_not_one_model():
    """The first version of this check read empty as one and reported four
    agents that were fine. A lint that cries wolf is a lint people stop
    reading, and the next thing it says is the thing that mattered."""
    from jarvis_core.agents.base import RetryPolicy

    retry = RetryPolicy(alternative_model=True)
    assert "no_alternative_model" not in checks_for(
        spec_from(retry_policy=retry, allowed_models=())
    )
    assert "no_alternative_model" in checks_for(
        spec_from(retry_policy=retry, allowed_models=("claude-opus-5",))
    )


def test_a_model_the_catalogue_does_not_have_cannot_be_routed_to():
    from jarvis_core.verification import audit_agent

    findings = audit_agent(
        spec_from(allowed_models=("no-such-model", "claude-opus-5")),
        known_models=frozenset({"claude-opus-5"}),
    )
    assert any(f.check == "unknown_model" for f in findings)


def test_a_prompt_promising_a_tool_the_agent_cannot_reach_is_flagged():
    spec = spec_from(
        system_prompt="You search the web for what the request asks. " * 10 + " Never guess.",
        tool_capabilities=(),
    )
    assert "prompt_promises_a_tool" in checks_for(spec)


def test_a_missing_or_thin_prompt_is_reported():
    assert "no_prompt" in checks_for(spec_from(system_prompt=""))
    assert "thin_prompt" in checks_for(spec_from(system_prompt="You are an agent."))


def test_a_prompt_with_no_boundaries_is_a_note_not_a_failure():
    from jarvis_core.verification import Severity, audit_agent

    spec = spec_from(system_prompt="You do the thing well and carefully. " * 15)
    finding = next(f for f in audit_agent(spec) if f.check == "no_boundaries")
    assert finding.severity is Severity.NOTE


def test_a_smoke_task_unrelated_to_the_agent_proves_only_the_plumbing():
    spec = spec_from(
        capabilities=("pricing", "packaging"),
        role="sets a price",
        name="Pricing",
        smoke_task="Write a poem about the weather in a distant country somewhere",
    )
    assert "smoke_task_unrelated" in checks_for(spec)


def test_a_budget_too_small_for_the_schema_truncates_into_a_parse_failure():
    spec = spec_from(
        output_schema={f"key_{i}": "..." for i in range(10)}
        | {"summary": "...", "confidence": 0.0},
        max_tokens=500,
    )
    assert "output_budget" in checks_for(spec)


# -- the roster this repository actually ships ------------------------------


def test_every_shipped_definition_is_internally_consistent():
    """Sixteen breaking findings the day this check was written: eight agents
    requiring sources with nowhere to put them, eight declaring web or github
    capabilities without NETWORK. Every one would have failed against a real
    model, every time, and cost money to find out."""
    from jarvis_core.models.catalog import default_catalog
    from jarvis_core.verification import audit_roster

    known = frozenset(spec.id for spec in default_catalog().all())
    report = audit_roster(build_registry().all(), known_models=known)

    assert not report.breaking, "\n".join(f"{f.agent_id}: {f.what}" for f in report.breaking)
    assert not report.degrading, "\n".join(f"{f.agent_id}: {f.what}" for f in report.degrading)


def test_the_audit_does_not_claim_to_measure_quality():
    """It says what is broken, never what is good. Confusing the two is how a
    green lint becomes a claim nobody checked."""
    from jarvis_core.models.catalog import default_catalog
    from jarvis_core.verification import audit_roster

    known = frozenset(spec.id for spec in default_catalog().all())
    rendered = audit_roster(build_registry().all(), known_models=known).render()
    assert "says nothing about answer quality" in rendered
    assert "verify --real" in rendered


# -- sources may not eat the whole output budget ------------------------------
#
# `market_researcher` passed the definition audit and then truncated on a real
# run. The old floor asked for `keys * 500`, which its max_tokens of 8000
# cleared — but it declares web_search, and the search blocks come out of the
# same ceiling as the answer. 96 sources later there was no room left to write
# the JSON.
#
# The check now asks a searching agent for room for both. The multiplier is a
# heuristic fitted to that one case and says so; what it must do is flag an
# agent that would truncate.


def test_a_searching_agent_needs_room_for_the_search_and_the_answer():
    spec = spec_from(
        output_schema={f"k{i}": "" for i in range(9)},
        max_tokens=8000,
        tools=["web_search"],
        permissions=["read", "network"],
    )
    assert "output_budget" in checks_for(spec)


def test_the_same_schema_without_a_server_tool_is_fine_at_that_budget():
    """The floor rises because of the tool, not because nine keys is a lot."""
    spec = spec_from(output_schema={f"k{i}": "" for i in range(9)}, max_tokens=8000)
    assert "output_budget" not in checks_for(spec)


def test_the_whole_roster_now_clears_its_output_budget():
    """The six the widened check caught, fixed. This keeps them fixed."""
    from jarvis_core.models.catalog import default_catalog
    from jarvis_core.verification import audit_roster

    report = audit_roster(
        build_registry().all(),
        known_models=frozenset(m.id for m in default_catalog().all()),
    )
    offenders = [f.agent_id for f in report.findings if f.check == "output_budget"]
    assert offenders == [], offenders


def test_every_searching_agent_has_headroom_over_its_schema():
    """Stated as an invariant, so a new research agent cannot arrive without it."""
    for spec in build_registry().all():
        if not spec.server_tools:
            continue
        floor = len(spec.output_schema) * 1000 + 2000
        assert spec.max_tokens >= floor, f"{spec.id}: {spec.max_tokens} < {floor}"
