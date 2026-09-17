"""The model catalog, the router and the cost engine."""

from __future__ import annotations

import json

import pytest

from jarvis_core.models.base import Message, ModelRequest
from jarvis_core.models.catalog import ModelCatalog, ModelSpec, default_catalog
from jarvis_core.models.mock_provider import MockProvider
from jarvis_core.models.router import ModelRouter, TaskKind
from jarvis_core.observability.cost import BudgetExceeded, CostEngine


def test_the_catalog_holds_the_verified_models():
    catalog = default_catalog()
    assert catalog.require("claude-opus-5").tier == "frontier"
    assert catalog.require("claude-haiku-4-5").tier == "fast"


def test_an_unknown_model_raises_with_a_useful_message():
    with pytest.raises(KeyError, match="unknown model"):
        default_catalog().require("gpt-imaginary")


def test_pricing_is_per_million_tokens():
    spec = default_catalog().require("claude-opus-5")
    # 1M in and 1M out at $5 and $25.
    assert spec.cost_usd(1_000_000, 1_000_000) == pytest.approx(30.0)


def test_extra_models_load_from_a_file_without_a_code_change(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "some-future-model",
                        "provider": "future_co",
                        "context_tokens": 500_000,
                        "input_cost_per_mtok": 1.5,
                        "output_cost_per_mtok": 6.0,
                        "tier": "balanced",
                    }
                ]
            }
        )
    )
    catalog = ModelCatalog()
    assert catalog.load_extra(path) == 1
    assert catalog.require("some-future-model").provider == "future_co"


def test_a_malformed_extra_model_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": [{"provider": "x"}, {"id": "ok", "provider": "y"}]}))
    catalog = ModelCatalog()
    assert catalog.load_extra(path) == 1


def test_simple_work_routes_to_the_cheap_tier(router):
    decision = router.select(TaskKind.SIMPLE)
    assert decision.model.tier == "fast"


def test_planning_routes_to_the_frontier_tier(router):
    decision = router.select(TaskKind.PLANNING)
    assert decision.model.tier == "frontier"


def test_an_agent_allow_list_is_respected(router):
    decision = router.select(TaskKind.PLANNING, allowed_models=["claude-haiku-4-5"])
    assert decision.model.id == "claude-haiku-4-5"


def test_work_needing_server_tools_avoids_models_without_them(router):
    decision = router.select(TaskKind.SIMPLE, needs_server_tools=True)
    assert decision.model.supports_server_tools


def test_budget_pressure_steps_the_router_down_a_tier(router, settings):
    router.cost.record(
        model="claude-opus-5",
        provider="mock",
        input_tokens=2_000_000,
        output_tokens=0,
    )
    assert router.cost.budget_pressure() >= 0.7
    decision = router.select(TaskKind.PLANNING)
    assert decision.downgraded
    assert decision.model.tier != "frontier"


def test_high_value_work_keeps_its_tier_under_budget_pressure(router):
    router.cost.record(
        model="claude-opus-5", provider="mock", input_tokens=2_000_000, output_tokens=0
    )
    decision = router.select(TaskKind.HIGH_VALUE)
    assert decision.model.tier == "frontier"


def test_a_router_with_no_providers_says_so(settings):
    from jarvis_core.models.base import ModelUnavailable

    empty = ModelRouter(providers={}, settings=settings)
    with pytest.raises(ModelUnavailable, match="no model provider"):
        empty.select()


async def test_a_completed_call_is_priced_and_recorded(router):
    response = await router.complete(
        ModelRequest.single("hello"), kind=TaskKind.SIMPLE, agent="tester", task_id="tsk_1"
    )
    assert response.text
    assert router.cost.records
    assert router.cost.spent_on_task("tsk_1") > 0
    assert router.cost.spent_on_agent("tester") > 0


async def test_a_failing_provider_is_retried_on_another_model(cost, settings, audit):
    from dataclasses import replace

    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))
    flaky = MockProvider(fail_times=1)
    router = ModelRouter(
        providers={"mock": flaky}, catalog=catalog, cost=cost, settings=settings, audit=audit
    )
    response = await router.complete(ModelRequest.single("hi"), max_attempts=3)
    assert response.text
    assert len(flaky.calls) == 2


async def test_retries_are_bounded(cost, settings, audit):
    from dataclasses import replace

    from jarvis_core.models.base import ModelError

    catalog = default_catalog()
    for spec in list(catalog.all()):
        catalog.register(replace(spec, provider="mock"))
    always_fails = MockProvider(fail_times=99)
    router = ModelRouter(
        providers={"mock": always_fails}, catalog=catalog, cost=cost, settings=settings, audit=audit
    )
    with pytest.raises(ModelError, match="every model attempt failed"):
        await router.complete(ModelRequest.single("hi"), max_attempts=3)
    assert len(always_fails.calls) == 3


async def test_a_breached_daily_budget_stops_the_call(router, settings):
    router.cost.record(
        model="claude-opus-5",
        provider="mock",
        input_tokens=int(settings.daily_budget_usd * 1_000_000 / 5) + 1_000,
        output_tokens=0,
    )
    with pytest.raises(BudgetExceeded):
        await router.complete(ModelRequest.single("hello"))


def test_the_cost_summary_reports_by_model_and_agent(cost):
    cost.record(
        model="claude-opus-5",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=500,
        agent="web_researcher",
    )
    summary = cost.summary()
    assert summary["calls"] == 1
    assert summary["by_model"]["claude-opus-5"] > 0
    assert summary["by_agent"]["web_researcher"] > 0


def test_an_unpriced_model_is_recorded_as_zero_rather_than_crashing():
    engine = CostEngine(catalog=ModelCatalog(specs=[]))
    record = engine.record(model="mystery", provider="x", input_tokens=100, output_tokens=100)
    assert record.cost_usd == 0.0


async def test_the_mock_provider_answers_from_a_script():
    provider = MockProvider(scripted=["first", "second"])
    assert (await provider.complete(ModelRequest.single("a"))).text == "first"
    assert (await provider.complete(ModelRequest.single("b"))).text == "second"


async def test_the_mock_provider_counts_tokens_for_cost_bookkeeping():
    provider = MockProvider(scripted=["a longer answer with several words in it"])
    response = await provider.complete(
        ModelRequest(messages=[Message(role="user", content="a question")])
    )
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


# -- what the router is *not* doing ------------------------------------------
#
# "Stronger model, better answer" is the assumption a router quietly drifts
# into, and it is expensive and usually wrong. These say what the selection
# actually rests on: the kind of work picks a tier, the tier is walked
# downwards and never upwards, and inside the chosen tier price decides among
# the models that can do the job. None of that has been checked against a real
# model — whether the cheap tier really is good enough for simple work is a
# question `jarvis verify --real` answers and these tests cannot.


def _catalog_of(*specs):
    catalog = ModelCatalog()
    for spec in specs:
        catalog.register(spec)
    return catalog


def _spec(model_id, tier, price, **kwargs):
    return ModelSpec(
        id=model_id,
        provider="mock",
        context_tokens=200_000,
        input_cost_per_mtok=price,
        output_cost_per_mtok=price * 4,
        tier=tier,
        **kwargs,
    )


def test_inside_a_tier_the_cheapest_capable_model_wins(provider, cost, audit, settings):
    """Price decides among equals. A pricier model in the same tier is not better."""
    router = ModelRouter(
        providers={"mock": provider},
        catalog=_catalog_of(
            _spec("dear-one", "fast", 5.0),
            _spec("cheap-one", "fast", 0.5),
        ),
        cost=cost,
        settings=settings,
        audit=audit,
    )
    assert router.select(kind=TaskKind.SIMPLE).model.id == "cheap-one"


def test_a_declared_strength_outranks_price_inside_a_tier(provider, cost, audit, settings):
    """Cheapest *among the models that can do the work*, not cheapest overall."""
    router = ModelRouter(
        providers={"mock": provider},
        catalog=_catalog_of(
            _spec("cheap-generalist", "frontier", 1.0),
            _spec("dear-specialist", "frontier", 8.0, strengths=frozenset({"coding"})),
        ),
        cost=cost,
        settings=settings,
        audit=audit,
    )
    assert router.select(kind=TaskKind.CODING).model.id == "dear-specialist"
    # And with no strength to match, price decides again.
    assert router.select(kind=TaskKind.PLANNING).model.id == "cheap-generalist"


def test_simple_work_is_never_promoted_to_a_stronger_tier(provider, cost, audit, settings):
    """Even with the budget untouched and a frontier model sitting right there."""
    router = ModelRouter(
        providers={"mock": provider},
        catalog=_catalog_of(
            _spec("frontier-one", "frontier", 15.0),
            _spec("fast-one", "fast", 0.25),
        ),
        cost=cost,
        settings=settings,
        audit=audit,
    )
    assert cost.budget_pressure() == 0.0
    assert router.select(kind=TaskKind.SIMPLE).model.tier == "fast"


def test_the_walk_only_ever_goes_downwards(provider, cost, audit, settings):
    """A tier with nothing in it falls to a cheaper one, never to a dearer one."""
    router = ModelRouter(
        providers={"mock": provider},
        catalog=_catalog_of(
            _spec("frontier-one", "frontier", 15.0),
            _spec("fast-one", "fast", 0.25),
        ),
        cost=cost,
        settings=settings,
        audit=audit,
    )
    # RESEARCH wants "balanced", which this catalog does not have.
    decision = router.select(kind=TaskKind.RESEARCH)
    assert decision.model.id == "fast-one"


def test_every_routing_decision_says_why(provider, cost, audit, settings):
    """A choice with no stated reason cannot be argued with, or corrected."""
    router = ModelRouter(
        providers={"mock": provider},
        catalog=_catalog_of(_spec("only-one", "fast", 1.0)),
        cost=cost,
        settings=settings,
        audit=audit,
    )
    for kind in TaskKind:
        assert router.select(kind=kind).reason
