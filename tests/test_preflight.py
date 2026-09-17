"""One small call before the twenty-seven expensive ones.

Written because of a real event: a valid API key on an account with no credit.
Without a preflight the sweep made that discovery twenty-seven times and
recorded twenty-seven agent failures — every one of them a lie, because no
prompt was read and no schema was tested.

The distinction these tests exist to protect is between four kinds of "no",
which need four different things from four different people:

* no key at all
* a key that was rejected at the edge (HTTP 401, no request id)
* a key that authenticated and an account that cannot pay (HTTP 400 with a
  billing message, and a real request id — the id is the evidence that the
  request got past authentication)
* a network that will not carry the call

"The real sweep failed" collapses all four into one sentence and sends the
reader to fix whichever they guess.

Nothing here touches a network. The failures are constructed from the exact
error strings the provider returned on the day this was written.
"""

from __future__ import annotations

import pytest

from jarvis_core.models.base import ModelError, ProviderFault, classify_fault
from jarvis_core.verification.failures import FailureKind, classify
from jarvis_core.verification.preflight import (
    PreflightResult,
    Reachability,
    classify_error,
    preflight,
    request_id_of,
)
from jarvis_core.verification.status import VerificationRun

# The provider's own words, captured verbatim. A paraphrase here would test the
# paraphrase.
NO_CREDIT = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'Your credit balance is too low to access the Anthropic API. Please go "
    "to Plans & Billing to upgrade or purchase credits.'}, "
    "'request_id': 'req_011Cej3dizeyV1xgcr3XUy2c'}"
)
BAD_KEY = (
    "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
    "'message': 'API key is invalid.'}, 'request_id': None}"
)
RATE_LIMITED = "Error code: 429 - {'error': {'type': 'rate_limit_error'}}"


class FakeError(Exception):
    """Stands in for an SDK exception, which is all the classifier reads."""


# -- the four kinds of no ---------------------------------------------------


def test_an_empty_account_is_not_a_bad_key():
    """The whole point. Both are 4xx and they need different people."""
    assert classify_error(FakeError(NO_CREDIT)) is Reachability.NO_CREDIT
    assert classify_error(FakeError(BAD_KEY)) is Reachability.INVALID_CREDENTIAL


def test_a_rate_limit_is_not_a_broken_anything():
    assert classify_error(FakeError(RATE_LIMITED)) is Reachability.RATE_LIMITED


def test_an_unrecognised_failure_is_unreachable_rather_than_guessed():
    assert classify_error(FakeError("connection reset by peer")) is Reachability.UNREACHABLE


def test_the_reason_survives_the_routers_wrapper():
    """The router raises its own error. The diagnosis must travel in it."""
    wrapped = ModelError(f"every model attempt failed (tried: claude-haiku-4-5): {NO_CREDIT}")
    assert classify_error(wrapped) is Reachability.NO_CREDIT


def test_the_reason_survives_on_the_cause_alone():
    wrapped = ModelError("every model attempt failed (tried: x)")
    wrapped.__cause__ = FakeError(NO_CREDIT)
    assert classify_error(wrapped) is Reachability.NO_CREDIT


# -- the request id is itself evidence --------------------------------------


def test_a_billing_refusal_carries_a_request_id():
    """It got past authentication. That is what having an id means."""
    assert request_id_of(FakeError(NO_CREDIT)) == "req_011Cej3dizeyV1xgcr3XUy2c"


def test_a_rejected_key_carries_no_request_id():
    assert request_id_of(FakeError(BAD_KEY)) == ""


def test_the_request_id_is_found_through_the_wrapper():
    wrapped = ModelError(f"every model attempt failed (tried: x): {NO_CREDIT}")
    assert request_id_of(wrapped) == "req_011Cej3dizeyV1xgcr3XUy2c"


# -- what each state tells the reader to do ---------------------------------


@pytest.mark.parametrize("state", list(Reachability))
def test_every_unhappy_state_says_what_would_fix_it(state):
    result = PreflightResult(state=state)
    if state is Reachability.OK:
        assert result.remedy == ""
    else:
        assert result.remedy, f"{state} has no remedy"


def test_the_no_credit_remedy_does_not_tell_anyone_to_replace_a_working_key():
    remedy = PreflightResult(state=Reachability.NO_CREDIT).remedy
    assert "valid" in remedy and "credit" in remedy
    assert "Create a new one" not in remedy


def test_nothing_offers_to_buy_credit_on_efes_behalf():
    """Spending money is his. The remedy says so rather than implying it."""
    remedy = PreflightResult(state=Reachability.NO_CREDIT).remedy
    assert "nothing should try" in remedy.lower()


# -- the shared classification ----------------------------------------------


def test_the_models_layer_and_the_preflight_read_the_same_error_the_same_way():
    """Two copies of one fact is how the last matching bug got in."""
    assert classify_fault(FakeError(NO_CREDIT)) is ProviderFault.BILLING
    assert classify_fault(FakeError(BAD_KEY)) is ProviderFault.AUTH
    assert classify_error(FakeError(NO_CREDIT)) is Reachability.NO_CREDIT
    assert classify_error(FakeError(BAD_KEY)) is Reachability.INVALID_CREDENTIAL


def test_a_dead_credential_is_not_worth_another_model():
    """It fails identically on every model in the catalogue."""
    assert ProviderFault.AUTH.retryable is False
    assert ProviderFault.BILLING.retryable is False
    assert ProviderFault.RATE_LIMIT.retryable is True
    assert ProviderFault.TRANSIENT.retryable is True


# -- and none of it is the agent's fault ------------------------------------


def test_an_account_problem_is_never_recorded_as_an_agent_defect():
    run = VerificationRun(agent_id="a", task="t", real_model=True, ok=False, error=NO_CREDIT)
    kind = classify(run)
    assert kind is FailureKind.BILLING
    assert kind.is_environmental


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NO_CREDIT, FailureKind.BILLING),
        (BAD_KEY, FailureKind.AUTHENTICATION),
        (RATE_LIMITED, FailureKind.RATE_LIMIT),
    ],
)
def test_account_faults_are_named_rather_than_lumped_into_model_failure(error, expected):
    run = VerificationRun(agent_id="a", task="t", real_model=True, ok=False, error=error)
    assert classify(run) is expected


def test_a_genuine_model_refusal_is_still_the_models():
    run = VerificationRun(
        agent_id="a", task="t", real_model=True, ok=False, error="the model declined the request"
    )
    kind = classify(run)
    assert kind is FailureKind.MODEL_FAILURE
    assert not kind.is_environmental


# -- the probe itself -------------------------------------------------------


async def test_a_router_with_no_provider_needs_no_network_to_answer(router, monkeypatch):
    monkeypatch.setattr(router, "usable_models", lambda: [])
    result = await preflight(router)
    assert result.state is Reachability.NO_CREDENTIAL
    assert result.cost_usd == 0.0


async def test_a_working_provider_reports_what_the_probe_consumed(router, audit):
    result = await preflight(router, audit=audit)
    assert result.ok
    assert result.total_tokens > 0
    assert result.model
    assert "estimated" in result.render()


async def test_the_probe_uses_the_cheapest_model_available(router):
    """A reachability check should not be billed at frontier rates."""
    result = await preflight(router)
    cheapest = min(
        router.usable_models(), key=lambda s: s.input_cost_per_mtok + s.output_cost_per_mtok
    )
    assert result.model == cheapest.id


async def test_the_probe_is_audited_whichever_way_it_goes(router, audit):
    await preflight(router, audit=audit)
    actions = [event.action for event in audit.tail(20)]
    assert "verification.preflight" in actions


# -- and the sweep refuses to spend when it says no -------------------------


async def test_a_blocked_preflight_stops_the_sweep_before_any_agent_runs(
    router, audit, monkeypatch
):
    """Twenty-seven calls saved, and twenty-seven false failures not recorded."""
    from jarvis_core.agents.registry import build_registry
    from jarvis_core.verification import AgentVerifier

    async def refuse(*_args, **_kwargs):
        raise ModelError(f"every model attempt failed (tried: x): {NO_CREDIT}")

    verifier = AgentVerifier(build_registry(audit=audit), router, audit=audit)
    monkeypatch.setattr(router, "complete", refuse)

    await verifier.verify_all(real_model=True)

    assert verifier.preflight is not None
    assert verifier.preflight.state is Reachability.NO_CREDIT
    # It fell back to offline rather than recording failures against agents.
    assert not any(record.real_runs for record in verifier.records().values())
    assert all(
        not run.error for record in verifier.records().values() for run in record.offline_runs
    )


async def test_a_blocked_sweep_tells_each_agent_the_real_reason(router, audit, monkeypatch):
    """Not "set ANTHROPIC_API_KEY" when the key is set and correct."""
    from jarvis_core.agents.registry import build_registry
    from jarvis_core.verification import AgentStatus, AgentVerifier

    async def refuse(*_args, **_kwargs):
        raise ModelError(f"every model attempt failed (tried: x): {NO_CREDIT}")

    verifier = AgentVerifier(build_registry(audit=audit), router, audit=audit)
    monkeypatch.setattr(router, "complete", refuse)
    await verifier.verify_all(real_model=True)

    record = verifier.record("web_researcher")
    assert record is not None
    status, why = record.status()
    assert status is AgentStatus.OFFLINE_TESTED
    assert "credit" in why
    assert "set ANTHROPIC_API_KEY" not in why
