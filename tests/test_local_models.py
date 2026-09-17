"""The free tier, and the four things it must not become.

A local model changes the economics of every routing decision, which is why it
needs more care than a new vendor and not less. What is asserted here:

* it is absent unless a daemon actually answered, so nothing registers a
  provider that fails on first use;
* it takes simple, high-volume work and is what every tier steps down to when
  the paid budget runs hot — the useful property;
* it never takes planning, synthesis or high-value work, whatever the
  environment claims, because a free model that wins those wins them by being
  free and not by being right;
* a local call is priced at zero in the ledger and spends no budget, which is
  the literal truth rather than a discount.

Everything runs offline against an injected transport: no daemon, no HTTP, no
localhost. `conftest` switches the real one off for the whole suite so a
developer running Ollama does not get a different test run from CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis_core.config import get_settings, reset_settings_cache
from jarvis_core.models.base import Message, ModelError, ModelRequest, ModelUnavailable
from jarvis_core.models.catalog import TIERS, ModelCatalog, normalise_tier
from jarvis_core.models.local_provider import DEFAULT_CONTEXT_TOKENS, OllamaProvider
from jarvis_core.models.mock_provider import MockProvider
from jarvis_core.models.router import ModelRouter, TaskKind
from jarvis_core.observability.cost import CostEngine

PULLED = ("qwen2.5:7b", "llama3.1:8b", "nomic-embed-text:latest")


class Daemon:
    """A stand-in Ollama that records what it was sent."""

    def __init__(
        self,
        *,
        models: tuple[str, ...] = PULLED,
        answer: str = "  a local answer  ",
        chat: dict[str, Any] | None = None,
        fail: Exception | None = None,
    ) -> None:
        self.models = models
        self.answer = answer
        self.chat_override = chat
        self.fail = fail
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(
        self, method: str, url: str, body: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        self.calls.append((method, url, body))
        if self.fail is not None:
            raise self.fail
        if url.endswith("/api/tags"):
            return {"models": [{"model": name, "name": name} for name in self.models]}
        if self.chat_override is not None:
            return self.chat_override
        return {
            "model": (body or {}).get("model"),
            "message": {"role": "assistant", "content": self.answer},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 310,
            "eval_count": 88,
        }

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [b for _, _, b in self.calls if b is not None]


@pytest.fixture
def enabled(monkeypatch):
    """The setting the suite otherwise keeps off."""
    monkeypatch.setenv("OLLAMA_ENABLED", "1")
    reset_settings_cache()
    yield
    reset_settings_cache()


def provider(daemon: Daemon, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(transport=daemon, settings=get_settings(), **kwargs)


# -- presence ---------------------------------------------------------------


def test_no_daemon_means_no_provider(enabled):
    """The NVIDIA rule: unconfigured is absent, not present and failing."""
    dead = provider(Daemon(fail=OSError("connection refused")))
    assert dead.available() is False
    assert "install it from https://ollama.com" in (dead.why_unavailable() or "")
    assert dead.installed_models() == ()


def test_a_daemon_with_nothing_pulled_is_also_absent(enabled):
    empty = provider(Daemon(models=()))
    assert empty.available() is False
    assert empty.summary()["installed"] == []


def test_the_switch_beats_a_running_daemon(monkeypatch):
    """OLLAMA_ENABLED=0 must not be advisory: nothing is asked, nothing runs."""
    monkeypatch.setenv("OLLAMA_ENABLED", "0")
    reset_settings_cache()
    daemon = Daemon()
    off = provider(daemon)
    assert off.available() is False
    assert not daemon.calls, "a switched-off provider still called the daemon"
    assert "switched off" in (off.why_unavailable() or "")


def test_it_reports_what_the_machine_actually_has(enabled):
    live = provider(Daemon())
    assert live.available() is True
    assert live.installed_models() == tuple(sorted(PULLED))
    assert live.why_unavailable() is None


def test_a_pinned_model_that_is_not_pulled_says_which_ones_are(enabled):
    pinned = provider(Daemon(), model="deepseek-r1:70b")
    assert pinned.available() is False
    why = pinned.why_unavailable() or ""
    assert "deepseek-r1:70b" in why
    assert "qwen2.5:7b" in why, "the refusal did not say what could be used instead"


def test_the_daemon_is_asked_once_and_the_answer_is_cached(enabled):
    """`available()` is on the routing path; the probe must not be."""
    daemon = Daemon()
    live = provider(daemon)
    for _ in range(25):
        assert live.available() is True
    assert len(daemon.calls) == 1, f"the daemon was probed {len(daemon.calls)} times"
    assert live.installed_models(force=True) == tuple(sorted(PULLED))
    assert len(daemon.calls) == 2


def test_a_hostile_base_url_is_refused_rather_than_fetched(enabled, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "file:///etc/passwd")
    reset_settings_cache()
    daemon = Daemon()
    with pytest.raises(ModelUnavailable) as caught:
        provider(daemon)._url_for("/api/tags")
    assert "http(s)" in str(caught.value)
    assert not daemon.calls


def test_pointing_it_off_this_machine_is_allowed_but_said_out_loud(enabled, monkeypatch, caplog):
    """A deliberate setting, not something a request can do — and a disclosure.

    The prompt leaves this machine when the URL is remote, so the warning is
    the feature: it is the only place a reader of the logs finds out.
    """
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.40:11434")
    reset_settings_cache()
    daemon = Daemon()
    with caplog.at_level("WARNING"):
        remote = provider(daemon)
        assert remote.available() is True
        remote.available()
    warnings = [r for r in caplog.records if "not this machine" in r.getMessage()]
    assert len(warnings) == 1, "the disclosure was missing or repeated on every call"


# -- the call ---------------------------------------------------------------


async def test_it_sends_what_the_native_api_expects(enabled):
    daemon = Daemon()
    response = await provider(daemon).complete(
        ModelRequest(
            messages=[Message(role="user", content="which of these is cheapest?")],
            system="You are JARVIS.",
            model="qwen2.5:7b",
            max_tokens=512,
        )
    )
    body = daemon.bodies[-1]
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False, "a blocking call asked the daemon to stream"
    assert body["options"]["num_predict"] == 512
    assert body["messages"][0] == {"role": "system", "content": "You are JARVIS."}
    assert body["messages"][1]["content"] == "which of these is cheapest?"

    assert response.text == "a local answer"
    assert response.provider == "ollama"
    assert response.model == "qwen2.5:7b"
    assert response.usage.input_tokens == 310
    assert response.usage.output_tokens == 88
    assert response.latency_ms > 0


async def test_with_no_model_named_it_uses_one_the_machine_has(enabled):
    daemon = Daemon()
    response = await provider(daemon).complete(
        ModelRequest(messages=[Message(role="user", content="hello")])
    )
    assert response.model in PULLED


async def test_a_daemon_that_died_mid_call_is_unavailable_not_broken(enabled):
    """The distinction the retry logic runs on: try another model, not this one."""
    import urllib.error

    daemon = Daemon()
    live = provider(daemon)
    assert live.available() is True
    daemon.fail = urllib.error.URLError("connection refused")
    with pytest.raises(ModelUnavailable):
        await live.complete(ModelRequest(messages=[Message(role="user", content="hi")]))
    # And the cached "it is there" is dropped, so the next routing decision
    # does not keep choosing a daemon that has gone.
    daemon.fail = None
    assert live.installed_models() == tuple(sorted(PULLED))


async def test_an_empty_answer_is_an_error_and_not_an_empty_string(enabled):
    daemon = Daemon(chat={"model": "qwen2.5:7b", "message": {"content": ""}})
    with pytest.raises(ModelError):
        await provider(daemon).complete(ModelRequest(messages=[Message(role="user", content="hi")]))


# -- the catalogue ----------------------------------------------------------


def test_every_local_model_is_a_free_entry_in_the_fast_tier(enabled):
    specs = provider(Daemon()).catalog_specs()
    assert {s.id for s in specs} == set(PULLED)
    for spec in specs:
        assert spec.provider == "ollama"
        assert spec.input_cost_per_mtok == 0.0
        assert spec.output_cost_per_mtok == 0.0
        assert spec.tier == "fast"
        assert spec.supports_server_tools is False
        assert spec.context_tokens == DEFAULT_CONTEXT_TOKENS


def test_a_pinned_model_is_the_only_entry(enabled):
    specs = provider(Daemon(), model="llama3.1:8b").catalog_specs()
    assert [s.id for s in specs] == ["llama3.1:8b"]


def test_an_invented_tier_cannot_promote_a_free_model(enabled, monkeypatch):
    """The footgun this closes, by name.

    `select` falls back to "the cheapest usable model" when no tier matched. A
    spec carrying a tier the router does not know matches nothing, so a
    zero-priced one would win *every* fallback — including planning. An unknown
    tier is corrected to `fast` instead.
    """
    monkeypatch.setenv("OLLAMA_TIER", "frontier-plus")
    reset_settings_cache()
    specs = provider(Daemon()).catalog_specs()
    assert {s.tier for s in specs} == {"fast"}
    assert normalise_tier("frontier-plus", default="fast") == "fast"
    assert all(tier in TIERS for tier in ("frontier", "balanced", "fast"))


# -- routing ----------------------------------------------------------------


def _router(daemon: Daemon, **kwargs: Any) -> tuple[ModelRouter, OllamaProvider]:
    """A router holding one paid vendor and one local one."""
    local = provider(daemon, **kwargs)
    catalog = ModelCatalog()
    for spec in local.catalog_specs():
        catalog.register(spec)
    mock = MockProvider()
    mock.name = "anthropic"
    cost = CostEngine(catalog=catalog)
    router = ModelRouter(providers={"anthropic": mock, "ollama": local}, catalog=catalog, cost=cost)
    return router, local


def test_simple_work_goes_to_the_free_model(enabled):
    router, _ = _router(Daemon())
    decision = router.select(TaskKind.SIMPLE)
    assert decision.model.provider == "ollama"
    assert decision.model.cost_usd(1_000_000, 1_000_000) == 0.0


@pytest.mark.parametrize(
    "kind", [TaskKind.PLANNING, TaskKind.SYNTHESIS, TaskKind.CODING, TaskKind.HIGH_VALUE]
)
def test_the_work_that_must_be_right_is_never_routed_for_being_free(enabled, kind):
    router, _ = _router(Daemon())
    assert router.select(kind).model.provider == "anthropic"


def test_a_spent_budget_steps_down_to_free_instead_of_stopping(enabled):
    """What the local tier is actually for: the ceiling stops being a wall."""
    router, _ = _router(Daemon())
    paid = router.select(TaskKind.ANALYSIS)
    assert paid.model.provider == "anthropic"

    # Spend most of today's budget for real, through the ledger's own API: at
    # $10 per million output tokens that is 950k tokens for $9.50 of a $10 day.
    router.cost.record(
        model="claude-sonnet-5",
        provider="anthropic",
        input_tokens=0,
        output_tokens=950_000,
    )
    assert router.cost.budget_pressure() >= 0.7
    stepped = router.select(TaskKind.ANALYSIS)
    assert stepped.model.provider == "ollama"
    assert stepped.downgraded is True
    assert "budget" in stepped.reason


async def test_a_local_run_is_recorded_at_zero_and_spends_no_budget(enabled):
    router, _ = _router(Daemon())
    before = router.cost.spent_today()
    response = await router.complete(
        ModelRequest(messages=[Message(role="user", content="classify this")]),
        kind=TaskKind.SIMPLE,
    )
    assert response.provider == "ollama"
    record = router.cost.records[-1]
    assert record.provider == "ollama"
    assert record.cost_usd == 0.0
    assert record.input_tokens == 310, "local token counts were not recorded"
    assert router.cost.spent_today() == before


async def test_a_local_model_needs_no_credential_at_all(enabled, monkeypatch):
    """Not "a key that happens to be unset": no key is part of the protocol."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    daemon = Daemon()
    live = provider(daemon)
    assert live.available() is True
    response = await live.complete(ModelRequest(messages=[Message(role="user", content="hi")]))
    assert response.provider == "ollama"
    # And nothing resembling a credential was sent, because there is none to
    # send: the request body is the model, the messages and the options.
    sent = daemon.bodies[-1]
    assert set(sent) == {"model", "messages", "stream", "options"}


# -- the default wiring -----------------------------------------------------


def test_the_default_router_does_not_invent_a_local_tier(monkeypatch):
    """With the switch off — as in CI — nothing local reaches the catalogue."""
    from jarvis_core.models.router import build_default_router

    monkeypatch.setenv("OLLAMA_ENABLED", "0")
    reset_settings_cache()
    router = build_default_router()
    assert "ollama" not in router.providers
    assert [s for s in router.catalog.all() if s.provider == "ollama"] == []
