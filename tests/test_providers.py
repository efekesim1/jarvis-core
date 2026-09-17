"""Four Chat Completions vendors, and the honest label on each.

With one provider every model in the catalogue succeeds and fails together, so
`docs/JARVIS_OS_STATUS.md` has said ROUTER QUALITY UNMEASURED since it was written:
there is nothing to compare a routing decision against. NVIDIA NIM speaks the
Chat Completions protocol, so a second vendor costs a base URL rather than a
second provider implementation.

What these tests prove: the provider is configured from the environment only,
registers only when it is configured, sends what the protocol expects, reads
tokens back, and reports itself as `nvidia` in the cost ledger.

What they do NOT prove, and must not be read as proving: that the live endpoint
answers. This container's network policy blocks `integrate.api.nvidia.com`
(GitHub resolves, that host does not), so the remote half is untested here — it
sits at OFFLINE_TESTED on the same ladder every agent is held to, and
`jarvis smoke` on a machine without that restriction is what moves it.
"""

from __future__ import annotations

import pytest

from jarvis_core.models.base import Message, ModelRequest, ModelUnavailable
from jarvis_core.models.catalog import default_catalog
from jarvis_core.models.openai_provider import (
    GeminiProvider,
    GrokProvider,
    NvidiaProvider,
    OpenAIProvider,
)


class FakeCompletions:
    """Records what the provider sent, and answers as the API would."""

    def __init__(self, recorder):
        self._recorder = recorder

    async def create(self, **kwargs):
        self._recorder.append(kwargs)

        class Usage:
            prompt_tokens = 120
            completion_tokens = 45

        class Message_:
            content = "  an answer  "

        class Choice:
            message = Message_()
            finish_reason = "stop"

        class Completion:
            model = kwargs["model"]
            choices = [Choice()]
            usage = Usage()

        return Completion()


class FakeClient:
    def __init__(self, recorder):
        self.chat = type("Chat", (), {"completions": FakeCompletions(recorder)})()


# -- configuration comes from the environment, never from source --------------


def test_the_provider_reads_its_credential_from_settings(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    from jarvis_core.config import get_settings

    get_settings.cache_clear()
    try:
        assert NvidiaProvider().available()
    finally:
        get_settings.cache_clear()


def test_without_a_key_it_is_simply_not_available(monkeypatch):
    """An absent credential is an absent provider, not one that fails later."""
    assert NvidiaProvider(api_key=None, model="m").available() is False


def test_without_a_named_model_it_is_not_available():
    """A catalogue of hundreds changes faster than this code should."""
    assert NvidiaProvider(api_key="nvapi-x", model=None).available() is False


def test_no_credential_appears_anywhere_in_the_source():
    """Rule 1, checked rather than assumed.

    Looks for a key *shape*, not the bare `nvapi-` prefix. The prefix now
    legitimately appears in `logging.SECRET_PATTERNS`, where it is followed by a
    regex character class rather than a key — and a scanner that cannot tell a
    pattern definition from a credential is a scanner that gets softened the
    first time it fires. The same mistake, one layer up: the fix is to make the
    check shape-aware, never to loosen what it checks.
    """
    import re
    from pathlib import Path

    # A real key: the prefix followed by at least twenty key characters and
    # nothing that belongs to a regex.
    key_shaped = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")
    source = Path(__file__).resolve().parents[1] / "src" / "jarvis_core"
    for path in source.rglob("*.py"):
        found = key_shaped.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path}: {found[0][:12]}..."


# -- it identifies itself correctly ------------------------------------------


def test_it_reports_itself_as_nvidia_not_as_openai():
    """The cost ledger groups by provider; two vendors under one name is a lie."""
    assert NvidiaProvider(api_key="k", model="m").name == "nvidia"
    assert OpenAIProvider(api_key="k", model="m").name == "openai"


def test_its_error_names_the_provider_that_is_unconfigured():
    provider = NvidiaProvider(api_key=None, model=None)
    with pytest.raises(ModelUnavailable, match="nvidia"):
        provider._get_client()


# -- the base URL is what makes it a different vendor ------------------------


def test_the_base_url_is_nvidias_by_default(monkeypatch):
    from jarvis_core.config import get_settings

    provider = NvidiaProvider(api_key="k", model="m")
    assert provider._base_url == get_settings().nvidia_base_url
    assert "nvidia" in provider._base_url


def test_the_openai_provider_sends_no_base_url_so_it_reaches_openai():
    """Sharing the class must not quietly redirect the OpenAI provider."""
    assert OpenAIProvider(api_key="k", model="m")._base_url is None


# -- the protocol ------------------------------------------------------------


async def test_a_system_prompt_becomes_a_system_message():
    sent: list[dict] = []
    provider = NvidiaProvider(api_key="k", model="meta/llama-3.3-70b-instruct")
    provider._client = FakeClient(sent)

    await provider.complete(
        ModelRequest(
            messages=[Message(role="user", content="hello")],
            system="you are terse",
            max_tokens=256,
        )
    )
    assert sent[0]["messages"][0] == {"role": "system", "content": "you are terse"}
    assert sent[0]["messages"][1]["content"] == "hello"
    assert sent[0]["max_completion_tokens"] == 256


async def test_tokens_come_back_under_the_chat_completions_names():
    """prompt_tokens/completion_tokens, not input_tokens/output_tokens."""
    provider = NvidiaProvider(api_key="k", model="m")
    provider._client = FakeClient([])
    response = await provider.complete(
        ModelRequest(messages=[Message(role="user", content="x")], max_tokens=8)
    )
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 45
    assert response.usage.total_tokens == 165
    assert response.provider == "nvidia"
    assert response.text == "an answer"


async def test_the_requested_model_overrides_the_configured_one():
    sent: list[dict] = []
    provider = NvidiaProvider(api_key="k", model="configured")
    provider._client = FakeClient(sent)
    await provider.complete(
        ModelRequest(messages=[Message(role="user", content="x")], model="asked-for", max_tokens=8)
    )
    assert sent[0]["model"] == "asked-for"


# -- the catalogue -----------------------------------------------------------


def test_the_catalogue_carries_an_nvidia_model():
    nvidia = [m for m in default_catalog().all() if m.provider == "nvidia"]
    assert nvidia, "no NVIDIA model in config/models.json"


def test_no_nvidia_model_claims_server_side_tools():
    """Chat Completions has no equivalent of Anthropic's web search.

    An agent that needs one must never be routed here, and the router filters on
    this flag — so a wrong value would send a research agent to a model that
    cannot search and then blame the agent for citing nothing.
    """
    for model in default_catalog().all():
        if model.provider == "nvidia":
            assert model.supports_server_tools is False


def test_an_unpriced_model_is_recorded_as_zero_rather_than_invented():
    """The free tier is not metered per token, so a made-up price would be fiction.

    0.0 means "this model reports no cost", which is a different claim from
    "this model is free" and the only one the data supports.
    """
    for model in default_catalog().all():
        if model.provider == "nvidia":
            assert model.cost_usd(1_000_000, 1_000_000) == 0.0


# -- registration ------------------------------------------------------------


def test_the_default_router_registers_it_only_when_configured(monkeypatch):
    from jarvis_core.config import get_settings
    from jarvis_core.models.router import build_default_router

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    get_settings.cache_clear()
    try:
        router = build_default_router()
        assert "nvidia" not in router.providers
    finally:
        get_settings.cache_clear()


def test_the_router_can_reach_it_once_a_key_and_a_model_exist(monkeypatch):
    from jarvis_core.config import get_settings
    from jarvis_core.models.router import build_default_router

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    get_settings.cache_clear()
    try:
        router = build_default_router()
        assert "nvidia" in router.providers
        usable = {spec.provider for spec in router.usable_models()}
        assert "nvidia" in usable
    finally:
        get_settings.cache_clear()


# -- and the honest label ----------------------------------------------------


def test_the_live_endpoint_is_documented_as_unreached_rather_than_claimed():
    """The distinction this whole repository is built on.

    The code path above is tested. The remote endpoint is not, because this
    container's network policy blocks the host. Saying so in the status document
    is what keeps "we wrote a provider" from becoming "we have two vendors".
    """
    from pathlib import Path

    # Whichever document this distribution keeps its honest label in. The
    # filename is not the point — the claim being written down somewhere is —
    # and hard-coding one name made this test fail in a distribution that
    # keeps the same statement in a different file.
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "docs" / "JARVIS_OS_STATUS.md",
        root / "docs" / "CURRENT_STATE.md",
        root / "README.md",
    ]
    documents = [p.read_text(encoding="utf-8") for p in candidates if p.is_file()]
    assert documents, "no status document to check the claim against"
    assert any("NVIDIA" in text for text in documents), (
        "no document mentions the second vendor at all"
    )


# -- the other two Chat Completions vendors ----------------------------------
#
# xAI and Gemini arrive the same way NVIDIA did: a base URL, a credential and a
# name, over one shared request format. Parameterised together on purpose — a
# test that only covered one of three would let the next one ship with the
# OpenAI base URL and nobody would notice until a call went to the wrong vendor.
#
# The same honest label applies to all three: the code path is tested, the live
# endpoint is not. Grok and Gemini have never answered, from here or anywhere
# this system can see.

VENDORS = [
    (NvidiaProvider, "nvidia", "nvidia"),
    (GrokProvider, "grok", "x.ai"),
    (GeminiProvider, "gemini", "googleapis"),
]


@pytest.mark.parametrize(("cls", "name", "host_hint"), VENDORS)
def test_each_vendor_names_itself_and_points_somewhere_of_its_own(cls, name, host_hint):
    provider = cls(api_key="k", model="m")
    assert provider.name == name
    assert host_hint in provider._base_url


@pytest.mark.parametrize(("cls", "name", "host_hint"), VENDORS)
def test_no_vendor_inherits_the_openai_endpoint(cls, name, host_hint):
    """The failure this catches: a new subclass that forgets its base URL."""
    assert cls(api_key="k", model="m")._base_url is not None
    assert OpenAIProvider(api_key="k", model="m")._base_url is None


@pytest.mark.parametrize(("cls", "name", "host_hint"), VENDORS)
def test_a_vendor_without_both_a_key_and_a_model_is_not_available(cls, name, host_hint):
    assert cls(api_key=None, model="m").available() is False
    assert cls(api_key="k", model=None).available() is False


@pytest.mark.parametrize(("cls", "name", "host_hint"), VENDORS)
async def test_every_vendor_reads_tokens_back_the_same_way(cls, name, host_hint):
    """One usage shape for all of them; that is why they share a class."""
    provider = cls(api_key="k", model="m")
    provider._client = FakeClient([])
    response = await provider.complete(
        ModelRequest(messages=[Message(role="user", content="x")], max_tokens=8)
    )
    assert response.usage.total_tokens == 165
    assert response.provider == name


@pytest.mark.parametrize(("vendor", "model_id"), [("grok", "XAI"), ("gemini", "GEMINI")])
def test_the_router_reaches_a_vendor_once_its_key_and_model_exist(monkeypatch, vendor, model_id):
    from jarvis_core.config import get_settings
    from jarvis_core.models.catalog import default_catalog
    from jarvis_core.models.router import build_default_router

    catalogued = [m for m in default_catalog().all() if m.provider == vendor]
    assert catalogued, f"no {vendor} model in config/models.json"

    monkeypatch.setenv(f"{model_id}_API_KEY", "test-key-value")
    monkeypatch.setenv(f"{model_id}_MODEL", catalogued[0].id)
    get_settings.cache_clear()
    try:
        router = build_default_router()
        assert vendor in router.providers
        assert vendor in {spec.provider for spec in router.usable_models()}
    finally:
        get_settings.cache_clear()


def test_no_chat_completions_vendor_claims_server_side_tools():
    """Chat Completions has no equivalent of Anthropic's web search.

    The router filters on this flag, so a wrong value would route a research
    agent to a model that cannot search and then blame the agent for citing
    nothing.
    """
    from jarvis_core.models.catalog import default_catalog

    for model in default_catalog().all():
        if model.provider != "anthropic":
            assert model.supports_server_tools is False, model.id


def test_a_router_with_four_vendors_still_routes_by_tier_not_by_vendor(monkeypatch):
    """Offline proof that adding vendors did not change the selection rule."""
    from jarvis_core.config import get_settings
    from jarvis_core.models.router import TaskKind, build_default_router

    for vendor in ("NVIDIA", "XAI", "GEMINI"):
        monkeypatch.setenv(f"{vendor}_API_KEY", "test-key-value")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    monkeypatch.setenv("XAI_MODEL", "grok-4-fast")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-value")
    get_settings.cache_clear()
    try:
        router = build_default_router()
        assert len(router.providers) >= 4
        # Planning still draws the frontier tier, and only Anthropic has one.
        assert router.select(kind=TaskKind.PLANNING).model.tier == "frontier"
        # Work needing a server tool must not land on a vendor without one.
        chosen = router.select(kind=TaskKind.RESEARCH, needs_server_tools=True).model
        assert chosen.supports_server_tools
    finally:
        get_settings.cache_clear()
