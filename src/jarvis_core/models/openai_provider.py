"""The OpenAI-compatible providers: the second opinion, and the third.

Used for independent review, so a mistake in one vendor's model does not go
unchallenged. The model id is configured, not hard-coded, because model names
change faster than this code should.

`NvidiaProvider` is here rather than in its own file because NVIDIA's NIM
endpoint speaks the Chat Completions protocol: the only differences are the base
URL, the credential and the provider name. A separate copy of this file would
have been a second place for the same bug.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import get_settings
from ..logging import get_logger
from .base import ModelRequest, ModelResponse, ModelUnavailable, Usage

log = get_logger("models.openai")


class OpenAIProvider:
    """Calls the OpenAI Chat Completions API through the official SDK."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._client: Any = None

    def available(self) -> bool:
        """Requires both a key and an explicitly chosen model id."""
        if not self._api_key or not self._model:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelUnavailable(
                "the 'openai' package is not installed; run: uv pip install openai"
            ) from exc
        if not self._api_key:
            raise ModelUnavailable(f"no API key is set for the {self.name} provider")
        kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self._timeout_s}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        model = request.model or self._model
        if not model:
            raise ModelUnavailable(f"no model configured for the {self.name} provider")
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(m.as_dict() for m in request.messages)

        started = time.perf_counter()
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=request.max_tokens,
        )
        choice = completion.choices[0]
        raw_usage = getattr(completion, "usage", None)
        return ModelResponse(
            text=(choice.message.content or "").strip(),
            model=getattr(completion, "model", model),
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=getattr(choice, "finish_reason", None),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=completion,
        )


class NvidiaProvider(OpenAIProvider):
    """Calls NVIDIA's NIM endpoint, which speaks Chat Completions.

    A second vendor is the only thing that can turn "the router picks a tier"
    into "the router picks the better answer". With one provider, every routing
    decision is a decision between models that fail and succeed together, and no
    comparison is possible — which is why `docs/JARVIS_OS_STATUS.md` has said
    ROUTER QUALITY UNMEASURED since it was written.

    It inherits everything but the three things that differ: where to send the
    request, what to authenticate with, and what to call itself in the cost
    ledger. `supports_server_tools` is False for every NVIDIA model in the
    catalogue, because the Chat Completions protocol has no equivalent of
    Anthropic's server-side web search, and an agent that needs one must not be
    routed here.
    """

    name = "nvidia"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        settings = get_settings()
        super().__init__(
            api_key=api_key or settings.nvidia_api_key,
            model=model or settings.nvidia_model,
            base_url=base_url or settings.nvidia_base_url,
            timeout_s=timeout_s,
        )


class GrokProvider(OpenAIProvider):
    """Calls xAI's API, which speaks Chat Completions.

    Same shape as ``NvidiaProvider``: a base URL, a credential and a name. The
    three of them together are the argument for keeping these in one file —
    three copies of the request-building and usage-reading code would be three
    places for the same bug, and the Chat Completions surface is the same for
    all of them.
    """

    name = "grok"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        settings = get_settings()
        super().__init__(
            api_key=api_key or settings.xai_api_key,
            model=model or settings.xai_model,
            base_url=base_url or settings.xai_base_url,
            timeout_s=timeout_s,
        )


class GeminiProvider(OpenAIProvider):
    """Calls Gemini through Google's OpenAI-compatibility endpoint.

    Gemini has its own native API, and this deliberately does not use it. The
    compatibility endpoint speaks Chat Completions, which means Gemini costs a
    base URL here instead of a fourth request format, a fourth usage shape and a
    fourth set of error strings to teach `classify_fault` about.

    The trade is real and worth stating: a compatibility layer lags the native
    API, so Gemini-specific features (its own safety settings, its file API,
    its longer context modes) are not reachable this way. Nothing in this
    system asks for them yet, and when something does, a native provider is a
    new class rather than a rewrite of this one.
    """

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        settings = get_settings()
        super().__init__(
            api_key=api_key or settings.gemini_api_key,
            model=model or settings.gemini_model,
            base_url=base_url or settings.gemini_base_url,
            timeout_s=timeout_s,
        )
