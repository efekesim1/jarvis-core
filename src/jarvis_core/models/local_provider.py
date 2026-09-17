"""The local tier: whatever is already running on this machine.

Every other provider in this package is a bill. This one is a process. Ollama
serves models on `localhost`, needs no credential, costs no dollars and answers
with the network down, which changes what the router is choosing between: not
"which model is best" but "does this step need to be paid for at all".

Three decisions are worth stating, because each was a choice and not the only
option:

* **The standard library, not an SDK.** The cheapest path must not be the one
  that needs an install. `urllib` in a worker thread is enough for two POSTs
  and a GET, and it means the local tier works on a machine where `openai`
  and `anthropic` are both absent.
* **The daemon is asked what it has.** Key-bearing providers require their
  model named in the environment, because a vendor catalogue of hundreds
  changes faster than this code. A local daemon has a list, and reading it is
  one HTTP call, so every pulled model becomes a usable `$0` entry in the
  catalogue with nothing to configure. `OLLAMA_MODEL` still pins one when
  that is wanted.
* **Local models enter the `fast` tier and no higher.** An 8B model on a
  desktop is not a planning model, and a catalogue entry claiming otherwise
  would route planning to it and call the result frontier work. In `fast` it
  takes what `TaskKind.SIMPLE` asks for, and — because every tier steps down
  towards `fast` under budget pressure — it is also what the system falls back
  to when the paid budget is nearly spent. That is the useful property: the
  ceiling stops being a wall.

What this deliberately does not do is stream. The router hands a non-streaming
provider's text over in one piece, which is honest about what happened and
renders the same; calling a sink from the worker thread that does the HTTP
would touch the caller's event loop from off it, and a free token is not worth
that.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from ..config import Settings, get_settings
from ..logging import get_logger
from .base import ModelError, ModelRequest, ModelResponse, ModelUnavailable, Usage
from .catalog import ModelSpec, normalise_tier

log = get_logger("models.local")

# The endpoints this provider uses. Three, written out, because the set of
# things it can reach should be readable rather than derived from a request.
TAGS_PATH = "/api/tags"
CHAT_PATH = "/api/chat"

# A local model is slower per token than a hosted frontier one and has no
# queue to blame. Long, but not unbounded.
TIMEOUT_S = 300.0

# What the probe waits for. A daemon that is running answers `/api/tags` in
# milliseconds; one that is not refuses the connection immediately. This
# timeout only covers the case of something listening on the port that is not
# Ollama, and it is short because it is on the routing path.
PROBE_TIMEOUT_S = 2.0

# Understated on purpose. Ollama's default context is small and per-model, and
# `/api/tags` does not report it. The context budget trims a prompt to fit this
# number, so guessing low drops material visibly and guessing high truncates it
# silently inside the daemon.
DEFAULT_CONTEXT_TOKENS = 8192

# What the local tier claims to be good at. These are the names
# `ModelSpec.strengths` is matched against for simple, high-volume steps.
LOCAL_STRENGTHS = frozenset({"classification", "extraction", "summarization", "routing"})

# A transport takes a method, an absolute URL, an optional JSON body and a
# timeout, and returns the decoded response. Injectable so the tests exercise
# the argv-free request building, the usage mapping and the error taxonomy
# without a daemon, which is what keeps the suite offline.
Transport = Callable[[str, str, dict[str, Any] | None, float], dict[str, Any]]


def _request(method: str, url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    """One HTTP call to the daemon, with the standard library."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ModelUnavailable(f"local model transport only accepts http(s) URLs; got {url!r}")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(  # nosec B310 - scheme is validated above
        url, data=data, headers=headers, method=method
    )
    with urllib.request.urlopen(  # nosec B310 - scheme is validated above
        request, timeout=timeout
    ) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ModelError(f"the local daemon answered with {type(parsed).__name__}, not an object")
    return parsed


class OllamaProvider:
    """Calls a local Ollama daemon over its native API."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = (base_url or self._settings.ollama_base_url).rstrip("/")
        self._model = model or self._settings.ollama_model
        self._transport = transport or _request
        self._enabled = self._settings.ollama_enabled
        # The probe result and when it was taken. `available()` is consulted on
        # every routing decision; the daemon is not asked that often.
        self._probed_at = 0.0
        self._installed: tuple[str, ...] = ()
        self._warned_remote = False

    # -- where it points --------------------------------------------------

    def _url_for(self, path: str) -> str:
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme not in ("http", "https"):
            raise ModelUnavailable(
                f"OLLAMA_BASE_URL must be an http(s) URL; got {self._base_url!r}"
            )
        if not self._warned_remote and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            # Not refused: pointing this at another machine is a deliberate
            # setting, not something a request can do. Said once, because the
            # prompt then leaves this machine and that is worth knowing.
            log.warning(
                "the local model tier is pointed at %s, which is not this machine; "
                "prompts will be sent there",
                parsed.hostname,
            )
            self._warned_remote = True
        return f"{self._base_url}{path}"

    # -- what is actually here --------------------------------------------

    def _probe(self, *, force: bool = False) -> tuple[str, ...]:
        """Ask the daemon what it has, at most once every TTL."""
        now = time.monotonic()
        fresh = now - self._probed_at < self._settings.ollama_probe_ttl_s
        if not force and fresh:
            return self._installed
        self._probed_at = now
        if not self._enabled:
            self._installed = ()
            return self._installed
        try:
            payload = self._transport("GET", self._url_for(TAGS_PATH), None, PROBE_TIMEOUT_S)
        except ModelUnavailable:
            raise
        except Exception as exc:
            log.debug("no local model daemon at %s (%s)", self._base_url, exc)
            self._installed = ()
            return self._installed
        found: list[str] = []
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("model") or entry.get("name")
            if isinstance(name, str) and name:
                found.append(name)
        self._installed = tuple(sorted(found))
        return self._installed

    def installed_models(self, *, force: bool = False) -> tuple[str, ...]:
        """The models this daemon has pulled, newest probe first."""
        try:
            return self._probe(force=force)
        except ModelUnavailable:
            return ()

    def available(self) -> bool:
        """Reachable, enabled, and holding at least one usable model.

        All three, because a provider registered without them answers every
        call with the same error, and the router cannot tell "no daemon" from
        "a daemon with nothing pulled" once a call has already failed.
        """
        installed = self.installed_models()
        if not installed:
            return False
        if self._model:
            return self._model in installed
        return True

    def why_unavailable(self) -> str | None:
        if not self._enabled:
            return "the local model tier is switched off (OLLAMA_ENABLED=0)"
        installed = self.installed_models()
        if not installed:
            return (
                f"no Ollama daemon answered at {self._base_url}; install it from "
                "https://ollama.com, then run: ollama pull qwen2.5:7b"
            )
        if self._model and self._model not in installed:
            return (
                f"OLLAMA_MODEL is {self._model!r}, which is not pulled; "
                f"available: {', '.join(installed)}"
            )
        return None

    def catalog_specs(self) -> list[ModelSpec]:
        """A free catalogue entry per local model the router may use.

        Generated rather than written into `config/models.json`, because what
        is on one machine is not what is on another and a hand-maintained list
        would be wrong on both. Zero cost is the literal truth for the cost
        engine: a local call is not billed. The electricity is real and is not
        an API cost, which is why it is not invented here as one.
        """
        models = [self._model] if self._model else list(self.installed_models())
        tier = normalise_tier(self._settings.ollama_tier, default="fast")
        return [
            ModelSpec(
                id=model,
                provider=self.name,
                context_tokens=DEFAULT_CONTEXT_TOKENS,
                input_cost_per_mtok=0.0,
                output_cost_per_mtok=0.0,
                tier=tier,
                strengths=LOCAL_STRENGTHS,
                supports_server_tools=False,
                notes=f"Runs locally through Ollama at {self._base_url}. No key, no bill.",
            )
            for model in models
            if model
        ]

    # -- invocation -------------------------------------------------------

    async def complete(self, request: ModelRequest) -> ModelResponse:
        model = request.model or self._model or (self.installed_models() or (None,))[0]
        if not model:
            raise ModelUnavailable(self.why_unavailable() or "no local model is available")

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(m.as_dict() for m in request.messages)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": request.max_tokens},
        }
        if request.temperature is not None:
            body["options"]["temperature"] = request.temperature

        url = self._url_for(CHAT_PATH)
        started = time.perf_counter()
        try:
            payload = await asyncio.to_thread(self._transport, "POST", url, body, TIMEOUT_S)
        except urllib.error.URLError as exc:
            # A refused connection is the daemon being off, which is an absent
            # provider rather than a failed model. The distinction is what
            # `classify_fault` uses to decide whether another model is worth
            # trying, and another model usually is.
            self._probed_at = 0.0
            raise ModelUnavailable(
                f"the local model daemon at {self._base_url} did not answer: {exc}"
            ) from exc

        message = payload.get("message")
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        if not text and isinstance(payload.get("response"), str):
            # `/api/generate`'s shape, in case something answers with it.
            text = payload["response"]
        if not text:
            raise ModelError(f"the local model {model} returned no text")

        return ModelResponse(
            text=text.strip(),
            model=str(payload.get("model") or model),
            provider=self.name,
            usage=Usage(
                input_tokens=int(payload.get("prompt_eval_count") or 0),
                output_tokens=int(payload.get("eval_count") or 0),
            ),
            stop_reason=str(payload.get("done_reason") or "stop"),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=payload,
        )

    def summary(self) -> dict[str, Any]:
        installed = self.installed_models()
        return {
            "provider": self.name,
            "base_url": self._base_url,
            "enabled": self._enabled,
            "installed": list(installed),
            "pinned_model": self._model,
            "tier": self._settings.ollama_tier,
            "unavailable_because": self.why_unavailable(),
        }
