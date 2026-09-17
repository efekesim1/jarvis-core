"""The Anthropic provider, built on the official ``anthropic`` SDK."""

from __future__ import annotations

import time
from typing import Any

from ..config import get_settings
from ..logging import get_logger
from .base import ModelRequest, ModelResponse, ModelUnavailable, TextSink, Usage

log = get_logger("models.anthropic")

# Server-side tools, named by the shorthand agents use in their definitions.
SERVER_TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {"type": "web_search_20260209", "name": "web_search"},
    "web_fetch": {"type": "web_fetch_20260209", "name": "web_fetch"},
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
}

# Turns can pause while a server tool runs; resume this many times at most.
MAX_PAUSE_RESUMES = 5


class AnthropicProvider:
    """Calls the Claude Messages API.

    The API key comes from the environment. It is never read from a file in
    this repository and never written to a log.
    """

    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, timeout_s: float = 300.0) -> None:
        self._api_key = api_key or get_settings().anthropic_api_key
        self._timeout_s = timeout_s
        self._client: Any = None

    def available(self) -> bool:
        if self._api_key:
            return True
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        # The SDK also resolves an `ant auth login` profile, so absence of the
        # env var is not proof that there are no credentials.
        return True

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelUnavailable(
                "the 'anthropic' package is not installed; run: uv pip install anthropic"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": self._timeout_s}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "model": request.model or settings.default_model,
            "max_tokens": request.max_tokens,
            "messages": [m.as_dict() for m in request.messages],
        }
        if request.system:
            # An agent's system prompt is identical on every run, so caching it
            # turns a repeated cost into a near-free read. The volatile part of
            # the request lives in `messages`, after this prefix, which is what
            # keeps the cached prefix stable.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if request.effort:
            kwargs["output_config"] = {"effort": request.effort}
        tools = [SERVER_TOOLS[name] for name in request.server_tools if name in SERVER_TOOLS]
        unknown = [name for name in request.server_tools if name not in SERVER_TOOLS]
        if unknown:
            log.warning("ignoring unknown server tool(s): %s", ", ".join(unknown))
        if tools:
            kwargs["tools"] = tools
        return kwargs

    @staticmethod
    def _extract(message: Any) -> tuple[str, list[dict[str, Any]]]:
        """Pull plain text and any citations out of a response."""
        parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for block in getattr(message, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(block.text)
                for citation in getattr(block, "citations", None) or []:
                    citations.append(
                        {
                            "title": getattr(citation, "title", None),
                            "url": getattr(citation, "url", None),
                            "cited_text": getattr(citation, "cited_text", None),
                        }
                    )
            elif btype == "web_search_tool_result":
                content = getattr(block, "content", None)
                # On success this is a list of results; on error it is one object.
                if isinstance(content, list):
                    for result in content:
                        citations.append(
                            {
                                "title": getattr(result, "title", None),
                                "url": getattr(result, "url", None),
                                "cited_text": None,
                            }
                        )
                else:
                    code = getattr(content, "error_code", None)
                    if code:
                        log.warning("web search returned error: %s", code)
        return "\n".join(p for p in parts if p).strip(), citations

    async def stream(self, request: ModelRequest, on_text: TextSink) -> ModelResponse:
        """Answer one request, handing over text as it arrives.

        The finished :class:`ModelResponse` is what comes back, identical in
        shape to ``complete``'s, so the router prices and audits one kind of
        thing. ``on_text`` is best-effort reporting: an exception from a sink
        is logged and swallowed, because a paid call must not fail over a
        rendering problem.

        A paused turn is not resumed here. Server-side tools pause and resume
        the turn, which a single stream cannot represent, so a request that
        asks for them goes to ``complete`` instead and arrives in one piece.
        That is a slower first token, not a missing answer, and it is why
        Quick Chat — which has no server tools — is the streaming path.
        """
        if request.server_tools:
            response = await self.complete(request)
            self._feed(on_text, response.text)
            return response

        client = self._get_client()
        kwargs = self._build_kwargs(request)
        started = time.perf_counter()
        parts: list[str] = []
        usage = Usage()
        model = kwargs["model"]
        stop_reason: str | None = None

        async with client.messages.stream(**kwargs) as stream:
            async for delta in stream.text_stream:
                if delta:
                    parts.append(delta)
                    self._feed(on_text, delta)
            message = await stream.get_final_message()

        raw_usage = getattr(message, "usage", None)
        if raw_usage is not None:
            usage = Usage(
                input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
                output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
            )
        text, citations = self._extract(message)
        stop_reason = getattr(message, "stop_reason", None)
        return ModelResponse(
            # Prefer the assembled message over the concatenated deltas: they
            # agree, and the message is the thing the API considers final.
            text=text or "".join(parts).strip(),
            model=getattr(message, "model", model),
            provider=self.name,
            usage=usage,
            stop_reason=stop_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            citations=citations,
            raw=message,
        )

    @staticmethod
    def _feed(on_text: TextSink, chunk: str) -> None:
        """Hand a delta to the sink without letting it break the call."""
        if not chunk:
            return
        try:
            on_text(chunk)
        except Exception:  # pragma: no cover - a sink that raises is a bug elsewhere
            log.warning("a text sink raised; dropping the delta and continuing", exc_info=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        messages = list(kwargs["messages"])
        started = time.perf_counter()
        usage = Usage()
        text = ""
        citations: list[dict[str, Any]] = []
        message: Any = None

        for _ in range(MAX_PAUSE_RESUMES + 1):
            kwargs["messages"] = messages
            message = await client.messages.create(**kwargs)
            raw_usage = getattr(message, "usage", None)
            if raw_usage is not None:
                usage = usage + Usage(
                    input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
                    cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
                    cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
                )
            chunk, chunk_citations = self._extract(message)
            if chunk:
                text = f"{text}\n{chunk}".strip() if text else chunk
            citations.extend(chunk_citations)
            if getattr(message, "stop_reason", None) != "pause_turn":
                break
            # A server tool is still working. Hand the paused turn back so it continues.
            messages = [*messages, {"role": "assistant", "content": message.content}]
        else:
            log.warning("gave up resuming a paused turn after %d attempts", MAX_PAUSE_RESUMES)

        return ModelResponse(
            text=text,
            model=getattr(message, "model", kwargs["model"]),
            provider=self.name,
            usage=usage,
            stop_reason=getattr(message, "stop_reason", None),
            latency_ms=(time.perf_counter() - started) * 1000,
            citations=citations,
            raw=message,
        )
