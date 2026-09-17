"""A deterministic in-process provider.

The whole system can be exercised end to end with this provider: no network,
no API key, no cost. Every test uses it, which is what makes the test suite
runnable on any machine and in CI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..logging import get_logger
from .base import ModelRequest, ModelResponse, TextSink, Usage

log = get_logger("models.mock")

Responder = Callable[[ModelRequest], str]


def _estimate_tokens(text: str) -> int:
    """A rough token count. Good enough for cost bookkeeping in tests."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


class MockProvider:
    """Answers from a scripted queue, a rule table, or a default echo."""

    name = "mock"

    def __init__(
        self,
        *,
        scripted: Sequence[str] | None = None,
        rules: dict[str, str] | None = None,
        responder: Responder | None = None,
        default_model: str = "claude-haiku-4-5",
        fail_times: int = 0,
    ) -> None:
        self._scripted = list(scripted or [])
        self._rules = dict(rules or {})
        self._responder = responder
        self._default_model = default_model
        self._fail_times = fail_times
        self.calls: list[ModelRequest] = []

    def available(self) -> bool:
        return True

    def _answer(self, request: ModelRequest) -> str:
        if self._responder is not None:
            return self._responder(request)
        if self._scripted:
            return self._scripted.pop(0)
        last = request.messages[-1].content if request.messages else ""
        for needle, reply in self._rules.items():
            if needle.lower() in last.lower():
                return reply
        return f"[mock:{request.model or self._default_model}] {last[:400]}"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("mock provider transient failure")
        text = self._answer(request)
        prompt_text = (request.system or "") + " ".join(m.content for m in request.messages)
        return ModelResponse(
            text=text,
            model=request.model or self._default_model,
            provider=self.name,
            usage=Usage(
                input_tokens=_estimate_tokens(prompt_text),
                output_tokens=_estimate_tokens(text),
            ),
            stop_reason="end_turn",
            latency_ms=0.0,
        )

    async def stream(self, request: ModelRequest, on_text: TextSink) -> ModelResponse:
        """The same answer, handed over in pieces.

        Streaming has to be exercised offline or the path that runs in front
        of the operator is the one path no test covers. The chunking is by whitespace
        rather than by character so a test can assert that several deltas
        arrived and that they reassemble into exactly the finished text.
        """
        response = await self.complete(request)
        for index, word in enumerate(response.text.split(" ")):
            on_text(word if index == 0 else f" {word}")
        return response


def agent_reply(summary: str = "a finding", confidence: float = 0.8, **extra: Any) -> str:
    """A well-formed agent response, for the mock provider to return.

    Lives here rather than in the test fixtures because verification needs the
    same thing at runtime: an offline pass has to produce output the real
    parser and the real evaluator will accept, or it proves nothing about
    either. Two copies of this would drift, and the drift would look like an
    agent regression.
    """
    payload = {"summary": summary, "confidence": confidence, **extra}
    return json.dumps(payload, ensure_ascii=False)
