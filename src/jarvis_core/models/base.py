"""The model provider abstraction.

Nothing above this module knows which company serves a request. A provider
takes a :class:`ModelRequest` and returns a :class:`ModelResponse`; swapping
Anthropic for OpenAI, a local model, or a future provider means registering a
new provider, not rewriting the system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

# Roughly four characters per token for English and Turkish prose. This is not
# a tokenizer: it exists to decide whether to trim a prompt before paying for
# it, and a 20% error either way does not change that answer. The exact count
# comes back from the provider afterwards.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """A cheap pre-flight token estimate."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_request_tokens(system: str | None, prompt: str) -> int:
    """Estimate what one request will cost in input tokens."""
    return estimate_tokens(system or "") + estimate_tokens(prompt)


class ModelError(RuntimeError):
    """A model call failed."""


class ModelUnavailable(ModelError):
    """The provider could not be reached or is not configured."""


class ModelRefusal(ModelError):
    """The model declined to answer."""


@dataclass(slots=True)
class Message:
    """One turn of a conversation."""

    role: str  # "user" or "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ModelRequest:
    """Everything a provider needs to answer one question."""

    messages: list[Message]
    system: str | None = None
    model: str | None = None
    max_tokens: int = 8000
    effort: str | None = None
    server_tools: list[str] = field(default_factory=list)
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single(cls, prompt: str, **kwargs: Any) -> ModelRequest:
        return cls(messages=[Message(role="user", content=prompt)], **kwargs)


@dataclass(slots=True)
class Usage:
    """Token counts for one call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(slots=True)
class ModelResponse:
    """What a provider gives back."""

    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    latency_ms: float = 0.0
    citations: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class ProviderFault(StrEnum):
    """Why a provider call failed, at the only granularity control flow needs.

    The router retries on another model, which helps for exactly one of these.
    An expired key and an empty account are the same on every model in the
    catalogue, so retrying them buys three identical refusals and three round
    trips. `TRANSIENT` is the only fault worth another attempt.

    Read from the message rather than the exception class on purpose: the same
    ``BadRequestError`` carries "your credit balance is too low" and every
    ordinary malformed-request problem, and those are not the same news.
    """

    AUTH = "auth"  # the credential was rejected
    BILLING = "billing"  # the credential is fine and the account cannot pay
    RATE_LIMIT = "rate_limit"  # reachable, and not now
    TRANSIENT = "transient"  # anything that another attempt might survive

    @property
    def retryable(self) -> bool:
        return self in (ProviderFault.TRANSIENT, ProviderFault.RATE_LIMIT)


# Ordered, because a billing refusal arrives as an HTTP 400 and so does a
# malformed request; the billing wording is what tells them apart.
_FAULT_MARKERS: tuple[tuple[ProviderFault, tuple[str, ...]], ...] = (
    (
        ProviderFault.BILLING,
        ("credit balance", "billing", "insufficient funds", "quota exceeded", "plans & billing"),
    ),
    (
        ProviderFault.AUTH,
        (
            "authentication_error",
            "api key is invalid",
            "invalid x-api-key",
            "invalid api key",
            "unauthorized",
            "permission_error",
        ),
    ),
    (ProviderFault.RATE_LIMIT, ("rate_limit", "rate limit", "overloaded", "too many requests")),
)


def classify_fault(exc: BaseException) -> ProviderFault:
    """Which fault an exception from a provider describes."""
    text = f"{type(exc).__name__} {exc}".lower()
    for fault, markers in _FAULT_MARKERS:
        if any(marker in text for marker in markers):
            return fault
    return ProviderFault.TRANSIENT


# Where a streaming provider puts text as it arrives. Synchronous and
# returning nothing on purpose: it is called from inside the provider's read
# loop, so anything that can block or raise there would stall or fail a paid
# call over a reporting problem. Sinks append to a queue and nothing else.
TextSink = Callable[[str], None]


@runtime_checkable
class ModelProvider(Protocol):
    """The one interface every model backend implements."""

    name: str

    def available(self) -> bool:
        """True when the provider is configured well enough to be called."""
        ...

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer one request."""
        ...


@runtime_checkable
class StreamingProvider(Protocol):
    """A provider that can hand over text before the answer is finished.

    Optional. The router checks for it at call time and falls back to
    ``complete`` plus one whole-text emission, so a provider that cannot
    stream is never a missing feature anywhere above this module — only a
    slower first token.

    ``stream`` returns the same :class:`ModelResponse` ``complete`` does. That
    is what keeps cost accounting on one code path: the router prices and
    audits the finished response identically whether it watched it arrive or
    waited for it.
    """

    name: str

    def available(self) -> bool: ...

    async def stream(self, request: ModelRequest, on_text: TextSink) -> ModelResponse:
        """Answer one request, calling ``on_text`` with each delta."""
        ...
