"""One small call, before the twenty-seven expensive ones.

A real sweep without this check fails twenty-seven times for one reason and
records twenty-seven agent failures. Every one of them is a lie about the
agent: the prompt was never read, the schema was never tested, and the column
of red says the roster is broken when the account has no credit.

So the sweep asks one question first, with the cheapest model and a handful of
tokens: *can this system reach a model at all?* The answer is one of six
states, and the four unhappy ones are told apart because they need four
different things from four different people:

* ``NO_CREDENTIAL`` — nothing to authenticate with. Set ANTHROPIC_API_KEY.
* ``INVALID_CREDENTIAL`` — the key was rejected at the edge (HTTP 401, and no
  request id, because the request never got far enough to have one).
* ``NO_CREDIT`` — the key is **valid** and the account cannot pay (HTTP 400
  with a billing message, and a real request id, because the request was
  authenticated and then refused). This is the state that most needs its own
  name: it is indistinguishable from a broken key to anyone reading "the real
  sweep failed", and the fix is a payment method rather than a credential.
* ``RATE_LIMITED`` — reachable, and not now.
* ``UNREACHABLE`` — network, proxy, timeout, or the SDK missing.

None of these is an agent failure and none of them is recorded as one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..logging import AuditEvent, AuditLog, get_logger
from ..models.base import Message, ModelRequest, ModelUnavailable, ProviderFault, classify_fault

log = get_logger("verification.preflight")

# The smallest question that still proves the whole path: authentication,
# connectivity, a parsed response, a token count and a price.
PROBE_PROMPT = "Reply with exactly: OK"
PROBE_MAX_TOKENS = 16


class Reachability(StrEnum):
    """Whether a real model can be reached, and if not, whose problem it is."""

    OK = "ok"
    NO_CREDENTIAL = "no_credential"
    INVALID_CREDENTIAL = "invalid_credential"
    NO_CREDIT = "no_credit"
    RATE_LIMITED = "rate_limited"
    UNREACHABLE = "unreachable"

    @property
    def can_spend(self) -> bool:
        return self is Reachability.OK


# What each state means and what closes it. Kept next to the enum so the CLI,
# the doctor and the status document all say the same sentence.
REMEDY: dict[Reachability, str] = {
    Reachability.OK: "",
    Reachability.NO_CREDENTIAL: (
        "no API key. Put ANTHROPIC_API_KEY in jarvis/.env — see .env.example"
    ),
    Reachability.INVALID_CREDENTIAL: (
        "the API key was rejected. Create a new one at console.anthropic.com "
        "and replace ANTHROPIC_API_KEY in jarvis/.env"
    ),
    Reachability.NO_CREDIT: (
        "the API key is valid and the account has no credit. Add a payment "
        "method or buy credits at console.anthropic.com under Plans & Billing. "
        "Nothing in this repository can do that, and nothing should try"
    ),
    Reachability.RATE_LIMITED: "rate limited right now; wait and run it again",
    Reachability.UNREACHABLE: "the API could not be reached at all",
}

# The provider fault the models layer read, mapped to what it means for a
# person about to run a sweep. The substrings live down there, in one place,
# because the router needs the same reading to decide whether another model is
# worth trying — and two copies of one fact is how the last matching bug got
# in.
_FROM_FAULT: dict[ProviderFault, Reachability] = {
    ProviderFault.BILLING: Reachability.NO_CREDIT,
    ProviderFault.AUTH: Reachability.INVALID_CREDENTIAL,
    ProviderFault.RATE_LIMIT: Reachability.RATE_LIMITED,
    ProviderFault.TRANSIENT: Reachability.UNREACHABLE,
}


@dataclass(slots=True)
class PreflightResult:
    """What the one probe call established, and what it consumed doing it."""

    state: Reachability
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    request_id: str = ""
    text: str = ""
    error: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def ok(self) -> bool:
        return self.state.can_spend

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remedy(self) -> str:
        return REMEDY[self.state]

    def render(self) -> str:
        if self.ok:
            return (
                f"reachable: {self.model} answered in {self.latency_ms:.0f}ms, "
                f"{self.total_tokens} tokens, ${self.cost_usd:.6f} (estimated from the "
                f"price list, not billed)"
            )
        line = f"{self.state.value.replace('_', ' ')}: {self.remedy}"
        return f"{line}\n  {self.error}" if self.error else line

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "ok": self.ok,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd_estimated": round(self.cost_usd, 8),
            "latency_ms": round(self.latency_ms, 1),
            "request_id": self.request_id,
            "error": self.error,
            "remedy": self.remedy,
            "at": self.at.isoformat(),
        }


def classify_error(exc: BaseException) -> Reachability:
    """Which state a provider exception describes.

    Read from the message rather than the exception class, because the same
    ``BadRequestError`` carries both "your credit balance is too low" and every
    ordinary malformed-request problem, and those are not the same news.
    """
    # A ModelError from the router wraps the provider's own words in its
    # message *and* keeps the original on __cause__. Read both: the wrapper
    # alone would classify a billing refusal as merely unreachable.
    for candidate in (exc, exc.__cause__):
        if candidate is None:
            continue
        fault = classify_fault(candidate)
        if fault is not ProviderFault.TRANSIENT:
            return _FROM_FAULT[fault]
    return Reachability.UNREACHABLE


_REQUEST_ID = re.compile(r"'?request_id'?\s*[:=]\s*'?(req_[A-Za-z0-9]+)")


def request_id_of(payload: Any) -> str:
    """The provider's own id for a call, when it gave one.

    A billing refusal carries one and an authentication failure does not, which
    is itself the evidence that the first was authenticated and the second was
    not. Worth recording for exactly that reason.

    Four places to look, because the router wraps a provider exception in its
    own and the id can survive in any of them — including only inside the
    message text, which is why there is a regex here rather than four getattrs.
    """
    for candidate in (payload, getattr(payload, "__cause__", None)):
        if candidate is None:
            continue
        for attribute in ("_request_id", "request_id"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, str) and value:
                return value
        response = getattr(candidate, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers is not None:
            value = headers.get("request-id") or headers.get("x-request-id")
            if isinstance(value, str) and value:
                return value
    if match := _REQUEST_ID.search(str(payload)):
        return match.group(1)
    return ""


async def preflight(
    router: Any, *, model: str = "", audit: AuditLog | None = None
) -> PreflightResult:
    """Ask one cheap question, and report what came back.

    ``router`` is the real router. The probe deliberately goes through it
    rather than around it, so that a misconfigured router is caught here too
    rather than twenty-seven calls later.
    """
    usable = router.usable_models()
    if not usable:
        return PreflightResult(
            state=Reachability.NO_CREDENTIAL,
            error="no model provider is configured",
        )

    # The cheapest usable model. A reachability probe should not be billed at
    # frontier rates, and any model answering proves the same thing.
    chosen = model or min(usable, key=lambda s: s.input_cost_per_mtok + s.output_cost_per_mtok).id

    request = ModelRequest(
        messages=[Message(role="user", content=PROBE_PROMPT)],
        max_tokens=PROBE_MAX_TOKENS,
        model=chosen,
        metadata={"purpose": "preflight"},
    )
    started = time.perf_counter()
    try:
        response = await router.complete(request, allowed_models=[chosen], max_attempts=1)
    except (ModelUnavailable, Exception) as exc:  # noqa: BLE001 - every failure is an answer
        state = classify_error(exc)
        result = PreflightResult(
            state=state,
            model=chosen,
            latency_ms=(time.perf_counter() - started) * 1000,
            request_id=request_id_of(exc),
            # The message can carry the account's own words back. It never
            # carries the key: the SDK does not put it in an error, and the
            # audit log redacts anything that looks like one regardless.
            error=str(exc)[:400],
        )
        log.warning("preflight: %s", result.render())
    else:
        result = PreflightResult(
            state=Reachability.OK,
            model=response.model or chosen,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=router.cost.price(
                response.model or chosen,
                response.usage.input_tokens,
                response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
            ),
            latency_ms=response.latency_ms,
            request_id=request_id_of(response.raw),
            text=response.text[:200],
        )
        log.info("preflight: %s", result.render())

    if audit is not None:
        audit.record(
            AuditEvent(
                action="verification.preflight",
                actor="verifier",
                subject=chosen,
                result=result.state.value,
                reason=result.remedy or "reachable",
            )
        )
    return result
