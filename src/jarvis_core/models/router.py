"""The Model Router.

Not every job deserves the strongest model. The router picks one from the task
kind, the risk, how loaded today's budget already is, and which providers are
actually reachable. It is also the single place that records cost and retries
a failed call on an alternative model.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..config import Settings, get_settings
from ..logging import AuditLog, get_logger
from ..observability.cost import BudgetExceeded, CostEngine
from .base import (
    ModelError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelUnavailable,
    TextSink,
    classify_fault,
    estimate_request_tokens,
)
from .catalog import TIERS, ModelCatalog, ModelSpec, default_catalog

log = get_logger("models.router")


class TaskKind(StrEnum):
    """What sort of work a model call is doing."""

    SIMPLE = "simple"  # classification, extraction, routing
    RESEARCH = "research"  # gathering and summarizing sources
    ANALYSIS = "analysis"  # reasoning over gathered material
    PLANNING = "planning"  # decomposing a request into a task graph
    CODING = "coding"
    REVIEW = "review"  # an independent second opinion
    SYNTHESIS = "synthesis"  # merging many agent outputs into one answer
    HIGH_VALUE = "high_value"  # correctness matters more than cost


# Which model tier each kind of work wants, before budget pressure is applied.
_TIER_BY_KIND: dict[TaskKind, str] = {
    TaskKind.SIMPLE: "fast",
    TaskKind.RESEARCH: "balanced",
    TaskKind.ANALYSIS: "balanced",
    TaskKind.PLANNING: "frontier",
    TaskKind.CODING: "frontier",
    TaskKind.REVIEW: "balanced",
    TaskKind.SYNTHESIS: "frontier",
    TaskKind.HIGH_VALUE: "frontier",
}

_TIER_ORDER = TIERS

# Above this share of the daily budget, step down one tier.
BUDGET_STEPDOWN_AT = 0.7


@dataclass(slots=True)
class RoutingDecision:
    """Which model was chosen and why."""

    model: ModelSpec
    reason: str
    kind: TaskKind
    downgraded: bool = False


class ModelRouter:
    """Chooses a model, calls its provider, prices the call, and retries."""

    def __init__(
        self,
        *,
        providers: dict[str, ModelProvider] | None = None,
        catalog: ModelCatalog | None = None,
        cost: CostEngine | None = None,
        settings: Settings | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or default_catalog()
        self.cost = cost or CostEngine(settings=self.settings, catalog=self.catalog)
        self.audit = audit or AuditLog()
        self.providers: dict[str, ModelProvider] = dict(providers or {})

    # -- registration -----------------------------------------------------

    def register_provider(self, provider: ModelProvider) -> None:
        self.providers[provider.name] = provider

    def provider_for(self, spec: ModelSpec) -> ModelProvider | None:
        provider = self.providers.get(spec.provider)
        if provider is not None and provider.available():
            return provider
        return None

    def usable_models(self) -> list[ModelSpec]:
        """Models whose provider is registered and configured."""
        return [s for s in self.catalog.all() if self.provider_for(s) is not None]

    # -- selection --------------------------------------------------------

    def _tier_for(self, kind: TaskKind) -> str:
        tier = _TIER_BY_KIND.get(kind, "balanced")
        # High-value work keeps its tier even when the budget is tight; the
        # cost engine's hard limit is what stops it.
        if kind is TaskKind.HIGH_VALUE:
            return tier
        if self.cost.budget_pressure() >= BUDGET_STEPDOWN_AT:
            index = _TIER_ORDER.index(tier)
            return _TIER_ORDER[min(index + 1, len(_TIER_ORDER) - 1)]
        return tier

    def select(
        self,
        kind: TaskKind = TaskKind.ANALYSIS,
        *,
        allowed_models: list[str] | None = None,
        needs_server_tools: bool = False,
        prefer: str | None = None,
    ) -> RoutingDecision:
        """Pick the model for one call.

        ``allowed_models`` is the agent's own allow-list: an agent may never be
        routed to a model it does not permit.
        """
        pool = self.usable_models()
        if not pool:
            raise ModelUnavailable(
                "no model provider is configured; set ANTHROPIC_API_KEY or register a provider"
            )
        if allowed_models:
            allowed = set(allowed_models)
            restricted = [s for s in pool if s.id in allowed]
            if restricted:
                pool = restricted
            else:
                log.warning(
                    "none of the agent's allowed models are usable (%s); falling back to the full pool",
                    ", ".join(sorted(allowed)),
                )
        if needs_server_tools:
            with_tools = [s for s in pool if s.supports_server_tools]
            if with_tools:
                pool = with_tools

        if prefer:
            for spec in pool:
                if spec.id == prefer:
                    return RoutingDecision(spec, f"explicitly requested {prefer}", kind)
            log.warning("preferred model %s is not usable; routing normally", prefer)

        wanted_tier = self._tier_for(kind)
        downgraded = wanted_tier != _TIER_BY_KIND.get(kind, "balanced")

        # Walk from the wanted tier towards cheaper tiers until something exists.
        start = _TIER_ORDER.index(wanted_tier)
        for tier in _TIER_ORDER[start:]:
            candidates = [s for s in pool if s.tier == tier]
            if candidates:
                # Prefer a model whose declared strengths match the work.
                matched = [s for s in candidates if kind.value in s.strengths]
                chosen = self.catalog.cheapest(matched or candidates)
                assert chosen is not None
                reason = f"{kind.value} work routed to the {tier} tier"
                if downgraded:
                    reason += (
                        f" (stepped down: {self.cost.budget_pressure():.0%} of today's budget used)"
                    )
                return RoutingDecision(chosen, reason, kind, downgraded)

        chosen = self.catalog.cheapest(pool)
        assert chosen is not None
        return RoutingDecision(chosen, "no tier matched; using the cheapest usable model", kind)

    # -- execution --------------------------------------------------------

    async def complete(
        self,
        request: ModelRequest,
        *,
        kind: TaskKind = TaskKind.ANALYSIS,
        allowed_models: list[str] | None = None,
        agent: str | None = None,
        task_id: str | None = None,
        max_attempts: int = 2,
        on_text: TextSink | None = None,
    ) -> ModelResponse:
        """Route one request, priced and retried.

        On failure it tries the next usable model rather than hammering the one
        that just failed. Attempts are capped; there is no unbounded retry.

        ``on_text`` asks for the answer as it is produced. It changes how the
        text arrives and nothing else: selection, pricing, auditing, retry and
        the returned response are the same objects on the same code path,
        because a second path would be the one that drifts. A provider that
        cannot stream is handed the whole text once, so a caller never has to
        ask whether streaming is available.
        """
        self.cost.check_budgets(task_id)
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        estimated_input = estimate_request_tokens(
            request.system, "".join(m.content for m in request.messages)
        )
        needs_tools = bool(request.server_tools)
        tried: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(1, max(1, max_attempts) + 1):
            decision = self.select(
                kind,
                allowed_models=allowed_models,
                needs_server_tools=needs_tools,
                prefer=request.model if attempt == 1 else None,
            )
            spec = decision.model
            if spec.id in tried:
                # Selection is deterministic, so drop the failed model and re-pick.
                remaining = [
                    s.id
                    for s in self.usable_models()
                    if s.id not in tried and (not allowed_models or s.id in allowed_models)
                ]
                if not remaining:
                    break
                decision = self.select(
                    kind, allowed_models=remaining, needs_server_tools=needs_tools
                )
                spec = decision.model
            tried.add(spec.id)

            provider = self.provider_for(spec)
            if provider is None:  # pragma: no cover - usable_models already filtered
                continue

            call = ModelRequest(
                messages=request.messages,
                system=request.system,
                model=spec.id,
                max_tokens=request.max_tokens,
                effort=request.effort,
                server_tools=request.server_tools,
                temperature=request.temperature,
                metadata=request.metadata,
            )
            try:
                response = await self._call(provider, call, on_text)
            except BudgetExceeded:
                raise
            except Exception as exc:
                last_error = exc
                fault = classify_fault(exc)
                log.warning(
                    "model call failed on %s (attempt %d/%d, %s): %s",
                    spec.id,
                    attempt,
                    max_attempts,
                    fault.value,
                    exc,
                )
                self.audit.emit(
                    "model.call_failed",
                    agent=agent,
                    model=spec.id,
                    task_id=task_id,
                    result="error",
                    reason=str(exc),
                    metadata={"attempt": attempt, "request_id": request_id, "fault": fault.value},
                )
                # An expired key and an empty account fail identically on every
                # model in the catalogue. Walking the rest of it buys the same
                # refusal two more times and hides the first one behind a
                # message about attempts.
                if not fault.retryable:
                    break
                continue

            record = self.cost.record(
                model=response.model,
                provider=response.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                agent=agent,
                task_id=task_id,
                request_id=request_id,
            )
            self.audit.emit(
                "model.call",
                agent=agent,
                model=response.model,
                task_id=task_id,
                result=response.stop_reason or "ok",
                reason=decision.reason,
                metadata={
                    "request_id": request_id,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_tokens,
                    "estimated_input_tokens": estimated_input,
                    "cost_usd": round(record.cost_usd, 6),
                    "latency_ms": round(response.latency_ms, 1),
                    "attempt": attempt,
                },
            )
            return response

        # The reason has to travel in the message. Chaining it onto __cause__
        # keeps it for a debugger and loses it for every caller that reads
        # str(exc) — which is all of them, including the audit log and the
        # failure taxonomy. A router that reports "every attempt failed" and
        # drops "your credit balance is too low" turns one fixable fact into
        # twenty-seven unexplained agent failures.
        detail = f": {last_error}" if last_error is not None else ""
        raise ModelError(
            f"every model attempt failed (tried: {', '.join(sorted(tried)) or 'none'}){detail}"
        ) from last_error

    @staticmethod
    async def _call(
        provider: ModelProvider, call: ModelRequest, on_text: TextSink | None
    ) -> ModelResponse:
        """Make the call, streaming when both sides can.

        The fallback emits the finished text in one piece rather than
        pretending to stream it. A consumer sees one large delta instead of
        many small ones, which is honest about what happened and renders the
        same.
        """
        if on_text is None:
            return await provider.complete(call)
        streamer = getattr(provider, "stream", None)
        if streamer is None:
            response = await provider.complete(call)
            on_text(response.text)
            return response
        return await streamer(call, on_text)

    async def complete_many(
        self,
        requests: list[ModelRequest],
        *,
        kind: TaskKind = TaskKind.ANALYSIS,
        **kwargs: Any,
    ) -> list[ModelResponse | BaseException]:
        """Run several independent calls at once."""
        return await asyncio.gather(
            *(self.complete(r, kind=kind, **kwargs) for r in requests),
            return_exceptions=True,
        )


def build_default_router(
    *,
    settings: Settings | None = None,
    cost: CostEngine | None = None,
    audit: AuditLog | None = None,
) -> ModelRouter:
    """A router wired to whichever real providers are configured."""
    from .anthropic_provider import AnthropicProvider
    from .local_provider import OllamaProvider
    from .openai_provider import (
        GeminiProvider,
        GrokProvider,
        NvidiaProvider,
        OpenAIProvider,
    )

    settings = settings or get_settings()
    router = ModelRouter(settings=settings, cost=cost, audit=audit)
    # Each one registers only if it is configured, so an absent credential is an
    # absent provider rather than a provider that fails on first use.
    local = OllamaProvider(settings=settings)
    candidates = (
        AnthropicProvider(),
        OpenAIProvider(),
        NvidiaProvider(),
        GrokProvider(),
        GeminiProvider(),
        local,
    )
    for provider in candidates:
        if provider.available():
            router.register_provider(provider)
            log.info("registered model provider: %s", provider.name)
        else:
            log.info("model provider %s is not configured; skipping", provider.name)

    # The local tier brings its own catalogue. Every other provider draws on a
    # published price list that is the same on every machine; what Ollama can
    # serve is whatever this machine has pulled, so the specs are read from the
    # daemon rather than written down. They are registered only when the daemon
    # answered, which is also why nothing changes in an offline test run.
    if local.available():
        for spec in local.catalog_specs():
            router.catalog.register(spec)
            log.info("local model available at no cost: %s (tier %s)", spec.id, spec.tier)
    return router
