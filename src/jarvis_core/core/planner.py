"""The Planner.

Turns one sentence from the operator into a task graph: which agents run, in what
order, and which of them can run at the same time.

Two planners exist on purpose. The rule-based one is free, instant and always
available, and handles the shapes of request that repeat. The model planner
handles everything else. Simple requests must stay simple: a single lookup
should not spawn six agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..agents.registry import AgentRegistry
from ..logging import get_logger
from ..models.base import Message, ModelRequest
from ..models.router import ModelRouter, TaskKind
from ..text import terms

log = get_logger("core.planner")


class Intent(StrEnum):
    """What the request is really asking for."""

    RESEARCH = "research"
    BUSINESS = "business"
    BUILD = "build"
    QUESTION = "question"
    SYSTEM = "system"


@dataclass(slots=True)
class PlanStep:
    """One agent run inside a plan."""

    step_id: str
    agent_id: str
    goal: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "goal": self.goal,
            "inputs": dict(self.inputs),
            "depends_on": list(self.depends_on),
        }


@dataclass(slots=True)
class Plan:
    """The whole task graph for one request."""

    request: str
    intent: Intent
    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""
    synthesizer: str | None = None
    source: str = "rules"

    def step(self, step_id: str) -> PlanStep | None:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def validate(self, registry: AgentRegistry) -> list[str]:
        """Return every problem with this plan. An empty list means it is sound."""
        problems: list[str] = []
        ids = [s.step_id for s in self.steps]
        if not ids:
            problems.append("the plan has no steps")
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            problems.append(f"duplicate step ids: {', '.join(sorted(duplicates))}")
        known = set(ids)
        for step in self.steps:
            if registry.get(step.agent_id) is None:
                problems.append(f"step {step.step_id} names unknown agent {step.agent_id!r}")
            for dep in step.depends_on:
                if dep not in known:
                    problems.append(f"step {step.step_id} depends on unknown step {dep!r}")
                if dep == step.step_id:
                    problems.append(f"step {step.step_id} depends on itself")
        if cycle := self._find_cycle():
            problems.append(f"the plan has a dependency cycle: {' -> '.join(cycle)}")
        return problems

    def _find_cycle(self) -> list[str] | None:
        graph = {s.step_id: list(s.depends_on) for s in self.steps}
        state: dict[str, int] = {}  # 0 unvisited, 1 in progress, 2 done
        path: list[str] = []

        def visit(node: str) -> list[str] | None:
            if state.get(node, 0) == 1:
                start = path.index(node)
                return [*path[start:], node]
            if state.get(node, 0) == 2:
                return None
            state[node] = 1
            path.append(node)
            for dep in graph.get(node, []):
                if dep in graph and (found := visit(dep)):
                    return found
            path.pop()
            state[node] = 2
            return None

        for node in graph:
            if found := visit(node):
                return found
        return None

    def waves(self) -> list[list[PlanStep]]:
        """Group steps into rounds. Everything in a round runs at once."""
        remaining = {s.step_id: s for s in self.steps}
        done: set[str] = set()
        rounds: list[list[PlanStep]] = []
        while remaining:
            ready = [
                s
                for s in remaining.values()
                if all(dep in done or dep not in remaining for dep in s.depends_on)
            ]
            if not ready:  # a cycle slipped through; run the rest sequentially
                ready = [next(iter(remaining.values()))]
            rounds.append(sorted(ready, key=lambda s: s.step_id))
            for step in ready:
                done.add(step.step_id)
                del remaining[step.step_id]
        return rounds

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "intent": self.intent.value,
            "source": self.source,
            "rationale": self.rationale,
            "synthesizer": self.synthesizer,
            "steps": [s.as_dict() for s in self.steps],
        }


# Words that hint at what the request is for. Turkish and English, because
# that is what the operator actually types.
_INTENT_HINTS: dict[Intent, tuple[str, ...]] = {
    Intent.BUSINESS: (
        "business",
        "opportunity",
        "market",
        "revenue",
        "pricing",
        "sales",
        "customer",
        "e-commerce",
        "ecommerce",
        "competitor",
        "iş",
        "fırsat",
        "pazar",
        "gelir",
        "fiyat",
        "satış",
        "müşteri",
        "rakip",
        "ticaret",
        "margin",
        "lead",
        "marketing",
        "kâr",
        "pazarlama",
    ),
    Intent.BUILD: (
        "build",
        "code",
        "implement",
        "deploy",
        "refactor",
        "bug",
        "api",
        "test",
        "repository",
        "kod",
        "yaz",
        "geliştir",
        "hata",
        "dağıt",
        "kur",
        "architecture",
        "security",
        "review",
        "mimari",
        "güvenlik",
    ),
    Intent.RESEARCH: (
        "research",
        "find",
        "investigate",
        "compare",
        "latest",
        "news",
        "trend",
        "araştır",
        "bul",
        "incele",
        "karşılaştır",
        "haber",
        "trend",
        "son",
    ),
}

# English puts the question word first; Turkish puts it last. Matching only the
# opening word classified every "X nedir?" as a research project.
_SIMPLE_QUESTION = re.compile(
    r"^\s*(what|who|when|where|which|why|how much|how many)\b"
    r"|\b(nedir|kimdir|ne demek|ne kadar|kaç tane|hangisi|nasıl)\b",
    re.IGNORECASE,
)


def detect_intent(request: str) -> Intent:
    """Classify a request without spending anything."""
    lowered = request.lower()
    scores = {
        intent: sum(1 for hint in hints if hint in lowered)
        for intent, hints in _INTENT_HINTS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return Intent.QUESTION if _SIMPLE_QUESTION.search(request) else Intent.RESEARCH
    # A short factual question stays a question even if it mentions a topic word.
    if _SIMPLE_QUESTION.match(request) and len(request.split()) <= 12 and scores[best] <= 1:
        return Intent.QUESTION
    return best


# How many declared capability words a request has to share before an agent
# is worth staffing. One is noise — "analysis" appears in half the roster — and
# staffing on it costs a model call to discover that.
MIN_CAPABILITY_HITS = 2

# How much being in the intent's own swarm is worth. A preference rather than a
# filter: enough to break a tie between two comparable matches, not enough to
# hide a specialist in another swarm that named the work exactly.
SWARM_PREFERENCE = 1.3


class RulePlanner:
    """Builds a plan from templates. Free, instant, and good enough most days."""

    name = "rules"

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def _pick(self, *candidates: str) -> list[str]:
        """Keep only the agents that are actually registered."""
        return [a for a in candidates if self.registry.get(a) is not None]

    def _matching(self, request: str, *, swarm: str = "", limit: int = 3) -> list[str]:
        """Agents whose declared capabilities match what was asked.

        The first version had a fixed trio per intent, which meant nineteen of
        the twenty-seven agents were reachable only through the model planner.
        A pricing question got the market researcher, because that is who was
        on the list — the pricing agent existed and was never staffed.

        Matching is against what each agent *declared* it does: capabilities,
        role, name. Not its system prompt, which is long and full of incidental
        words. Same rule as skills: an agent says what it is for, and that is
        what gets matched.

        The swarm is a **preference, not a filter**. Filtering by it lost the
        pricing agent on "set a price and defend the number", because the
        intent detector read that as research and the pricing agent lives in
        business. Intent is a guess, and a guess should not be able to hide a
        specialist that named the work exactly.

        A weak match is no match. One shared word means nothing — "analysis"
        appears across half the roster — and staffing on it costs a model call
        to discover that.
        """
        wanted = terms(request)
        if not wanted:
            return []

        scored: list[tuple[float, str]] = []
        for spec in self.registry.all():
            declared = terms(f"{spec.name} {spec.role} {' '.join(spec.capabilities)}")
            if not declared:
                continue
            hits = wanted & declared
            if len(hits) < MIN_CAPABILITY_HITS:
                continue
            relevance = len(hits) / min(len(wanted), len(declared))
            if swarm and spec.swarm == swarm:
                relevance *= SWARM_PREFERENCE
            scored.append((relevance, spec.id))

        scored.sort(reverse=True)
        return [agent_id for _, agent_id in scored[:limit]]

    def plan(self, request: str, *, intent: Intent | None = None) -> Plan:
        intent = intent or detect_intent(request)

        if intent is Intent.QUESTION:
            # One agent, but the right one. A question about runway should go
            # to the finance agent rather than to the web researcher just
            # because the web researcher is the default — the specialist is a
            # better answer at the same cost.
            matched = self._matching(request, limit=1)
            agents = (
                matched or self._pick("web_researcher") or [s.id for s in self.registry.all()[:1]]
            )
            steps = [PlanStep(step_id="s1", agent_id=agents[0], goal=request)] if agents else []
            return Plan(
                request=request,
                intent=intent,
                steps=steps,
                rationale=(
                    "a single factual question needs one agent, not a swarm"
                    + (f"; {agents[0]} declared this work" if matched else "")
                ),
                source=self.name,
            )

        # Three gatherers at most. A fourth rarely finds something the others
        # missed and always costs another model call.
        swarm = {Intent.BUSINESS: "business", Intent.BUILD: "build"}.get(intent, "research")
        if intent is Intent.BUSINESS:
            fallback = self._pick(
                "market_researcher", "competitor_analyst", "trend_analyst", "product_hunter"
            )
            analyst = self._pick("business_analyst", "business_strategist")
        elif intent is Intent.BUILD:
            fallback = self._pick("software_architect", "github_agent", "security_agent")
            analyst = self._pick("qa_agent", "codex_developer", "research_synthesizer")
        else:
            fallback = self._pick(
                "web_researcher", "news_researcher", "social_researcher", "market_researcher"
            )
            analyst = self._pick("research_synthesizer", "business_analyst")

        # Whoever actually declared the work, first — then the swarm's usual
        # trio to fill out the wave. A specialist that matched is worth more
        # than a generalist that is always on the list, and the fallback is
        # what stops a request nobody declared from staffing nobody.
        matched = self._matching(request, swarm=swarm)
        gatherers: list[str] = []
        for agent_id in matched + fallback:
            if agent_id not in gatherers and agent_id not in analyst[:1]:
                gatherers.append(agent_id)
        gatherers = gatherers[:3]

        if not gatherers:
            gatherers = [s.id for s in self.registry.all()[:1]]

        steps: list[PlanStep] = [
            PlanStep(step_id=f"s{i}", agent_id=agent, goal=request)
            for i, agent in enumerate(gatherers, start=1)
        ]
        gather_ids = [s.step_id for s in steps]

        verifier = self._pick("source_verifier")
        if verifier and len(steps) > 1:
            steps.append(
                PlanStep(
                    step_id="verify",
                    agent_id=verifier[0],
                    goal=f"Check the sources behind the findings for: {request}",
                    depends_on=gather_ids,
                )
            )

        synthesizer_id: str | None = None
        if analyst:
            synthesizer_id = analyst[0]
            steps.append(
                PlanStep(
                    step_id="synthesis",
                    agent_id=synthesizer_id,
                    goal=f"Merge every finding into one answer for: {request}",
                    depends_on=[s.step_id for s in steps],
                )
            )

        return Plan(
            request=request,
            intent=intent,
            steps=steps,
            rationale=(
                f"{len(gather_ids)} agent(s) gather in parallel, then the findings are merged"
                + (f"; {', '.join(matched)} matched the request directly" if matched else "")
            ),
            synthesizer=synthesizer_id,
            source=self.name,
        )


class ModelPlanner:
    """Asks a model to build the task graph, and falls back to rules if it cannot."""

    name = "model"

    def __init__(self, registry: AgentRegistry, router: ModelRouter) -> None:
        self.registry = registry
        self.router = router
        self._fallback = RulePlanner(registry)

    SYSTEM = (
        "You are the planner inside a personal AI operating system. Given a request and the "
        "list of available agents, produce the smallest task graph that answers it well.\n"
        "Rules: keep simple requests to one step; only add an agent when it contributes "
        "something the others cannot; make independent steps depend on nothing so they run "
        "in parallel; finish with one synthesis step when there is more than one finding.\n"
        'Reply with one JSON object: {"intent": "...", "rationale": "...", "steps": '
        '[{"step_id": "s1", "agent_id": "...", "goal": "...", "depends_on": []}]}'
    )

    async def plan(self, request: str, *, intent: Intent | None = None) -> Plan:
        prompt = (
            f"## Request\n{request}\n\n## Available agents\n{self.registry.catalog_for_planner()}"
        )
        try:
            response = await self.router.complete(
                ModelRequest(
                    messages=[Message(role="user", content=prompt)],
                    system=self.SYSTEM,
                    max_tokens=4000,
                ),
                kind=TaskKind.PLANNING,
                agent="planner",
            )
        except Exception as exc:
            log.warning("model planning failed (%s); using the rule planner", exc)
            return self._fallback.plan(request, intent=intent)

        from ..agents.base import extract_json

        parsed = extract_json(response.text)
        if not parsed or not isinstance(parsed.get("steps"), list) or not parsed["steps"]:
            log.warning("the planner returned no usable steps; using the rule planner")
            return self._fallback.plan(request, intent=intent)

        steps: list[PlanStep] = []
        for index, raw in enumerate(parsed["steps"], start=1):
            if not isinstance(raw, dict) or not raw.get("agent_id"):
                continue
            steps.append(
                PlanStep(
                    step_id=str(raw.get("step_id") or f"s{index}"),
                    agent_id=str(raw["agent_id"]),
                    goal=str(raw.get("goal") or request),
                    inputs=dict(raw.get("inputs", {})),
                    depends_on=[str(d) for d in raw.get("depends_on", [])],
                )
            )
        if not steps:
            return self._fallback.plan(request, intent=intent)

        try:
            parsed_intent = Intent(str(parsed.get("intent", "research")))
        except ValueError:
            parsed_intent = intent or detect_intent(request)

        plan = Plan(
            request=request,
            intent=parsed_intent,
            steps=steps,
            rationale=str(parsed.get("rationale", "")),
            synthesizer=steps[-1].agent_id if len(steps) > 1 else None,
            source=self.name,
        )
        if problems := plan.validate(self.registry):
            log.warning(
                "the model's plan was rejected (%s); using the rule planner", "; ".join(problems)
            )
            return self._fallback.plan(request, intent=intent)
        return plan
