"""Agent definitions and the runtime that executes them.

An agent is data, not code. Its identity, permissions, allowed models, prompt
and schemas live in a YAML file; a small runtime class turns that data into a
model call. That is what lets the system grow from 27 agents to a hundred
without writing a hundred modules, and it is why a generated agent can be
reviewed as a diff before anyone approves it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.context import (
    ContextBudget,
    compress_findings,
    estimate_request_tokens,
    render_findings,
    sanitize_untrusted,
)
from ..core.messages import AgentMessage, MessageBus, MessageType
from ..core.task import Task
from ..logging import AuditLog, get_logger
from ..models.base import Message, ModelRequest
from ..models.router import ModelRouter, TaskKind
from ..observability.tracing import Trace, Tracer
from ..security.guardian import SecurityGuardian
from ..security.permissions import (
    Permission,
    PermissionSet,
    RiskLevel,
    parse_permissions,
    parse_risk,
)

log = get_logger("agents")

# What each tool actually does, expressed as the permission it needs. Reaching
# the open web is a network action; running code is an execute action.
TOOL_PERMISSIONS: dict[str, PermissionSet] = {
    "web_search": frozenset({Permission.NETWORK}),
    "web_fetch": frozenset({Permission.NETWORK}),
    "code_execution": frozenset({Permission.EXECUTE}),
}


@dataclass(slots=True)
class RetryPolicy:
    """How hard an agent tries before giving up. Never unbounded."""

    max_attempts: int = 2
    backoff_s: float = 1.0
    alternative_model: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RetryPolicy:
        raw = raw or {}
        attempts = int(raw.get("max_attempts", 2))
        if attempts < 1:
            raise ValueError("retry_policy.max_attempts must be at least 1")
        if attempts > 5:
            raise ValueError("retry_policy.max_attempts is capped at 5; infinite retry is banned")
        return cls(
            max_attempts=attempts,
            backoff_s=float(raw.get("backoff_s", 1.0)),
            alternative_model=bool(raw.get("alternative_model", True)),
        )


@dataclass(slots=True)
class EvaluationPolicy:
    """How an agent's output is judged before it counts as done."""

    criteria: tuple[str, ...] = ("correctness", "relevance", "completeness")
    min_confidence: float = 0.5
    require_sources: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> EvaluationPolicy:
        raw = raw or {}
        return cls(
            criteria=tuple(raw.get("criteria", ("correctness", "relevance", "completeness"))),
            min_confidence=float(raw.get("min_confidence", 0.5)),
            require_sources=bool(raw.get("require_sources", False)),
        )


@dataclass(slots=True)
class AgentSpec:
    """Everything that defines one agent."""

    id: str
    name: str
    role: str
    swarm: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    permissions: PermissionSet = field(default_factory=frozenset)
    risk_level: RiskLevel = RiskLevel.LOW
    system_prompt: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 180.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    evaluation_policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)
    runtime: str = "llm"
    task_kind: TaskKind = TaskKind.ANALYSIS
    # What this agent may reach through the gateway, and how far. Capabilities
    # rather than tool ids, so a tool can be replaced behind one without
    # touching the agent.
    tool_capabilities: tuple[str, ...] = ()
    max_tool_risk: str = "read_only"
    max_tokens: int = 8000
    effort: str | None = None
    enabled: bool = True
    version: str = "1.0.0"
    # A real task this agent can do, used to verify it works rather than
    # merely loads. Without one the agent stays at DEFINED however good its
    # definition looks — nothing can be run, so nothing can be shown.
    smoke_task: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source: str = "<memory>") -> AgentSpec:
        """Build a spec from a YAML mapping, failing loudly on bad data."""
        missing = [k for k in ("id", "name", "role", "swarm") if not raw.get(k)]
        if missing:
            raise ValueError(f"{source}: agent definition is missing: {', '.join(missing)}")
        agent_id = str(raw["id"])
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,48}", agent_id):
            raise ValueError(
                f"{source}: agent id {agent_id!r} must be lowercase letters, digits and underscores"
            )
        try:
            kind = TaskKind(str(raw.get("task_kind", "analysis")))
        except ValueError as exc:
            valid = ", ".join(k.value for k in TaskKind)
            raise ValueError(f"{source}: unknown task_kind; valid values are {valid}") from exc

        return cls(
            id=agent_id,
            name=str(raw["name"]),
            role=str(raw["role"]),
            swarm=str(raw["swarm"]),
            description=str(raw.get("description", "")),
            capabilities=tuple(raw.get("capabilities", ())),
            tools=tuple(raw.get("tools", ())),
            allowed_models=tuple(raw.get("allowed_models", ())),
            permissions=parse_permissions(raw.get("permissions")),
            risk_level=parse_risk(raw.get("risk_level")),
            system_prompt=str(raw.get("system_prompt", "")).strip(),
            smoke_task=str(raw.get("smoke_task", "")).strip(),
            input_schema=dict(raw.get("input_schema", {})),
            output_schema=dict(raw.get("output_schema", {})),
            timeout_s=float(raw.get("timeout_s", 180.0)),
            retry_policy=RetryPolicy.from_dict(raw.get("retry_policy")),
            evaluation_policy=EvaluationPolicy.from_dict(raw.get("evaluation_policy")),
            runtime=str(raw.get("runtime", "llm")),
            task_kind=kind,
            tool_capabilities=tuple(raw.get("tool_capabilities", ())),
            max_tool_risk=str(raw.get("max_tool_risk", "read_only")),
            max_tokens=int(raw.get("max_tokens", 8000)),
            effort=raw.get("effort"),
            enabled=bool(raw.get("enabled", True)),
            version=str(raw.get("version", "1.0.0")),
        )

    @property
    def server_tools(self) -> list[str]:
        """The subset of declared tools that are model server-side tools."""
        known = {"web_search", "web_fetch", "code_execution"}
        return [t for t in self.tools if t in known]

    def summary(self) -> str:
        perms = ",".join(sorted(p.value for p in self.permissions)) or "none"
        return f"{self.id} ({self.swarm}/{self.role}, risk={self.risk_level.value}, perms={perms})"


@dataclass(slots=True)
class AgentResult:
    """What an agent hands back."""

    agent_id: str
    task_id: str
    ok: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    model: str | None = None
    latency_ms: float = 0.0
    # What the call actually consumed, as the provider reported it. Cost is
    # derived from these, so a report that carries the money without the tokens
    # cannot be checked against the price list. Verification and the cost ledger
    # both read them from here rather than re-estimating.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    # Why the model stopped. `max_tokens` means the answer was cut off, which
    # looks identical to a badly-behaved model once the text is all you have —
    # and the two need opposite fixes: raise the ceiling, or fix the prompt.
    # It was being read for `refused` and then discarded.
    stop_reason: str | None = None
    error: str | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    # Which skills this run was given, so the cost of loading them is visible
    # next to what they cost. An unnamed prompt addition is a cost nobody audits.
    skills: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "ok": self.ok,
            "text": self.text,
            "data": self.data,
            "confidence": self.confidence,
            "sources": self.sources,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "evaluation": self.evaluation,
            "skills": list(self.skills),
        }


@dataclass(slots=True)
class AgentContext:
    """The shared services an agent is allowed to use."""

    router: ModelRouter
    guardian: SecurityGuardian
    audit: AuditLog
    tracer: Tracer
    trace: Trace
    bus: MessageBus = field(default_factory=MessageBus)
    budget: ContextBudget = field(default_factory=ContextBudget)
    # The only way an agent reaches the outside world. None means this run has
    # no tools, not that the agent may find its own way to one.
    gateway: Any = None
    # Instruction that applies to some tasks and not others. None means the
    # run gets none, which is the old behaviour and always safe: a skill adds
    # guidance, never access.
    skills: Any = None
    # How much instruction this run may load. None means the registry's own
    # default; an A/B measurement narrows it so the two arms cannot differ by
    # which *other* skills happened to fit.
    skill_budget: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    async def call_tool(self, spec: Any, task: Any, tool_id: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on behalf of an agent.

        Every check lives in the gateway, so this is a thin hand-off rather
        than a place where an agent could talk its way past something.
        """
        from ..mcp.errors import ToolErrorCode, ToolFailure
        from ..mcp.models import ToolRequest, ToolResult

        if self.gateway is None:
            return ToolResult(
                request_id="",
                tool_id=tool_id,
                ok=False,
                error=ToolFailure(
                    code=ToolErrorCode.TOOL_NOT_FOUND,
                    message="no tool gateway is attached to this run",
                ),
            )
        return await self.gateway.call(
            ToolRequest(
                tool_id=tool_id,
                arguments=arguments,
                agent_id=spec.id,
                task_id=getattr(task, "task_id", None),
                trace_id=self.trace.trace_id,
            ),
            agent_permissions=spec.permissions,
            trace=self.trace,
        )


class AgentRuntime(Protocol):
    """How a spec is actually executed."""

    name: str

    async def run(self, spec: AgentSpec, task: Task, ctx: AgentContext) -> AgentResult: ...


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that parsing the raw
    string is not reliable.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class LLMAgentRuntime:
    """The default runtime: build a prompt, route it, read the answer back."""

    name = "llm"

    def _build_prompt(
        self, spec: AgentSpec, task: Task, budget: ContextBudget
    ) -> tuple[str, dict[str, Any]]:
        """Build the prompt and report what the context budget did.

        Findings from earlier steps are the only untrusted material an agent
        sees, so they are compressed, sanitised and fenced. Everything else in
        ``inputs`` came from the plan, which JARVIS wrote.
        """
        lines = [f"# Task\n{task.goal}"]
        report: dict[str, Any] = {"findings_in": 0, "findings_used": 0, "injection_hits": []}

        inputs = dict(task.inputs)
        raw_findings = inputs.pop("findings", None) or []
        if raw_findings:
            report["findings_in"] = len(raw_findings)
            findings, sanitation = compress_findings(raw_findings, budget)
            report["findings_used"] = len(findings)
            report["injection_hits"] = sanitation.hits
            rendered = render_findings(findings)
            if rendered:
                lines.append("\n" + rendered)

        if inputs:
            body = json.dumps(inputs, ensure_ascii=False, indent=2, default=str)
            safe_body, sanitation = sanitize_untrusted(body)
            report["injection_hits"].extend(sanitation.hits)
            lines.append("\n# Inputs\n" + safe_body)

        if spec.output_schema:
            lines.append(
                "\n# Required output\n"
                "Reply with a single JSON object matching this shape, and nothing else:\n"
                + json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
            )
        else:
            lines.append(
                "\n# Required output\n"
                'Reply with a single JSON object holding at least "summary" (string) '
                'and "confidence" (a number from 0 to 1).'
            )
        return "\n".join(lines), report

    async def run(self, spec: AgentSpec, task: Task, ctx: AgentContext) -> AgentResult:
        required: PermissionSet = spec.permissions or frozenset({Permission.READ})
        verdict = await ctx.guardian.authorize(
            action=f"run agent {spec.id}",
            granted=spec.permissions,
            required=required,
            risk=spec.risk_level,
            agent=spec.id,
            task_id=task.task_id,
            reason=task.goal,
        )
        if not verdict.allowed:
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=False,
                error=verdict.reason,
            )

        # A tool is authorised separately from the agent. An agent that holds
        # READ does not thereby get to reach the network, so a definition that
        # asks for web search without NETWORK loses the tool, not the run.
        allowed_tools: list[str] = []
        for tool in spec.server_tools:
            tool_verdict = ctx.guardian.check(
                action=f"use tool {tool}",
                granted=spec.permissions,
                required=TOOL_PERMISSIONS.get(tool, frozenset({Permission.NETWORK})),
                risk=spec.risk_level,
                agent=spec.id,
                task_id=task.task_id,
            )
            if tool_verdict.allowed:
                allowed_tools.append(tool)
            else:
                log.warning("agent %s may not use %s: %s", spec.id, tool, tool_verdict.reason)

        prompt, context_report = self._build_prompt(spec, task, ctx.budget)
        if context_report["injection_hits"]:
            # Someone upstream tried to talk past this agent. The text was
            # already neutralised; this makes it visible.
            ctx.audit.emit(
                "security.injection_blocked",
                agent=spec.id,
                task_id=task.task_id,
                result="blocked",
                reason="instruction-shaped text found in untrusted findings",
                metadata={"patterns": context_report["injection_hits"][:10]},
            )

        # Skills that apply to this task, if any are loaded. They are filtered
        # by what this agent may actually call, so a skill telling it to search
        # the web is not loaded into an agent that has no network.
        system = spec.system_prompt
        loaded_skills = None
        if ctx.skills is not None:
            loaded_skills = ctx.skills.load_for(
                task.goal,
                agent_id=spec.id,
                swarm=spec.swarm,
                available_tools=frozenset(spec.tool_capabilities),
                budget=ctx.skill_budget,
            )
            if block := loaded_skills.render():
                system = f"{system}\n\n{block}" if system else block

        request = ModelRequest(
            messages=[Message(role="user", content=prompt)],
            system=system or None,
            max_tokens=spec.max_tokens,
            effort=spec.effort,
            server_tools=allowed_tools,
            metadata={"agent": spec.id, "task_id": task.task_id},
        )
        estimated_input = estimate_request_tokens(system, prompt)

        with ctx.tracer.span(
            ctx.trace,
            spec.id,
            kind="agent",
            task_id=task.task_id,
            swarm=spec.swarm,
            estimated_input_tokens=estimated_input,
            findings_in=context_report["findings_in"],
            findings_used=context_report["findings_used"],
        ) as span:
            response = await ctx.router.complete(
                request,
                kind=spec.task_kind,
                allowed_models=list(spec.allowed_models) or None,
                agent=spec.id,
                task_id=task.task_id,
                max_attempts=spec.retry_policy.max_attempts
                if spec.retry_policy.alternative_model
                else 1,
            )
            span.set(model=response.model, output_tokens=response.usage.output_tokens)

        parsed = extract_json(response.text) or {}
        confidence = parsed.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else 0.6
        except (TypeError, ValueError):
            confidence_value = 0.6
        confidence_value = min(1.0, max(0.0, confidence_value))

        cost = ctx.router.cost.price(
            response.model, response.usage.input_tokens, response.usage.output_tokens
        )
        sources = list(response.citations)
        for entry in parsed.get("sources", []) or []:
            if isinstance(entry, dict):
                sources.append(entry)
            elif isinstance(entry, str):
                sources.append({"url": entry, "title": None})

        result = AgentResult(
            agent_id=spec.id,
            task_id=task.task_id,
            ok=not response.refused,
            text=parsed.get("summary") or response.text,
            data=parsed,
            confidence=confidence_value,
            sources=sources,
            cost_usd=cost,
            model=response.model,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            stop_reason=response.stop_reason,
            error="the model declined the request" if response.refused else None,
            skills=[s.skill_id for s in loaded_skills.skills] if loaded_skills else [],
        )
        ctx.bus.publish(
            AgentMessage(
                task_id=task.task_id,
                from_agent=spec.id,
                to_agent="jarvis",
                message_type=MessageType.RESULT if result.ok else MessageType.ERROR,
                payload={"summary": result.text[:500], "sources": len(result.sources)},
                confidence=result.confidence,
            )
        )
        return result
