"""The Tool Gateway: the one door between agents and the world.

An agent never holds a transport, never sees a server, and never calls MCP.
It hands the gateway a request and gets back a result. Everything between
those two points happens here, in a fixed order, with no way around it:

    lookup -> server -> policy -> risk -> budget -> rate limit -> approval
           -> argument validation -> duplicate check -> cache
           -> execution with timeout and bounded retry
           -> result validation -> secret redaction -> audit -> return

Failures are returned, not raised. An agent deciding what to do about a rate
limit should read a result, not catch an exception, and a tool that explodes
must never take a run down with it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..logging import AuditLog, get_logger
from ..mcp.errors import ToolError, ToolErrorCode, ToolFailure
from ..mcp.models import MCPTool, ToolRequest, ToolResult
from ..mcp.registry import MCPServerRegistry
from ..mcp.transport import call_with_timeout
from ..observability.cache import ResultCache, work_key
from ..observability.cost import BudgetExceeded, CostEngine
from ..observability.tracing import Trace, Tracer
from ..security.permissions import PermissionSet
from .approval import ApprovalEngine, ApprovalStatus
from .policy import PolicyEngine
from .ratelimit import RateLimiter
from .registry import ToolRegistry
from .validation import require_valid_arguments, validate_result, wrap_untrusted

log = get_logger("tools.gateway")


@dataclass(slots=True)
class GatewayStats:
    """What the gateway has done, for the dashboard and the morning report."""

    calls: int = 0
    allowed: int = 0
    denied: int = 0
    failed: int = 0
    cache_hits: int = 0
    approvals_required: int = 0
    security_blocks: int = 0
    rate_limited: int = 0
    duplicates_blocked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "allowed": self.allowed,
            "denied": self.denied,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "approvals_required": self.approvals_required,
            "security_blocks": self.security_blocks,
            "rate_limited": self.rate_limited,
            "duplicates_blocked": self.duplicates_blocked,
        }


class ToolGateway:
    """Every tool call in the system goes through one instance of this."""

    def __init__(
        self,
        *,
        servers: MCPServerRegistry,
        tools: ToolRegistry,
        policy: PolicyEngine,
        approvals: ApprovalEngine | None = None,
        cost: CostEngine | None = None,
        audit: AuditLog | None = None,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.audit = audit or AuditLog()
        self.servers = servers
        self.tools = tools
        self.policy = policy
        self.approvals = approvals or ApprovalEngine(audit=self.audit)
        self.cost = cost or CostEngine(settings=self.settings)
        self.tracer = tracer or Tracer()
        self.limiter = rate_limiter or RateLimiter()
        self.stats = GatewayStats()
        self.cache: ResultCache[ToolResult] = ResultCache(
            max_entries=self.settings.cache_max_entries, ttl_s=self.settings.cache_ttl_s
        )
        # Write calls already made, so a repeat cannot create a second thing.
        self._completed_writes: dict[str, ToolResult] = {}

    # -- what an agent may see --------------------------------------------

    def available_tools(self, agent_id: str) -> list[MCPTool]:
        """The tools this agent could actually call.

        Discovery is not authorisation, so an agent listing tools sees its own
        grants rather than everything the servers publish.
        """
        return self.policy.visible_tools(agent_id, self.tools.all(available_only=True))

    # -- the one entry point ----------------------------------------------

    async def call(
        self,
        request: ToolRequest,
        *,
        agent_permissions: PermissionSet = frozenset(),
        trace: Trace | None = None,
    ) -> ToolResult:
        """Run one tool call through every gate. Never raises."""
        self.stats.calls += 1
        started = time.perf_counter()
        span_trace = trace or self.tracer.start_trace(f"tool:{request.tool_id}")

        try:
            with self.tracer.span(
                span_trace,
                request.tool_id,
                kind="tool",
                request_id=request.request_id,
                agent=request.agent_id,
                task_id=request.task_id,
            ) as span:
                result = await self._call_inner(request, agent_permissions)
                span.set(
                    ok=result.ok,
                    from_cache=result.from_cache,
                    attempts=result.attempts,
                )
                return result
        except ToolError as error:
            return self._deny(request, error, started)
        except Exception as error:
            # A gateway bug must not take the caller down either.
            log.exception("the gateway failed handling %s", request.tool_id)
            return self._deny(
                request,
                ToolError(
                    ToolErrorCode.EXTERNAL_ERROR,
                    f"the gateway failed: {type(error).__name__}: {error}",
                ),
                started,
            )

    async def _call_inner(
        self, request: ToolRequest, agent_permissions: PermissionSet
    ) -> ToolResult:
        # 1. The tool must exist, be enabled, and be usable.
        tool = self.tools.require(request.tool_id)

        # 2. So must its server. Unknown, disabled or unhealthy all stop here.
        server = self.servers.require_usable(tool.server_id)

        # 3. Policy. Default deny; a grant is a capability, not a tool list.
        verdict = self.policy.check(
            agent_id=request.agent_id,
            tool=tool,
            server=server,
            agent_permissions=agent_permissions,
        )
        if not verdict.allowed:
            self.audit.emit(
                "tool.denied",
                agent=request.agent_id,
                tool=tool.tool_id,
                task_id=request.task_id,
                result="denied",
                reason=verdict.reason,
                metadata={"request_id": request.request_id, "risk": tool.risk.value},
            )
            raise ToolError(
                verdict.code or ToolErrorCode.PERMISSION_DENIED,
                verdict.reason,
                details={"risk": tool.risk.value},
            )

        # 4. Budget. A breached budget stops the call before it costs anything.
        try:
            self.cost.check_budgets(request.task_id)
        except BudgetExceeded as exc:
            self.audit.emit(
                "tool.budget_blocked",
                agent=request.agent_id,
                tool=tool.tool_id,
                task_id=request.task_id,
                result="blocked",
                reason=str(exc),
            )
            raise ToolError(ToolErrorCode.BUDGET_EXCEEDED, str(exc)) from exc

        # 5. Rate limit.
        #
        # `acquire` rather than `check`, because checking now and recording
        # after the call leaves a window two callers both pass through — and
        # with a shared store that window spans processes, which is exactly
        # what the shared store exists to close. A call that is refused later
        # (approval, validation) has been counted, which is the safe direction:
        # over-counting refuses a call that could have gone; under-counting
        # lets one through that should not have.
        limit_verdict = self.limiter.acquire(
            tool_id=tool.tool_id,
            agent_id=request.agent_id,
            task_id=request.task_id,
            limit=tool.rate_limit,
        )
        if not limit_verdict.allowed:
            self.stats.rate_limited += 1
            self.stats.denied -= 1  # counted once, in _deny, not twice
            self.audit.emit(
                "tool.rate_limited",
                agent=request.agent_id,
                tool=tool.tool_id,
                task_id=request.task_id,
                result="blocked",
                reason=limit_verdict.reason,
            )
            raise ToolError(
                ToolErrorCode.RATE_LIMITED,
                limit_verdict.reason,
                details=limit_verdict.as_dict(),
            )

        # 6. Arguments, before approval: a human should not be asked to approve
        #    a call that was never going to work.
        require_valid_arguments(tool.input_schema, request.arguments)

        # 7. Approval, for anything that cannot be undone by trying again.
        if verdict.needs_approval:
            allowed, reason, status, bypass_attempt = self.approvals.consume(request)
            if not allowed:
                if bypass_attempt:
                    # A replayed approval, or one granted for a different tool
                    # or different arguments. That is an attempted bypass.
                    self.stats.security_blocks += 1
                    raise ToolError(ToolErrorCode.SECURITY_BLOCKED, reason)
                if status is None or status is ApprovalStatus.PENDING:
                    self.stats.approvals_required += 1
                    approval = self.approvals.require(
                        request, tool.risk, reason=f"{request.agent_id} wants {tool.tool_id}"
                    )
                    raise ToolError(
                        ToolErrorCode.APPROVAL_REQUIRED,
                        f"{tool.risk.value} calls need the operator's approval: {reason}",
                        details={"approval_id": approval.approval_id},
                    )
                if status is ApprovalStatus.EXPIRED:
                    raise ToolError(ToolErrorCode.APPROVAL_EXPIRED, reason)
                raise ToolError(ToolErrorCode.APPROVAL_REJECTED, reason)

        # 8. A write that already happened must not happen twice.
        idempotency_key = self._idempotency_key(tool, request)
        if idempotency_key is not None:
            previous = self._completed_writes.get(idempotency_key)
            if previous is not None:
                self.stats.duplicates_blocked += 1
                self.audit.emit(
                    "tool.duplicate_blocked",
                    agent=request.agent_id,
                    tool=tool.tool_id,
                    task_id=request.task_id,
                    result="blocked",
                    reason="this write has already been performed",
                )
                raise ToolError(
                    ToolErrorCode.DUPLICATE_REQUEST,
                    f"{tool.tool_id} has already run with these arguments",
                    details={"original_request_id": previous.request_id},
                )

        self.audit.emit(
            "tool.authorized",
            agent=request.agent_id,
            tool=tool.tool_id,
            task_id=request.task_id,
            result="allowed",
            metadata={"request_id": request.request_id, "risk": tool.risk.value},
        )

        # 9. Execute, through the cache when the tool is a read.
        if tool.is_cacheable:
            key = work_key("tool", tool.tool_id, tool.version, request.arguments)
            result, from_cache = await self.cache.run(
                key,
                lambda: self._execute(tool, request),
                cacheable=lambda r: r.ok,
            )
            if from_cache:
                self.stats.cache_hits += 1
                self.audit.emit(
                    "tool.cache_hit",
                    agent=request.agent_id,
                    tool=tool.tool_id,
                    task_id=request.task_id,
                    metadata={"request_id": request.request_id},
                )
                result = ToolResult(
                    request_id=request.request_id,
                    tool_id=result.tool_id,
                    ok=result.ok,
                    content=result.content,
                    text=result.text,
                    duration_ms=0.0,
                    attempts=0,
                    from_cache=True,
                    truncated=result.truncated,
                    sanitized=result.sanitized,
                    warnings=list(result.warnings),
                )
            else:
                self.audit.emit(
                    "tool.cache_miss",
                    agent=request.agent_id,
                    tool=tool.tool_id,
                    task_id=request.task_id,
                )
        else:
            result = await self._execute(tool, request)

        if idempotency_key is not None and result.ok:
            self._completed_writes[idempotency_key] = result

        self.stats.allowed += 1
        return result

    # -- execution ---------------------------------------------------------

    def _idempotency_key(self, tool: MCPTool, request: ToolRequest) -> str | None:
        """The key under which a completed write is remembered.

        Only writes get one. Reads are safe to repeat, and remembering them
        here would duplicate what the cache already does.
        """
        if not tool.risk.is_write:
            return None
        return request.idempotency_key or work_key(
            "write", tool.tool_id, request.agent_id, request.task_id, request.arguments
        )

    async def _execute(self, tool: MCPTool, request: ToolRequest) -> ToolResult:
        """Call the tool, with a timeout and a bounded retry."""
        transport = self.servers.transport_for(tool.server_id)
        if transport is None:
            raise ToolError(
                ToolErrorCode.SERVER_NOT_FOUND,
                f"the server {tool.server_id!r} has no transport configured",
            )
        server = self.servers.get(tool.server_id)
        assert server is not None

        attempts = tool.retry_policy.attempts_for(tool.risk)
        started = time.perf_counter()
        last_error: ToolError | None = None

        self.audit.emit(
            "tool.started",
            agent=request.agent_id,
            tool=tool.tool_id,
            task_id=request.task_id,
            metadata={"request_id": request.request_id, "attempts_allowed": attempts},
        )

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                # The first attempt was counted by `acquire` above. A retry is
                # a second call the tool actually sees, so it counts too.
                self.limiter.record(
                    tool_id=tool.tool_id, agent_id=request.agent_id, task_id=request.task_id
                )
            try:
                if not transport.connected:
                    await transport.connect(server)
                raw = await call_with_timeout(
                    transport, tool.name, request.arguments, tool.timeout_s
                )
            except ToolError as error:
                last_error = error
                if not error.retryable or attempt >= attempts:
                    break
                log.info(
                    "%s failed (%s); retrying %d/%d",
                    tool.tool_id,
                    error.code.value,
                    attempt + 1,
                    attempts,
                )
                await asyncio.sleep(tool.retry_policy.backoff_s)
                continue
            except Exception as error:
                # A transport that raises something unexpected is an external
                # failure, not a crash of this process.
                last_error = ToolError(
                    ToolErrorCode.EXTERNAL_ERROR,
                    f"{type(error).__name__}: {error}",
                )
                if attempt >= attempts:
                    break
                await asyncio.sleep(tool.retry_policy.backoff_s)
                continue

            # The call returned. Now decide whether what came back is usable.
            duration_ms = (time.perf_counter() - started) * 1000
            try:
                text, structured, report = validate_result(
                    raw, output_schema=tool.output_schema or None
                )
            except ToolError as error:
                last_error = error
                break

            if report.secrets_found:
                self.stats.security_blocks += 1
                self.audit.emit(
                    "tool.security_blocked",
                    agent=request.agent_id,
                    tool=tool.tool_id,
                    task_id=request.task_id,
                    result="redacted",
                    reason="the result contained something that looked like a credential",
                    metadata={"kinds": report.secrets_found},
                )
            if not report.ok:
                last_error = ToolError(ToolErrorCode.INVALID_RESULT, "; ".join(report.errors))
                break

            self.audit.emit(
                "tool.completed",
                agent=request.agent_id,
                tool=tool.tool_id,
                task_id=request.task_id,
                result="ok",
                metadata={
                    "request_id": request.request_id,
                    "duration_ms": round(duration_ms, 1),
                    "attempts": attempt,
                    "truncated": report.truncated,
                    "sanitized": report.sanitized,
                },
            )
            return ToolResult(
                request_id=request.request_id,
                tool_id=tool.tool_id,
                ok=True,
                content=structured,
                # Fenced, so whatever reads it knows it is evidence.
                text=wrap_untrusted(tool.tool_id, text),
                duration_ms=duration_ms,
                attempts=attempt,
                truncated=report.truncated,
                sanitized=report.sanitized,
                warnings=report.warnings,
            )

        assert last_error is not None
        self.stats.failed += 1
        duration_ms = (time.perf_counter() - started) * 1000
        self.audit.emit(
            "tool.timeout" if last_error.code is ToolErrorCode.TIMEOUT else "tool.failed",
            agent=request.agent_id,
            tool=tool.tool_id,
            task_id=request.task_id,
            result="error",
            reason=last_error.message[:300],
            metadata={
                "request_id": request.request_id,
                "code": last_error.code.value,
                "duration_ms": round(duration_ms, 1),
            },
        )
        return ToolResult(
            request_id=request.request_id,
            tool_id=tool.tool_id,
            ok=False,
            error=ToolFailure.from_error(last_error),
            duration_ms=duration_ms,
            attempts=attempts,
        )

    # -- denial ------------------------------------------------------------

    def _deny(self, request: ToolRequest, error: ToolError, started: float) -> ToolResult:
        """Turn a gate refusal into a result the agent can read."""
        if error.code is not ToolErrorCode.APPROVAL_REQUIRED:
            self.stats.denied += 1
        if error.code is ToolErrorCode.SECURITY_BLOCKED:
            self.stats.security_blocks += 1
        return ToolResult(
            request_id=request.request_id,
            tool_id=request.tool_id,
            ok=False,
            error=ToolFailure.from_error(error),
            duration_ms=(time.perf_counter() - started) * 1000,
            attempts=0,
        )

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "gateway": self.stats.as_dict(),
            "cache": self.cache.stats.as_dict(),
            "tools": self.tools.summary(),
            "servers": len(self.servers.all()),
            "pending_approvals": len(self.approvals.pending),
        }
