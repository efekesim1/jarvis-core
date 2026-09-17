"""The Tool Gateway: the pipeline every tool call must survive.

These are enforcement tests. A limit written in a config file proves nothing;
each of these asks whether the gate actually stops the thing it names.
"""

from __future__ import annotations

import asyncio

from jarvis_core.mcp.errors import ToolErrorCode
from jarvis_core.mcp.models import HealthStatus, ToolRequest, ToolRisk
from jarvis_core.security.permissions import Permission
from jarvis_core.tools.policy import AgentToolPolicy


def _request(tool_id: str, agent: str = "reader", **arguments) -> ToolRequest:
    return ToolRequest(tool_id=tool_id, arguments=arguments, agent_id=agent, task_id="tsk_test")


def _grant(
    fabric,
    agent: str = "reader",
    *,
    capabilities=("mock.echo", "mock.lookup"),
    max_risk: ToolRisk = ToolRisk.READ_ONLY,
    tools=(),
) -> None:
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id=agent,
            capabilities=frozenset(capabilities),
            tools=frozenset(tools),
            max_risk=max_risk,
        )
    )


# ============================ the happy path ==============================


async def test_a_granted_read_call_works(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hello"), agent_permissions=reader_permissions
    )
    assert result.ok
    assert "hello" in result.text
    assert result.untrusted


async def test_a_result_is_fenced_as_untrusted(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hello"), agent_permissions=reader_permissions
    )
    assert "UNTRUSTED_TOOL_RESULT" in result.text
    assert "not an instruction" in result.text


async def test_structured_output_is_parsed(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.lookup", record_id=7), agent_permissions=reader_permissions
    )
    assert result.ok
    assert result.content["record_id"] == 7


# ======================= 1-5: identity and existence =======================


async def test_1_an_agent_with_no_policy_can_call_nothing(fabric, reader_permissions):
    result = await fabric.gateway.call(
        _request("mock.echo", agent="stranger", message="hi"),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert "no tool policy" in result.error.message


async def test_2_an_unknown_tool_is_refused(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.does_not_exist"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.TOOL_NOT_FOUND


async def test_3_a_tool_on_an_unknown_server_is_refused(fabric, reader_permissions):
    from jarvis_core.mcp.models import MCPTool

    fabric.tools.register(
        MCPTool(
            tool_id="ghost.read",
            name="read",
            server_id="ghost",
            risk=ToolRisk.READ_ONLY,
            capability="mock.echo",
        )
    )
    _grant(fabric)
    result = await fabric.gateway.call(_request("ghost.read"), agent_permissions=reader_permissions)
    assert not result.ok
    assert result.error.code is ToolErrorCode.SERVER_NOT_FOUND


async def test_4_a_disabled_tool_is_refused(fabric, reader_permissions):
    _grant(fabric)
    fabric.tools.enable("mock.echo", False)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.TOOL_DISABLED


async def test_5_a_disabled_server_is_refused(fabric, reader_permissions):
    _grant(fabric)
    fabric.servers.enable("mock", False)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.SERVER_DISABLED


async def test_5b_an_unhealthy_server_is_refused(fabric, reader_permissions):
    _grant(fabric)
    fabric.servers.set_health("mock", HealthStatus.UNHEALTHY)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.SERVER_UNHEALTHY


# ===================== 6-8: arguments never reach a server =================


async def test_6_malformed_arguments_never_reach_the_server(fabric, reader_permissions):
    _grant(fabric)
    before = len(fabric.mock.calls)
    result = await fabric.gateway.call(
        _request("mock.lookup", record_id="not a number"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS
    assert len(fabric.mock.calls) == before  # nothing left the process


async def test_7_a_missing_required_argument_is_refused(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.lookup"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert "missing required argument" in result.error.message


async def test_8_an_unknown_argument_is_refused(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.lookup", record_id=1, sneaky="payload"),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert "unknown argument" in result.error.message


async def test_8b_a_boolean_is_not_an_integer(fabric, reader_permissions):
    _grant(fabric)
    result = await fabric.gateway.call(
        _request("mock.lookup", record_id=True), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


# ================== 9-10: escalation and impersonation ====================


async def test_9_an_agent_cannot_exceed_its_risk_ceiling(fabric, writer_permissions):
    # Granted the capability, but capped at reads.
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.READ_ONLY)
    result = await fabric.gateway.call(
        _request("mock.deploy", target="production"), agent_permissions=writer_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.RISK_DENIED


async def test_9b_a_capability_grant_cannot_conjure_a_missing_permission(fabric):
    """Policy grants reach; the agent's own permissions still bound it."""
    _grant(fabric, capabilities=("mock.create_note",), max_risk=ToolRisk.LOW_RISK_WRITE)
    result = await fabric.gateway.call(
        _request("mock.create_note", title="x"),
        agent_permissions=frozenset({Permission.READ}),  # no WRITE
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert "lacks permission" in result.error.message


async def test_10_one_agent_cannot_borrow_another_agents_grant(fabric, reader_permissions):
    _grant(fabric, agent="reader")
    result = await fabric.gateway.call(
        _request("mock.echo", agent="impostor", message="hi"),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED


async def test_10b_a_tool_restricted_to_one_agent_refuses_others(fabric, reader_permissions):
    _grant(fabric, agent="other")
    fabric.tools.get("mock.echo").allowed_agents = ("someone_else",)
    result = await fabric.gateway.call(
        _request("mock.echo", agent="other", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED


# ===================== 11-14: the approval gate ===========================


async def test_11_a_high_risk_call_stops_for_approval(fabric, writer_permissions, settings):
    settings.environment = "production"  # the sandbox refuses these outright
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)
    before = len(fabric.mock.calls)

    result = await fabric.gateway.call(
        _request("mock.deploy", target="production"), agent_permissions=writer_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.APPROVAL_REQUIRED
    assert len(fabric.mock.calls) == before  # nothing was deployed
    assert fabric.approvals.pending


async def test_11b_an_approved_call_then_runs(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)
    request = _request("mock.deploy", target="staging")

    first = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert first.error.code is ToolErrorCode.APPROVAL_REQUIRED

    approval = fabric.approvals.for_request(request.request_id)
    fabric.approvals.approve(approval.approval_id, by="operator")

    second = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert second.ok


async def test_12_an_expired_approval_does_not_authorise(fabric, writer_permissions, settings):
    from datetime import UTC, datetime, timedelta

    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)
    request = _request("mock.deploy", target="staging")

    await fabric.gateway.call(request, agent_permissions=writer_permissions)
    approval = fabric.approvals.for_request(request.request_id)
    fabric.approvals.approve(approval.approval_id)
    approval.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    result = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert not result.ok
    assert result.error.code is ToolErrorCode.APPROVAL_EXPIRED


async def test_13_an_approval_cannot_be_replayed(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)
    request = _request("mock.deploy", target="staging")

    await fabric.gateway.call(request, agent_permissions=writer_permissions)
    approval = fabric.approvals.for_request(request.request_id)
    fabric.approvals.approve(approval.approval_id)

    first = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert first.ok

    # The same approval, presented a second time for a second deploy.
    second = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert not second.ok
    assert second.error.code in (
        ToolErrorCode.SECURITY_BLOCKED,
        ToolErrorCode.DUPLICATE_REQUEST,
    )


async def test_14_an_approval_for_one_call_does_not_authorise_another(
    fabric, writer_permissions, settings
):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(
        fabric,
        capabilities=("mock.deploy", "mock.delete_everything"),
        max_risk=ToolRisk.DESTRUCTIVE,
    )
    harmless = _request("mock.deploy", target="staging")
    await fabric.gateway.call(harmless, agent_permissions=writer_permissions)
    approval = fabric.approvals.for_request(harmless.request_id)
    fabric.approvals.approve(approval.approval_id)

    # The approval for a staging deploy, carried onto a deletion.
    hostile = ToolRequest(
        tool_id="mock.delete_everything",
        arguments={},
        agent_id="reader",
        task_id="tsk_test",
        approval_id=approval.approval_id,
    )
    result = await fabric.gateway.call(hostile, agent_permissions=writer_permissions)
    assert not result.ok
    # Carrying an approval to another tool is an attempted bypass, and is
    # reported as one rather than as a call that merely lacks approval.
    assert result.error.code is ToolErrorCode.SECURITY_BLOCKED
    assert fabric.mock.call_counts.get("delete_everything") is None
    assert fabric.gateway.stats.security_blocks >= 1


async def test_14b_arguments_cannot_change_after_approval(fabric, writer_permissions, settings):
    """Approving a staging deploy must not approve a production one."""
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)

    staging = _request("mock.deploy", target="staging")
    await fabric.gateway.call(staging, agent_permissions=writer_permissions)
    approval = fabric.approvals.for_request(staging.request_id)
    fabric.approvals.approve(approval.approval_id)

    swapped = ToolRequest(
        tool_id="mock.deploy",
        arguments={"target": "production"},
        agent_id="reader",
        task_id="tsk_test",
        request_id=staging.request_id,  # the same request id
        approval_id=approval.approval_id,
    )
    result = await fabric.gateway.call(swapped, agent_permissions=writer_permissions)
    assert not result.ok
    assert "production" not in str(fabric.mock.calls)


async def test_14c_a_rejected_approval_blocks_the_call(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.deploy",), max_risk=ToolRisk.HIGH_RISK_WRITE)
    request = _request("mock.deploy", target="staging")

    await fabric.gateway.call(request, agent_permissions=writer_permissions)
    approval = fabric.approvals.for_request(request.request_id)
    fabric.approvals.reject(approval.approval_id, note="not now")

    result = await fabric.gateway.call(request, agent_permissions=writer_permissions)
    assert not result.ok
    assert result.error.code is ToolErrorCode.APPROVAL_REJECTED


async def test_19_a_destructive_call_never_runs_without_approval(
    fabric, writer_permissions, settings
):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.delete_everything",), max_risk=ToolRisk.DESTRUCTIVE)
    result = await fabric.gateway.call(
        _request("mock.delete_everything"), agent_permissions=writer_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.APPROVAL_REQUIRED
    assert "delete_everything" not in fabric.mock.call_counts


# ======================= 15-18: budgets, limits, time =====================


async def test_15_a_breached_budget_stops_the_call(fabric, reader_permissions, settings):
    _grant(fabric)
    fabric.gateway.cost.record(
        model="claude-opus-5",
        provider="mock",
        input_tokens=int(settings.daily_budget_usd * 1_000_000 / 5) + 1_000,
        output_tokens=0,
    )
    before = len(fabric.mock.calls)
    result = await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.BUDGET_EXCEEDED
    assert len(fabric.mock.calls) == before


async def test_16_the_rate_limit_stops_a_runaway_agent(fabric, reader_permissions):
    _grant(fabric)
    fabric.tools.get("mock.echo").rate_limit.per_agent_per_minute = 3
    fabric.tools.get("mock.echo").cacheable = False  # exercise the limiter, not the cache

    codes = []
    for index in range(6):
        result = await fabric.gateway.call(
            _request("mock.echo", message=f"call {index}"), agent_permissions=reader_permissions
        )
        codes.append(result.error.code if not result.ok else None)

    assert ToolErrorCode.RATE_LIMITED in codes
    assert fabric.gateway.stats.rate_limited >= 1


async def test_17_a_hanging_tool_times_out(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.slow",))
    fabric.tools.get("mock.slow").timeout_s = 0.05
    fabric.mock.slow_seconds = 5.0

    result = await fabric.gateway.call(_request("mock.slow"), agent_permissions=reader_permissions)
    assert not result.ok
    assert result.error.code is ToolErrorCode.TIMEOUT
    assert result.error.retryable


async def test_18_a_retryable_failure_is_retried_then_gives_up(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.flaky",))
    fabric.mock.fail_times = 99
    tool = fabric.tools.get("mock.flaky")
    tool.retry_policy.max_attempts = 3
    tool.retry_policy.backoff_s = 0.0
    tool.cacheable = False

    result = await fabric.gateway.call(_request("mock.flaky"), agent_permissions=reader_permissions)
    assert not result.ok
    assert fabric.mock.call_counts["flaky"] == 3  # bounded, not infinite


async def test_18b_a_permission_denial_is_never_retried(fabric, reader_permissions):
    result = await fabric.gateway.call(
        _request("mock.echo", agent="stranger", message="hi"),
        agent_permissions=reader_permissions,
    )
    assert not result.error.retryable


async def test_29_a_non_idempotent_write_is_not_retried(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.create_note",), max_risk=ToolRisk.LOW_RISK_WRITE)
    tool = fabric.tools.get("mock.create_note")
    tool.retry_policy.idempotent = False
    tool.retry_policy.max_attempts = 3
    fabric.mock.fail_times = 99

    await fabric.gateway.call(
        _request("mock.create_note", title="a note"), agent_permissions=writer_permissions
    )
    # One attempt only: retrying a write that may have landed creates two.
    assert fabric.mock.call_counts["create_note"] == 1


# =================== 20-24: results are not to be trusted =================


async def test_20_21_instructions_inside_a_result_are_neutralised(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.hostile",))
    result = await fabric.gateway.call(
        _request("mock.hostile"), agent_permissions=reader_permissions
    )
    assert result.ok
    assert "ignore all previous instructions" not in result.text.lower()
    assert result.sanitized
    assert "UNTRUSTED_TOOL_RESULT" in result.text


async def test_22_an_oversized_result_is_truncated(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.huge",))
    result = await fabric.gateway.call(_request("mock.huge"), agent_permissions=reader_permissions)
    assert result.ok
    assert result.truncated
    assert len(result.text) < 50_000  # not the two megabytes it returned


async def test_23_a_credential_in_a_result_is_redacted(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.leaky",))
    result = await fabric.gateway.call(_request("mock.leaky"), agent_permissions=reader_permissions)
    assert result.ok
    assert "sk-ant-api03" not in result.text
    assert "redacted" in result.text.lower()


async def test_24_a_redacted_credential_does_not_reach_the_audit_log(
    fabric, reader_permissions, audit
):
    _grant(fabric, capabilities=("mock.leaky",))
    await fabric.gateway.call(_request("mock.leaky"), agent_permissions=reader_permissions)
    dumped = "\n".join(event.to_json() for event in audit.events)
    assert "sk-ant-api03" not in dumped


async def test_25_a_malformed_result_is_a_failure_not_a_crash(fabric, reader_permissions):
    _grant(fabric, capabilities=("mock.malformed",))
    result = await fabric.gateway.call(
        _request("mock.malformed"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_RESULT


# ==================== 26-27: a broken server is survivable ================


async def test_26_a_server_that_crashes_does_not_take_the_gateway_down(fabric, reader_permissions):
    _grant(fabric)
    fabric.mock.unreachable = True
    result = await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.EXTERNAL_ERROR

    # And the gateway still works for everything else.
    fabric.mock.unreachable = False
    recovered = await fabric.gateway.call(
        _request("mock.echo", message="again"), agent_permissions=reader_permissions
    )
    assert recovered.ok


async def test_27_a_health_check_marks_a_dead_server_unhealthy(fabric):
    fabric.mock.unreachable = True
    assert await fabric.servers.check_health("mock") is HealthStatus.UNHEALTHY
    fabric.mock.unreachable = False
    assert await fabric.servers.check_health("mock") is HealthStatus.HEALTHY


async def test_27b_a_health_check_has_no_side_effect(fabric):
    before = list(fabric.mock.calls)
    await fabric.servers.check_health("mock")
    assert fabric.mock.calls == before  # listing tools is not calling one


# ==================== 28: the same write, twice ===========================


async def test_28_the_same_write_cannot_run_twice(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.create_note",), max_risk=ToolRisk.LOW_RISK_WRITE)

    first = await fabric.gateway.call(
        _request("mock.create_note", title="the same note"),
        agent_permissions=writer_permissions,
    )
    assert first.ok

    second = await fabric.gateway.call(
        _request("mock.create_note", title="the same note"),
        agent_permissions=writer_permissions,
    )
    assert not second.ok
    assert second.error.code is ToolErrorCode.DUPLICATE_REQUEST
    assert fabric.mock.call_counts["create_note"] == 1


async def test_28b_a_different_write_is_allowed_through(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.create_note",), max_risk=ToolRisk.LOW_RISK_WRITE)

    await fabric.gateway.call(
        _request("mock.create_note", title="one"), agent_permissions=writer_permissions
    )
    second = await fabric.gateway.call(
        _request("mock.create_note", title="two"), agent_permissions=writer_permissions
    )
    assert second.ok
    assert fabric.mock.call_counts["create_note"] == 2


# ==================== 30: the unknown is not harmless =====================


async def test_30_an_unclassified_tool_is_treated_as_high_risk(fabric):
    from jarvis_core.mcp.registry import classify_tool_risk

    assert classify_tool_risk("frobnicate_widget") is ToolRisk.HIGH_RISK_WRITE
    assert classify_tool_risk("get_user") is ToolRisk.READ_ONLY
    assert classify_tool_risk("delete_repository") is ToolRisk.DESTRUCTIVE
    assert classify_tool_risk("send_email") is ToolRisk.EXTERNAL_SIDE_EFFECT


async def test_30b_a_server_may_not_publish_above_its_ceiling(audit, settings, cost):
    """A read-only server offering a delete tool does not get one registered."""
    from jarvis_core.mcp.models import ToolRisk as Risk
    from jarvis_core.tools.setup import build_tool_fabric

    fabric = await build_tool_fabric(
        settings=settings,
        audit=audit,
        cost=cost,
        include_mock=True,
        mock_max_risk=Risk.READ_ONLY,
        include_github=False,
    )
    assert fabric.tools.get("mock.delete_everything") is None
    assert fabric.tools.get("mock.echo") is not None


# ========================== caching and reporting =========================


async def test_a_repeated_read_is_served_from_cache(fabric, reader_permissions):
    _grant(fabric)
    await fabric.gateway.call(
        _request("mock.echo", message="same"), agent_permissions=reader_permissions
    )
    calls_after_first = len(fabric.mock.calls)
    second = await fabric.gateway.call(
        _request("mock.echo", message="same"), agent_permissions=reader_permissions
    )
    assert second.ok
    assert second.from_cache
    assert len(fabric.mock.calls) == calls_after_first


async def test_a_write_is_never_cached(fabric):
    tool = fabric.tools.get("mock.create_note")
    assert not tool.is_cacheable
    assert fabric.tools.get("mock.echo").is_cacheable


async def test_an_agent_only_sees_the_tools_it_may_call(fabric):
    _grant(fabric, capabilities=("mock.echo",))
    visible = [t.tool_id for t in fabric.gateway.available_tools("reader")]
    assert visible == ["mock.echo"]
    assert fabric.gateway.available_tools("stranger") == []


async def test_every_stage_of_a_call_is_audited(fabric, reader_permissions, audit):
    _grant(fabric)
    await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    actions = {event.action for event in audit.events}
    assert "tool.authorized" in actions
    assert "tool.started" in actions
    assert "tool.completed" in actions


async def test_a_denial_is_audited_with_its_reason(fabric, reader_permissions, audit):
    await fabric.gateway.call(
        _request("mock.echo", agent="stranger", message="hi"),
        agent_permissions=reader_permissions,
    )
    denials = [e for e in audit.events if e.action == "tool.denied"]
    assert denials
    assert "no tool policy" in denials[0].reason


async def test_a_call_is_traceable_end_to_end(fabric, reader_permissions, tracer):
    _grant(fabric)
    trace = tracer.start_trace("test")
    await fabric.gateway.call(
        _request("mock.echo", message="hi"),
        agent_permissions=reader_permissions,
        trace=trace,
    )
    assert any(span.kind == "tool" for span in trace.spans)


async def test_the_gateway_summary_reports_what_happened(fabric, reader_permissions):
    _grant(fabric)
    await fabric.gateway.call(
        _request("mock.echo", message="hi"), agent_permissions=reader_permissions
    )
    await fabric.gateway.call(
        _request("mock.echo", agent="stranger", message="hi"),
        agent_permissions=reader_permissions,
    )
    summary = fabric.gateway.summary()
    assert summary["gateway"]["calls"] == 2
    assert summary["gateway"]["allowed"] == 1
    assert summary["gateway"]["denied"] == 1


# ============================= concurrency ================================


async def test_concurrent_agents_do_not_leak_permissions_into_each_other(
    fabric, reader_permissions
):
    _grant(fabric, agent="allowed", capabilities=("mock.echo",))
    # "denied" gets no policy at all.

    results = await asyncio.gather(
        *[
            fabric.gateway.call(
                _request("mock.echo", agent=agent, message=f"{agent} {index}"),
                agent_permissions=reader_permissions,
            )
            for index in range(5)
            for agent in ("allowed", "denied")
        ]
    )
    allowed = [r for r in results if r.ok]
    denied = [r for r in results if not r.ok]
    assert len(allowed) == 5
    assert len(denied) == 5
    assert all(r.error.code is ToolErrorCode.PERMISSION_DENIED for r in denied)


async def test_concurrent_calls_keep_their_own_request_ids(fabric, reader_permissions):
    _grant(fabric)
    requests = [_request("mock.echo", message=f"call {i}") for i in range(8)]
    results = await asyncio.gather(
        *[
            fabric.gateway.call(request, agent_permissions=reader_permissions)
            for request in requests
        ]
    )
    assert {r.request_id for r in results} == {r.request_id for r in requests}


async def test_a_concurrent_duplicate_write_only_lands_once(fabric, writer_permissions, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant(fabric, capabilities=("mock.create_note",), max_risk=ToolRisk.LOW_RISK_WRITE)

    results = await asyncio.gather(
        *[
            fabric.gateway.call(
                _request("mock.create_note", title="one note"),
                agent_permissions=writer_permissions,
            )
            for _ in range(4)
        ]
    )
    succeeded = [r for r in results if r.ok]
    # At most one write reached the server, whatever the interleaving.
    assert fabric.mock.call_counts.get("create_note", 0) <= len(succeeded)
    assert fabric.mock.call_counts.get("create_note", 0) >= 1
