"""End to end: a request from the operator reaching a tool and coming back.

These tests exercise the whole chain rather than one gate, including the two
that matter most for autonomy: a high-risk call stopping for approval, and the
night shift refusing to approve on the operator's behalf.
"""

from __future__ import annotations

import pytest
from fake_credentials import fake_github_token

from jarvis_core.agents.base import AgentContext, AgentResult, AgentSpec
from jarvis_core.core.orchestrator import AgentOrchestrator
from jarvis_core.core.planner import Intent, Plan, PlanStep
from jarvis_core.core.scheduler import JobStatus, ScheduledJob, Scheduler
from jarvis_core.core.task import Task, TaskEngine
from jarvis_core.evaluation.engine import EvaluationEngine
from jarvis_core.mcp.errors import ToolErrorCode
from jarvis_core.mcp.models import ToolRisk
from jarvis_core.security.permissions import Permission
from jarvis_core.tools.policy import AgentToolPolicy


def _grant_deploy_to_everyone(jarvis, fabric) -> None:
    """Let every agent try the deploy tool, so the test does not depend on
    which of them the planner happens to staff the request with."""
    for spec in jarvis.registry.all():
        spec.permissions = spec.permissions | {Permission.WRITE}
        fabric.policy.grant(
            AgentToolPolicy(
                agent_id=spec.id,
                capabilities=frozenset({"mock.deploy"}),
                max_risk=ToolRisk.HIGH_RISK_WRITE,
            )
        )


@pytest.fixture
def tooled_jarvis(jarvis, fabric):
    jarvis.attach_tools(fabric)
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="web_researcher",
            capabilities=frozenset({"mock.echo", "mock.lookup", "github.read"}),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    return jarvis


# ===================== the full chain, mocked end to end ==================


async def test_an_agent_reaches_a_tool_through_the_gateway(tooled_jarvis, fabric, tracer):
    """USER -> JARVIS -> agent -> gateway -> policy -> MCP -> result."""
    spec = tooled_jarvis.registry.require("web_researcher")
    ctx = AgentContext(
        router=tooled_jarvis.router,
        guardian=tooled_jarvis.guardian,
        audit=tooled_jarvis.audit,
        tracer=tracer,
        trace=tracer.start_trace("e2e"),
        gateway=fabric.gateway,
    )
    task = Task(goal="look something up")

    result = await ctx.call_tool(spec, task, "mock.echo", {"message": "hello from the agent"})

    assert result.ok
    assert "hello from the agent" in result.text
    assert result.untrusted


async def test_an_agent_with_no_gateway_gets_a_failure_not_a_crash(tracer, jarvis):
    spec = jarvis.registry.require("web_researcher")
    ctx = AgentContext(
        router=jarvis.router,
        guardian=jarvis.guardian,
        audit=jarvis.audit,
        tracer=tracer,
        trace=tracer.start_trace("e2e"),
        gateway=None,
    )
    result = await ctx.call_tool(spec, Task(goal="x"), "mock.echo", {"message": "hi"})
    assert not result.ok
    assert result.error.code is ToolErrorCode.TOOL_NOT_FOUND


async def test_an_agent_cannot_call_a_tool_outside_its_grant(tooled_jarvis, fabric, tracer):
    spec = tooled_jarvis.registry.require("web_researcher")
    ctx = AgentContext(
        router=tooled_jarvis.router,
        guardian=tooled_jarvis.guardian,
        audit=tooled_jarvis.audit,
        tracer=tracer,
        trace=tracer.start_trace("e2e"),
        gateway=fabric.gateway,
    )
    result = await ctx.call_tool(spec, Task(goal="x"), "mock.delete_everything", {})
    assert not result.ok
    assert result.error.code in (ToolErrorCode.PERMISSION_DENIED, ToolErrorCode.RISK_DENIED)
    assert "delete_everything" not in fabric.mock.call_counts


async def test_the_full_run_carries_the_gateway_to_every_agent(tooled_jarvis, fabric):
    """A whole JARVIS run: every agent gets the same gateway, nobody a transport."""
    seen: list[object] = []

    class Recording:
        name = "llm"

        async def run(self, spec, task, ctx):
            seen.append(ctx.gateway)
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=True,
                text="a finding with enough words to score properly " * 8,
                data={"summary": "ok"},
                confidence=0.8,
            )

    tooled_jarvis.registry.register_runtime(Recording())
    response = await tooled_jarvis.handle("research the AI agent market")

    assert response.ok
    assert seen
    assert all(gateway is fabric.gateway for gateway in seen)


async def test_the_status_report_includes_the_tool_fabric(tooled_jarvis):
    status = tooled_jarvis.status()
    assert status["tools"]["tools"]["tools"] > 0
    assert status["tools"]["servers"] >= 1


async def test_a_run_without_a_fabric_still_works(jarvis):
    """Tools are an addition, not a requirement. Nothing regressed."""
    response = await jarvis.handle("what is MCP?")
    assert response.ok
    assert jarvis.status()["tools"] == {"attached": False}


# ======================== the approval flow ===============================


async def test_a_high_risk_call_blocks_then_runs_once_approved(
    tooled_jarvis, fabric, tracer, settings
):
    settings.environment = "production"
    fabric.policy.environment = "production"
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="deployment_agent",
            capabilities=frozenset({"mock.deploy"}),
            max_risk=ToolRisk.HIGH_RISK_WRITE,
        )
    )
    spec = AgentSpec.from_dict(
        {
            "id": "deployment_agent",
            "name": "Deployment",
            "role": "deploys things",
            "swarm": "build",
            "permissions": ["read", "write", "deploy"],
        }
    )
    ctx = AgentContext(
        router=tooled_jarvis.router,
        guardian=tooled_jarvis.guardian,
        audit=tooled_jarvis.audit,
        tracer=tracer,
        trace=tracer.start_trace("approval"),
        gateway=fabric.gateway,
    )
    task = Task(goal="deploy the service")

    # The agent asks. Nothing is deployed.
    blocked = await ctx.call_tool(spec, task, "mock.deploy", {"target": "staging"})
    assert not blocked.ok
    assert blocked.error.code is ToolErrorCode.APPROVAL_REQUIRED
    assert "deploy" not in fabric.mock.call_counts

    # the operator decides.
    approval = fabric.approvals.pending[0]
    assert approval.tool_id == "mock.deploy"
    assert "staging" in approval.arguments_summary
    fabric.approvals.approve(approval.approval_id, by="operator")

    # The same request, now approved.
    from jarvis_core.mcp.models import ToolRequest

    approved = await fabric.gateway.call(
        ToolRequest(
            tool_id="mock.deploy",
            arguments={"target": "staging"},
            agent_id="deployment_agent",
            task_id=task.task_id,
            request_id=approval.request_id,
            approval_id=approval.approval_id,
        ),
        agent_permissions=spec.permissions | {Permission.ADMIN},
    )
    assert approved.ok
    assert fabric.mock.call_counts["deploy"] == 1


# ======================= the night shift flow =============================


async def test_the_night_shift_blocks_rather_than_approving_for_efe(
    tooled_jarvis, fabric, settings
):
    """NIGHT SHIFT -> tool needs approval -> BLOCKED -> morning report."""
    settings.environment = "production"
    fabric.policy.environment = "production"
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="web_researcher",
            capabilities=frozenset({"mock.deploy"}),
            max_risk=ToolRisk.HIGH_RISK_WRITE,
        )
    )

    # Whichever agents the planner staffs this with need the grant and the
    # permission, or the run stops one gate earlier than the one under test.
    _grant_deploy_to_everyone(tooled_jarvis, fabric)

    class WantsToDeploy:
        name = "llm"

        async def run(self, spec, task, ctx):
            # The agent tries something that needs a human.
            await ctx.call_tool(spec, task, "mock.deploy", {"target": "production"})
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=True,
                text="I tried to deploy and was stopped, which is correct. " * 8,
                data={"summary": "blocked"},
                confidence=0.8,
            )

    tooled_jarvis.registry.register_runtime(WantsToDeploy())
    scheduler = Scheduler(tooled_jarvis)
    scheduler.submit_request("deploy the new build")

    report = await scheduler.run_shift(budget_usd=5.0)

    assert report.blocked, "the job should be blocked, not completed"
    assert report.blocked[0].status is JobStatus.BLOCKED_APPROVAL
    assert fabric.approvals.pending
    # Nothing was deployed while the operator slept.
    assert "deploy" not in fabric.mock.call_counts


async def test_the_morning_report_names_what_is_waiting_on_efe(tooled_jarvis, fabric, settings):
    settings.environment = "production"
    fabric.policy.environment = "production"
    _grant_deploy_to_everyone(tooled_jarvis, fabric)

    class WantsToDeploy:
        name = "llm"

        async def run(self, spec, task, ctx):
            await ctx.call_tool(spec, task, "mock.deploy", {"target": "production"})
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=True,
                text="stopped for approval " * 12,
                data={"summary": "x"},
                confidence=0.8,
            )

    tooled_jarvis.registry.register_runtime(WantsToDeploy())
    scheduler = Scheduler(tooled_jarvis)
    scheduler.submit_request("deploy the new build")
    report = await scheduler.run_shift(budget_usd=5.0)

    text = report.render()
    assert "Waiting for your approval" in text
    assert "Approvals to decide" in text
    assert "mock.deploy" in text
    assert "jarvis approvals approve" in text


async def test_a_read_only_job_runs_unattended_without_asking(tooled_jarvis, fabric):
    """Low-risk work does not wake anyone up. That is the point of the shift."""

    class Reads:
        name = "llm"

        async def run(self, spec, task, ctx):
            result = await ctx.call_tool(spec, task, "mock.echo", {"message": "overnight"})
            assert result.ok
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=True,
                text="read something overnight without waking anyone " * 8,
                data={"summary": "ok"},
                confidence=0.8,
            )

    tooled_jarvis.registry.register_runtime(Reads())
    scheduler = Scheduler(tooled_jarvis)
    scheduler.submit(ScheduledJob(request="read something", budget_usd=1.0))
    report = await scheduler.run_shift(budget_usd=5.0)

    assert report.completed
    assert not report.blocked
    assert not fabric.approvals.pending


# ==================== the first real integration ==========================


async def test_the_github_tools_are_registered_read_only(fabric):
    tools = fabric.tools.by_server("github")
    assert {t.name for t in tools} == {
        "get_repository",
        "list_issues",
        "search_repositories",
    }
    assert all(t.risk is ToolRisk.READ_ONLY for t in tools)


async def test_github_works_without_a_token(fabric, reader_permissions, monkeypatch):
    """No credential is not a failure: CI has none and must still pass."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="reader",
            capabilities=frozenset({"github.read"}),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    from jarvis_core.mcp.models import ToolRequest

    result = await fabric.gateway.call(
        ToolRequest(
            tool_id="github.get_repository",
            arguments={"owner": "anthropics", "repo": "anthropic-sdk-python"},
            agent_id="reader",
        ),
        agent_permissions=reader_permissions,
    )
    assert result.ok
    assert result.content["_fixture"] is True


async def test_github_arguments_are_validated_before_any_request(fabric, reader_permissions):
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="reader",
            capabilities=frozenset({"github.read"}),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    from jarvis_core.mcp.models import ToolRequest

    result = await fabric.gateway.call(
        ToolRequest(
            tool_id="github.list_issues",
            arguments={"owner": "x", "repo": "y", "limit": 5000},  # over the maximum
            agent_id="reader",
        ),
        agent_permissions=reader_permissions,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


async def test_no_github_write_tool_exists_yet(fabric):
    """Writing to GitHub is a later decision, not an oversight."""
    names = {t.name for t in fabric.tools.by_server("github")}
    for forbidden in ("create_issue", "delete_repo", "push", "merge_pull_request"):
        assert forbidden not in names


async def test_the_github_token_never_appears_in_the_audit_log(
    fabric, reader_permissions, audit, monkeypatch
):
    token = fake_github_token("SHOULDNOTAPPEARINTHEAUDITLOG")
    monkeypatch.setenv("GITHUB_TOKEN", token)
    fabric.policy.grant(
        AgentToolPolicy(
            agent_id="reader",
            capabilities=frozenset({"github.read"}),
            max_risk=ToolRisk.READ_ONLY,
        )
    )
    from jarvis_core.mcp.models import ToolRequest

    await fabric.gateway.call(
        ToolRequest(
            tool_id="github.get_repository",
            arguments={"owner": "x", "repo": "y"},
            agent_id="reader",
        ),
        agent_permissions=reader_permissions,
    )
    dumped = "\n".join(event.to_json() for event in audit.events)
    assert token not in dumped
    assert "SHOULDNOTAPPEARINTHEAUDITLOG" not in dumped
    # And the server record holds a reference, never a value.
    server = fabric.servers.get("github")
    assert token[:4] not in str(server.as_dict())


# ==================== orchestrated, with tools ============================


async def test_a_whole_plan_runs_with_tools_attached(
    tooled_jarvis, fabric, audit, settings, tracer
):
    calls: list[str] = []

    class UsesTools:
        name = "llm"

        async def run(self, spec, task, ctx):
            result = await ctx.call_tool(spec, task, "mock.lookup", {"record_id": 42})
            calls.append(spec.id)
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=result.ok,
                text=f"looked up record 42 and found something worth reporting {result.ok} " * 6,
                data={"summary": "ok"},
                confidence=0.8,
            )

    tooled_jarvis.registry.register_runtime(UsesTools())
    for agent_id in ("web_researcher", "market_researcher", "trend_analyst"):
        fabric.policy.grant(
            AgentToolPolicy(
                agent_id=agent_id,
                capabilities=frozenset({"mock.lookup"}),
                max_risk=ToolRisk.READ_ONLY,
            )
        )

    orchestrator = AgentOrchestrator(
        registry=tooled_jarvis.registry,
        tasks=TaskEngine(),
        evaluator=EvaluationEngine(router=tooled_jarvis.router),
        audit=audit,
        settings=settings,
    )
    ctx = AgentContext(
        router=tooled_jarvis.router,
        guardian=tooled_jarvis.guardian,
        audit=audit,
        tracer=tracer,
        trace=tracer.start_trace("plan"),
        gateway=fabric.gateway,
    )
    plan = Plan(
        request="x",
        intent=Intent.RESEARCH,
        steps=[
            PlanStep(step_id="a", agent_id="web_researcher", goal="one"),
            PlanStep(step_id="b", agent_id="market_researcher", goal="two"),
            PlanStep(step_id="c", agent_id="trend_analyst", goal="three"),
        ],
    )
    report = await orchestrator.run_plan(plan, ctx)

    assert len(report.successful) == 3
    assert len(calls) == 3
    # Three agents asked the same question: the cache answered two of them.
    assert fabric.gateway.stats.cache_hits + fabric.gateway.cache.stats.coalesced >= 2


async def test_a_dead_server_does_not_stop_a_run(tooled_jarvis, fabric, audit, settings, tracer):
    fabric.mock.unreachable = True

    class UsesTools:
        name = "llm"

        async def run(self, spec, task, ctx):
            result = await ctx.call_tool(spec, task, "mock.echo", {"message": "hi"})
            assert not result.ok  # the tool failed, and the agent carried on
            return AgentResult(
                agent_id=spec.id,
                task_id=task.task_id,
                ok=True,
                text="the tool was unavailable, so here is what I know without it " * 6,
                data={"summary": "degraded"},
                confidence=0.5,
            )

    tooled_jarvis.registry.register_runtime(UsesTools())
    response = await tooled_jarvis.handle("research the AI agent market")
    assert response.ok  # JARVIS survived a dead server
