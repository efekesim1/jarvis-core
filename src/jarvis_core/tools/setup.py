"""Assembling the tool fabric.

One function that wires servers, discovery, the registry, policy and the
gateway together, so the CLI, JARVIS and the tests all get the same thing
rather than three slightly different arrangements.

The defaults are the safe ones: every server starts capped at read-only, every
shipped agent starts with read-only grants, and nothing is registered that
nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..integrations.codegraph import CodeGraphServer
from ..integrations.files import WorkspaceFilesServer
from ..integrations.github import GitHubReadOnlyServer
from ..integrations.web import WebServer
from ..logging import AuditLog, get_logger
from ..mcp.mock_server import MOCK_RISKS, MockMCPServer
from ..mcp.models import ToolRisk, TrustLevel
from ..mcp.registry import MCPServerRegistry, ToolDiscovery
from ..observability.cost import CostEngine
from ..observability.tracing import Tracer
from .approval import ApprovalEngine
from .gateway import ToolGateway
from .policy import PolicyEngine, default_policies
from .ratelimit import RateLimiter
from .registry import ToolRegistry

log = get_logger("tools.setup")


@dataclass(slots=True)
class ToolFabric:
    """Everything the tool layer needs, assembled and wired."""

    servers: MCPServerRegistry
    tools: ToolRegistry
    policy: PolicyEngine
    approvals: ApprovalEngine
    gateway: ToolGateway
    discovery: ToolDiscovery
    mock: MockMCPServer | None = None
    github: GitHubReadOnlyServer | None = None
    files: WorkspaceFilesServer | None = None
    web: WebServer | None = None
    codegraph: CodeGraphServer | None = None

    def summary(self) -> dict[str, Any]:
        return self.gateway.summary()

    def coverage(self, specs: Any) -> list[Any]:
        """Which agents can actually do what they declare."""
        from .policy import capability_coverage

        return capability_coverage(specs, self.tools.all())


async def build_tool_fabric(
    *,
    settings: Settings | None = None,
    audit: AuditLog | None = None,
    cost: CostEngine | None = None,
    tracer: Tracer | None = None,
    router: Any | None = None,
    include_mock: bool = False,
    mock_max_risk: ToolRisk = ToolRisk.READ_ONLY,
    include_github: bool = True,
    workspace: Path | None = None,
    storage: Any | None = None,
) -> ToolFabric:
    """Build the fabric and discover what the registered servers publish.

    ``include_mock`` is off outside tests: a server that returns hostile
    content on request has no business in a running system.
    """
    settings = settings or get_settings()
    audit = audit or AuditLog()
    servers = MCPServerRegistry(audit=audit)
    tools = ToolRegistry(audit=audit)
    discovery = ToolDiscovery(servers, audit=audit)
    policy = PolicyEngine(policies=default_policies(), environment=settings.environment)
    # With storage, an approval raised here can be decided from another
    # process. Without it, only this one can see it.
    approvals = ApprovalEngine(
        audit=audit, store=storage.approvals if storage is not None else None
    )

    mock: MockMCPServer | None = None
    github: GitHubReadOnlyServer | None = None
    files: WorkspaceFilesServer | None = None
    web: WebServer | None = None
    codegraph: CodeGraphServer | None = None

    if router is not None:
        # Web reading behind the gateway: one small model call per distinct
        # query, and after that the cache, an audit line and a provenance
        # record that the model's own in-agent search never produced.
        web = WebServer(router)
        if web.available:
            servers.register_in_process(
                "web",
                web,
                name="Web (search and fetch)",
                trust_level=TrustLevel.LIMITED,
                max_risk=ToolRisk.READ_ONLY,
            )
            tools.register_many(await discovery.discover("web", capability_prefix="web.read"))
        else:
            log.info("no model with web access is configured; the web tools are not registered")
            web = None

    if workspace is not None:
        files = WorkspaceFilesServer(workspace)
        servers.register_in_process(
            "files",
            files,
            name=f"Workspace files ({files.workspace.name})",
            trust_level=TrustLevel.TRUSTED,
            max_risk=ToolRisk.READ_ONLY,
        )
        tools.register_many(await discovery.discover("files", capability_prefix="files.read"))

        # The code graph answers questions about this same workspace by
        # traversal rather than by grep. It is an external program and an
        # optional one: when it is absent, or nothing is indexed yet, the
        # tools are not registered at all rather than registered and failing.
        # No agent declares `graph.read`, so it is default-denied even when
        # present — granting it is a separate, deliberate edit.
        codegraph = CodeGraphServer(files.workspace)
        if codegraph.available:
            servers.register_in_process(
                "codegraph",
                codegraph,
                name=f"Code graph ({codegraph.workspace.name})",
                trust_level=TrustLevel.TRUSTED,
                max_risk=ToolRisk.READ_ONLY,
            )
            tools.register_many(
                await discovery.discover("codegraph", capability_prefix="graph.read")
            )
        else:
            log.info("the code graph is not available: %s", codegraph.why_unavailable())
            codegraph = None

    if include_mock:
        mock = MockMCPServer()
        servers.register_in_process(
            "mock",
            mock,
            name="Mock MCP server",
            trust_level=TrustLevel.LIMITED,
            # The mock publishes destructive tools on purpose. Tests raise
            # this ceiling to exercise them; a running system does not.
            max_risk=mock_max_risk,
        )
        tools.register_many(
            await discovery.discover(
                "mock",
                capability_prefix="mock",
                risk_overrides={name: ToolRisk(risk) for name, risk in MOCK_RISKS.items()},
            )
        )

    if include_github:
        github = GitHubReadOnlyServer()
        servers.register_in_process(
            "github",
            github,
            name="GitHub (read-only)",
            trust_level=TrustLevel.TRUSTED,
            max_risk=ToolRisk.READ_ONLY,
        )
        discovered = await discovery.discover("github", capability_prefix="github.read")
        tools.register_many(discovered)
        if not github.has_credentials:
            log.info(
                "no GITHUB_TOKEN is set; the GitHub tools will answer from the offline fixture"
            )

    # With storage, the rate limit window is shared between every process on
    # this machine. Without it, a limit of thirty a minute is thirty per
    # process, and two processes make it sixty — which is not a limit.
    limiter = RateLimiter(store=storage.rate_limits if storage is not None else None)
    if not limiter.shared:
        log.info("rate limits are per-process; set JARVIS_DATABASE_PATH to share them")

    gateway = ToolGateway(
        servers=servers,
        tools=tools,
        policy=policy,
        approvals=approvals,
        rate_limiter=limiter,
        cost=cost,
        audit=audit,
        tracer=tracer,
        settings=settings,
    )
    log.info("tool fabric ready: %d server(s), %d tool(s)", len(servers.all()), len(tools))
    return ToolFabric(
        servers=servers,
        tools=tools,
        policy=policy,
        approvals=approvals,
        gateway=gateway,
        discovery=discovery,
        mock=mock,
        github=github,
        files=files,
        web=web,
        codegraph=codegraph,
    )
