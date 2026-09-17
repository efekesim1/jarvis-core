"""The command line.

The primary interface, and the one that needs nothing installed beyond the
package itself.

    jarvis status
    jarvis agents
    jarvis models
    jarvis plan "research the market for AI note taking apps"
    jarvis ask  "research the market for AI note taking apps"

`plan` costs nothing and is the honest way to see what a request would do
before paying for it. `ask` spends money.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from .agents.registry import build_registry
from .config import get_settings
from .core.planner import RulePlanner
from .core.runtime import build_runtime, build_runtime_with_tools
from .logging import configure_logging
from .models.router import build_default_router
from .security.approval import ApprovalDecision, ApprovalRequest

app = typer.Typer(
    name="jarvis",
    help="jarvis-core: a multi-agent runtime with permissions, budgets and approvals.",
    no_args_is_help=True,
    add_completion=False,
)
approvals_app = typer.Typer(help="What is waiting for a human, and deciding it.")
app.add_typer(approvals_app, name="approvals")
console = Console()


class ConsoleApprovalGate:
    """Asks the person at the terminal before anything risky runs."""

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        console.print(
            Panel(
                f"[bold]{request.action}[/bold]\n\n{request.reason}",
                title=f"Approval needed - risk: {request.risk}",
                border_style="yellow",
            )
        )
        answer = await asyncio.to_thread(typer.confirm, "Approve?", default=False)
        return ApprovalDecision.APPROVED if answer else ApprovalDecision.DENIED


def _print_response(response: Any, as_json: bool) -> None:
    if as_json:
        console.print(JSON(json.dumps(response.as_dict(), ensure_ascii=False, default=str)))
        return

    console.print(Panel(response.answer or "(no answer)", title="JARVIS", border_style="cyan"))

    table = Table(title="Run", show_header=True, header_style="bold")
    table.add_column("step")
    table.add_column("agent")
    table.add_column("status")
    table.add_column("score", justify="right")
    table.add_column("cost", justify="right")
    for outcome in response.report.outcomes:
        score = f"{outcome.evaluation.overall:.2f}" if outcome.evaluation else "-"
        cost = f"${outcome.result.cost_usd:.4f}" if outcome.result else "-"
        state = "ok" if outcome.ok else ("skipped" if outcome.skipped else "failed")
        table.add_row(outcome.step_id, outcome.agent_id, state, score, cost)
    console.print(table)

    console.print(
        f"intent: [bold]{response.intent.value}[/bold]   "
        f"confidence: [bold]{response.confidence:.2f}[/bold]   "
        f"cost: [bold]${response.cost_usd:.4f}[/bold]   "
        f"trace: {response.trace_id}"
    )
    for source in response.sources[:12]:
        console.print(f"  - {source.get('title') or ''} {source.get('url') or ''}".rstrip())
    for warning in response.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


# -- the two that matter ----------------------------------------------------


@app.command()
def ask(
    request: Annotated[str, typer.Argument(help="What you want done.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the whole run as JSON.")] = False,
    approve: Annotated[
        bool, typer.Option("--approve/--no-approve", help="Ask at the terminal for risky actions.")
    ] = True,
) -> None:
    """Plan a request, run the agents it needs, and report what it cost.

    This spends money. `jarvis plan` shows what it would do for nothing.
    """
    configure_logging()
    runtime = asyncio.run(
        build_runtime_with_tools(approval_gate=ConsoleApprovalGate() if approve else None)
    )
    if not runtime.router.usable_models():
        console.print(
            "[red]No model provider is configured.[/red] Set ANTHROPIC_API_KEY, or run a "
            "local Ollama daemon, and try again. `jarvis models` shows what is usable."
        )
        raise typer.Exit(code=2)
    response = asyncio.run(runtime.handle(request))
    _print_response(response, as_json)
    raise typer.Exit(code=0 if response.ok else 1)


@app.command()
def plan(
    request: Annotated[str, typer.Argument(help="The request to plan, without running it.")],
) -> None:
    """Show the plan for a request without spending anything."""
    configure_logging()
    built = RulePlanner(build_registry()).plan(request)
    console.print(JSON(json.dumps(built.as_dict(), ensure_ascii=False, indent=2)))
    console.print("\n[bold]Execution waves[/bold] (each wave runs in parallel)")
    for index, wave in enumerate(built.waves(), start=1):
        console.print(f"  {index}. " + ", ".join(f"{s.step_id}:{s.agent_id}" for s in wave))


# -- what is here -----------------------------------------------------------


@app.command()
def agents(
    swarm: Annotated[str, typer.Option(help="Only this swarm.")] = "",
    capabilities: Annotated[
        bool, typer.Option("--capabilities", help="Show which declared tools actually exist.")
    ] = False,
) -> None:
    """The registered agents, their models, tools and permissions."""
    configure_logging()
    registry = build_registry()
    specs = [s for s in registry.all() if not swarm or s.swarm == swarm]

    table = Table(title=f"{len(specs)} agent(s)", show_header=True, header_style="bold")
    table.add_column("id")
    table.add_column("name")
    table.add_column("swarm")
    table.add_column("permissions")
    table.add_column("tools" if capabilities else "role")
    for spec in specs:
        table.add_row(
            spec.id,
            spec.name,
            spec.swarm,
            ", ".join(p.value for p in spec.permissions) or "-",
            ", ".join(spec.tool_capabilities) or "-" if capabilities else spec.role,
        )
    console.print(table)
    if capabilities:
        console.print(
            "\nDeclaring a capability is not being granted it: the tool policy decides "
            "what an agent may actually call."
        )


@app.command()
def models() -> None:
    """Which models the router can actually use, and at what price."""
    configure_logging()
    router = build_default_router()
    table = Table(title="Model catalogue", show_header=True, header_style="bold")
    table.add_column("model")
    table.add_column("provider")
    table.add_column("tier")
    table.add_column("in $/Mtok", justify="right")
    table.add_column("out $/Mtok", justify="right")
    table.add_column("usable")
    usable = {s.id for s in router.usable_models()}
    for spec in router.catalog.all():
        table.add_row(
            spec.id,
            spec.provider,
            spec.tier,
            f"{spec.input_cost_per_mtok:.2f}",
            f"{spec.output_cost_per_mtok:.2f}",
            "yes" if spec.id in usable else "no",
        )
    console.print(table)

    # The local tier is the only one whose absence is worth a sentence: every
    # other provider is missing a credential, which is a decision, while this
    # one is missing a process, which is usually an oversight.
    from .models.local_provider import OllamaProvider

    local = OllamaProvider()
    blocked = local.why_unavailable()
    if blocked is None:
        free = ", ".join(local.installed_models())
        console.print(f"\n[green]Local models at no cost[/green]: {free}")
    else:
        console.print(f"\n[yellow]No local model tier:[/yellow] {blocked}")


@app.command()
def status(
    as_json: Annotated[bool, typer.Option("--json", help="Print the snapshot as JSON.")] = False,
) -> None:
    """Everything the runtime can account for, in one snapshot."""
    configure_logging()
    snapshot = build_runtime().status()
    if as_json:
        console.print(JSON(json.dumps(snapshot, ensure_ascii=False, default=str)))
        return
    for section, value in snapshot.items():
        console.print(f"[bold]{section}[/bold]: {value}")


@app.command()
def cost() -> None:
    """What has been spent, by day, model, agent and task."""
    configure_logging()
    console.print(JSON(json.dumps(build_runtime().cost.summary(), ensure_ascii=False, default=str)))


@app.command()
def memory(
    query: Annotated[str, typer.Argument(help="What to look for.")] = "",
    limit: Annotated[int, typer.Option(help="How many items at most.")] = 10,
) -> None:
    """Search what has been remembered."""
    configure_logging()
    runtime = build_runtime()
    if not query:
        console.print(JSON(json.dumps(runtime.memory.stats(), ensure_ascii=False, default=str)))
        return
    for item in runtime.memory.recall(query)[: max(1, limit)]:
        console.print(f"  [{item.layer.value}] {item.content[:160]}")


@app.command()
def tools() -> None:
    """What the gateway publishes, and what the policy says about it."""
    configure_logging()
    runtime = asyncio.run(build_runtime_with_tools())
    if runtime.tools is None:
        console.print("[yellow]No tool fabric is attached.[/yellow]")
        raise typer.Exit(code=2)
    console.print(JSON(json.dumps(runtime.tools.summary(), ensure_ascii=False, default=str)))


@app.command()
def verify(
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Where each agent stands on the verification ladder.

    Offline. It proves nothing is broken; it does not show any agent is good.
    """
    configure_logging()
    report = build_runtime().verification.summary()
    if as_json:
        console.print(JSON(json.dumps(report, ensure_ascii=False, default=str)))
        return
    for key, value in report.items():
        console.print(f"[bold]{key}[/bold]: {value}")


@app.command()
def doctor() -> None:
    """Say what is configured, what is missing, and what that costs you."""
    configure_logging()
    settings = get_settings()
    router = build_default_router()
    usable = router.usable_models()

    table = Table(title="Setup", show_header=True, header_style="bold")
    table.add_column("thing")
    table.add_column("state")
    table.add_column("what it means")
    table.add_row(
        "model provider",
        "ok" if usable else "missing",
        f"{len(usable)} usable model(s)" if usable else "nothing can run; see .env.example",
    )
    table.add_row(
        "database",
        "ok" if settings.database_path else "in memory",
        "durable tasks, memory, approvals and audit"
        if settings.database_path
        else "state is lost when the process exits",
    )
    table.add_row(
        "environment",
        settings.environment,
        "the sandbox refuses money, deploys and admin rights whatever an agent asks for",
    )
    table.add_row(
        "daily budget",
        f"${settings.daily_budget_usd:.2f}",
        "the router steps down a tier at 70% and stops at 100%",
    )
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8000,
) -> None:
    """Run the HTTP interface.

    It refuses to start without `JARVIS_API_TOKEN`, because an unauthenticated
    port exposing what has been spent and what is awaiting approval is a data
    leak with a REST interface.
    """
    configure_logging()
    try:
        import uvicorn
    except ImportError:
        console.print('[red]The API extra is not installed.[/red] pip install "jarvis-core[api]"')
        raise typer.Exit(code=2) from None

    from .api.security import MissingToken, new_token
    from .api.server import create_app

    try:
        application = create_app()
    except MissingToken as exc:
        console.print(f"[red]{exc}[/red]\n\nA token worth using: {new_token()}")
        raise typer.Exit(code=2) from None
    uvicorn.run(application, host=host, port=port, log_level="info")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
