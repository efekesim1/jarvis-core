# Documentation

Start with the [README](../README.md) for what this is and how to run it.
These pages go a layer deeper.

| Page | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The layers, what depends on what, and how one request travels |
| [SECURITY.md](SECURITY.md) | The boundaries, the threat model, and what each control is actually for |
| [security/tool-security.md](security/tool-security.md) | The attacks the tool gateway defends against, one by one |
| [architecture/mcp.md](architecture/mcp.md) | The tool fabric: servers, transports, discovery, the error taxonomy |
| [MODEL_ROUTING.md](MODEL_ROUTING.md) | How a model is chosen, and what happens when one fails |
| [COST_CONTROL.md](COST_CONTROL.md) | Budgets, the cache, coalescing, and what the controls measurably save |
| [MEMORY.md](MEMORY.md) | The layers, the bar for writing, and the budget for recall |
| [EVALUATION.md](EVALUATION.md) | How an agent's output is scored before the orchestrator accepts it |
| [agents/roster.md](agents/roster.md) | The agents that ship, and the two ways they reach the world |
| [operations/storage.md](operations/storage.md) | What is durable, what is not, and how to back it up |
| [operations/experiments.md](operations/experiments.md) | The sandbox, and why a passing run is evidence rather than consent |

Two things worth knowing before reading any of it:

**Everything here runs offline.** The test suite, the demos and the
benchmarks use an in-process provider. If a page describes behaviour you want
to check, there is almost always a command that shows it without a credential.

**The refusals are the design.** Much of this system is about what it will
not do — default-deny tools, bounded retries, approvals that cannot be
replayed, a sandbox that gets no rights. Where a page explains a limit, the
limit is the feature.
