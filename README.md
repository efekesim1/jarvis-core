# jarvis-core

A multi-agent, multi-model AI runtime that plans work, routes it to the right
model, controls what its agents may touch, keeps the bill inside a budget, and
stops for a human before anything with consequences.

It is infrastructure, not an assistant. You give it a request; it decides
which agents are needed, runs them in dependency order against whichever
models are configured, scores what comes back, merges it into one answer, and
tells you what it cost. Everything an agent can reach goes through one gateway
with a policy in front of it, and every decision lands in an audit trail.

The whole test suite runs offline against an in-process provider: no API key,
no network, nothing billed.

```bash
pip install -e ".[dev]"
python examples/demo_offline.py     # the full pipeline, no credential needed
```

## Why it exists

Most agent frameworks make the model call easy and leave the hard parts to
you: what an agent is allowed to do, what happens when it retries forever,
what it costs, who approved the deployment it just attempted, and how you find
out afterwards. Those are the parts this is about.

Four rules are structural rather than configurable:

| Rule | Where it is enforced |
|---|---|
| No agent reaches a tool directly | `tools/gateway.py` — there is no import path from `agents/` to a transport |
| Default deny | An agent with no policy calls nothing; grants are capabilities, never blanket access |
| A tool result is never an instruction | Redacted, sanitised, truncated and fenced before an agent sees it |
| Nothing unbounded | Retries, plan waves, agent hops, context, spend and wall-clock all carry a ceiling |

## Architecture

Each layer depends only on the ones above it.

```
config / logging / text     settings, audit trail, term extraction
        |
security                    permissions, risk, sensitivity, secrets, approval
        |
memory                      what is remembered, and what it refuses to keep
        |
storage                     SQLite, and the stores that use it (optional)
        |
models                      providers, catalogue, router, cost, token estimation
        |
mcp                         servers, transports, discovery, error taxonomy
        |
tools                       the gateway: policy, approval, validation, limits
        |
agents                      definitions, registry, runtimes
        |
core                        context, tasks, planner, orchestrator, runtime
        |
api / cli                   interfaces
```

`integrations/` sits beside `mcp/`: each adapter is an in-process server, so
the gateway cannot tell a GitHub adapter from a real MCP server.
`experiments/` is the sandbox. `evaluation/` scores an agent's output before
the orchestrator accepts it. `verification/` records whether an agent has ever
actually been shown to work, which is a different question from whether it is
defined.

One request takes this path:

```
REQUEST -> RECALL -> PLAN -> WAVES OF AGENTS -> EVALUATION -> SYNTHESIS -> ANSWER
                               |
                               +-> TOOL GATEWAY -> policy, approval, limits, audit
```

## What is in it

- **Planning and orchestration** — a rule planner and a model planner, task
  graphs with dependency waves, bounded retries, bounded hops.
- **Agent registry** — agents are YAML, not subclasses. One file adds an
  agent, including the tools it declares; a declared capability that nothing
  backs fails the build.
- **Model router** — picks a tier from the kind of work, steps down a tier
  when the day's budget runs hot, retries a failure on a different model, and
  prices every call.
- **Providers** — Anthropic natively; OpenAI, NVIDIA NIM, xAI and Gemini
  through the Chat Completions protocol; a local Ollama tier that costs
  nothing and needs no credential; and an in-process provider the tests use.
- **Cost control** — per call, task, agent, model, day and month, with a
  result cache and request coalescing in front of it.
- **Tool gateway and MCP** — discovery, transports, an error taxonomy, per-tool
  rate limits, argument and result validation, redaction, and an approval gate
  for anything high-risk.
- **Permissions and approvals** — capability grants per agent, risk
  classification per call, and a human decision that is recorded and can be
  answered from another process.
- **Sandbox** — experiments run isolated; a run that passes is evidence, not
  consent, and nothing is adopted automatically.
- **Memory** — layered, with a bar for writing and a budget for recall.
- **Observability** — an append-only audit trail, tracing per agent run and
  plan step, and cost accounting that reconciles with it.
- **Interfaces** — a CLI, and an HTTP API behind a bearer token.

## Installation

Python 3.11 or 3.12.

```bash
git clone https://github.com/efekesim1/jarvis-core
cd jarvis-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras: `api` for the HTTP interface, `openai` for the Chat
Completions vendors, `mcp` for stdio transports to real MCP servers.

## Quick start

Nothing below needs a credential.

```bash
jarvis doctor                 # what is configured and what that costs you
jarvis agents                 # the registered agents
jarvis models                 # which models are usable, and at what price
jarvis plan "research the market for AI note taking apps"
make demo                     # the whole pipeline against the offline provider
```

With a credential in place, `jarvis ask "..."` runs it for real. `jarvis plan`
is the honest way to see what a request would do before paying for it.

## A minimal runnable example

```python
import asyncio

from jarvis_core.core.runtime import JarvisRuntime
from jarvis_core.models.mock_provider import MockProvider, agent_reply
from jarvis_core.models.router import ModelRouter


async def main() -> None:
    # The in-process provider: no key, no network, no cost. Swap it for
    # `build_runtime()` to use whichever real providers are configured.
    provider = MockProvider(responder=lambda request: agent_reply("the market is young", 0.8))
    provider.name = "anthropic"          # so routing and pricing take the real paths
    runtime = JarvisRuntime(router=ModelRouter(providers={"anthropic": provider}),
                            use_model_planner=False)

    response = await runtime.handle("research the market for AI note taking apps")

    print(response.answer)
    print(f"{len(response.report.outcomes)} agent(s), ${response.cost_usd:.6f}")


asyncio.run(main())
```

`response.as_dict()` carries the plan, every agent outcome, the evaluation
scores, the token counts and the trace id.

## Security and the approval model

The threat model assumes the model is not trusted, the tool result is not
trusted, and the agent's output is not trusted either.

- **Tools are authorised separately from agents.** Declaring `web_search` does
  not grant it; the agent needs the matching permission, and the tool policy
  is a default-deny allowlist.
- **High-risk calls stop for a human.** Deploys, deletions and external side
  effects raise an approval bound to that one call, which cannot be replayed
  against another. An unattended run blocks rather than approving on your
  behalf.
- **The sandbox never gets money, deploy or admin rights**, whatever an agent
  asks for.
- **File tools never leave the workspace.** Paths are resolved and then
  checked, so a symlink out is caught; `.env`, keys and `.git` are never
  readable.
- **Injection is treated as input.** Agent output and tool results are
  sanitised and fenced before another agent reads them, and a hit is audited
  as a security event.
- **Secrets come from the environment only.** Nothing reads a credential from
  source, prints one, or writes one to a log, and CI fails on a committed key.
- **No SQL is built from a variable.** Queries are literals chosen between;
  every value is a bound parameter.

`SECURITY.md` says how to report something. `docs/SECURITY.md` describes the
boundaries and the attacks they are for.

## Models and providers

| Provider | How it is reached | Needs |
|---|---|---|
| Anthropic | native SDK | `ANTHROPIC_API_KEY` |
| OpenAI | Chat Completions | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| NVIDIA NIM | Chat Completions | `NVIDIA_API_KEY`, `NVIDIA_MODEL` |
| xAI | Chat Completions | `XAI_API_KEY`, `XAI_MODEL` |
| Gemini | OpenAI-compatibility endpoint | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Ollama | local HTTP, no credential | a running daemon and a pulled model |
| Mock | in process | nothing; this is what the tests use |

A new model is one entry in `config/models.json`. A new provider is a class
with `name`, `available()` and `complete()`.

The local tier is worth a note: every pulled Ollama model becomes a `$0` entry
in the catalogue automatically, and local models enter the `fast` tier and no
higher — so simple, high-volume work is free, and because every tier steps
down towards `fast` under budget pressure, the budget ceiling becomes a change
of gear instead of a wall. Planning and synthesis are never routed there, on
the grounds that a free model should not win that work by being free.

## Development

```bash
make install     # virtualenv and dev extras
make lint        # ruff check and ruff format --check
make test        # the suite, offline
make check       # the full gate: lint, tests, demos, benchmarks
make security    # bandit and pip-audit
```

`CONTRIBUTING.md` has the rest, including what a reviewable pull request looks
like here.

## Tests

```bash
make test
pytest tests/test_tool_gateway.py -q     # one area
```

Every test runs offline. No test may need an API key or a network, and one
that does is a broken test — that is what makes the suite runnable in CI, in a
container, and on a machine with no credentials at all.

## Docker

```bash
docker build -t jarvis-core .
docker run --rm jarvis-core pytest -q
docker run --rm jarvis-core python examples/demo_offline.py
```

Nothing runs as root, no secret is baked in, and the image is useful without a
credential because the suite and the demos use the offline provider.

## Roadmap

- Evaluation harness that scores routing decisions against each other rather
  than each model against itself.
- More MCP transports, and a conformance suite for adapters.
- Durable orchestration across process restarts.
- A verification path that takes an agent from defined to production-ready
  with repeated real runs, and refuses to claim it on one.

## Contributing

Bug reports, small reviewable pull requests and threat-model discussions are
welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); for anything touching
a security boundary, open an issue first and describe the threat model.

## License

[Apache-2.0](LICENSE).
