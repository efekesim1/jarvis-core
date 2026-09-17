# Architecture

## The layers

Each layer depends only on the ones above it. Nothing reaches upward.

```
config / logging          settings, audit trail
        |
security                  permissions, risk, approval
        |
models                    providers, catalog, router, cost
        |
agents                    definitions, registry, runtimes
        |
core                      tasks, planner, orchestrator, JARVIS
        |
cli / api / dashboard     interfaces
```

The one place this is enforced in code is `core/__init__.py`, which resolves
its exports lazily. `agents` needs `core.task`; `core.jarvis` needs `agents`.
Importing eagerly would close that loop.

## The request path

```
request
  -> intent detection        cheap keyword pass, Turkish and English
  -> planner                 rules for known shapes, a model for the rest
  -> plan validation         unknown agents, cycles and duplicates rejected
  -> waves                   independent steps grouped to run together
  -> orchestrator            asyncio.gather per wave, capped by a semaphore
       -> guardian           permissions and approval, per agent run
       -> router             model chosen by work kind, risk and budget
       -> provider           the actual API call
       -> cost engine        priced and recorded
       -> evaluation         scored before it counts
  -> synthesis               the plan's own synthesizer, or one merge call
  -> response                answer, cost, confidence, sources, trace
```

## Why the pieces are shaped this way

### Agents are YAML, not classes

An agent is fifteen fields of metadata plus a prompt. Writing that as a Python
class buys nothing and costs plenty: a hundred agents becomes a hundred modules,
and a generated agent becomes generated code that has to be reviewed as code.

As data, a new agent is a diff a human can read in a minute. The agent factory
in Phase 12 writes YAML, and the approval step reads it.

The `runtime` field says which class executes the definition. `llm` is the only
one today. A future agent that needs to drive a browser or a Claude Agent SDK
session registers its own runtime without touching the registry.

### The router, not the caller, picks the model

If every call site named its own model, changing model strategy would mean
touching every call site. Instead a caller says what kind of work this is
(`TaskKind.RESEARCH`, `TaskKind.PLANNING`, ...) and the router decides.

That gives three things for free: an agent's `allowed_models` list is enforced
in one place, budget pressure can step the whole system down a tier at once,
and a failed call can retry on a different model rather than hammering the one
that just failed.

### Cost is measured from the first line

At one agent, spend is noise. At a hundred it is the main operational risk, and
retrofitting cost accounting into a running system means missing exactly the
calls you most want to see.

Budgets are enforced at three scopes. Below 70% of the daily budget nothing
changes. Above it the router drops a tier. At 100% work stops with a
`BudgetExceeded`, which the orchestrator deliberately does not retry.

### Security is in the first commit

Adding permissions later means auditing every call site that already exists. So
the Guardian is in the path from the start, and every agent declares the
smallest permission set it needs.

Three rules matter most:

- Money, deploys and admin always need a human, whatever risk level was claimed.
- The sandbox refuses those permissions outright, even when granted.
- With nobody watching, the default gate denies rather than assumes.

### Results are scored, not trusted

An agent reporting success proves nothing. The evaluation engine grades every
result on seven criteria before it counts. Two are hard failures rather than
deductions: a leaked credential, and a research agent that cites no sources.

The free pass is deterministic and costs nothing. A model judge exists for
high-value work but is not used by default; paying a model to grade every
routine result is not worth it.

## What is deliberately not here yet

**Memory and knowledge.** The interfaces are defined in `src/jarvis_core/memory/`
so the code above them can already be written, but nothing stores anything
durably. Phase 4 brings PostgreSQL with pgvector.

**MCP.** Phase 3. Today an agent's tools are the model's own server-side tools
(web search, web fetch, code execution). MCP becomes the standard for
everything else so that adding a server never edits JARVIS.

**A database.** Tasks live in memory and vanish when the process exits. The
`TaskStore` protocol is what PostgreSQL will implement.

**The other 21 agents.** Six agents forming one complete pipeline teach more
than twenty-seven that have never run.

## Extending it

**A new agent:** add a YAML file under `agents/<swarm>/`. Nothing else.

**A new model:** add an entry to `config/models.json`. Nothing else.

**A new provider:** write a class with `name`, `available()` and
`complete()`, then register it on the router.

**A new tool:** Phase 3, through MCP.
