# The agent roster

Twenty-seven agents in three swarms. Each is a YAML file under
`agents/<swarm>/`, and everything about it lives there: role, capabilities,
tools, permissions, model policy, budget, timeout, evaluation criteria and
failure policy.

## The swarms

| Research | Business | Build |
|---|---|---|
| `web_researcher` | `business_strategist` | `software_architect` |
| `market_researcher` | `ecommerce_agent` | `claude_developer` |
| `news_researcher` | `product_hunter` | `codex_developer` |
| `trend_analyst` | `sales_agent` | `github_agent` |
| `competitor_analyst` | `lead_generation_agent` | `qa_agent` |
| `social_researcher` | `marketing_agent` | `security_agent` |
| `data_collector` | `finance_agent` | `api_agent` |
| `source_verifier` | `pricing_agent` | `automation_agent` |
| `research_synthesizer` | `business_analyst` | `deployment_agent` |

## How an agent reaches the world

Through the gateway, and nothing else. `web.read`, `github.read` and
`files.read` are declared in an agent's `tool_capabilities:` list, and every
call runs the full pipeline: policy, risk, budget, rate limit, approval,
argument validation, result validation, audit.

The `tools:` list is a separate, older path: `web_search`, `web_fetch` and
`code_execution` run inside the model's own call on the provider's
infrastructure, gated by `TOOL_PERMISSIONS` in `agents/base.py` rather than by
the gateway. Web reading used to work that way. It now has an adapter, so a
repeated search is free, shared across agents, and leaves an audit line.

The two paths still exist and it matters which one a call took, because only
one of them was seen by the fabric.

## Capabilities are checked, not assumed

```bash
jarvis agents --capabilities
```

Every agent is `ready`, `partial`, `no tools available`, or `no tools needed`.
A test fails the build if any agent declares a capability nothing backs, so
"the YAML exists" can never be mistaken for "the agent works".

## Least privilege, enforced

- No agent holds `FINANCIAL`. The finance agent models money and cannot move it.
- No agent holds `ADMIN`.
- Only `deployment_agent` holds `DEPLOY`, and every deploy is a high-risk write
  that stops for approval and is refused outright in the sandbox.
- Every agent's `max_tool_risk` is `read_only`. Nothing shipped writes to the
  world; that stays a per-agent decision with an approval path already in place.
- No business-swarm agent can reach a repository.

Each of those is a test.

## What a definition must carry

```yaml
id: something_agent          # lowercase, unique
name: Something Agent
role: One line on what it is for
swarm: research | business | build
capabilities: [...]          # what it is good at, for the planner
tools: [web_search]          # model server-side tools
tool_capabilities: [...]     # gateway capabilities
max_tool_risk: read_only
permissions: [read, network] # the smallest set that works
risk_level: low | medium | high
allowed_models: [...]        # empty means the router decides
timeout_s: 240
retry_policy: { max_attempts: 2 }   # capped at 5 by the loader
evaluation_policy:
  criteria: [...]
  min_confidence: 0.4
  require_sources: true      # research agents must cite
system_prompt: |
  ...
output_schema:
  ...
```

Adding an agent is one file. Nothing else changes.

## How the planner staffs a request

The rule planner covers the common shapes: at most three gatherers in parallel,
then a verifier, then one agent that merges. A fourth gatherer rarely finds
something the others missed and always costs another model call.

| Intent | Gatherers | Merges |
|---|---|---|
| business | market, competitor, trend | business analyst |
| build | architect, github, security | QA |
| research | web, news, social | synthesizer |
| question | one agent | nobody |

Anything else goes to the model planner, which sees the whole catalogue.

Turkish and English put the question word at opposite ends of a sentence, so
`detect_intent` matches both: "what is MCP?" and "MCP nedir?" are each one
agent, not a swarm.

## The imported shelf

`agents/imported/` holds 256 definitions converted from
[`msitarzewski/agency-agents`](https://github.com/msitarzewski/agency-agents)
(MIT). They are on disk, they load, and **none of them is enabled**.

| | |
|---|---|
| converted | 256 of 264 |
| refused | 18 — 6 for injection-shaped prose, 10 not agent files, 2 duplicates of ours |
| breaking audit findings | 0 |
| live roster | still exactly 27 |
| what each arrives with | READ permission, no tools, no network, `enabled: false` |

What the conversion supplies, because the source has none of it: `output_schema`,
`permissions`, `tool_capabilities`, `allowed_models`, `evaluation_policy`,
`retry_policy`, `risk_level`, `task_kind`, `smoke_task`.

Their divisions map onto our three swarms:

| Swarm | Their divisions |
|---|---|
| research | academic, research, gis, spatial-computing, healthcare |
| build | engineering, testing, security, integrations, game-development, design, product, specialized |
| business | finance, marketing, paid-media, sales, strategy, support, project-management |

### Enabling one

It is an edit to its file — deliberately, so the decision has a name on it:

```bash
jarvis agents --all | grep <something>       # find it
$EDITOR agents/imported/<swarm>/<id>.yaml    # enabled: true
jarvis verify --agent <id>                   # does it run?
```

Read the prompt first. It is 200-odd lines someone else wrote, and 14 of them
name tools the agent cannot reach — the appended system rules say so, but the
prose would read better edited. Anything from the `security` division carries a
note asking for exactly that.

### Importing more

```bash
jarvis import-agents <checkout>                        # dry run, reports only
jarvis import-agents <checkout> --division finance --write
```

Nothing is written without `--write`, and nothing is enabled ever.
