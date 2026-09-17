# MCP and the tool fabric

## The shape

```
agent
  -> ToolGateway            the only door
       -> ToolRegistry      does this tool exist
       -> MCPServerRegistry does its server exist, and is it usable
       -> PolicyEngine      may this agent call it
       -> CostEngine        is there budget left
       -> RateLimiter       has it been called too often
       -> ApprovalEngine    does a human need to say yes
       -> validation        are the arguments the right shape
       -> MCPTransport      the actual call, with a timeout
       -> validation        is the result usable, safe, and small enough
       -> AuditLog          what happened
  <- ToolResult
```

An agent holds a gateway and nothing else. It never sees a transport, never
holds a server record, and never learns a credential. That is enforced by
construction: `AgentContext.call_tool` is the only path, and it takes a tool
id and arguments.

## The layers

| Layer | Module | What it does |
|---|---|---|
| Errors | `mcp/errors.py` | One taxonomy for every external failure |
| Models | `mcp/models.py` | Servers, tools, requests, results |
| Transport | `mcp/transport.py` | In-process, stdio, HTTP |
| Server registry | `mcp/registry.py` | Which servers exist and their health |
| Discovery | `mcp/registry.py` | What a server publishes, normalised |
| Tool registry | `tools/registry.py` | Every tool, by id and capability |
| Policy | `tools/policy.py` | Who may call what |
| Approval | `tools/approval.py` | One decision, one call |
| Validation | `tools/validation.py` | Arguments in, results out |
| Rate limit | `tools/ratelimit.py` | Per tool, agent and task |
| Gateway | `tools/gateway.py` | The pipeline above, in order |
| Adapters | `integrations/` | Real services, behind the same door |

## Risk levels

Five, because "write" spans filing an issue and deleting a repository.

| Level | Example | Approval |
|---|---|---|
| `read_only` | read a repository | no |
| `low_risk_write` | file an issue | no, if policy allows |
| `high_risk_write` | deploy | yes |
| `destructive` | delete a repository | yes |
| `external_side_effect` | send money, send email | yes |

A tool cannot raise its own risk. Classification happens on this side, from the
tool name, and anything unrecognised is treated as a high-risk write. Matching
is on whole word segments: "widget" contains "get", and a tool called
`frobnicate_widget` is not a read.

A server also carries a ceiling. A read-only server publishing a delete tool
does not get one registered.

## Permissions

An agent is granted **capabilities**, not tools, so a tool can be replaced
behind the same capability without touching any agent. `github.read` covers
`github.read.list_issues` by prefix; it never covers `github.write`.

Default deny at every level:

- an agent with no policy may call nothing
- a capability grant cannot conjure a permission the agent's definition lacks
- an untrusted server may only serve reads
- the sandbox refuses high-risk tools outright

## Approvals

An approval is bound to a request id, a tool id, an agent id, and a hash of the
arguments. It expires, and it is consumed when used.

That combination closes the three bypasses: replaying a spent approval, moving
one to a different tool, and getting one approved for harmless arguments then
calling with different ones. Each is reported as a security event, not as a
missing approval.

## Results are never instructions

Every result is `untrusted`, in that order:

1. credentials redacted, before anything else, so one cannot survive past a cut
2. instruction-shaped text neutralised
3. truncated to 20,000 characters
4. fenced in a block that says it is data

## Failure handling

The gateway returns failures rather than raising them. A dead server, a hung
tool, a malformed result: each becomes a `ToolResult` with a typed error, and
the run carries on.

Retries are bounded and selective. Only timeouts, unhealthy servers, transport
errors and rate limits are retried at all. A non-idempotent write is never
retried automatically, because retrying a call that may already have landed is
how one issue becomes three.

## Duplicate writes

A completed write is remembered by tool, agent, task and arguments. The same
write again is refused with `duplicate_request` rather than performed twice.
Reads are cached instead, which is the same idea for something safe to repeat.

## Transports

`in_process` backs the mock server and the adapters, and is what the whole test
suite runs on. `stdio` speaks to a real MCP server through the official SDK,
which is an optional dependency: install with `pip install 'jarvis-core[mcp]'`.
`http` is declared and not implemented, deliberately.

## Adding a server

```python
registry.register_in_process("myservice", handler, max_risk=ToolRisk.READ_ONLY)
tools.register_many(await discovery.discover("myservice"))
policy.grant_capabilities("some_agent", ("myservice.read",))
```

Nothing in JARVIS changes.
