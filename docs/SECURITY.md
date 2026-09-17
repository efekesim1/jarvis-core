# Security boundaries

`SECURITY.md` in the repository root says how to report a vulnerability. This
page describes what the boundaries are and what each one is for.

The threat model has three assumptions, and everything else follows from them:

1. **The model is not trusted.** It can be wrong, and it can be steered by
   something in its input.
2. **A tool result is not trusted.** Anything fetched, read or returned by a
   server may be hostile text written by whoever controls that source.
3. **An agent's output is not trusted by the next agent.** A finding is input,
   not instruction.

## What an agent may reach

**No agent reaches a tool directly.** Every call goes through
`ToolGateway.call`. There is no import path from `agents/` to a transport, and
adding one is a security regression rather than a shortcut — a test asserts
the absence of that path by name.

**Default deny.** An agent with no policy calls nothing. A grant is a
capability (`web.read`, `files.read`), never blanket access, and declaring a
capability in an agent's YAML does not grant it: the agent also needs the
matching permission, and a declared capability that nothing backs fails the
build rather than failing at run time.

**File tools never leave the workspace.** Paths are resolved first and checked
second, which is what catches a symlink pointing out. `.env`, private keys and
`.git` are never readable, whatever an agent asks for.

**The sandbox never gets money, deploy or admin rights.** Not by default, and
not on request.

## What stops for a human

**High-risk calls raise an approval.** Deploys, deletions and external side
effects need a decision that is bound to that one call: an approval token
cannot be replayed against a different tool, a different argument set or a
later call. An unattended run blocks rather than approving on anyone's
behalf, and with a database configured the decision can be made from another
process — which is what makes an overnight run answerable in the morning.

**Risk is classified per call, not per agent.** The same tool can be
low-risk reading and high-risk writing, and the gateway decides from the
arguments.

## What is done to text

**A tool result is never an instruction.** Before an agent sees it, it is
redacted (credentials and tokens removed), sanitised (injection attempts
neutralised), truncated to a budget, and fenced so the model sees clearly
where untrusted content begins and ends. A hit on an injection pattern is
recorded as a security event rather than silently dropped.

**Agent output gets the same treatment before another agent reads it**, for
the same reason: the second agent has no way to tell a finding from an
instruction, so the system tells it.

## What is bounded

Nothing here is allowed to be unbounded, because every one of these has a
failure mode that is expensive rather than merely wrong:

| Bound | Where |
|---|---|
| Retries per task | agent definition, capped by the loader |
| Agents per plan, waves per plan | planner and orchestrator |
| Context per step | `core/context.py`, against a budget |
| Spend per call, task, day and month | `observability/cost.py` |
| Calls per tool per window | the gateway's rate limits |
| Wall-clock per agent run and per sweep | settings |

An autonomous job that carried no ceiling would not be allowed to exist; the
scheduler requires a budget, a deadline and an iteration cap.

## Credentials

**No secret in code.** Everything comes from the environment. Nothing reads a
credential from source, prints one, or writes one to a log or an audit entry,
and CI fails the build on a committed key.

**The HTTP interface refuses to start without a token.** `JARVIS_API_TOKEN` is
required and has a minimum length; comparison is constant-time. Binding to
localhost instead is not a substitute — it is wrong on any machine with a
container, a VPN or a second user, and it fails silently on all of them.

**Spending over HTTP is off by default.** `POST /runs` answers 403 until
`JARVIS_API_ALLOW_RUNS` is set, because a request that bills the operator
should not be one misconfigured reverse proxy away.

## Storage

**No SQL is built from a variable.** Queries are literals chosen between;
every value is a bound parameter.

**Nothing that failed is cached.** A transient error must not become the
stored answer.

## What this does not protect against

- A model that is simply wrong. That is an evaluation problem; see
  [EVALUATION.md](EVALUATION.md).
- A credential you have already lost.
- A tool you granted deliberately doing exactly what you granted it.
- Costs inside the budgets you set.
