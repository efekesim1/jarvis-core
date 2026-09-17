# Tool security

## The principles

- **Default deny.** No policy means no tools. Not fewer tools: none.
- **Least privilege.** Capabilities, scoped and named, never blanket access.
- **No direct access.** An agent cannot reach a transport. There is no import
  path from `agents/` to `mcp/transport`.
- **No secret in a log.** Credentials live in the environment, are read at the
  moment of use, and are redacted from results and audit lines.
- **No trust of external content.** A tool result is evidence, never an
  instruction.
- **No automatic high-risk side effect.** Deploys, deletions and anything with
  an external consequence stop for a human.
- **No blind retry of a non-idempotent action.**

## What is tested

Thirty attacks, each with a test that fails if the defence is removed.

| # | Attack | Defence |
|---|---|---|
| 1 | An agent with no grant calls a tool | Default deny |
| 2 | A tool that does not exist | Registry lookup |
| 3 | A tool on an unregistered server | Server lookup |
| 4-5 | A disabled tool or server | Enabled checks |
| 6-8 | Bad, missing or extra arguments | Schema validation before any call |
| 9 | An agent exceeding its risk ceiling | Policy ceiling |
| 9b | A capability grant used to gain a permission | Permission intersection |
| 10 | One agent using another's grant | Per-agent policy |
| 11 | A high-risk call with no approval | Approval gate |
| 12 | An expired approval | Expiry check |
| 13 | A replayed approval | Single use |
| 14 | An approval moved to another tool | Request binding |
| 14b | Arguments changed after approval | Argument hash |
| 15 | A call over budget | Budget check before execution |
| 16 | A runaway agent | Rate limit |
| 17 | A tool that hangs | Timeout |
| 18 | Endless retries | Bounded attempts |
| 19 | A destructive call unapproved | Approval gate |
| 20-21 | Instructions inside a result | Sanitisation and fencing |
| 22 | A two-megabyte result | Truncation |
| 23-24 | A credential in a result or a log | Redaction before anything else |
| 25 | A malformed result | Typed failure |
| 26-27 | A server that crashes or dies | Isolated failure, health check |
| 28 | The same write twice | Idempotency record |
| 29 | A non-idempotent write retried | Retry policy |
| 30 | An unclassified tool | Safe default |

Plus concurrency: parallel agents do not leak permissions into each other,
request ids stay distinct, and a duplicated write lands once whatever the
interleaving.

## Credentials

A server record holds an `auth_reference`, never a value. The GitHub adapter
reads `GITHUB_TOKEN` inside the method that builds its headers, and the token
does not leave that method. A dump of the registry, a trace, or an audit line
cannot contain one.

The test suite deletes `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` from the environment, so a token the machine happens to hold
cannot turn an offline test into a live call.

## What this does not do

Sanitisation is pattern-based. A novel phrasing will pass it. The permission
model is the layer that actually limits damage: an agent that cannot reach a
write tool cannot be talked into using one.

There is no sandbox for executing untrusted code. That is why the knowledge
acquisition pipeline refuses executable candidates outright.
