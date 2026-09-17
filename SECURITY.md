# Security Policy

Security boundaries are part of the architecture here, not a hardening phase
that comes later. A report that could expose credentials, unauthorised tool
access, a sandbox escape, an approval bypass, a cost-control bypass or an
unintended external write should be treated as a security issue.

## Supported versions

The project is pre-1.0. Fixes land on `main`, and a release is cut from there.
Please report against `main` or a released version.

## Reporting a vulnerability

Please do **not** open a public issue, and do not include exploitable detail,
secrets, personal data or a working payload in anything public.

Use GitHub's private vulnerability reporting:
[open a private advisory](https://github.com/efekesim1/jarvis-core/security/advisories/new).

A useful report contains:

- the affected component and the version or commit;
- minimal reproduction steps;
- expected versus observed behaviour;
- realistic impact;
- whether credentials, network access, persistence, approvals, money,
  deployment or sandbox boundaries are involved;
- any fix or mitigation you can suggest.

You will get an acknowledgement, and a decision on whether it is in scope,
before anything is published. Please give a reasonable window for a fix before
disclosing.

## What is in scope

- An agent reaching a tool without going through the gateway.
- A capability grant that is wider than the policy says.
- An approval that can be replayed, forged, or bypassed.
- A budget, rate limit, retry cap or context cap that can be exceeded.
- A path traversal or symlink escape from the workspace in a file tool.
- Prompt injection that changes what the system does rather than what it says
  — for example a tool result that causes a tool call.
- A credential reaching a log, a prompt, a response body or the repository.
- A sandbox escape from `experiments/`.
- SQL injection, or a query built from an unbound value.

## What is not in scope

- A model producing a wrong or low-quality answer. That is an evaluation
  problem, and an issue is the right place for it.
- Costs incurred by a configuration you chose, within the budgets you set.
- Anything requiring an already-compromised machine or a credential you
  already hold.
- Denial of service against your own deployment through your own credentials.
- Findings from a scanner with no demonstrated impact.

## Hardening notes for operators

- `JARVIS_API_TOKEN` is required for the HTTP interface; it refuses to start
  without one. Binding to localhost is not a substitute.
- Keep `JARVIS_ENVIRONMENT=sandbox` unless you mean otherwise: the sandbox
  refuses money, deploys and admin rights whatever an agent asks for.
- Set a database path if you want approvals raised by an unattended run to be
  answerable from another process. Without one they are visible only to the
  process that raised them.
- `JARVIS_API_ALLOW_RUNS` is off by default. Turning it on lets an
  authenticated caller spend money.
