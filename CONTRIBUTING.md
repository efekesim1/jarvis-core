# Contributing

Thanks for looking at this. The goal is not to maximise pull-request count; it
is to make small, reviewable changes that leave the runtime safer and easier
to understand.

## Before you start

1. Read the architecture overview in [README.md](README.md) and the relevant
   pages under `docs/`.
2. Check the open issues and pull requests so two people do not write the same
   patch.
3. For a large feature, an architecture change, a new external write
   capability, or anything that moves a security boundary, **open an issue
   first** and describe both the behaviour you want and the threat model.
4. Never commit credentials, `.env` files, personal data, production logs, or
   copied proprietary material.

## Development setup

```bash
git clone https://github.com/<your-fork>/jarvis-core
cd jarvis-core
make install
source .venv/bin/activate
make check
```

`make check` is the gate CI runs: lint, the offline suite, the demos and the
benchmarks. `make security` runs bandit and pip-audit on top of it.

## What a good pull request looks like

It is narrow enough to review in one sitting, and it answers four questions:

- **What changed?** The observable behaviour, not a list of file names.
- **Why?** The issue, bug, benchmark or maintenance reason.
- **How was it verified?** The exact commands, and their output.
- **What could go wrong?** Permissions, approvals, model routing, spend,
  persistence, concurrency, network access, sandboxing, migrations.

New behaviour normally comes with a regression test. A bug fix without one
should say why a reliable test is not practical.

## House rules in the code

These are not style preferences; a change that breaks one of them will be
asked to change.

- **No agent reaches a tool directly.** Every call goes through
  `ToolGateway.call`. Adding an import path from `agents/` to a transport is a
  security regression, not a shortcut.
- **Default deny.** A new tool is unreachable until a policy grants the
  capability. Grants are capabilities, never blanket access.
- **A tool result is never an instruction.** It is redacted, sanitised,
  truncated and fenced before an agent sees it.
- **Nothing unbounded.** Retries, plan waves, agent hops, context size, spend
  and wall-clock time all carry a ceiling, and the ceiling is enforced where
  the work happens rather than trusted to the caller.
- **No secret in code.** Credentials come from the environment. Nothing prints
  one or writes one to a log.
- **No SQL built from a variable.** Queries are literals chosen between; every
  value is a bound parameter.
- **Tests run offline.** No test may need an API key or a network.
- **Layering.** Each layer depends only on the ones above it in the README's
  diagram. Two circular imports have already been caused by breaking that.

## Style

`ruff` decides formatting and import order; `make format` applies it. Beyond
that: say why in a comment when the reason is not obvious from the code, name
things after what they are for, and prefer a boring implementation that a
reader can check over a clever one they have to trust.

## Tests

```bash
make test
pytest tests/test_tool_gateway.py -q
pytest -k approval -q
```

A test that asserts a refusal is as valuable as one that asserts a success —
most of the suite here is the former.

## Reporting a vulnerability

Do not open a public issue. See [SECURITY.md](SECURITY.md).
