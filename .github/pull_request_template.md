## What

<!-- The observable change, in one or two sentences. -->

## Why

<!-- The issue, bug, benchmark, or maintenance reason. -->

## Verification

- [ ] `make lint`
- [ ] `make test`
- [ ] Focused tests for the changed behaviour
- [ ] Documentation updated where behaviour changed

<!-- Paste the commands and the results that matter. -->

## Risk and boundaries

<!-- Say what this changes about permissions, approvals, external writes,
     model routing, spend, persistence, concurrency, network access or
     sandboxing. Write "None" only after checking. -->

## Checklist

- [ ] No credentials, `.env` files, personal data or private logs are included.
- [ ] New behaviour has a regression test, or the PR says why one is not practical.
- [ ] Retries, autonomous work, timeouts and spend stay bounded.
- [ ] No agent reaches a tool except through the gateway.
- [ ] I have read the code in this PR, including any AI-assisted parts.
