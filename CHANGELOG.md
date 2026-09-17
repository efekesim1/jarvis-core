# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first public extraction of the runtime. Nothing has been released yet;
this section describes what is on `main`.

### Added

- **Planning and orchestration.** A rule planner and a model planner, task
  graphs with dependency waves, bounded retries and bounded agent hops.
- **Agent registry.** Agents are YAML rather than subclasses; 27 generic
  agents across three swarms ship as examples, and a declared tool capability
  that nothing backs fails the build.
- **Model router.** Tier selection from the kind of work, a step down a tier
  under budget pressure, a retry on a different model rather than the one that
  just failed, and pricing for every call.
- **Providers.** Anthropic natively; OpenAI, NVIDIA NIM, xAI and Gemini over
  the Chat Completions protocol; a local Ollama tier that needs no credential
  and costs nothing; and an in-process provider the tests run against.
- **Cost control.** Accounting per call, task, agent, model, day and month,
  with a result cache and request coalescing in front of it, and a benchmark
  that measures what the controls save.
- **Tool gateway and MCP.** Discovery, transports, an error taxonomy, per-tool
  rate limits, argument and result validation, redaction, and an approval gate
  for high-risk calls.
- **Permissions and approvals.** Capability grants per agent, risk
  classification per call, and a human decision that is recorded and can be
  answered from another process.
- **Sandbox.** Isolated experiment runs, where a pass is evidence and not
  consent: nothing is adopted automatically.
- **Memory and storage.** Layered memory with a bar for writing and a budget
  for recall, over optional SQLite.
- **Observability.** An append-only audit trail, tracing per agent run and
  plan step, and cost figures that reconcile with both.
- **Interfaces.** A CLI, and an HTTP API behind a bearer token whose only
  money-spending endpoint is off until it is explicitly enabled.
- **CI.** Lint, the offline suite on 3.11 and 3.12, the demos, the
  benchmarks, a container build that runs the suite inside the image, bandit,
  pip-audit and a credential scan.

[Unreleased]: https://github.com/efekesim1/jarvis-core/commits/main
