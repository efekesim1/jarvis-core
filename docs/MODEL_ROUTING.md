# Model routing

## What

One request, several models, and a rule for choosing between them.

`ModelRouter` picks a model per call from the kind of work, the budget left,
and what the agent is allowed to use — then records what it chose and why.

## Why

The naive version sends everything to the best model. On this workload that is
roughly ten times the cost for no measurable gain on the easy two thirds:
classification, extraction, formatting, short synthesis. The measured saving
from routing plus the other controls is 66% fewer tokens and 58% lower cost
(`examples/benchmark_cost.py`, which CI fails on if it regresses).

The opposite mistake is worse. A cheap model on a reasoning task produces an
answer that looks fine and is wrong, and the saving is spent several times over
finding out.

## How

**Tier by work kind.** `TaskKind` maps to a tier: `FAST` for classification and
extraction, `BALANCED` for analysis and synthesis, `FRONTIER` for planning and
hard reasoning.

**Narrow by what the agent may use.** An agent's `allowed_models` is a ceiling,
not a preference. A definition that lists only Sonnet does not get Opus because
the router thought the task was hard.

**Step down under budget pressure.** As the daily budget fills, the router
drops a tier rather than stopping. Work continuing on a cheaper model is
usually better than work stopping, and the step-down is audited so the
degradation is visible rather than mysterious.

**Retry across models, bounded.** A provider failure retries on the next usable
model, up to the agent's `max_attempts`. Never unbounded — a failing provider
would otherwise spend the whole budget on retries.

**Every choice is logged.** Model, provider, tier, reason, tokens, cost.

## Providers

| Provider | State |
|---|---|
| Anthropic | Used. Prompt caching on system prompts, `pause_turn` resume. |
| OpenAI | Implemented, NOT VERIFIED against the live API |
| Mock | Deterministic, for tests and offline verification |

Adding one is a class with `name`, `available()` and `complete()`. Nothing else
in the system names a vendor.

## Example

```
TASK    research competitor pricing
KIND    analysis
TIER    balanced
MODEL   claude-sonnet-5
WHY     medium complexity, network needed, low risk, budget at 34%
COST    $0.0081 estimated, $0.0074 actual
```

## What is not measured

Whether the tiering is *right*. The mapping from work kind to tier is
judgement, and no experiment has compared the tiers on the same task. This is
in KNOWN_ISSUES and it needs a credential to fix.
