# Cost control

## What

Twelve mechanisms that between them cut tokens by 66% and cost by 58% on a
representative workload, measured rather than claimed.

## Why

An agent system's cost is not a line item, it is the constraint. Without
controls, a six-request session with three distinct questions pays for six, an
agent receives the whole conversation to answer one question, and the same
research is done twice because nobody checked.

The benchmark is in `examples/benchmark_cost.py` and runs in CI. If the saving
disappears, the build fails.

## How

| Control | What it stops |
|---|---|
| Model routing | Frontier prices for classification |
| Prompt caching | Re-sending an agent's system prompt every call |
| Result cache | Paying twice for identical work |
| In-flight coalescing | Two identical calls racing, both paying |
| Context budget | An agent receiving what it does not need |
| Finding compression | Full text where a summary would do |
| Per-step isolation | Every agent getting every other agent's output |
| Retrieval budget | Memory expanding to fill the window |
| Hop and wave limits | A plan that expands without bound |
| Task budget | One request consuming the day |
| Daily and monthly budgets | A bad week consuming the month |
| Duplicate detection | The same tool call twice in one run |

**Nothing that failed is cached.** A transient error must not become the stored
answer — that is the cache turning one bad minute into a permanent wrong
answer.

**Cost is recorded per task, agent, model, day and month**, so "what did that
cost" has an answer, and so the self-improvement engine can see an agent that
is expensive relative to its peers.

## Example

```
$ make benchmark
Without controls: 6 calls, 24,300 tokens, $0.0912
With controls:    3 calls,  8,160 tokens, $0.0382
                  66.4% fewer tokens, 58.1% lower cost
```

## What it does not do

- **Semantic caching.** The cache is exact-key. A rephrased question pays
  again.
- **Token estimation is approximate.** Four characters per token, adjusted per
  provider. Good enough for budgets, not for billing reconciliation.
- **The gateway costs 0.23ms per call.** Measured by `make gateway-cost`, which
  CI fails over 2ms. Eleven checks against a network round trip: not the thing
  to optimise.
