# Evaluation

## What

Every agent result is scored before it is used, on seven criteria, and two of
them are hard failures.

## Why

"The agent returned something" is not the same as "the agent did the job", and
a multi-agent system that cannot tell them apart compounds the difference: a
weak finding from one agent becomes the input to three others, and by the time
it reaches an answer nobody can see where it came from.

## How

| Criterion | Asks |
|---|---|
| correctness | Is it internally consistent, and does it answer what was asked? |
| relevance | Is it about the question? |
| completeness | Does it cover what the output schema asked for? |
| source quality | Are the citations real and appropriate? |
| safety | Does it contain anything that should not be passed on? |
| cost | Was it worth what it cost? |
| latency | Did it take longer than it should have? |

Two hard failures, which no score can outweigh:

- **A research agent citing no sources.** Its own `evaluation_policy` says it
  must; an answer without them is not a cheaper answer, it is a different kind
  of thing.
- **A result that failed.** An agent reporting an error scores zero rather than
  being partially credited for the text it produced first.

Scores below `pass_threshold` do not stop the run — they lower the confidence
that reaches the synthesis step and are recorded, so `jarvis review` can see an
agent that is consistently weak.

## Example

```
web_researcher            0.78  passed
  correctness      0.80
  relevance        0.90
  completeness     0.70
  source_quality   0.85
  notes: two of four claims are supported by the cited pages
```

## What is not measured

**The evaluator itself.** It scores agents; nothing scores it. Its criteria
weights were chosen by judgement and have never been compared against a human
rating of the same outputs. This is in KNOWN_ISSUES.
