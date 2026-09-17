# Memory

## What

Nine layers, each with a different lifetime and a different bar to get into.

## Why

Memory is not a transcript. A system that remembers everything has the same
problem as one that remembers nothing — it cannot find the part that matters,
and it pays for the rest on every prompt.

So three questions, and only these three:

- **Is this worth keeping?** Most of what happens is not.
- **What is relevant now?** Retrieval is scored, ranked and capped. Nothing is
  pulled into a prompt because it happens to exist.
- **When does it stop being true?** Working memory dies with its task, an
  episode fades, a decision stands until superseded.

## How

| Layer | Lifetime | Bar to enter |
|---|---|---|
| `WORKING` | 1 hour, or the task | anything |
| `CONVERSATION` | 7 days | 0.3 |
| `EPISODIC` | 90 days | 0.4 |
| `AGENT` | 180 days | 0.5 |
| `SYSTEM` | 30 days | 0.4 |
| `PROJECT` | forever | 0.5 |
| `KNOWLEDGE` | forever | 0.5 |
| `USER` | forever | 0.6 |
| `DECISION` | forever | 0.6 |

A write below its layer's bar is declined, and declining is the common case
rather than an error. A passing remark about the weather is not a user fact; a
chosen database is.

**Retrieval is budgeted** — at most five items, 2000 characters, and nothing
below a relevance floor. An unrelated question gets an empty list rather than
the closest available noise. Retrieval that ignores a budget is just a slower
way of passing the whole history.

**Scoring** combines word overlap, the importance it was written with, its
confidence, and how recently it was written, with recency halving every 30
days.

**Working memory is dropped with its task.** Leaving it is how a context window
fills with material nobody asked for.

## Example

```
$ jarvis memory "which database did we choose"
layer      score  content
decision   0.72   Chose SQLite over Postgres: one machine, no ops, WAL is enough
project    0.41   Storage is optional throughout; unset means in-memory
```

## What it is not

- **Semantic.** Retrieval is lexical with a crude stemmer. A paraphrase misses,
  quietly and in the safe direction. Embeddings would fix it and are not built.
- **Consolidating.** Nothing merges near-duplicates or ages out a memory nobody
  has corroborated. The corpus grows and retrieval gets noisier.
- **Shared.** One SQLite file, one machine.
