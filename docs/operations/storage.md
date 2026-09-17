# Storage

## What it is

A SQLite file holding tasks, memory, approvals, the audit trail and cost.
Optional: with no path configured everything stays in memory and the system
behaves exactly as it did before.

```bash
# in jarvis/.env
JARVIS_DATABASE_PATH=var/jarvis.db
```

`jarvis doctor` says which mode you are in.

## Why SQLite

It needs no server, so local development and CI work with nothing installed.
It is a real database, so a restart no longer loses in-flight work. And two
processes on one machine can share it, which is the problem that was actually
blocking autonomy: an approval raised by a night shift used to be invisible to
every other process, so `jarvis approvals approve` could not be honest.

PostgreSQL earns its place when there are two machines, or when embeddings
arrive and recall should be semantic. The store protocols are what let it slot
in then.

## The overnight loop

```
23:00  a shift hits a high-risk tool call
       -> the approval is written to the database
       -> the job is BLOCKED_APPROVAL, and the shift moves on

07:00  jarvis approvals            (a different terminal, a different process)
       jarvis approvals approve <id>

07:01  the shift picks the decision up and the call runs, once
```

`examples/demo_overnight_approval.py` runs all four stages as separate Python
processes and asserts each one.

## Tables

| Table | Holds |
|---|---|
| `tasks` | The task lifecycle, dependencies, cost, error |
| `memory` | Every layer, with importance, confidence and expiry |
| `approvals` | One row per decision, with its argument hash |
| `audit` | Append-only. No update or delete, deliberately |
| `costs` | Priced model calls |

Schema version 1. A mismatch logs a warning rather than migrating silently.

## Concurrency

Write-ahead logging is on, so a reader never blocks a writer. Writes are
serialised behind a lock within a process and behind `BEGIN IMMEDIATE` between
them. Connections are per thread, because SQLite objects cannot be shared
across them.

Deciding an approval is a single statement guarded on `status = 'pending'`, and
consuming one is guarded on `status = 'approved'`. Two terminals racing to
decide the same approval cannot both win, and a replayed approval cannot be
spent twice.

## Queries are literals

No SQL string is assembled from a variable anywhere in this layer. Where a
query varies, the variants are written out in full and chosen between; every
value is a bound parameter. The memory search filters by scope and expiry in
SQL and matches words in Python, because a variable number of `LIKE` clauses
would mean building a query in a loop.

## Backup

The file, plus its `-wal` and `-shm` siblings. Copy them together, or use
`sqlite3 var/jarvis.db ".backup out.db"` while the system is running.
