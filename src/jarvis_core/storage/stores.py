"""Durable implementations of the store protocols.

Each class here is a drop-in for one of the in-memory stores the system
already used, so nothing above them changes. That was the point of putting
those protocols in from the start.

The approval store is the one that mattered most: an approval raised by a
night shift in one process was invisible to every other, which made
`jarvis approvals approve` impossible to implement honestly.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.task import Task, TaskStatus
from ..logging import get_logger
from ..memory.base import MemoryItem, MemoryScope
from ..security.permissions import RiskLevel
from ..tools.ratelimit import LimitVerdict
from ..verification.status import AgentVerification, VerificationRun
from .database import Database, dumps, from_iso, loads, to_iso

log = get_logger("storage.stores")

# -- tasks ------------------------------------------------------------------


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        goal=row["goal"],
        task_id=row["task_id"],
        parent_task_id=row["parent_task_id"],
        priority=row["priority"],
        status=TaskStatus(row["status"]),
        assigned_agent=row["assigned_agent"],
        dependencies=loads(row["dependencies"], []),
        inputs=loads(row["inputs"], {}),
        outputs=loads(row["outputs"], {}),
        created_at=from_iso(row["created_at"]) or datetime.now(UTC),
        started_at=from_iso(row["started_at"]),
        completed_at=from_iso(row["completed_at"]),
        retry_count=row["retry_count"],
        cost_usd=row["cost_usd"],
        risk=RiskLevel(row["risk"]),
        error=row["error"],
        trace_id=row["trace_id"],
        goal_id=row["goal_id"],
    )


class SqliteTaskStore:
    """Tasks that survive a restart."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def add(self, task: Task) -> Task:
        return self.update(task)

    def update(self, task: Task) -> Task:
        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, parent_task_id, goal, priority, status, assigned_agent,
                    dependencies, inputs, outputs, created_at, started_at, completed_at,
                    retry_count, cost_usd, risk, error, trace_id, goal_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    parent_task_id=excluded.parent_task_id,
                    goal=excluded.goal,
                    priority=excluded.priority,
                    status=excluded.status,
                    assigned_agent=excluded.assigned_agent,
                    dependencies=excluded.dependencies,
                    inputs=excluded.inputs,
                    outputs=excluded.outputs,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    retry_count=excluded.retry_count,
                    cost_usd=excluded.cost_usd,
                    risk=excluded.risk,
                    error=excluded.error,
                    trace_id=excluded.trace_id,
                    goal_id=excluded.goal_id
                """,
                (
                    task.task_id,
                    task.parent_task_id,
                    task.goal,
                    task.priority,
                    task.status.value,
                    task.assigned_agent,
                    dumps(task.dependencies),
                    dumps(task.inputs),
                    dumps(task.outputs),
                    to_iso(task.created_at),
                    to_iso(task.started_at),
                    to_iso(task.completed_at),
                    task.retry_count,
                    task.cost_usd,
                    task.risk.value,
                    task.error,
                    task.trace_id,
                    task.goal_id,
                ),
            )
        return task

    def get(self, task_id: str) -> Task | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        return _row_to_task(row) if row else None

    # Four filters, four literal statements. Writing them out is duller than
    # assembling a WHERE clause and leaves nothing to get wrong.
    _LIST_ALL = "SELECT * FROM tasks ORDER BY priority ASC, created_at ASC"
    _LIST_BY_STATUS = "SELECT * FROM tasks WHERE status = ? ORDER BY priority ASC, created_at ASC"
    _LIST_BY_PARENT = (
        "SELECT * FROM tasks WHERE parent_task_id = ? ORDER BY priority ASC, created_at ASC"
    )
    _LIST_BY_BOTH = (
        "SELECT * FROM tasks WHERE status = ? AND parent_task_id = ? "
        "ORDER BY priority ASC, created_at ASC"
    )

    def list(self, *, status: TaskStatus | None = None, parent: str | None = None) -> list[Task]:
        if status is not None and parent is not None:
            sql: str = self._LIST_BY_BOTH
            params: tuple[Any, ...] = (status.value, parent)
        elif status is not None:
            sql, params = self._LIST_BY_STATUS, (status.value,)
        elif parent is not None:
            sql, params = self._LIST_BY_PARENT, (parent,)
        else:
            sql, params = self._LIST_ALL, ()
        return [_row_to_task(row) for row in self.db.query(sql, params)]


# -- memory -----------------------------------------------------------------


def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        content=row["content"],
        scope=MemoryScope(row["scope"]),
        source=row["source"],
        item_id=row["item_id"],
        timestamp=from_iso(row["timestamp"]) or datetime.now(UTC),
        confidence=row["confidence"],
        importance=row["importance"],
        expiration=from_iso(row["expiration"]),
        embedding=loads(row["embedding"], None),
        metadata=loads(row["metadata"], {}),
    )


class SqliteMemoryStore:
    """Memory that outlives the process that formed it."""

    _SELECT_LIVE = (
        "SELECT * FROM memory WHERE (expiration IS NULL OR expiration > ?) "
        "ORDER BY importance DESC, timestamp DESC"
    )
    _SELECT_LIVE_IN_SCOPE = (
        "SELECT * FROM memory WHERE (expiration IS NULL OR expiration > ?) AND scope = ? "
        "ORDER BY importance DESC, timestamp DESC"
    )

    def __init__(self, database: Database, *, importance_floor: float = 0.0) -> None:
        self.db = database
        self.importance_floor = importance_floor

    def write(self, item: MemoryItem) -> MemoryItem:
        if item.importance < self.importance_floor:
            return item
        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO memory (
                    item_id, content, scope, source, timestamp,
                    confidence, importance, expiration, embedding, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(item_id) DO UPDATE SET
                    content=excluded.content,
                    confidence=excluded.confidence,
                    importance=excluded.importance,
                    expiration=excluded.expiration,
                    embedding=excluded.embedding,
                    metadata=excluded.metadata
                """,
                (
                    item.item_id,
                    item.content,
                    item.scope.value,
                    item.source,
                    to_iso(item.timestamp),
                    item.confidence,
                    item.importance,
                    to_iso(item.expiration),
                    dumps(item.embedding) if item.embedding else None,
                    dumps(item.metadata),
                ),
            )
        return item

    def read(self, item_id: str) -> MemoryItem | None:
        row = self.db.query_one("SELECT * FROM memory WHERE item_id = ?", (item_id,))
        if row is None:
            return None
        item = _row_to_memory(row)
        if item.is_expired():
            self.forget(item_id)
            return None
        return item

    def search(
        self, query: str, *, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemoryItem]:
        """Keyword matching over the live rows.

        The database narrows by expiry and scope; the word matching happens
        here. A variable number of LIKE clauses would mean assembling a query
        in a loop, and matching in Python costs nothing at this scale. A larger
        corpus needs a real index, which arrives with embeddings.
        """
        words = [w.lower() for w in query.split() if len(w) > 2][:8]
        if not words:
            return []
        now = to_iso(datetime.now(UTC))
        if scope is not None:
            rows = self.db.query(self._SELECT_LIVE_IN_SCOPE, (now, scope.value))
        else:
            rows = self.db.query(self._SELECT_LIVE, (now,))

        found: list[MemoryItem] = []
        for row in rows:
            content = row["content"].lower()
            if any(word in content for word in words):
                found.append(_row_to_memory(row))
                if len(found) >= limit:
                    break
        return found

    def forget(self, item_id: str) -> bool:
        with self.db.write() as connection:
            cursor = connection.execute("DELETE FROM memory WHERE item_id = ?", (item_id,))
            return cursor.rowcount > 0

    def all(self) -> list[MemoryItem]:
        """Every unexpired item. Used by the manager for ranked retrieval."""
        return [
            _row_to_memory(row)
            for row in self.db.query(self._SELECT_LIVE, (to_iso(datetime.now(UTC)),))
        ]

    def forget_where(self, scope: MemoryScope, task_id: str) -> int:
        with self.db.write() as connection:
            cursor = connection.execute(
                "DELETE FROM memory WHERE scope = ? AND json_extract(metadata, '$.task_id') = ?",
                (scope.value, task_id),
            )
            return cursor.rowcount

    def prune_expired(self) -> int:
        with self.db.write() as connection:
            cursor = connection.execute(
                "DELETE FROM memory WHERE expiration IS NOT NULL AND expiration <= ?",
                (to_iso(datetime.now(UTC)),),
            )
            return cursor.rowcount


# -- approvals --------------------------------------------------------------


class SqliteApprovalStore:
    """Approvals that another process can see and decide.

    This is what makes `jarvis approvals approve` real. A night shift raises an
    approval and goes on with its queue; in the morning the operator decides it from a
    different terminal, and the decision is there when the shift next looks.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def save(self, approval: Any) -> Any:
        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, request_id, tool_id, agent_id, task_id, risk,
                    argument_hash, reason, arguments_summary, status,
                    created_at, expires_at, decided_at, decided_by, note
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    status=excluded.status,
                    decided_at=excluded.decided_at,
                    decided_by=excluded.decided_by,
                    note=excluded.note
                """,
                (
                    approval.approval_id,
                    approval.request_id,
                    approval.tool_id,
                    approval.agent_id,
                    approval.task_id,
                    approval.risk.value,
                    approval.argument_hash,
                    approval.reason,
                    approval.arguments_summary,
                    approval.status.value,
                    to_iso(approval.created_at),
                    to_iso(approval.expires_at),
                    to_iso(approval.decided_at),
                    approval.decided_by,
                    approval.note,
                ),
            )
        return approval

    def _row_to_approval(self, row: sqlite3.Row) -> Any:
        # Imported here: tools depends on storage-free modules, and storage
        # should not pull the tool layer in at import time.
        from ..mcp.models import ToolRisk
        from ..tools.approval import ApprovalStatus, ToolApproval

        return ToolApproval(
            request_id=row["request_id"],
            tool_id=row["tool_id"],
            agent_id=row["agent_id"],
            task_id=row["task_id"],
            risk=ToolRisk(row["risk"]),
            argument_hash=row["argument_hash"],
            reason=row["reason"],
            arguments_summary=row["arguments_summary"],
            approval_id=row["approval_id"],
            status=ApprovalStatus(row["status"]),
            created_at=from_iso(row["created_at"]) or datetime.now(UTC),
            expires_at=from_iso(row["expires_at"]),
            decided_at=from_iso(row["decided_at"]),
            decided_by=row["decided_by"],
            note=row["note"] or "",
        )

    def get(self, approval_id: str) -> Any | None:
        row = self.db.query_one("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))
        return self._row_to_approval(row) if row else None

    def for_request(self, request_id: str) -> Any | None:
        row = self.db.query_one("SELECT * FROM approvals WHERE request_id = ?", (request_id,))
        return self._row_to_approval(row) if row else None

    def pending(self) -> list[Any]:
        rows = self.db.query(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at ASC"
        )
        return [self._row_to_approval(row) for row in rows]

    def all(self, *, limit: int = 100) -> list[Any]:
        rows = self.db.query("SELECT * FROM approvals ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row_to_approval(row) for row in rows]

    def expire_stale(self) -> int:
        with self.db.write() as connection:
            cursor = connection.execute(
                "UPDATE approvals SET status = 'expired' "
                "WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at <= ?",
                (to_iso(datetime.now(UTC)),),
            )
            return cursor.rowcount

    def decide(self, approval_id: str, status: str, *, by: str, note: str = "") -> tuple[bool, str]:
        """Record a decision, refusing one that was already made.

        The check and the write are one statement, so two terminals deciding
        the same approval at the same moment cannot both succeed.
        """
        with self.db.write() as connection:
            row = connection.execute(
                "SELECT status, expires_at FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                return False, f"no approval {approval_id!r}"
            if row["status"] != "pending":
                return False, f"this approval is already {row['status']}"
            expires_at = from_iso(row["expires_at"])
            if expires_at is not None and datetime.now(UTC) >= expires_at:
                connection.execute(
                    "UPDATE approvals SET status = 'expired' WHERE approval_id = ?",
                    (approval_id,),
                )
                return False, "this approval expired before it was decided"
            connection.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, note = ? "
                "WHERE approval_id = ? AND status = 'pending'",
                (status, to_iso(datetime.now(UTC)), by, note, approval_id),
            )
        return True, status

    def consume(self, approval_id: str) -> bool:
        """Spend an approved decision. Only the first caller wins."""
        with self.db.write() as connection:
            cursor = connection.execute(
                "UPDATE approvals SET status = 'consumed' "
                "WHERE approval_id = ? AND status = 'approved'",
                (approval_id,),
            )
            return cursor.rowcount == 1


# -- audit ------------------------------------------------------------------


class SqliteAuditSink:
    """Writes audit events to the database as well as the log file.

    Append only: there is no update or delete on this table, deliberately.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, event: Any) -> None:
        from ..logging import redact

        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO audit (
                    at, action, actor, subject, reason, agent, model, tool,
                    permission, task_id, result, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.at,
                    event.action,
                    event.actor,
                    event.subject,
                    event.reason,
                    event.agent,
                    event.model,
                    event.tool,
                    event.permission,
                    event.task_id,
                    event.result,
                    dumps(redact(event.metadata)),
                ),
            )

    def tail(self, count: int = 50, *, action: str | None = None) -> list[dict[str, Any]]:
        if action:
            rows = self.db.query(
                "SELECT * FROM audit WHERE action = ? ORDER BY id DESC LIMIT ?",
                (action, count),
            )
        else:
            rows = self.db.query("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (count,))
        return [dict(row) for row in reversed(rows)]

    def count(self, *, action: str | None = None) -> int:
        if action:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM audit WHERE action = ?", (action,))
        else:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM audit")
        return int(row["n"]) if row else 0


# -- cost -------------------------------------------------------------------


class SqliteCostSink:
    """Priced model calls, kept so spend survives a restart."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, entry: Any) -> None:
        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO costs (
                    at, model, provider, input_tokens, output_tokens,
                    cost_usd, agent, task_id, request_id
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    to_iso(entry.at),
                    entry.model,
                    entry.provider,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.cost_usd,
                    entry.agent,
                    entry.task_id,
                    entry.request_id,
                ),
            )

    def spent_since(self, since: datetime) -> float:
        row = self.db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM costs WHERE at >= ?",
            (to_iso(since),),
        )
        return float(row["total"]) if row else 0.0

    def spent_on_task(self, task_id: str) -> float:
        row = self.db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM costs WHERE task_id = ?",
            (task_id,),
        )
        return float(row["total"]) if row else 0.0


# -- agent verification -----------------------------------------------------


class SqliteVerificationStore:
    """What each agent has been shown to do, across runs and restarts.

    Verification against a real model costs money, so losing the record means
    paying for it again. It is stored as one row per agent with the runs as
    JSON: the runs are only ever read together, and a table of them would be
    a join for no benefit.
    """

    def __init__(self, database: Database, *, keep_runs: int = 25) -> None:
        self.db = database
        # Enough to see a trend, few enough that a row stays small. Old runs
        # are dropped oldest-first rather than the row growing forever.
        self.keep_runs = keep_runs

    def save(self, record: AgentVerification) -> AgentVerification:
        runs = record.runs[-self.keep_runs :]
        with self.db.write() as connection:
            connection.execute(
                """
                INSERT INTO agent_verification (
                    agent_id, capabilities_backed, has_smoke_task, runs, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    capabilities_backed = excluded.capabilities_backed,
                    has_smoke_task = excluded.has_smoke_task,
                    runs = excluded.runs,
                    updated_at = excluded.updated_at
                """,
                (
                    record.agent_id,
                    1 if record.capabilities_backed else 0,
                    1 if record.has_smoke_task else 0,
                    dumps([run.as_dict() for run in runs]),
                    to_iso(datetime.now(UTC)),
                ),
            )
        return record

    def load(self) -> list[AgentVerification]:
        records: list[AgentVerification] = []
        for row in self.db.query("SELECT * FROM agent_verification ORDER BY agent_id"):
            # Every field a run carries is read back. Dropping one here is not a
            # cosmetic loss: `blocked` distinguishes "the system refused this on
            # purpose" from "this agent failed", and a reload that forgets it
            # turns a working safety rule into a defect report against the
            # agent obeying it. A test asserts the round trip is lossless.
            runs = [
                VerificationRun(
                    agent_id=raw["agent_id"],
                    task=raw["task"],
                    real_model=bool(raw["real_model"]),
                    ok=bool(raw["ok"]),
                    score=float(raw.get("score", 0.0)),
                    passed=bool(raw.get("passed", False)),
                    confidence=float(raw.get("confidence", 0.0)),
                    cost_usd=float(raw.get("cost_usd", 0.0)),
                    latency_ms=float(raw.get("latency_ms", 0.0)),
                    model=raw.get("model", ""),
                    sources=int(raw.get("sources", 0)),
                    input_tokens=int(raw.get("input_tokens", 0)),
                    output_tokens=int(raw.get("output_tokens", 0)),
                    schema_valid=bool(raw.get("schema_valid", True)),
                    missing_keys=list(raw.get("missing_keys", [])),
                    answer_chars=int(raw.get("answer_chars", 0)),
                    truncated=bool(raw.get("truncated", False)),
                    unsupported_claims=list(raw.get("unsupported_claims", [])),
                    error=raw.get("error", ""),
                    failure=raw.get("failure", ""),
                    blocked=bool(raw.get("blocked", False)),
                    at=from_iso(raw.get("at")) or datetime.now(UTC),
                )
                for raw in loads(row["runs"], [])
            ]
            records.append(
                AgentVerification(
                    agent_id=row["agent_id"],
                    runs=runs,
                    capabilities_backed=bool(row["capabilities_backed"]),
                    has_smoke_task=bool(row["has_smoke_task"]),
                )
                # `real_blocked_reason` is deliberately not persisted: it
                # describes the environment at the moment of a sweep, not the
                # agent, and a stale "no credit" restored months later would be
                # a claim about an account nobody has checked.
            )
        return records


# -- the shared rate limit window -------------------------------------------


class SqliteRateLimitStore:
    """A rate limit window shared between every process on this machine.

    In memory, a limit of thirty calls a minute is thirty *per process*: a
    night shift and a terminal session each enforce it separately and the tool
    on the other end sees sixty. That is not a limit, it is a limit-shaped
    number.

    Two things differ from the in-memory version, and both are forced:

    * **Wall clock.** ``time.monotonic()`` is right for one process and
      meaningless between two, since each has its own zero. Rows are stamped in
      UTC. The cost is that a clock adjustment moves the window; the benefit is
      that there is a window at all.
    * **One transaction.** Checking and then inserting leaves a gap two
      processes both pass through. ``acquire`` does the count and the insert
      inside one ``BEGIN IMMEDIATE``, so the second one loses.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    # Every statement is a full literal, chosen between rather than assembled.
    _COUNT_TOOL = (
        "SELECT COUNT(*) AS n, MIN(at) AS first FROM tool_calls WHERE tool_id = ? AND at >= ?"
    )
    _COUNT_AGENT = (
        "SELECT COUNT(*) AS n, MIN(at) AS first FROM tool_calls "
        "WHERE agent_id = ? AND tool_id = ? AND at >= ?"
    )
    _COUNT_TASK = "SELECT COUNT(*) AS n FROM tool_calls WHERE task_id = ? AND tool_id = ?"
    _INSERT = "INSERT INTO tool_calls (at, tool_id, agent_id, task_id) VALUES (?, ?, ?, ?)"

    def _verdict(
        self,
        connection: sqlite3.Connection,
        *,
        tool_id: str,
        agent_id: str,
        task_id: str | None,
        limit: Any,
        window_s: float,
    ) -> LimitVerdict:
        now = datetime.now(UTC)
        cutoff = to_iso(now - timedelta(seconds=window_s))

        row = connection.execute(self._COUNT_TOOL, (tool_id, cutoff)).fetchone()
        if row["n"] >= limit.per_minute:
            return LimitVerdict(
                False,
                f"{tool_id} has been called {row['n']} times in the last minute, "
                f"and its limit is {limit.per_minute}",
                "tool",
                _seconds_left(row["first"], now, window_s),
            )

        row = connection.execute(self._COUNT_AGENT, (agent_id, tool_id, cutoff)).fetchone()
        if row["n"] >= limit.per_agent_per_minute:
            return LimitVerdict(
                False,
                f"{agent_id} has called {tool_id} {row['n']} times in the last minute, "
                f"and its limit is {limit.per_agent_per_minute}",
                "agent",
                _seconds_left(row["first"], now, window_s),
            )

        if task_id is not None:
            row = connection.execute(self._COUNT_TASK, (task_id, tool_id)).fetchone()
            if row["n"] >= limit.per_task:
                return LimitVerdict(
                    False,
                    f"task {task_id} has called {tool_id} {row['n']} times, "
                    f"and its limit is {limit.per_task}",
                    "task",
                )
        return LimitVerdict(True)

    def check(
        self, *, tool_id: str, agent_id: str, task_id: str | None, limit: Any, window_s: float
    ) -> LimitVerdict:
        return self._verdict(
            self.db.connection,
            tool_id=tool_id,
            agent_id=agent_id,
            task_id=task_id,
            limit=limit,
            window_s=window_s,
        )

    def acquire(
        self, *, tool_id: str, agent_id: str, task_id: str | None, limit: Any, window_s: float
    ) -> LimitVerdict:
        """Count and insert together. The second caller through loses."""
        with self.db.write() as connection:
            verdict = self._verdict(
                connection,
                tool_id=tool_id,
                agent_id=agent_id,
                task_id=task_id,
                limit=limit,
                window_s=window_s,
            )
            if verdict.allowed:
                connection.execute(
                    self._INSERT, (to_iso(datetime.now(UTC)), tool_id, agent_id, task_id)
                )
            return verdict

    def record(self, *, tool_id: str, agent_id: str, task_id: str | None) -> None:
        with self.db.write() as connection:
            connection.execute(
                self._INSERT, (to_iso(datetime.now(UTC)), tool_id, agent_id, task_id)
            )

    def reset_task(self, task_id: str) -> None:
        with self.db.write() as connection:
            connection.execute("DELETE FROM tool_calls WHERE task_id = ?", (task_id,))

    def reset(self) -> None:
        with self.db.write() as connection:
            connection.execute("DELETE FROM tool_calls")

    def prune(self, *, older_than_s: float = 3600.0) -> int:
        """Drop rows outside any window anyone could still be asking about.

        The table is append-only during normal operation, so without this it
        grows for as long as the install runs. An hour is far outside the
        one-minute window and leaves enough to debug a rate limit that fired.
        """
        cutoff = to_iso(datetime.now(UTC) - timedelta(seconds=older_than_s))
        with self.db.write() as connection:
            cursor = connection.execute("DELETE FROM tool_calls WHERE at < ?", (cutoff,))
            return cursor.rowcount


def _seconds_left(first_at: str | None, now: datetime, window_s: float) -> float:
    """How long until the oldest call in the window falls out of it."""
    oldest = from_iso(first_at)
    if oldest is None:
        return window_s
    return max(0.0, window_s - (now - oldest).total_seconds())
