"""Durable storage.

Everything the system knows has been living in memory: tasks, approvals,
memory, the audit tail. That was fine while one process did everything, and it
stopped being fine the moment a night shift could raise an approval that
nobody else could see.

SQLite, not PostgreSQL, and that is a deliberate choice. It needs no server, so
local development and CI keep working with nothing installed; it is a real
database, so a restart no longer loses in-flight work; and two processes on one
machine can share it, which is the actual problem. PostgreSQL earns its place
when there are two machines or when vector search arrives, and this module's
shape is what lets it slot in then.

Write-ahead logging is on, so a reader never blocks a writer. Connections are
per-thread, because SQLite objects are not safe to move between them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging import get_logger

log = get_logger("storage")

SCHEMA_VERSION = 8

# One statement per table. Kept as plain SQL rather than behind an ORM: the
# schema is small, and being able to read it is worth more than the layer.
# The version table stands alone, because everything else depends on knowing
# which version this database is before touching it.
_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    parent_task_id  TEXT,
    goal            TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 5,
    status          TEXT NOT NULL,
    assigned_agent  TEXT,
    dependencies    TEXT NOT NULL DEFAULT '[]',
    inputs          TEXT NOT NULL DEFAULT '{}',
    outputs         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    risk            TEXT NOT NULL DEFAULT 'low',
    error           TEXT,
    trace_id        TEXT,
    goal_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id);

CREATE TABLE IF NOT EXISTS memory (
    item_id     TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    scope       TEXT NOT NULL,
    source      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    importance  REAL NOT NULL DEFAULT 0.5,
    expiration  TEXT,
    embedding   TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory(scope);
CREATE INDEX IF NOT EXISTS idx_memory_expiration ON memory(expiration);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id       TEXT PRIMARY KEY,
    request_id        TEXT NOT NULL,
    tool_id           TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    task_id           TEXT,
    risk              TEXT NOT NULL,
    argument_hash     TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT '',
    arguments_summary TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    expires_at        TEXT,
    decided_at        TEXT,
    decided_by        TEXT,
    note              TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_request ON approvals(request_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);

CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    action    TEXT NOT NULL,
    actor     TEXT,
    subject   TEXT,
    reason    TEXT,
    agent     TEXT,
    model     TEXT,
    tool      TEXT,
    permission TEXT,
    task_id   TEXT,
    result    TEXT,
    metadata  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit(action);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit(at);

CREATE TABLE IF NOT EXISTS costs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    model         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    agent         TEXT,
    task_id       TEXT,
    request_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_costs_at ON costs(at);

CREATE TABLE IF NOT EXISTS twin_traits (
    key               TEXT PRIMARY KEY,
    domain            TEXT NOT NULL,
    statement         TEXT NOT NULL,
    observations      TEXT NOT NULL DEFAULT '[]',
    superseded        TEXT NOT NULL DEFAULT '[]',
    contested         INTEGER NOT NULL DEFAULT 0,
    contested_by      TEXT NOT NULL DEFAULT '',
    contested_reason  TEXT NOT NULL DEFAULT '',
    resolved_by       TEXT NOT NULL DEFAULT '',
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_twin_domain ON twin_traits(domain);
CREATE INDEX IF NOT EXISTS idx_twin_contested ON twin_traits(contested);

CREATE TABLE IF NOT EXISTS goals (
    goal_id        TEXT PRIMARY KEY,
    parent_id      TEXT,
    title          TEXT NOT NULL,
    why            TEXT NOT NULL DEFAULT '',
    horizon        TEXT NOT NULL,
    status         TEXT NOT NULL,
    measure_kind   TEXT NOT NULL DEFAULT 'none',
    measure_target REAL NOT NULL DEFAULT 0,
    measure_current REAL NOT NULL DEFAULT 0,
    measure_unit   TEXT NOT NULL DEFAULT '',
    measure_note   TEXT NOT NULL DEFAULT '',
    due            TEXT,
    weight         REAL NOT NULL DEFAULT 1,
    blocked_by     TEXT NOT NULL DEFAULT '',
    tags           TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    effort_tasks   INTEGER NOT NULL DEFAULT 0,
    effort_cost    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

CREATE TABLE IF NOT EXISTS tool_calls (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT NOT NULL,
    tool_id  TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_at ON tool_calls(at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_id, at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_id, tool_id, at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id, tool_id);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    thesis         TEXT NOT NULL DEFAULT '',
    stage          TEXT NOT NULL,
    assessments    TEXT NOT NULL DEFAULT '{}',
    notes          TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opportunities_stage ON opportunities(stage);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    turns           TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);

CREATE TABLE IF NOT EXISTS agent_verification (
    agent_id            TEXT PRIMARY KEY,
    capabilities_backed INTEGER NOT NULL DEFAULT 1,
    has_smoke_task      INTEGER NOT NULL DEFAULT 1,
    runs                TEXT NOT NULL DEFAULT '[]',
    updated_at          TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class AddColumn:
    """A column added to a table that already exists.

    Idempotent on purpose. A new table needs no migration at all — the
    ``CREATE TABLE IF NOT EXISTS`` pass makes it — so the only statements that
    reach here are column additions, and those have to survive being tried
    twice: a migration that fails halfway leaves the database at its old
    version with some of the work done, and the next open must be able to
    finish rather than crash on the part that succeeded.

    ``statement`` is a full literal, never assembled, so the no-built-SQL rule
    holds. The existence check uses ``pragma_table_info``, which does take a
    bound parameter, unlike the ``PRAGMA`` statement form.
    """

    table: str
    column: str
    statement: str

    def apply(self, connection: sqlite3.Connection) -> bool:
        """Add the column if it is missing. Returns whether anything changed.

        A table that does not exist yet is skipped rather than being an error.
        That happens on a long upgrade: a v1 database jumping to v4 has no
        ``goals`` table when the v4 step runs, because ``_SCHEMA`` creates it
        afterwards — already carrying the columns this step would have added.
        """
        columns = {
            row["name"]
            for row in connection.execute("SELECT name FROM pragma_table_info(?)", (self.table,))
        }
        if not columns or self.column in columns:
            return False
        connection.execute(self.statement)
        return True


# Schema changes, applied in order to a database that predates them. A fresh
# database is created at the current shape by _SCHEMA above and skips these
# entirely.
_MIGRATIONS: dict[int, tuple[AddColumn, ...]] = {
    # v2 added the digital twin, v3 added goals. Both are whole new tables,
    # made by the IF NOT EXISTS pass, so there is nothing to alter — the
    # migration exists only to move the version stamp.
    2: (),
    3: (),
    # v4 linked work to goals, which does need columns on existing tables.
    # v5 added agent verification, v6 opportunities, v7 the shared rate limit
    # window and v8 chat conversations. All whole new tables, made by the
    # IF NOT EXISTS pass, so none of them has anything to alter.
    5: (),
    6: (),
    7: (),
    8: (),
    4: (
        AddColumn("tasks", "goal_id", "ALTER TABLE tasks ADD COLUMN goal_id TEXT"),
        AddColumn(
            "goals",
            "effort_tasks",
            "ALTER TABLE goals ADD COLUMN effort_tasks INTEGER NOT NULL DEFAULT 0",
        ),
        AddColumn(
            "goals",
            "effort_cost",
            "ALTER TABLE goals ADD COLUMN effort_cost REAL NOT NULL DEFAULT 0",
        ),
    ),
}


def to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    # Rows written by an older build may lack a zone; treat those as UTC
    # rather than comparing naive and aware datetimes at read time.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class Database:
    """A SQLite file, with the schema applied and connections per thread."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # An in-memory database is per-connection, so it must be shared
        # explicitly or every thread gets its own empty one.
        self._shared_memory = str(self.path) == ":memory:"
        self._memory_connection: sqlite3.Connection | None = None
        self._apply_schema()

    # -- connections ------------------------------------------------------

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=10.0,
            isolation_level=None,  # autocommit; transactions are explicit
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self._shared_memory:
            # Readers do not block the writer, which is what makes two
            # processes on one machine workable.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._shared_memory:
            if self._memory_connection is None:
                self._memory_connection = self._new_connection()
            return self._memory_connection
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._new_connection()
            self._local.connection = existing
        return existing

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A write transaction. Serialised, and rolled back on failure."""
        with self._write_lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.write() as connection:
            connection.execute(sql, params)

    # -- schema -----------------------------------------------------------

    def _apply_schema(self) -> None:
        """Bring the database to the current shape.

        The order matters and cost a bug to get right. ``_SCHEMA`` used to run
        first, which broke as soon as a new column got an index: the
        ``CREATE INDEX ... ON tasks(goal_id)`` line ran against a database
        whose ``tasks`` table predated the column, and every open failed with
        "no such column". Migrations have to add the columns before anything
        indexes them.

        So: make the version table, read the stamp, migrate, then apply the
        schema. A fresh database has no stamp, skips the migrations entirely,
        and is created at the current shape by ``_SCHEMA``.
        """
        connection = self.connection
        connection.executescript(_VERSION_TABLE)
        row = connection.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            connection.executescript(_SCHEMA)
            connection.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            log.debug("storage ready at %s", self.path)
            return

        found = int(row["version"])
        if found == SCHEMA_VERSION:
            connection.executescript(_SCHEMA)
            log.debug("storage ready at %s", self.path)
            return
        if found > SCHEMA_VERSION:
            # An older build opening a newer database. Refusing is the only
            # safe answer: writing v2 shapes into a v3 file corrupts it.
            raise RuntimeError(
                f"the database at {self.path} is schema version {found}, "
                f"newer than this build ({SCHEMA_VERSION}). Upgrade rather than downgrade."
            )

        for version in range(found + 1, SCHEMA_VERSION + 1):
            changed = [
                step.column for step in _MIGRATIONS.get(version, ()) if step.apply(connection)
            ]
            log.info(
                "migrated %s to schema version %s%s",
                self.path,
                version,
                f" (added {', '.join(changed)})" if changed else "",
            )
        # Only now: every column the indexes below refer to exists.
        connection.executescript(_SCHEMA)
        connection.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        connection.commit()
        log.debug("storage ready at %s", self.path)

    @property
    def version(self) -> int:
        row = self.query_one("SELECT version FROM schema_version")
        return int(row["version"]) if row else 0

    # One literal statement per table, so no query is ever built from a
    # variable. A new table needs a line here, which is the point.
    _COUNTS: dict[str, str] = {
        "tasks": "SELECT COUNT(*) AS n FROM tasks",
        "memory": "SELECT COUNT(*) AS n FROM memory",
        "approvals": "SELECT COUNT(*) AS n FROM approvals",
        "audit": "SELECT COUNT(*) AS n FROM audit",
        "costs": "SELECT COUNT(*) AS n FROM costs",
        "twin_traits": "SELECT COUNT(*) AS n FROM twin_traits",
        "goals": "SELECT COUNT(*) AS n FROM goals",
        "agent_verification": "SELECT COUNT(*) AS n FROM agent_verification",
        "opportunities": "SELECT COUNT(*) AS n FROM opportunities",
        "tool_calls": "SELECT COUNT(*) AS n FROM tool_calls",
        "conversations": "SELECT COUNT(*) AS n FROM conversations",
    }

    def counts(self) -> dict[str, int]:
        """How much is stored, per table."""
        return {table: int(self.query_one(sql)["n"]) for table, sql in self._COUNTS.items()}

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None
