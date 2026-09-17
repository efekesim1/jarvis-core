"""Opening the database and handing out the stores that use it.

One function, so every entry point gets the same arrangement. Storage is
optional throughout: with no path configured this returns nothing and the
system runs in memory exactly as it did before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..logging import get_logger
from .database import Database
from .stores import (
    SqliteApprovalStore,
    SqliteAuditSink,
    SqliteCostSink,
    SqliteMemoryStore,
    SqliteRateLimitStore,
    SqliteTaskStore,
    SqliteVerificationStore,
)

# The stores for the owner-specific subsystems, when this distribution has
# them. A build without the twin, the goal tree, the opportunity pipeline and
# chat has no use for their tables, and importing their types would couple
# the storage layer to subsystems that are not there. Optional, so the same
# `setup.py` serves both: absent means those four attributes are None, which
# a caller can check, rather than an ImportError at start-up.
try:
    from .private_stores import (
        SqliteConversationStore,
        SqliteGoalStore,
        SqliteOpportunityStore,
        SqliteTwinStore,
    )

    HAS_PRIVATE_STORES = True
except ImportError:  # pragma: no cover - exercised by the public distribution
    HAS_PRIVATE_STORES = False

log = get_logger("storage.setup")


@dataclass(slots=True)
class Storage:
    """An open database and the stores backed by it."""

    database: Database
    tasks: SqliteTaskStore
    memory: SqliteMemoryStore
    approvals: SqliteApprovalStore
    audit: SqliteAuditSink
    costs: SqliteCostSink
    verification: SqliteVerificationStore
    rate_limits: SqliteRateLimitStore
    # Present only where the subsystems they serve are. `None` is a real
    # answer: it says this build has no twin rather than that the database
    # is broken.
    twin: Any = None
    goals: Any = None
    opportunities: Any = None
    conversations: Any = None

    @property
    def path(self) -> Path:
        return self.database.path

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.database.path),
            "schema_version": self.database.version,
            "rows": self.database.counts(),
        }

    def close(self) -> None:
        self.database.close()


def open_storage(
    path: Path | str | None = None, *, settings: Settings | None = None
) -> Storage | None:
    """Open the database, or return ``None`` when none is configured.

    Returning ``None`` rather than falling back to an in-memory database is
    deliberate: a caller that thinks it has durable storage and does not
    should find out here, not after a restart loses something.
    """
    settings = settings or get_settings()
    target = path if path is not None else settings.database_path
    if target is None:
        return None

    database = Database(target)
    log.info("storage open at %s (schema %d)", database.path, database.version)
    return Storage(
        database=database,
        tasks=SqliteTaskStore(database),
        memory=SqliteMemoryStore(database),
        approvals=SqliteApprovalStore(database),
        audit=SqliteAuditSink(database),
        costs=SqliteCostSink(database),
        verification=SqliteVerificationStore(database),
        rate_limits=SqliteRateLimitStore(database),
        twin=SqliteTwinStore(database) if HAS_PRIVATE_STORES else None,
        goals=SqliteGoalStore(database) if HAS_PRIVATE_STORES else None,
        opportunities=SqliteOpportunityStore(database) if HAS_PRIVATE_STORES else None,
        conversations=SqliteConversationStore(database) if HAS_PRIVATE_STORES else None,
    )


# How long each append-only table is kept, by default.
#
# These tables only grow. Audit rows are the record of what the system did and
# are worth keeping longest; cost rows back the monthly figures and are worth
# keeping past a billing cycle; the rate limit window is meaningless after an
# hour and is kept only long enough to explain a limit that fired.
RETENTION: dict[str, int] = {
    "audit": 365,
    "costs": 400,
    "tool_calls": 1,
}


@dataclass(slots=True)
class PruneReport:
    """What a retention pass removed, per table."""

    removed: dict[str, int] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.removed.values())

    def render(self) -> str:
        if not self.total:
            return "Nothing to prune. Every table is inside its retention window."
        lines = [f"Removed {self.total} row(s):"]
        for table, count in sorted(self.removed.items()):
            if count:
                lines.append(f"  {table}: {count} removed, {self.kept.get(table, 0)} kept")
        return "\n".join(lines)


def prune(storage: Storage, *, retention: dict[str, int] | None = None) -> PruneReport:
    """Drop rows past their retention window.

    Deliberately a function someone calls rather than something that happens
    on open. A delete that runs implicitly is a delete nobody decided to run,
    and this one removes the record of what the system did — the thing you
    most want to still have after something goes wrong.

    Rows are deleted, not vacuumed. Reclaiming the file space needs an
    exclusive lock on the whole database, and taking that on a system another
    process may be using is a worse problem than a slightly large file.
    """
    from datetime import UTC, datetime, timedelta

    windows = {**RETENTION, **(retention or {})}
    report = PruneReport()

    # One literal statement per table, chosen between rather than assembled,
    # so the no-built-SQL rule holds here too.
    statements = {
        "audit": "DELETE FROM audit WHERE at < ?",
        "costs": "DELETE FROM costs WHERE at < ?",
        "tool_calls": "DELETE FROM tool_calls WHERE at < ?",
    }

    for table, days in windows.items():
        statement = statements.get(table)
        if statement is None:
            log.warning("no retention statement for %r; skipping", table)
            continue
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with storage.database.write() as connection:
            cursor = connection.execute(statement, (cutoff,))
            report.removed[table] = max(0, cursor.rowcount)

    counts = storage.database.counts()
    report.kept = {table: counts.get(table, 0) for table in report.removed}
    log.info("pruned %d row(s) past their retention window", report.total)
    return report
