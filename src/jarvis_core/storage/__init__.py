"""Durable storage.

SQLite, chosen because it needs no server, works in CI, survives a restart,
and can be shared between two processes on one machine. PostgreSQL earns its
place when there are two machines or when vector search arrives; the store
protocols are what let it slot in then.

Storage is optional. With no database configured the system runs entirely in
memory, exactly as before.

The stores come in two halves. The generic ones — tasks, memory, approvals,
the audit sink, cost, verification, rate limits — are in `stores.py`. The four
that persist owner-specific subsystems are in `private_stores.py` and are
imported only when that module is present, so a distribution without those
subsystems is not coupled to their types.
"""

from .database import SCHEMA_VERSION, Database
from .setup import HAS_PRIVATE_STORES, RETENTION, PruneReport, Storage, open_storage, prune
from .stores import (
    SqliteApprovalStore,
    SqliteAuditSink,
    SqliteCostSink,
    SqliteMemoryStore,
    SqliteRateLimitStore,
    SqliteTaskStore,
    SqliteVerificationStore,
)

__all__ = [
    "HAS_PRIVATE_STORES",
    "RETENTION",
    "SCHEMA_VERSION",
    "Database",
    "PruneReport",
    "SqliteApprovalStore",
    "SqliteAuditSink",
    "SqliteCostSink",
    "SqliteMemoryStore",
    "SqliteRateLimitStore",
    "SqliteTaskStore",
    "SqliteVerificationStore",
    "Storage",
    "open_storage",
    "prune",
]

if HAS_PRIVATE_STORES:  # pragma: no cover - always true in this distribution
    from .private_stores import (
        SqliteConversationStore,
        SqliteGoalStore,
        SqliteOpportunityStore,
        SqliteTwinStore,
    )

    __all__ += [
        "SqliteConversationStore",
        "SqliteGoalStore",
        "SqliteOpportunityStore",
        "SqliteTwinStore",
    ]
