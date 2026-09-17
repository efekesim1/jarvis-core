"""Durable storage: it survives a restart, and two processes share it.

The question these answer is not "does SQLite work". It is whether the thing
that was blocking autonomy is actually fixed: an approval raised by a night
shift, decided from a different terminal, honoured by the shift.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fake_credentials import fake_anthropic_key

from jarvis_core.core.task import TaskEngine, TaskStatus
from jarvis_core.memory.base import MemoryScope
from jarvis_core.memory.manager import MemoryManager
from jarvis_core.security.permissions import RiskLevel
from jarvis_core.storage import Database, open_storage
from jarvis_core.storage.database import SCHEMA_VERSION
from jarvis_core.storage.stores import SqliteTaskStore


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "jarvis.db"


@pytest.fixture
def storage(db_path):
    handle = open_storage(db_path)
    yield handle
    handle.close()


# -- the database -----------------------------------------------------------


def test_a_new_database_has_the_current_schema(storage):
    assert storage.database.version == SCHEMA_VERSION
    assert storage.summary()["rows"] == {
        "tasks": 0,
        "memory": 0,
        "approvals": 0,
        "audit": 0,
        "costs": 0,
        "twin_traits": 0,
        "goals": 0,
        "agent_verification": 0,
        "opportunities": 0,
        "tool_calls": 0,
        "conversations": 0,
    }


def test_no_configured_path_means_no_storage():
    """Silently falling back to memory would hide a missing database."""
    assert open_storage(None) is None


def test_the_schema_is_applied_only_once(db_path):
    first = open_storage(db_path)
    first.close()
    second = open_storage(db_path)
    assert second.database.version == SCHEMA_VERSION
    second.close()


# -- tasks ------------------------------------------------------------------


def test_a_task_survives_a_restart(db_path):
    engine = TaskEngine(SqliteTaskStore(Database(db_path)))
    task = engine.create("research something", priority=2, risk=RiskLevel.MEDIUM)
    engine.transition(task, TaskStatus.RUNNING)
    engine.add_cost(task, 0.25)

    # A fresh process, pointing at the same file.
    reopened = TaskEngine(SqliteTaskStore(Database(db_path)))
    found = reopened.get(task.task_id)

    assert found is not None
    assert found.goal == "research something"
    assert found.status is TaskStatus.RUNNING
    assert found.cost_usd == pytest.approx(0.25)
    assert found.risk is RiskLevel.MEDIUM


def test_task_inputs_and_outputs_round_trip(db_path):
    engine = TaskEngine(SqliteTaskStore(Database(db_path)))
    task = engine.create("x", inputs={"findings": [{"a": 1}], "turkish": "araştır"})
    engine.transition(task, TaskStatus.RUNNING)
    engine.complete(task, {"answer": "something", "sources": ["https://example.com"]})

    found = TaskEngine(SqliteTaskStore(Database(db_path))).get(task.task_id)
    assert found.inputs["findings"] == [{"a": 1}]
    assert found.inputs["turkish"] == "araştır"
    assert found.outputs["answer"] == "something"


def test_children_are_found_after_a_restart(db_path):
    engine = TaskEngine(SqliteTaskStore(Database(db_path)))
    parent = engine.create("parent")
    engine.create("child one", parent=parent)
    engine.create("child two", parent=parent)

    reopened = TaskEngine(SqliteTaskStore(Database(db_path)))
    assert len(reopened.children(parent)) == 2


def test_tasks_can_be_listed_by_status(storage):
    engine = TaskEngine(storage.tasks)
    done = engine.create("done")
    engine.transition(done, TaskStatus.RUNNING)
    engine.complete(done, {})
    engine.create("queued")

    assert len(engine.store.list(status=TaskStatus.COMPLETED)) == 1
    assert len(engine.store.list(status=TaskStatus.QUEUED)) == 1


def test_the_snapshot_reads_from_the_database(storage):
    engine = TaskEngine(storage.tasks)
    task = engine.create("a")
    engine.transition(task, TaskStatus.RUNNING)
    engine.add_cost(task, 0.5)
    engine.complete(task, {})

    snapshot = engine.snapshot()
    assert snapshot["total"] == 1
    assert snapshot["total_cost_usd"] == pytest.approx(0.5)


# -- memory -----------------------------------------------------------------


def test_memory_survives_a_restart(db_path):
    storage = open_storage(db_path)
    memory = MemoryManager(storage.memory)
    memory.remember(
        "the operator chose PostgreSQL for this project",
        scope=MemoryScope.DECISION,
        source="conversation",
        importance=0.9,
    )
    storage.close()

    reopened = open_storage(db_path)
    found = MemoryManager(reopened.memory).recall("postgresql project")
    assert found
    assert "PostgreSQL" in found[0].item.content
    reopened.close()


def test_the_write_policy_still_refuses_most_writes(storage):
    memory = MemoryManager(storage.memory)
    assert (
        memory.remember("a trivial aside", scope=MemoryScope.USER, source="x", importance=0.1)
        is None
    )
    assert storage.database.counts()["memory"] == 0


def test_an_expired_memory_is_not_returned_from_the_database(storage):
    memory = MemoryManager(storage.memory)
    item = memory.remember(
        "postgres note that has expired",
        scope=MemoryScope.KNOWLEDGE,
        source="x",
        importance=0.9,
        expires_in=timedelta(seconds=-1),
    )
    assert item is not None
    assert memory.recall("postgres note") == []


def test_pruning_removes_expired_rows(storage):
    memory = MemoryManager(storage.memory)
    memory.remember(
        "expired",
        scope=MemoryScope.KNOWLEDGE,
        source="x",
        importance=0.9,
        expires_in=timedelta(seconds=-1),
    )
    memory.remember("kept postgres fact", scope=MemoryScope.KNOWLEDGE, source="x", importance=0.9)
    assert memory.prune() == 1
    assert storage.database.counts()["memory"] == 1


def test_working_memory_is_dropped_by_task_in_the_database(storage):
    memory = MemoryManager(storage.memory)
    memory.remember(
        "scratch",
        scope=MemoryScope.WORKING,
        source="task",
        importance=0.5,
        metadata={"task_id": "tsk_1"},
    )
    memory.remember(
        "other",
        scope=MemoryScope.WORKING,
        source="task",
        importance=0.5,
        metadata={"task_id": "tsk_2"},
    )
    assert memory.forget_task("tsk_1") == 1
    assert storage.database.counts()["memory"] == 1


# -- approvals: the thing that was blocking --------------------------------


async def test_an_approval_raised_in_one_process_is_visible_in_another(storage):
    """The night shift raises it; a different terminal sees it."""
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    shift = ApprovalEngine(store=storage.approvals)
    request = ToolRequest(
        tool_id="mock.deploy",
        arguments={"target": "production"},
        agent_id="deployment_agent",
        task_id="tsk_night",
    )
    raised = shift.require(request, ToolRisk.HIGH_RISK_WRITE, reason="deploy the build")

    # A separate engine, as another process would have.
    terminal = ApprovalEngine(store=storage.approvals)
    pending = terminal.pending
    assert [a.approval_id for a in pending] == [raised.approval_id]
    assert pending[0].arguments_summary


async def test_a_decision_in_one_process_is_honoured_in_another(storage):
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine, ApprovalStatus

    shift = ApprovalEngine(store=storage.approvals)
    request = ToolRequest(tool_id="mock.deploy", arguments={"target": "staging"}, agent_id="agent")
    raised = shift.require(request, ToolRisk.HIGH_RISK_WRITE)

    # the operator decides from a different terminal.
    ApprovalEngine(store=storage.approvals).approve(raised.approval_id, by="operator", note="go")

    # The shift picks it up and spends it.
    allowed, reason, status, bypass = shift.consume(request)
    assert allowed, reason
    assert status is ApprovalStatus.APPROVED
    assert not bypass


async def test_an_approval_cannot_be_spent_twice_across_processes(storage):
    """The replay defence has to hold when two processes race for it."""
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    first = ApprovalEngine(store=storage.approvals)
    request = ToolRequest(tool_id="mock.deploy", arguments={"target": "staging"}, agent_id="agent")
    raised = first.require(request, ToolRisk.HIGH_RISK_WRITE)
    first.approve(raised.approval_id)

    second = ApprovalEngine(store=storage.approvals)
    assert first.consume(request)[0] is True
    allowed, reason, _, bypass = second.consume(request)
    assert not allowed
    assert bypass
    assert "already been used" in reason


async def test_two_terminals_cannot_both_decide_the_same_approval(storage):
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    raiser = ApprovalEngine(store=storage.approvals)
    raised = raiser.require(
        ToolRequest(tool_id="mock.deploy", arguments={"t": "x"}, agent_id="a"),
        ToolRisk.HIGH_RISK_WRITE,
    )

    ApprovalEngine(store=storage.approvals).approve(raised.approval_id, by="operator")
    with pytest.raises(ValueError, match="already"):
        ApprovalEngine(store=storage.approvals).reject(raised.approval_id, by="someone_else")


async def test_a_rejection_is_honoured_across_processes(storage):
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine, ApprovalStatus

    shift = ApprovalEngine(store=storage.approvals)
    request = ToolRequest(tool_id="mock.deploy", arguments={"t": "x"}, agent_id="a")
    raised = shift.require(request, ToolRisk.HIGH_RISK_WRITE)
    ApprovalEngine(store=storage.approvals).reject(raised.approval_id, by="operator", note="no")

    allowed, _, status, _ = shift.consume(request)
    assert not allowed
    assert status is ApprovalStatus.REJECTED


async def test_an_expired_approval_cannot_be_decided(storage):
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    engine = ApprovalEngine(store=storage.approvals, ttl_s=-1)
    raised = engine.require(
        ToolRequest(tool_id="mock.deploy", arguments={"t": "x"}, agent_id="a"),
        ToolRisk.HIGH_RISK_WRITE,
    )
    with pytest.raises(ValueError, match="expired"):
        engine.approve(raised.approval_id)


async def test_stale_approvals_expire_in_the_database(storage):
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    engine = ApprovalEngine(store=storage.approvals, ttl_s=-1)
    engine.require(
        ToolRequest(tool_id="mock.deploy", arguments={"t": "x"}, agent_id="a"),
        ToolRisk.HIGH_RISK_WRITE,
    )
    assert engine.pending == []


# -- audit ------------------------------------------------------------------


def test_audit_events_are_queryable_after_a_restart(db_path):
    from jarvis_core.logging import AuditLog

    storage = open_storage(db_path)
    log = AuditLog(path=db_path.parent / "audit.jsonl", sink=storage.audit)
    log.emit("tool.denied", agent="stranger", reason="no policy")
    log.emit("tool.completed", agent="reader")
    storage.close()

    reopened = open_storage(db_path)
    assert reopened.audit.count() == 2
    assert reopened.audit.count(action="tool.denied") == 1
    assert reopened.audit.tail(1)[0]["action"] == "tool.completed"
    reopened.close()


def test_a_credential_never_reaches_the_audit_table(storage):
    from jarvis_core.logging import AuditLog

    log = AuditLog(sink=storage.audit)
    log.emit("model.call", metadata={"api_key": fake_anthropic_key("SHOULDNOTAPPEAR0123456789")})
    stored = json.dumps(storage.audit.tail(1))
    assert "SHOULDNOTAPPEAR" not in stored
    assert "redacted" in stored


def test_a_failing_sink_does_not_stop_the_thing_being_audited(storage):
    from jarvis_core.logging import AuditLog

    class Broken:
        def record(self, event):
            raise RuntimeError("the disk is on fire")

    log = AuditLog(sink=Broken())
    event = log.emit("tool.completed", agent="reader")
    assert event.action == "tool.completed"
    assert log.events  # the in-memory tail still has it


# -- concurrency ------------------------------------------------------------


def test_concurrent_writers_do_not_lose_rows(storage):
    """Two threads writing at once must not drop or corrupt anything."""
    engine = TaskEngine(storage.tasks)
    errors: list[Exception] = []

    def write(offset: int) -> None:
        try:
            for index in range(25):
                engine.create(f"task {offset}-{index}")
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert storage.database.counts()["tasks"] == 100


def test_a_failed_write_rolls_back(storage):
    with pytest.raises(RuntimeError), storage.database.write() as connection:
        connection.execute(
            "INSERT INTO tasks (task_id, goal, status, created_at) VALUES (?,?,?,?)",
            ("tsk_rollback", "x", "queued", datetime.now(UTC).isoformat()),
        )
        raise RuntimeError("something went wrong halfway")
    assert storage.database.counts()["tasks"] == 0


# -- a real second process --------------------------------------------------

_CHILD = """
import sys
sys.path.insert(0, {src!r})
from jarvis_core.storage import open_storage
from jarvis_core.tools.approval import ApprovalEngine

storage = open_storage({db!r})
engine = ApprovalEngine(store=storage.approvals)
pending = engine.pending
print("PENDING", len(pending))
if pending:
    engine.approve(pending[0].approval_id, by="operator-from-another-process", note="ok")
    print("APPROVED", pending[0].approval_id)
storage.close()
"""


async def test_a_genuinely_separate_process_can_decide_an_approval(db_path, tmp_path):
    """Not two objects in one process: a second Python interpreter."""
    from jarvis_core.mcp.models import ToolRequest, ToolRisk
    from jarvis_core.tools.approval import ApprovalEngine

    storage = open_storage(db_path)
    shift = ApprovalEngine(store=storage.approvals)
    request = ToolRequest(
        tool_id="mock.deploy", arguments={"target": "production"}, agent_id="agent"
    )
    raised = shift.require(request, ToolRisk.HIGH_RISK_WRITE, reason="overnight deploy")

    src = str(Path(__file__).resolve().parent.parent / "src")
    script = tmp_path / "decide.py"
    script.write_text(_CHILD.format(src=src, db=str(db_path)))

    result = subprocess.run(  # noqa: S603 - a fixed script, no shell
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "PENDING 1" in result.stdout
    assert "APPROVED" in result.stdout

    # Back in the shift: the decision is there, and the call may proceed.
    allowed, reason, _, _ = shift.consume(request)
    assert allowed, reason
    assert shift.get(raised.approval_id).decided_by == "operator-from-another-process"
    storage.close()


# -- end to end -------------------------------------------------------------


async def test_a_whole_run_persists_and_is_readable_afterwards(
    registry, router, guardian, audit, settings, db_path
):
    from jarvis_core.core.runtime import JarvisRuntime
    from jarvis_core.evaluation.engine import EvaluationEngine

    storage = open_storage(db_path)
    jarvis = JarvisRuntime(
        registry=registry,
        router=router,
        tasks=TaskEngine(storage.tasks),
        memory=MemoryManager(storage.memory),
        guardian=guardian,
        evaluator=EvaluationEngine(router=router),
        audit=audit,
        settings=settings,
        use_model_planner=False,
    )
    response = await jarvis.handle("research an AI e-commerce business opportunity")
    assert response.ok
    storage.close()

    # A fresh handle, as a later process would open.
    reopened = open_storage(db_path)
    tasks = TaskEngine(reopened.tasks)
    root = tasks.get(response.task.task_id)
    assert root is not None
    assert root.status is TaskStatus.COMPLETED
    assert len(tasks.children(root)) == len(response.plan.steps)

    recalled = MemoryManager(reopened.memory).recall("e-commerce business opportunity")
    assert recalled, "the run should have left something worth remembering"
    reopened.close()


def test_a_database_from_an_older_build_is_migrated_forward(tmp_path):
    """An existing database gains the new tables rather than being abandoned.

    The version stamp is the whole reason to care: without the migration step
    a v1 file kept its old stamp forever and every open logged a mismatch
    warning, which is a fine way to train people to ignore warnings.
    """
    path = tmp_path / "old.db"
    database = Database(path)
    database.execute("UPDATE schema_version SET version = 1")
    database.execute("DROP TABLE twin_traits")
    database.execute("DROP TABLE goals")
    database.close()

    reopened = Database(path)
    assert reopened.version == SCHEMA_VERSION
    assert "twin_traits" in reopened.counts()
    assert "goals" in reopened.counts()
    reopened.close()


def test_a_half_applied_migration_can_be_finished_on_the_next_open(tmp_path):
    """A migration that dies partway must not wedge the database.

    Adding a column is not transactional across a crash: the first ALTER can
    land and the second not, leaving the version stamp behind. The next open
    then re-runs the whole step, and if the statements were not idempotent it
    would crash on "duplicate column name" every time from then on.
    """
    path = tmp_path / "halfway.db"
    database = Database(path)
    # The current shape, but stamped as if the v4 step never finished.
    database.execute("UPDATE schema_version SET version = 3")
    database.close()

    reopened = Database(path)
    assert reopened.version == SCHEMA_VERSION
    columns = {row["name"] for row in reopened.query("SELECT name FROM pragma_table_info('goals')")}
    assert {"effort_tasks", "effort_cost"} <= columns
    reopened.close()


def test_a_database_from_a_newer_build_is_refused(tmp_path):
    """Writing today's shapes into tomorrow's file corrupts it. Refuse instead."""
    path = tmp_path / "future.db"
    database = Database(path)
    database.execute(f"UPDATE schema_version SET version = {SCHEMA_VERSION + 5}")
    database.close()

    with pytest.raises(RuntimeError, match="newer than this build"):
        Database(path)


def test_a_genuinely_old_database_upgrades_all_the_way(tmp_path):
    """A v1 file, with only v1 tables and v1 columns, opened by this build.

    The earlier migration test cheated: it stamped a *current* database as v1
    and dropped a table, so `tasks` already had the columns a real v1 file
    would lack. That hid a bug for a whole schema version — `_SCHEMA` ran
    before the migrations, so `CREATE INDEX ... ON tasks(goal_id)` hit a table
    that did not have the column yet and every open failed.

    This builds the v1 shape by hand, which is the only way to be sure the
    upgrade path works rather than the test-fixture path.
    """
    import sqlite3

    path = tmp_path / "v1.db"
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            parent_task_id TEXT,
            goal TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 5,
            status TEXT NOT NULL,
            assigned_agent TEXT,
            dependencies TEXT NOT NULL DEFAULT '[]',
            inputs TEXT NOT NULL DEFAULT '{}',
            outputs TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0,
            risk TEXT NOT NULL DEFAULT 'low',
            error TEXT,
            trace_id TEXT
        );
        INSERT INTO tasks (task_id, goal, status, created_at)
        VALUES ('tsk_old', 'something from before', 'completed', '2026-01-01T00:00:00+00:00');
        """
    )
    raw.commit()
    raw.close()

    database = Database(path)
    assert database.version == SCHEMA_VERSION

    task_columns = {
        row["name"] for row in database.query("SELECT name FROM pragma_table_info('tasks')")
    }
    assert "goal_id" in task_columns

    goal_columns = {
        row["name"] for row in database.query("SELECT name FROM pragma_table_info('goals')")
    }
    assert {"effort_tasks", "effort_cost"} <= goal_columns

    # The row that was already there is still there.
    assert database.counts()["tasks"] == 1
    database.close()


def test_two_processes_share_one_rate_limit_window(tmp_path):
    """In memory, a limit of thirty a minute is thirty *per process*.

    A night shift and a terminal session each enforce it separately and the
    tool on the other end sees sixty. That is not a limit, it is a
    limit-shaped number.

    This runs a real second interpreter against the same database, because two
    limiters in one process would share a lock and prove nothing.
    """
    path = tmp_path / "limits.db"
    first = open_storage(path)

    from jarvis_core.mcp.models import RateLimit
    from jarvis_core.tools.ratelimit import RateLimiter

    limiter = RateLimiter(store=first.rate_limits)
    limit = RateLimit(per_minute=3, per_agent_per_minute=3, per_task=10)
    assert limiter.shared

    # Two calls from this process.
    for _ in range(2):
        assert limiter.acquire(
            tool_id="web.search", agent_id="a", task_id=None, limit=limit
        ).allowed
    first.database.close()

    # A third from another process, which should be the last one allowed.
    script = """
import sys
sys.path.insert(0, "src")
from jarvis_core.mcp.models import RateLimit
from jarvis_core.storage import open_storage
from jarvis_core.tools.ratelimit import RateLimiter

storage = open_storage(sys.argv[1])
limiter = RateLimiter(store=storage.rate_limits)
limit = RateLimit(per_minute=3, per_agent_per_minute=3, per_task=10)
third = limiter.acquire(tool_id="web.search", agent_id="a", task_id=None, limit=limit)
fourth = limiter.acquire(tool_id="web.search", agent_id="a", task_id=None, limit=limit)
print(f"{third.allowed} {fourth.allowed} {fourth.scope}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True False tool", result.stdout


def test_a_per_process_limiter_says_it_is_not_shared():
    """A caller that thinks a limit is global and is not is wrong by a factor
    of however many processes are running, and nothing else would tell it."""
    from jarvis_core.tools.ratelimit import RateLimiter

    assert RateLimiter().shared is False


def test_the_rate_limit_window_can_be_pruned(tmp_path):
    """The table is append-only, so without this it grows for as long as the
    install runs."""
    storage = open_storage(tmp_path / "limits.db")
    for index in range(5):
        storage.rate_limits.record(tool_id="t", agent_id=f"a{index}", task_id=None)
    assert storage.database.counts()["tool_calls"] == 5

    assert storage.rate_limits.prune(older_than_s=0) == 5
    assert storage.database.counts()["tool_calls"] == 0
    storage.close()


def test_the_credential_fixture_is_actually_key_shaped():
    """The guard on the guard.

    A CI credential scan started flagging these fixtures, and the first fix
    rewrote them as `fake-ant-key-...` — a string the scanner ignores and the
    redactor *also* ignores. The tests stayed green and stopped testing the
    pattern: the audit-table test then passed only because the metadata key was
    named `api_key`, so the redactor's whole pattern-matching branch lost its
    coverage.

    This asserts the fixture still looks like a key to the code that has to
    catch one. If it fails, a fixture was softened and the tests above are no
    longer evidence of anything.
    """
    from jarvis_core.security.secrets import looks_like_credential

    assert looks_like_credential(fake_anthropic_key("SHOULDNOTAPPEAR0123456789"))
    # And the shape the first fix used, which is the thing to never go back to.
    assert not looks_like_credential("fake-ant-key-SHOULDNOTAPPEAR0123456789")


def test_redaction_is_not_only_about_the_field_name(storage):
    """A secret in a value nobody named `api_key` must still be redacted."""
    from jarvis_core.logging import AuditLog

    log = AuditLog(sink=storage.audit)
    log.emit("model.call", reason=f"loaded {fake_anthropic_key('SHOULDNOTAPPEAR9876')} from env")
    stored = json.dumps(storage.audit.tail(1))
    assert "SHOULDNOTAPPEAR9876" not in stored
