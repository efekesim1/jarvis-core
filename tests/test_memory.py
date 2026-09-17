"""Memory: what is kept, what is refused, and what comes back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jarvis_core.memory.base import MemoryScope
from jarvis_core.memory.manager import MemoryManager, RetrievalBudget


def _manager() -> MemoryManager:
    return MemoryManager()


# -- the write policy -------------------------------------------------------


def test_an_important_user_fact_is_kept():
    memory = _manager()
    item = memory.remember(
        "the operator prefers PostgreSQL over MongoDB for this project",
        scope=MemoryScope.USER,
        source="conversation",
        importance=0.8,
    )
    assert item is not None
    assert memory.writes_accepted == 1


def test_a_trivial_remark_is_not_kept_as_a_user_fact():
    memory = _manager()
    item = memory.remember(
        "the weather was mentioned", scope=MemoryScope.USER, source="chat", importance=0.2
    )
    assert item is None
    assert memory.writes_rejected == 1


def test_working_memory_accepts_anything_because_it_dies_quickly():
    memory = _manager()
    assert memory.remember(
        "a scratch note", scope=MemoryScope.WORKING, source="task", importance=0.0
    )


def test_empty_content_is_never_stored():
    memory = _manager()
    assert memory.remember("   ", scope=MemoryScope.USER, source="x", importance=1.0) is None


def test_each_layer_has_its_own_bar():
    memory = _manager()
    # The same importance clears the episodic bar and misses the user bar.
    assert memory.remember(
        "something happened", scope=MemoryScope.EPISODIC, source="s", importance=0.45
    )
    assert (
        memory.remember("something happened", scope=MemoryScope.USER, source="s", importance=0.45)
        is None
    )


# -- retrieval --------------------------------------------------------------


def test_a_relevant_memory_comes_back():
    memory = _manager()
    memory.remember(
        "The pgvector extension stores embeddings inside PostgreSQL",
        scope=MemoryScope.KNOWLEDGE,
        source="research",
        importance=0.8,
        confidence=0.9,
    )
    found = memory.recall("how does pgvector work with postgres")
    assert found
    assert "pgvector" in found[0].item.content


def test_an_unrelated_question_returns_nothing():
    memory = _manager()
    memory.remember(
        "The pgvector extension stores embeddings",
        scope=MemoryScope.KNOWLEDGE,
        source="research",
        importance=0.8,
    )
    assert memory.recall("what is the price of coffee in Brazil") == []


def test_retrieval_is_capped_by_the_budget():
    memory = _manager()
    for index in range(20):
        memory.remember(
            f"pgvector fact number {index} about postgres embeddings",
            scope=MemoryScope.KNOWLEDGE,
            source="research",
            importance=0.8,
        )
    found = memory.recall("pgvector postgres embeddings", budget=RetrievalBudget(max_items=3))
    assert len(found) == 3


def test_retrieval_respects_the_character_budget():
    memory = _manager()
    for index in range(10):
        memory.remember(
            "pgvector postgres embeddings " + "detail " * 100 + str(index),
            scope=MemoryScope.KNOWLEDGE,
            source="research",
            importance=0.8,
        )
    found = memory.recall(
        "pgvector postgres embeddings", budget=RetrievalBudget(max_items=10, max_chars=1500)
    )
    assert sum(len(r.item.content) for r in found) <= 1500


def test_a_more_important_memory_outranks_a_less_important_one():
    memory = _manager()
    memory.remember(
        "postgres pgvector minor aside",
        scope=MemoryScope.KNOWLEDGE,
        source="s",
        importance=0.5,
        confidence=0.5,
    )
    memory.remember(
        "postgres pgvector central decision",
        scope=MemoryScope.KNOWLEDGE,
        source="s",
        importance=1.0,
        confidence=1.0,
    )
    found = memory.recall("postgres pgvector")
    assert "central" in found[0].item.content


def test_an_older_memory_ranks_below_an_equal_newer_one():
    memory = MemoryManager(half_life_days=1.0)
    old = memory.remember(
        "postgres pgvector established fact",
        scope=MemoryScope.KNOWLEDGE,
        source="s",
        importance=0.8,
        confidence=0.8,
    )
    assert old is not None
    old.timestamp = datetime.now(UTC) - timedelta(days=30)
    memory.remember(
        "postgres pgvector recent fact",
        scope=MemoryScope.KNOWLEDGE,
        source="s",
        importance=0.8,
        confidence=0.8,
    )
    found = memory.recall("postgres pgvector fact")
    assert "recent" in found[0].item.content


def test_retrieval_can_be_limited_to_one_layer():
    memory = _manager()
    memory.remember(
        "postgres decision made", scope=MemoryScope.DECISION, source="s", importance=0.9
    )
    memory.remember(
        "postgres episode happened", scope=MemoryScope.EPISODIC, source="s", importance=0.9
    )
    found = memory.recall("postgres", scopes=(MemoryScope.DECISION,))
    assert len(found) == 1
    assert found[0].item.scope is MemoryScope.DECISION


def test_a_query_of_only_stopwords_returns_nothing():
    memory = _manager()
    memory.remember("postgres pgvector", scope=MemoryScope.KNOWLEDGE, source="s", importance=0.9)
    assert memory.recall("the and or of") == []


# -- forgetting -------------------------------------------------------------


def test_an_expired_memory_is_not_returned():
    memory = _manager()
    item = memory.remember(
        "postgres pgvector short-lived note",
        scope=MemoryScope.KNOWLEDGE,
        source="s",
        importance=0.9,
        expires_in=timedelta(seconds=1),
    )
    assert item is not None
    item.expiration = datetime.now(UTC) - timedelta(seconds=1)
    assert memory.recall("postgres pgvector") == []


def test_working_memory_is_dropped_with_its_task():
    memory = _manager()
    memory.remember(
        "scratch note for the task",
        scope=MemoryScope.WORKING,
        source="task",
        importance=0.5,
        metadata={"task_id": "tsk_1"},
    )
    memory.remember(
        "note for another task",
        scope=MemoryScope.WORKING,
        source="task",
        importance=0.5,
        metadata={"task_id": "tsk_2"},
    )
    assert memory.forget_task("tsk_1") == 1
    assert memory.stats()["items"] == 1


def test_pruning_removes_what_has_expired():
    memory = _manager()
    item = memory.remember("postgres note", scope=MemoryScope.KNOWLEDGE, source="s", importance=0.9)
    assert item is not None
    item.expiration = datetime.now(UTC) - timedelta(days=1)
    assert memory.prune() == 1
    assert memory.stats()["items"] == 0


def test_a_permanent_layer_does_not_expire_on_its_own():
    memory = _manager()
    item = memory.remember(
        "the operator owns this project", scope=MemoryScope.USER, source="s", importance=0.9
    )
    assert item is not None
    assert item.expiration is None


# -- rendering --------------------------------------------------------------


def test_rendered_memory_is_labelled_as_context_not_instruction():
    memory = _manager()
    memory.remember(
        "postgres was chosen for this project",
        scope=MemoryScope.DECISION,
        source="s",
        importance=0.9,
    )
    rendered = memory.render(memory.recall("postgres"))
    assert "not as a fresh instruction" in rendered
    assert "postgres" in rendered


def test_rendering_nothing_produces_nothing():
    assert _manager().render([]) == ""


def test_stats_report_what_was_kept_and_refused():
    memory = _manager()
    memory.remember("kept postgres fact", scope=MemoryScope.KNOWLEDGE, source="s", importance=0.9)
    memory.remember("dropped", scope=MemoryScope.USER, source="s", importance=0.1)
    stats = memory.stats()
    assert stats["writes_accepted"] == 1
    assert stats["writes_rejected"] == 1
    assert stats["by_scope"]["knowledge"] == 1
