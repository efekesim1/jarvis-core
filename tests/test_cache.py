"""The result cache, duplicate-work detection and the saving they produce."""

from __future__ import annotations

import asyncio

import pytest

from jarvis_core.observability.cache import ResultCache, work_key

# -- keys -------------------------------------------------------------------


def test_the_same_work_produces_the_same_key():
    assert work_key("web_researcher", "what is MCP") == work_key("web_researcher", "what is MCP")


def test_whitespace_and_case_do_not_make_two_identical_questions_different():
    assert work_key("agent", "  What  IS   MCP ") == work_key("agent", "what is mcp")


def test_key_order_in_inputs_does_not_change_the_key():
    assert work_key("agent", {"b": 2, "a": 1}) == work_key("agent", {"a": 1, "b": 2})


def test_different_work_produces_different_keys():
    assert work_key("agent", "question one") != work_key("agent", "question two")
    assert work_key("agent_a", "same") != work_key("agent_b", "same")


# -- basic caching ----------------------------------------------------------


async def test_a_repeated_run_is_answered_without_working_again():
    calls = {"n": 0}
    cache: ResultCache[str] = ResultCache()

    async def produce() -> str:
        calls["n"] += 1
        return "answer"

    first, cached_first = await cache.run("k", produce)
    second, cached_second = await cache.run("k", produce)

    assert first == second == "answer"
    assert not cached_first
    assert cached_second
    assert calls["n"] == 1


async def test_the_cache_reports_what_it_saved():
    cache: ResultCache[str] = ResultCache()

    async def produce() -> str:
        return "answer"

    await cache.run("k", produce, cost_of=lambda _: 0.25)
    await cache.run("k", produce, cost_of=lambda _: 0.25)

    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.saved_usd == pytest.approx(0.25)
    assert cache.stats.hit_rate == pytest.approx(0.5)


async def test_two_identical_runs_starting_together_share_one_call():
    started = {"n": 0}
    cache: ResultCache[str] = ResultCache()

    async def produce() -> str:
        started["n"] += 1
        await asyncio.sleep(0.05)
        return "answer"

    results = await asyncio.gather(
        cache.run("k", produce), cache.run("k", produce), cache.run("k", produce)
    )

    assert [r[0] for r in results] == ["answer"] * 3
    assert started["n"] == 1
    assert cache.stats.coalesced == 2


async def test_an_entry_expires_after_its_time_to_live():
    calls = {"n": 0}
    cache: ResultCache[str] = ResultCache(ttl_s=0.01)

    async def produce() -> str:
        calls["n"] += 1
        return "answer"

    await cache.run("k", produce)
    await asyncio.sleep(0.05)
    _, cached = await cache.run("k", produce)

    assert not cached
    assert calls["n"] == 2


async def test_the_cache_stays_within_its_size_limit():
    cache: ResultCache[str] = ResultCache(max_entries=3)

    async def produce() -> str:
        return "answer"

    for index in range(10):
        await cache.run(f"k{index}", produce)

    assert len(cache) == 3
    assert cache.stats.evictions >= 7


# -- failure handling -------------------------------------------------------


async def test_a_failure_is_never_stored_as_the_answer():
    calls = {"n": 0}
    cache: ResultCache[str] = ResultCache()

    async def failing() -> str:
        calls["n"] += 1
        raise RuntimeError("the provider fell over")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cache.run("k", failing)

    # Both attempts really ran; the first failure did not become a cached answer.
    assert calls["n"] == 2
    assert len(cache) == 0


async def test_a_result_the_caller_rejects_is_not_stored():
    calls = {"n": 0}
    cache: ResultCache[str] = ResultCache()

    async def produce() -> str:
        calls["n"] += 1
        return ""

    await cache.run("k", produce, cacheable=lambda v: bool(v))
    await cache.run("k", produce, cacheable=lambda v: bool(v))

    assert calls["n"] == 2


async def test_a_waiter_sees_the_failure_rather_than_hanging():
    cache: ResultCache[str] = ResultCache()

    async def failing() -> str:
        await asyncio.sleep(0.02)
        raise RuntimeError("boom")

    results = await asyncio.gather(
        cache.run("k", failing), cache.run("k", failing), return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_invalidating_a_key_forces_the_work_to_run_again():
    calls = {"n": 0}
    cache: ResultCache[str] = ResultCache()

    async def produce() -> str:
        calls["n"] += 1
        return "answer"

    await cache.run("k", produce)
    assert cache.invalidate("k")
    await cache.run("k", produce)
    assert calls["n"] == 2
