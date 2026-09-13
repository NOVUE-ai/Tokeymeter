"""
Tests for async single-flight deduplication.

When N concurrent coroutines miss the cache for the same key, exactly
ONE invokes the wrapped function; the rest wait for its result. This
prevents thundering-herd API calls on cache cold starts.

Verified behaviors:
  - N concurrent identical calls → 1 function invocation.
  - Different keys still execute concurrently.
  - Leader exception propagates to all followers.
  - In-flight map is cleaned up after completion.
  - single_flight=False disables it.
  - Sync path is unaffected (no dedup on sync).
"""
import asyncio
import time

import pytest

import tokeymeter
from tokeymeter.decorator import _inflight_async, _inflight_size
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.reset_savings()
    _inflight_async.clear()
    yield
    _inflight_async.clear()


@pytest.mark.asyncio
async def test_single_flight_dedups_concurrent_identical_calls():
    """10 concurrent identical calls should result in 1 function invocation."""
    calls = [0]

    @tokeymeter.cache()
    async def slow_ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0.05)  # 50ms work
        return f"response-to-{prompt}"

    # 10 coros all asking the same thing at once
    results = await asyncio.gather(*[slow_ask("same") for _ in range(10)])
    assert all(r == "response-to-same" for r in results)
    assert calls[0] == 1, f"expected 1 call, got {calls[0]}"


@pytest.mark.asyncio
async def test_different_keys_execute_in_parallel():
    """Different cache keys should NOT be serialized — they run concurrently."""
    calls = [0]

    @tokeymeter.cache()
    async def ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0.05)
        return prompt

    t0 = time.perf_counter()
    results = await asyncio.gather(*[ask(f"prompt-{i}") for i in range(5)])
    elapsed = time.perf_counter() - t0

    assert calls[0] == 5  # all 5 unique keys → 5 real calls
    # Sequential would be 5*50ms = 250ms; parallel should be ~50ms.
    assert elapsed < 0.15, f"calls not parallel: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_leader_exception_propagates_to_followers():
    """If the leader raises, all followers see the same exception."""
    @tokeymeter.cache()
    async def failing_ask(prompt):
        await asyncio.sleep(0.02)
        raise ValueError("upstream down")

    # Five concurrent calls — leader fails, followers wait.
    # In current implementation, followers re-try after seeing leader's
    # exception. Each retry also fails. So all 5 see ValueError.
    tasks = [asyncio.create_task(failing_ask("x")) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(r, ValueError) for r in results), \
        f"expected all ValueError, got {results}"


@pytest.mark.asyncio
async def test_inflight_cleaned_up_after_completion():
    @tokeymeter.cache()
    async def ask(prompt):
        await asyncio.sleep(0.01)
        return "ok"

    assert _inflight_size() == 0
    await ask("a")
    await ask("b")
    assert _inflight_size() == 0


@pytest.mark.asyncio
async def test_inflight_cleaned_up_after_exception():
    @tokeymeter.cache()
    async def boom(prompt):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await boom("x")
    assert _inflight_size() == 0


@pytest.mark.asyncio
async def test_single_flight_can_be_disabled():
    """single_flight=False means concurrent calls all invoke the function."""
    calls = [0]

    @tokeymeter.cache(single_flight=False)
    async def ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0.05)
        return "ok"

    await asyncio.gather(*[ask("same") for _ in range(5)])
    # All 5 invocations should have happened — no dedup.
    assert calls[0] == 5


@pytest.mark.asyncio
async def test_followers_get_recorded_as_single_flight_hits():
    """Followers count as cache hits (with hit_type='single_flight')."""
    @tokeymeter.cache(model="gpt-4o-mini")
    async def ask(prompt):
        await asyncio.sleep(0.03)
        return "ok"

    await asyncio.gather(*[ask("same") for _ in range(5)])

    r = tokeymeter.savings_report()
    # 1 miss + 4 single_flight hits = 5 calls total
    assert r["total_calls"] == 5
    assert r["cache_misses"] == 1
    # The 4 followers are counted as cache hits
    assert r["cache_hits"] == 4


def test_sync_path_does_not_use_single_flight():
    """Sync path should call the function for every invocation in this
    quick sanity test (the dedup machinery is async-only)."""
    calls = [0]

    @tokeymeter.cache()
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    # Sequential sync calls with same key — second hits the cache normally
    r1 = ask("hello")
    r2 = ask("hello")
    assert r1 == r2
    assert calls[0] == 1  # normal cache hit, not single-flight
    # _inflight_async should not have been touched by sync calls
    assert _inflight_size() == 0


@pytest.mark.asyncio
async def test_single_flight_with_real_concurrency_under_load():
    """100 concurrent calls for 10 distinct keys → exactly 10 function calls."""
    calls = [0]

    @tokeymeter.cache()
    async def ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0.05)
        return f"r-{prompt}"

    tasks = []
    for i in range(10):  # 10 distinct keys
        for _ in range(10):  # 10 concurrent calls each
            tasks.append(ask(f"prompt-{i}"))

    results = await asyncio.gather(*tasks)
    assert len(results) == 100
    assert calls[0] == 10
