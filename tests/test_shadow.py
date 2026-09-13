"""
Tests for shadow mode.

In shadow mode:
  - Cache lookups happen normally and would-be hits ARE recorded.
  - The wrapped function is ALWAYS called and its result is returned.
  - The savings report shows `would_have_saved_usd` for shadow hits.
  - The user sees zero behavior change vs. having no cache.

Use case: a team that wants to measure savings risk-free for 7-14 days
before flipping shadow=False and actually returning cached results.
"""
import asyncio

import numpy as np
import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.semantic import SemanticCache
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield


def test_shadow_always_calls_wrapped_function():
    calls = [0]

    @tokeymeter.cache(shadow=True, model="gpt-4o-mini")
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = ask("hello")
    r2 = ask("hello")
    r3 = ask("hello")

    # All three calls invoke the function (no cached results returned)
    assert calls[0] == 3
    assert r1 != r2 != r3


def test_shadow_records_would_have_been_hits():
    @tokeymeter.cache(shadow=True, model="gpt-4o-mini")
    def ask(prompt):
        return "x" * 200  # ~50 output tokens

    ask("hello")           # miss
    ask("hello")           # would have been an exact hit
    ask("hello")           # would have been an exact hit
    ask("different")       # miss

    r = tokeymeter.savings_report()
    assert r["shadow"]["total_shadow_lookups"] == 2
    assert r["shadow"]["shadow_exact_hits"] == 2
    assert r["shadow"]["would_have_saved_usd"] > 0


def test_shadow_does_not_pollute_live_savings():
    @tokeymeter.cache(shadow=True)
    def ask(prompt):
        return "ok"

    ask("hello")
    ask("hello")  # shadow hit; real call ran

    r = tokeymeter.savings_report()
    # No real cache hits because we always called the function
    assert r["cache_hits"] == 0
    assert r["estimated_saved_usd"] == 0.0


def test_shadow_emits_shadow_event_type():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(shadow=True)
    def ask(prompt):
        return "ok"

    ask("hello")  # miss
    ask("hello")  # would-be hit
    ask("hello")  # would-be hit

    shadow_events = [e for e in seen if e.hit_type and e.hit_type.startswith("shadow_")]
    assert len(shadow_events) == 2
    assert all(e.shadow for e in shadow_events)


@pytest.mark.asyncio
async def test_shadow_disables_single_flight():
    """In shadow mode, concurrent calls should each hit the API
    (single-flight disabled so we measure the real load)."""
    calls = [0]

    @tokeymeter.cache(shadow=True)
    async def ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0.02)
        return "ok"

    await asyncio.gather(*[ask("same") for _ in range(5)])
    assert calls[0] == 5  # all 5 hit the API


@pytest.mark.asyncio
async def test_shadow_async_records_correctly():
    @tokeymeter.cache(shadow=True, model="gpt-4o-mini")
    async def ask(prompt):
        return "ok"

    for _ in range(5):
        await ask("hello")

    r = tokeymeter.savings_report()
    assert r["shadow"]["total_shadow_lookups"] == 4  # first was miss, next 4 = shadow hits
    assert r["shadow"]["shadow_exact_hits"] == 4


def test_shadow_with_semantic_layer(tmp_path):
    """Semantic shadow hits should be recorded as shadow_semantic."""
    def bag(t):
        t = t.lower()
        v = np.zeros(26, dtype=np.float32)
        for ch in t:
            if "a" <= ch <= "z":
                v[ord(ch) - ord("a")] += 1
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    sem = SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.95,
        encoder=bag,
    )

    calls = [0]

    @tokeymeter.cache(shadow=True, semantic=True, semantic_cache=sem, prompt_arg="prompt")
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    ask(prompt="abcdef")     # miss (and would-be store)
    ask(prompt="fedcba")     # would-be semantic hit
    assert calls[0] == 2     # both called the function

    r = tokeymeter.savings_report()
    assert r["shadow"]["shadow_semantic_hits"] == 1


def test_can_flip_off_shadow_to_serve_cached():
    """After shadow=False, the same cache entries should hit normally."""
    calls = [0]

    @tokeymeter.cache(shadow=True, namespace="shadow-workload")
    def ask_shadow(prompt):
        calls[0] += 1
        return "stable"

    ask_shadow("hello")  # writes to cache (shadow still stores)
    ask_shadow("hello")  # shadow hit, call invoked
    assert calls[0] == 2

    # Now make a new decorator with shadow=False using the same cache
    @tokeymeter.cache(shadow=False, namespace="shadow-workload")
    def ask_live(prompt):
        calls[0] += 1
        return "stable"

    ask_live("hello")  # should HIT the cache populated by shadow runs
    assert calls[0] == 2  # no new invocation
