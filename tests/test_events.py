"""
Tests for the observability event system.

Verified behaviors:
  - Subscribers fire on every lookup decision (hit AND miss).
  - Event fields match the cache decision (hit_type, model, latency, etc).
  - Subscriber errors NEVER crash the cache.
  - last_event() returns the most recent.
  - Subscribers can be removed.
  - Same subscriber registered twice fires once.
"""
import asyncio
import time

import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield
    events.clear_subscribers()


def test_event_fires_on_miss_and_hit():
    seen = []

    @events.on_event
    def collect(ev):
        seen.append(ev)

    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(prompt):
        return "ok"

    ask("hello")
    ask("hello")

    assert len(seen) == 2
    assert seen[0].hit is False
    assert seen[0].event_type == "lookup_miss"
    assert seen[0].model == "gpt-4o-mini"
    assert seen[1].hit is True
    assert seen[1].hit_type == "exact"


def test_event_contains_prompt_preview():
    # This test specifically verifies the opt-in "full" preview mode; the safe
    # default is "hashed" (no content), restored by the autouse fixture.
    from tokeymeter import decorator as _dec
    _dec.set_event_preview_policy("full")
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("This is the user's question about something specific.")

    assert seen[0].prompt_preview is not None
    assert "user's question" in seen[0].prompt_preview


def test_subscriber_error_does_not_crash_wrapped_call():
    def broken_subscriber(event):
        raise RuntimeError("subscriber on fire")

    events.on_event(broken_subscriber)

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    # Must not raise despite the broken subscriber
    assert ask("hello") == "ok"
    assert ask("hello") == "ok"


def test_unsubscribe_stops_events():
    seen = []

    def handler(ev):
        seen.append(ev)

    events.subscribe(handler)

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("a")
    assert len(seen) == 1

    events.unsubscribe(handler)
    ask("b")
    assert len(seen) == 1  # no new event


def test_idempotent_subscribe():
    seen = []

    def handler(ev):
        seen.append(ev)

    events.subscribe(handler)
    events.subscribe(handler)
    events.subscribe(handler)

    assert events.subscriber_count() == 1

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("x")
    assert len(seen) == 1  # not 3


def test_last_event_returns_most_recent():
    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(prompt):
        return "ok"

    ask("a")
    ev1 = events.last_event()
    assert ev1.model == "gpt-4o-mini"
    assert ev1.hit is False

    ask("a")
    ev2 = events.last_event()
    assert ev2.hit is True
    assert ev2.hit_type == "exact"


def test_event_latency_recorded():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache()
    def slow(prompt):
        time.sleep(0.02)
        return "ok"

    slow("hello")
    slow("hello")  # cached

    assert seen[0].latency_ms >= 15  # was a real ~20ms call
    assert seen[1].latency_ms < 10   # cache hit


@pytest.mark.asyncio
async def test_event_fires_for_async_path():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache()
    async def ask(prompt):
        return "ok"

    await ask("hello")
    await ask("hello")

    assert len(seen) == 2
    assert seen[1].hit is True


@pytest.mark.asyncio
async def test_event_fires_for_single_flight_followers():
    """All 10 concurrent calls should produce 10 events (1 miss + 9 single_flight hits)."""
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache()
    async def ask(prompt):
        await asyncio.sleep(0.02)
        return "ok"

    await asyncio.gather(*[ask("same") for _ in range(10)])

    assert len(seen) == 10
    misses = [e for e in seen if not e.hit]
    sf_hits = [e for e in seen if e.hit_type == "single_flight"]
    assert len(misses) == 1
    assert len(sf_hits) == 9


@pytest.mark.asyncio
async def test_event_fires_for_streaming():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache_stream()
    async def stream(prompt):
        for c in ["a", "b", "c"]:
            yield c

    _ = [c async for c in stream("x")]  # miss
    _ = [c async for c in stream("x")]  # hit

    assert len(seen) == 2
    assert seen[0].hit is False
    assert seen[1].hit is True
