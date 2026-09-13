"""
Tests for TTL (time-to-live) on cached entries.

Verified behaviors:
  - Entries hit within their TTL window.
  - Entries miss after TTL expires.
  - ttl=None (default) means never expires.
  - TTL works for sync, async, and streaming caches.
  - TTL works for both exact and semantic cache layers.
  - Legacy unwrapped values (no envelope) are treated as never-expiring.
"""
import asyncio
import time

import numpy as np
import pytest

import tokeymeter
from tokeymeter.envelope import is_expired, unwrap, wrap
from tokeymeter.semantic import SemanticCache
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.reset_savings()
    yield


# ---------- envelope unit tests ----------

def test_envelope_wrap_no_ttl_never_expires():
    env = wrap("hello", ttl=None)
    assert unwrap(env) == "hello"
    assert not is_expired(env)


def test_envelope_with_ttl_unwraps_within_window():
    env = wrap("hello", ttl=60)
    assert unwrap(env) == "hello"
    assert not is_expired(env)


def test_envelope_expired_returns_none():
    # Negative or zero ttl should not expire (treated as no-ttl)
    env = wrap("hello", ttl=0)
    assert unwrap(env) == "hello"

    # Manually construct expired envelope
    expired = {"__tokeymeter_v1__": ["hello", time.time() - 1]}
    assert unwrap(expired) is None
    assert is_expired(expired)


def test_envelope_legacy_passthrough():
    """Pre-v0.4 entries stored as raw values must still work."""
    assert unwrap("legacy-string") == "legacy-string"
    assert unwrap({"some": "dict"}) == {"some": "dict"}
    assert unwrap([1, 2, 3]) == [1, 2, 3]
    assert unwrap(None) is None


def test_envelope_malformed_returns_none():
    """Defensive: a stored envelope with wrong shape returns None, not raises."""
    bad = {"__tokeymeter_v1__": "not a list"}
    assert unwrap(bad) is None


# ---------- envelope metadata (self-host token recovery) ----------

def test_envelope_two_element_has_no_meta():
    from tokeymeter.envelope import meta
    e = wrap("value")                       # legacy 2-element form
    assert unwrap(e) == "value"
    assert meta(e) is None


def test_envelope_three_element_round_trips_meta():
    from tokeymeter.envelope import meta
    e = wrap("value", ttl=None, meta={"in": 340, "out": 128})
    assert unwrap(e) == "value"             # value unaffected by meta
    assert meta(e) == {"in": 340, "out": 128}


def test_envelope_meta_none_when_no_meta_supplied():
    from tokeymeter.envelope import meta
    assert meta(wrap("v", ttl=60)) is None


def test_envelope_meta_none_for_raw_and_malformed():
    from tokeymeter.envelope import meta
    assert meta("raw-legacy-value") is None
    assert meta({"__tokeymeter_v1__": "not a list"}) is None
    assert meta(None) is None


def test_envelope_meta_none_after_expiry():
    from tokeymeter.envelope import meta
    e = wrap("v", ttl=0.01, meta={"in": 5, "out": 5})
    time.sleep(0.02)
    assert unwrap(e) is None                # expired value
    assert meta(e) is None                  # and meta gated behind expiry too


def test_legacy_disk_envelope_still_unwraps():
    # A 2-element envelope as written by a prior version / persisted to disk.
    old = {"__tokeymeter_v1__": ["disk-value", None]}
    assert unwrap(old) == "disk-value"
    from tokeymeter.envelope import meta
    assert meta(old) is None


# ---------- decorator-level TTL tests ----------

def test_sync_ttl_hits_within_window():
    calls = [0]

    @tokeymeter.cache(ttl=60)
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = ask("hello")
    r2 = ask("hello")
    assert r1 == r2
    assert calls[0] == 1


def test_sync_ttl_misses_after_expiry():
    calls = [0]

    @tokeymeter.cache(ttl=0.05)  # 50ms TTL
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = ask("hello")
    time.sleep(0.08)
    r2 = ask("hello")  # expired → miss → real call
    assert r1 != r2
    assert calls[0] == 2


def test_sync_ttl_none_means_forever():
    calls = [0]

    @tokeymeter.cache(ttl=None)
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    ask("hello")
    time.sleep(0.05)
    ask("hello")
    assert calls[0] == 1  # still cached


@pytest.mark.asyncio
async def test_async_ttl_misses_after_expiry():
    calls = [0]

    @tokeymeter.cache(ttl=0.05)
    async def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    await ask("hello")
    await asyncio.sleep(0.08)
    await ask("hello")
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_stream_ttl_misses_after_expiry():
    calls = [0]

    @tokeymeter.cache_stream(ttl=0.05)
    async def stream(prompt):
        calls[0] += 1
        for w in ["a ", "b ", "c "]:
            yield w

    _ = [c async for c in stream("hello")]
    await asyncio.sleep(0.08)
    _ = [c async for c in stream("hello")]
    assert calls[0] == 2


def test_semantic_ttl_works(tmp_path):
    """Semantic-cached responses also respect TTL."""
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

    @tokeymeter.cache(semantic=True, semantic_cache=sem, prompt_arg="prompt", ttl=0.05)
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    ask(prompt="abcdef")          # miss
    ask(prompt="fedcba")          # semantic hit (cached)
    assert calls[0] == 1

    time.sleep(0.08)
    # Re-clearing the L1 exact cache so we test the semantic layer's TTL
    tokeymeter.set_default_store(MemoryStore())
    ask(prompt="fedcba")          # semantic entry expired → miss
    assert calls[0] == 2
