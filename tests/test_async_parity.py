"""Async-path parity (Phase-1 hardening, item 1).

v0.14's identity binding and budget-enforced keys were built and proven on
the sync block path. Production traffic is substantially async and streamed,
so every guarantee is re-proven here through `await` and `async for`:

  1.1  async miss stamps principal into CallRecord AND CacheEvent
  1.2  KeyBudgetExceeded raises BEFORE the awaited compute (async miss)
  1.3  async accrual: only real misses accrue — hits and shadow never do
  1.4  asyncio.gather with two principals → zero cross-bleed
  1.5  streaming: principal + key_name stamped; hard cap fires BEFORE the
       first chunk (for an async generator that means: at first iteration,
       with zero chunks produced and zero compute started)
  1.6  set_reported_usage inside async and streaming wrappers lands
       token_source == "reported"
  1.7  contract pin: enabled=False bypasses EVERYTHING — no enforcement,
       no accrual, no record (a cap that blocks without a ledger entry
       would be incoherent; see tokeymeter/keys.py docstring)
"""
import asyncio

import pytest

import tokeymeter
from tokeymeter import events as ev
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore
from tokeymeter.usage import set_reported_usage


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    K.clear_keys()
    tokeymeter.set_principal(None)
    yield
    tokeymeter.set_principal(None)
    K.clear_keys()
    tokeymeter.reset_savings()
    # restore the library default (sync-to-file); leaving in-memory mode on
    # leaks into later modules that depend on the default ledger file
    tokeymeter.set_in_memory_savings(False)


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


# ── 1.1 async miss stamps principal into record + event ─────────────────
async def test_async_miss_stamps_principal_record_and_event():
    seen = []
    cb = lambda e: seen.append((e.principal, e.hit))  # noqa: E731
    ev.subscribe(cb)
    try:
        @tokeymeter.cache(model="m")
        async def ask(p):
            await asyncio.sleep(0)
            return "x" * 50
        with tokeymeter.principal("person:priya"):
            await ask("async hello " * 5)
    finally:
        ev.unsubscribe(cb)
    rec = _records()[-1]
    assert rec["principal"] == "person:priya"
    assert seen and seen[-1] == ("person:priya", False)


async def test_async_miss_stamps_key_name():
    tokeymeter.register_key("ak", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache(model="m")
    async def ask(p):
        return "y"
    with tokeymeter.key("ak"):
        await ask("q " * 5)
    assert _records()[-1]["key_name"] == "ak"


# ── 1.2 hard cap raises BEFORE the awaited compute ───────────────────────
async def test_async_hard_cap_refuses_before_compute():
    tokeymeter.register_key("acap", "val-abcdefgh", monthly_cap_usd=0.001)
    computes = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    async def ask(p):
        computes["n"] += 1
        return "x" * 4000
    with pytest.raises(tokeymeter.KeyBudgetExceeded):
        with tokeymeter.key("acap"):
            for i in range(50):
                await ask(f"q{i} " * 200)
    # breach detected after first real spend; second compute never ran
    assert computes["n"] == 1
    assert tokeymeter.key_status("acap")["spent_usd"] > 0


# ── 1.3 accrual rules on the async path ──────────────────────────────────
async def test_async_accrual_ignores_hits_and_shadow():
    tokeymeter.register_key("ak2", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache(model="gpt-4o-mini")
    async def ask(p):
        return "y" * 200

    @tokeymeter.cache(model="gpt-4o-mini", shadow=True)
    async def shadow_ask(p):
        return "z" * 200

    with tokeymeter.key("ak2"):
        await ask("same prompt " * 20)     # miss → accrues (1)
        await ask("same prompt " * 20)     # exact hit → no accrual
        await shadow_ask("other " * 20)    # shadow → no accrual
    st = tokeymeter.key_status("ak2")
    assert st["calls"] == 1, st


# ── 1.4 gather: two principals, zero cross-bleed ─────────────────────────
async def test_gather_two_principals_no_cross_bleed():
    @tokeymeter.cache(model="m")
    async def ask(p):
        await asyncio.sleep(0.01)          # force interleaving
        return "r:" + p[:8]

    async def run_as(pid, marker):
        with tokeymeter.principal(pid):
            await ask(f"{marker} prompt " * 4)

    await asyncio.gather(run_as("agent:a", "alpha"),
                         run_as("agent:b", "beta"))
    recs = _records()
    by_marker = {}
    for r in recs:
        # prompt_preview isn't in the record; use principal↔count integrity
        by_marker.setdefault(r["principal"], 0)
        by_marker[r["principal"]] += 1
    assert by_marker == {"agent:a": 1, "agent:b": 1}, by_marker
    # and event-level check with markers, deterministically:
    seen = []
    cb = lambda e: seen.append((e.principal, e.hit))  # noqa: E731
    ev.subscribe(cb)
    try:
        await asyncio.gather(run_as("agent:a", "alpha"),   # exact hits now
                             run_as("agent:b", "beta"))
    finally:
        ev.unsubscribe(cb)
    assert sorted(seen) == [("agent:a", True), ("agent:b", True)]


# ── 1.5 streaming: stamps + cap-before-first-chunk ───────────────────────
async def test_stream_records_carry_principal_and_key():
    tokeymeter.register_key("sk1", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache_stream(model="m")
    async def gen(p):
        for i in range(3):
            yield f"chunk{i} "
    chunks = []
    with tokeymeter.principal("agent:streamer"), tokeymeter.key("sk1"):
        async for c in gen("stream me " * 5):
            chunks.append(c)
    assert chunks == ["chunk0 ", "chunk1 ", "chunk2 "]
    rec = _records()[-1]
    assert rec["principal"] == "agent:streamer"
    assert rec["key_name"] == "sk1"
    assert tokeymeter.key_status("sk1")["calls"] == 1   # streamed miss accrued


async def test_stream_hard_cap_fires_before_first_chunk():
    tokeymeter.register_key("scap", "val-abcdefgh", monthly_cap_usd=0.0005)
    produced = {"n": 0}

    @tokeymeter.cache_stream(model="gpt-4o")
    async def gen(p):
        for i in range(5):
            produced["n"] += 1
            yield "tok " * 200
    # exhaust the cap with one streamed miss
    with tokeymeter.key("scap"):
        async for _ in gen("first " * 50):
            pass
    assert tokeymeter.key_status("scap")["spent_usd"] > 0.0005
    # second stream must refuse at FIRST iteration, zero chunks produced
    before = produced["n"]
    with pytest.raises(tokeymeter.KeyBudgetExceeded):
        with tokeymeter.key("scap"):
            async for _ in gen("second " * 50):
                pytest.fail("no chunk may be yielded past an exhausted cap")
    assert produced["n"] == before, "generator body ran despite exhausted cap"


async def test_stream_hit_serves_without_accrual():
    tokeymeter.register_key("sk3", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache_stream(model="m")
    async def gen(p):
        yield "a "
        yield "b "
    with tokeymeter.key("sk3"):
        async for _ in gen("replay " * 5):
            pass
        out = []
        async for c in gen("replay " * 5):   # exact hit replays chunks
            out.append(c)
    assert out == ["a ", "b "]
    assert tokeymeter.key_status("sk3")["calls"] == 1


# ── 1.6 reported-usage plumbing through async + stream ───────────────────
async def test_async_reported_usage_lands():
    @tokeymeter.cache(model="gpt-4o")
    async def ask(p):
        set_reported_usage(321, 42)
        return "ok"
    await ask("anything async")
    rec = _records()[-1]
    assert (rec["input_tokens"], rec["output_tokens"]) == (321, 42)
    assert rec["token_source"] == "reported"


async def test_stream_reported_usage_lands():
    @tokeymeter.cache_stream(model="gpt-4o")
    async def gen(p):
        yield "x "
        set_reported_usage(1000, 250)      # provider's final-chunk usage
        yield "y "
    async for _ in gen("stream usage " * 4):
        pass
    rec = _records()[-1]
    assert rec["token_source"] == "reported"
    assert (rec["input_tokens"], rec["output_tokens"]) == (1000, 250)


# ── 1.7 contract pin: enabled=False bypasses everything ─────────────────
async def test_enabled_false_bypasses_enforcement_accrual_and_records():
    tokeymeter.register_key("off", "val-abcdefgh", monthly_cap_usd=0.000001)
    # exhaust it so any enforcement would raise
    @tokeymeter.cache(model="gpt-4o")
    async def metered(p):
        return "x" * 2000
    with pytest.raises(tokeymeter.KeyBudgetExceeded):
        with tokeymeter.key("off"):
            for i in range(3):
                await metered(f"q{i} " * 100)
    n_before = len(_records())
    spent_before = tokeymeter.key_status("off")["spent_usd"]

    @tokeymeter.cache(model="gpt-4o", enabled=False)
    async def stood_down(p):
        return "raw"
    with tokeymeter.key("off"):
        out = await stood_down("bypass " * 10)
    assert out == "raw"                                   # ran fine
    assert len(_records()) == n_before                    # no record
    assert tokeymeter.key_status("off")["spent_usd"] == spent_before


# ── 1.8 leader failure: fail-open takeover + no unretrieved futures ─────
async def test_leader_failure_failopen_takeover_and_clean_futures():
    """Pins the DESIGNED contract of async in-process single-flight under a
    failing leader (decorator.py ~1441): a follower that receives the
    leader's exception FAILS OPEN — it installs itself as the new leader and
    retries the compute — because a leader's failure may be transient and
    must never doom every waiter. Consequences pinned here:
      • every caller surfaces an exception (nobody hangs),
      • attempts are bounded by the caller count (1..N, not thundering ∞),
      • the exception type is the compute's own,
      • and (the fix this test guards) NO exception-holding future is left
        unretrieved — the suite runs warnings-as-errors, so a regression
        detonates in a later test."""
    calls = {"n": 0}

    @tokeymeter.cache(model="m")
    async def ask(p):
        calls["n"] += 1
        await asyncio.sleep(0.02)          # let followers attach to the leader
        raise ValueError("upstream provider exploded")

    results = await asyncio.gather(
        *[ask("same failing prompt " * 4) for _ in range(5)],
        return_exceptions=True)
    assert len(results) == 5
    assert all(isinstance(r, ValueError) for r in results), results
    assert 1 <= calls["n"] <= 5, calls      # bounded fail-open retries
    # a lone failing leader leaves no unretrieved-exception future behind:
    with pytest.raises(ValueError):
        await ask("solo failing prompt " * 4)
    import gc
    gc.collect()                            # would surface the GC warning
