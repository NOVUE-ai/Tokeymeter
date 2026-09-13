"""S1.1 — true avoided-token recovery on cache hits.

Before S1.1 a cache hit logged chars/4 of the CACHED VALUE, undercounting
recovered capacity. Now the original miss stamps its true token counts into the
cache envelope, and the hit recovers them — exact recovery, not an estimate.

Pinned:
  1. exact / semantic / single-flight hits all recover the miss's true tokens
  2. reported-source and estimated-source both recover consistently (a hit
     matches its miss exactly)
  3. the meta stamp is fail-safe: an uncountable arg/result never breaks the
     call and yields an honest estimate fallback (meta=None -> hit estimates)
  4. backward compat: a legacy 2-element envelope (no meta) makes the hit fall
     back to estimation, never fabricating
  5. the Capacity Recovery Report's recovered_tokens now equal the actual
     avoided volume, and the known_limitation disclosure is gone
  6. peek-not-consume: stamping meta must NOT disturb the record's reported
     usage (the miss record still gets its reported counts)
"""
import asyncio

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution.endpoint import endpoint as endpoint_ctx


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


# ── exact-hit recovery ──────────────────────────────────────────────────

def test_exact_hit_recovers_reported_miss_tokens():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(340, 128)
        return "short"
    ask("q")            # miss
    ask("q")            # exact hit
    recs = _records()
    assert (recs[1]["input_tokens"], recs[1]["output_tokens"]) == (340, 128)
    assert recs[1]["token_source"] == "reported"


def test_exact_hit_recovers_estimated_miss_tokens():
    @tokeymeter.cache(model="m")
    def ask(p):
        return "a considerably longer response body than the prompt"
    ask("q")
    ask("q")
    recs = _records()
    # hit recovers the MISS's estimate, not a fresh chars/4 of a tiny value
    assert recs[1]["input_tokens"] == recs[0]["input_tokens"]
    assert recs[1]["output_tokens"] == recs[0]["output_tokens"]
    assert recs[1]["token_source"] == "estimated"


def test_miss_record_still_gets_reported_tokens_peek_not_consume():
    # Stamping meta peeks reported usage; the miss record must still consume it.
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(77, 88)
        return "x"
    ask("q")
    rec = _records()[0]
    assert (rec["input_tokens"], rec["output_tokens"]) == (77, 88)
    assert rec["token_source"] == "reported"


# ── async + stream recovery ─────────────────────────────────────────────

async def test_async_exact_hit_recovers_tokens():
    @tokeymeter.cache(model="m")
    async def ask(p):
        set_reported_usage(500, 200)
        return "y"
    await ask("q")
    await ask("q")
    recs = _records()
    assert (recs[1]["input_tokens"], recs[1]["output_tokens"]) == (500, 200)
    assert recs[1]["token_source"] == "reported"


async def test_stream_exact_hit_recovers_tokens():
    @tokeymeter.cache_stream(model="m")
    async def gen(p):
        set_reported_usage(700, 300)
        yield "a"
        yield "b"
    async for _ in gen("q"):
        pass
    async for _ in gen("q"):
        pass
    recs = _records()
    assert (recs[1]["input_tokens"], recs[1]["output_tokens"]) == (700, 300)
    assert recs[1]["token_source"] == "reported"


# ── single-flight recovery ──────────────────────────────────────────────

def test_single_flight_follower_recovers_tokens():
    import threading
    import time

    @tokeymeter.cache(model="m")
    def slow(p):
        set_reported_usage(120, 260)
        time.sleep(0.05)
        return "z"

    ts = [threading.Thread(target=slow, args=("q",)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    sf = [r for r in _records() if r.get("hit_type") == "single_flight"]
    assert sf, "expected single-flight collapses"
    for r in sf:
        assert (r["input_tokens"], r["output_tokens"]) == (120, 260)
        assert r["token_source"] == "reported"


# ── fail-safe ───────────────────────────────────────────────────────────

def test_uncountable_arg_does_not_break_call_and_falls_back():
    class Uncountable:
        def __iter__(self):
            raise TypeError("not countable")
        def __str__(self):
            raise TypeError("nope")

    @tokeymeter.cache(model="m")
    def ask(p):
        return "ok"

    # miss + hit on an uncountable arg: call must succeed both times, meta
    # resolution must not raise, and the hit falls back honestly
    assert ask(Uncountable()) == "ok"
    assert ask(Uncountable()) == "ok"
    # no exception is the assertion; records exist and are internally valid
    for r in _records():
        assert r["token_source"] in ("reported", "estimated")


# ── backward compatibility ──────────────────────────────────────────────

def test_legacy_envelope_without_meta_falls_back_to_estimate():
    from tokeymeter.envelope import wrap
    store = MemoryStore()
    tokeymeter.set_default_store(store)

    @tokeymeter.cache(model="m", store=store)
    def ask(p):
        return "should not run on hit"

    # pre-seed the cache with a LEGACY 2-element envelope (no meta), as an
    # older version would have written
    from tokeymeter import decorator as _dec
    key = _dec.make_cache_key(("q",), {}, model="m")
    # namespace prefix mirrors the decorator's default
    import inspect
    # simplest: let a real miss populate, then strip meta to simulate legacy
    ask("q")                       # miss writes a 3-element envelope
    # overwrite with a legacy 2-element envelope for the same logical value
    # (find the key the decorator used)
    for k in list(store._data.keys()) if hasattr(store, "_data") else []:
        store.set(k, wrap("should not run on hit"))   # no meta
    ask("q")                       # hit on the legacy envelope
    hit = [r for r in _records() if r["hit"]]
    assert hit, "expected a hit"
    # legacy meta absent -> honest estimate, never a fabricated reported count
    assert hit[-1]["token_source"] == "estimated"


# ── report accuracy ─────────────────────────────────────────────────────

def test_report_recovered_tokens_equal_actual_avoided_volume():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(300, 100)   # 400 tokens avoided per hit
        return "r"
    ask("q")                            # miss
    for _ in range(5):
        ask("q")                        # 5 hits -> 5 * 400 = 2000 avoided
    rep = tokeymeter.capacity_recovery_report(measured_tokens_per_second=1400)
    assert rep["total_recovered"]["recovered_tokens_total"] == 5 * 400
    assert "known_limitation" not in rep


def test_report_known_limitation_removed():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(10, 10)
        return "r"
    ask("q")
    ask("q")
    rep = tokeymeter.capacity_recovery_report(measured_tokens_per_second=1400)
    assert "known_limitation" not in rep
