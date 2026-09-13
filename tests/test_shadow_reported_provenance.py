"""Shadow-hit token provenance (S0 self-host fix).

In shadow mode the wrapped function ALWAYS runs, so the provider's reported
usage exists in the same invocation as the shadow hit. The shadow-hit record
must therefore carry those reported counts (token_source == "reported"), not
the chars/4 estimate: the tokens a shadow hit would have avoided ARE the
tokens the real call actually spent. Shadow mode is the pre-sales evaluation
mode — its "would have saved" figure must rest on provider truth.

Pinned here:
  1. Sync/async/stream shadow hits carry reported counts when the real call
     reports usage, and remain honestly "estimated" when it doesn't.
  2. Ledger order is unchanged: shadow-hit record precedes its real-call record.
  3. Non-shadow hits are untouched (estimated by construction — avoided cost
     with no upstream call in the invocation).
  4. Failure paths never lose the shadow-hit record: an exception, hard-cap
     stop, or abandoned stream after the hit still emits it — and drains any
     reported usage the upstream set before failing (both to attach the truest
     counts available and to prevent stale usage bleeding into the NEXT call).
"""
import pytest

import tokeymeter
from tokeymeter import decorator as _dec
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore
from tokeymeter.usage import set_reported_usage


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    K.clear_keys()
    yield
    K.clear_keys()
    tokeymeter.reset_savings()


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


def _shadow_hits(recs):
    return [r for r in recs if r["hit"] and r.get("shadow")]


# ── sync ────────────────────────────────────────────────────────────────

def test_sync_shadow_hit_carries_reported_counts():
    @tokeymeter.cache(model="vllm-model", shadow=True)
    def ask(p):
        set_reported_usage(111, 222)
        return "response text here"

    ask("same prompt")
    ask("same prompt")   # shadow hit + real call in one invocation

    recs = _records()
    hits = _shadow_hits(recs)
    assert len(hits) == 1
    assert hits[0]["hit_type"] == "shadow_exact"
    assert (hits[0]["input_tokens"], hits[0]["output_tokens"]) == (111, 222)
    assert hits[0]["token_source"] == "reported"
    # the paired real-call record is unchanged
    assert recs[-1]["hit"] is False
    assert recs[-1]["token_source"] == "reported"


def test_sync_shadow_hit_precedes_real_record_in_ledger():
    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        set_reported_usage(10, 20)
        return "r"

    ask("p")
    ask("p")
    recs = _records()
    kinds = [(r["hit"], r.get("hit_type")) for r in recs]
    assert kinds == [(False, None), (True, "shadow_exact"), (False, None)]


def test_sync_shadow_hit_without_reported_stays_estimated():
    @tokeymeter.cache(model="bare-model", shadow=True)
    def ask(p):
        return "no usage reported by this callable"

    ask("p")
    ask("p")
    hits = _shadow_hits(_records())
    assert len(hits) == 1
    assert hits[0]["token_source"] == "estimated"


def test_non_shadow_hit_recovers_miss_provenance():
    # S1.1: a LIVE (non-shadow) exact hit now reports the ORIGINAL miss's token
    # counts and provenance, recovered from the cache envelope — not a chars/4
    # re-estimate of the cached value. Before S1.1 this read "estimated"; that
    # was the undercount bug. The miss reported 50/60, so the hit does too.
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(50, 60)
        return "cached value"

    ask("p")             # miss: reported 50/60, stamps meta
    ask("p")             # live hit: recovers 50/60 from the envelope
    recs = _records()
    assert recs[0]["token_source"] == "reported"
    assert recs[1]["hit"] is True and not recs[1].get("shadow")
    assert recs[1]["token_source"] == "reported"
    assert (recs[1]["input_tokens"], recs[1]["output_tokens"]) == (50, 60)


def test_no_bleed_into_following_call():
    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        set_reported_usage(111, 222)
        return "r"

    @tokeymeter.cache(model="other")
    def bare(p):
        return "no usage here"

    ask("p")
    ask("p")
    bare("q")
    last = _records()[-1]
    assert last["model"] == "other"
    assert last["token_source"] == "estimated"
    assert (last["input_tokens"], last["output_tokens"]) != (111, 222)


# ── sync failure paths ──────────────────────────────────────────────────

def test_exception_after_report_still_emits_hit_with_reported_and_no_bleed():
    calls = {"n": 0}

    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        calls["n"] += 1
        set_reported_usage(310, 640)
        if calls["n"] == 2:      # fail on the shadow-hit invocation,
            raise RuntimeError("upstream failed after reporting usage")
        return "r"

    ask("p")
    with pytest.raises(RuntimeError):
        ask("p")

    hits = _shadow_hits(_records())
    assert len(hits) == 1
    # drained from the failing upstream: the truest counts still available
    assert (hits[0]["input_tokens"], hits[0]["output_tokens"]) == (310, 640)
    assert hits[0]["token_source"] == "reported"

    @tokeymeter.cache(model="after")
    def bare(p):
        return "x"
    bare("q")
    assert _records()[-1]["token_source"] == "estimated"   # nothing bled


def test_exception_before_report_still_emits_hit_estimated():
    calls = {"n": 0}

    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("failed before reporting")
        set_reported_usage(10, 20)
        return "r"

    ask("p")
    with pytest.raises(RuntimeError):
        ask("p")

    hits = _shadow_hits(_records())
    assert len(hits) == 1
    assert hits[0]["token_source"] == "estimated"


def test_hard_cap_stop_still_emits_deferred_hit(monkeypatch):
    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        set_reported_usage(10, 20)
        return "r"

    ask("p")

    def _boom():
        raise RuntimeError("hard key-budget stop")
    monkeypatch.setattr(_dec._keys, "check_current", _boom)
    with pytest.raises(RuntimeError):
        ask("p")

    hits = _shadow_hits(_records())
    assert len(hits) == 1   # deferral must not lose the record on the stop


# ── async ───────────────────────────────────────────────────────────────

async def test_async_shadow_hit_carries_reported_counts():
    @tokeymeter.cache(model="m", shadow=True)
    async def ask(p):
        set_reported_usage(77, 88)
        return "r"

    await ask("p")
    await ask("p")
    hits = _shadow_hits(_records())
    assert len(hits) == 1
    assert (hits[0]["input_tokens"], hits[0]["output_tokens"]) == (77, 88)
    assert hits[0]["token_source"] == "reported"


async def test_async_exception_after_report_still_emits_hit():
    calls = {"n": 0}

    @tokeymeter.cache(model="m", shadow=True)
    async def ask(p):
        calls["n"] += 1
        set_reported_usage(31, 64)
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return "r"

    await ask("p")
    with pytest.raises(RuntimeError):
        await ask("p")
    hits = _shadow_hits(_records())
    assert len(hits) == 1
    assert hits[0]["token_source"] == "reported"
    assert (hits[0]["input_tokens"], hits[0]["output_tokens"]) == (31, 64)


# ── stream ──────────────────────────────────────────────────────────────

async def test_stream_shadow_hit_carries_reported_counts():
    @tokeymeter.cache_stream(model="m", shadow=True)
    async def gen(p):
        yield "a"
        yield "b"
        set_reported_usage(12, 34)

    async for _ in gen("p"):
        pass
    async for _ in gen("p"):   # shadow hit; real stream still runs
        pass

    hits = _shadow_hits(_records())
    assert len(hits) == 1
    assert hits[0]["hit_type"] == "shadow_exact"
    assert (hits[0]["input_tokens"], hits[0]["output_tokens"]) == (12, 34)
    assert hits[0]["token_source"] == "reported"


async def test_stream_abandoned_after_shadow_hit_still_emits_hit():
    @tokeymeter.cache_stream(model="m", shadow=True)
    async def gen(p):
        set_reported_usage(21, 43)
        yield "a"
        yield "b"
        yield "c"

    async for _ in gen("p"):
        pass

    agen = gen("p")            # shadow hit deferred at lookup
    await agen.__anext__()     # consume one chunk, then abandon
    await agen.aclose()

    hits = _shadow_hits(_records())
    assert len(hits) == 1      # fallback flush on abandon — record not lost
    assert hits[0]["token_source"] == "reported"   # drained from upstream
