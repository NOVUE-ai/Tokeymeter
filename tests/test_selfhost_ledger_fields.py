"""Self-hosted ledger fields (S0-2): endpoint_identity + queue_wait_ms.

These two fields are the join keys every downstream self-host service depends
on. Per-endpoint unit cost, duplicate-deployment consolidation, and queue
economics are impossible without them, so they are pinned hard here.

Invariants:
  endpoint_identity
    - operator-declared identifier, bound via contextvar OR @cache(endpoint=)
    - precedence: explicit per-call arg > bound contextvar > None
    - content-blind: a URL-shaped value is REJECTED at binding (never enters
      the record stream)
    - contextvar isolation across threads and async tasks
    - threads through the deferred shadow-hit record (S0-1 interaction)
  queue_wait_ms
    - miss-only, consume-once (never bleeds into the next record)
    - hits carry None by construction (nothing queued)
    - absent measurement reads as None, NEVER a fabricated 0
    - the extractor is schema-tolerant and normalizes seconds->ms
"""
import asyncio
import threading

import pytest

import tokeymeter
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore
from tokeymeter.usage import (
    set_queue_wait_ms, set_reported_usage, extract_queue_wait_ms,
)
from tokeymeter.engines.execution.endpoint import (
    set_endpoint, get_endpoint, endpoint as endpoint_ctx, resolve,
)


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    set_endpoint(None)
    K.clear_keys()
    yield
    set_endpoint(None)
    K.clear_keys()
    tokeymeter.reset_savings()


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


def _last():
    recs = _records()
    assert recs
    return recs[-1]


# ── endpoint: binding + validation ──────────────────────────────────────

def test_endpoint_binds_and_stamps_record():
    with endpoint_ctx("vllm-a100-pool"):
        @tokeymeter.cache(model="m")
        def ask(p):
            return "r"
        ask("hello")
    assert _last()["endpoint_identity"] == "vllm-a100-pool"


def test_endpoint_unset_is_none_not_error():
    @tokeymeter.cache(model="m")
    def ask(p):
        return "r"
    ask("hello")
    assert _last()["endpoint_identity"] is None


def test_endpoint_rejects_url_shaped_value():
    # A base_url can carry credentials and topology; it must never enter the
    # record stream. The grammar rejects "/" so a URL fails by construction.
    with pytest.raises(ValueError):
        set_endpoint("http://user:pass@10.0.0.1:8000/v1")


def test_endpoint_rejects_url_via_decorator_arg():
    # Malformed endpoint is a developer error — caught at decoration, loudly,
    # not swallowed by the non-raising record path.
    with pytest.raises(ValueError):
        @tokeymeter.cache(model="m", endpoint="https://host/v1")
        def ask(p):
            return "r"


def test_endpoint_accepts_declared_identifier_forms():
    for good in ("vllm-a100-pool", "tgi.spillover", "pool:us-east-1", "ep_01"):
        assert resolve(good) == good


# ── endpoint: precedence ────────────────────────────────────────────────

def test_explicit_arg_beats_contextvar():
    with endpoint_ctx("from-contextvar"):
        @tokeymeter.cache(model="m", endpoint="from-arg")
        def ask(p):
            return "r"
        ask("hello")
    assert _last()["endpoint_identity"] == "from-arg"


def test_contextvar_used_when_no_arg():
    with endpoint_ctx("from-contextvar"):
        @tokeymeter.cache(model="m")
        def ask(p):
            return "r"
        ask("hello")
    assert _last()["endpoint_identity"] == "from-contextvar"


def test_context_manager_restores_prior_binding():
    set_endpoint("outer")
    with endpoint_ctx("inner"):
        assert get_endpoint() == "inner"
    assert get_endpoint() == "outer"


def test_context_manager_restores_on_exception():
    set_endpoint("outer")
    with pytest.raises(RuntimeError):
        with endpoint_ctx("inner"):
            raise RuntimeError("boom")
    assert get_endpoint() == "outer"


# ── endpoint: contextvar isolation ──────────────────────────────────────

def test_thread_isolation():
    seen = {}

    def worker(name):
        set_endpoint(name)
        # a tiny yield so threads interleave if isolation were broken
        threading.Event().wait(0.001)
        seen[name] = get_endpoint()

    ts = [threading.Thread(target=worker, args=(f"ep-{i}",)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # each thread sees only its own binding
    assert seen == {f"ep-{i}": f"ep-{i}" for i in range(8)}


async def test_async_task_isolation():
    async def task(name):
        with endpoint_ctx(name):
            await asyncio.sleep(0.001)

            @tokeymeter.cache(model="m")
            async def ask(p):
                return "r"
            await ask(f"prompt-{name}")

    await asyncio.gather(*(task(f"ep-{i}") for i in range(6)))
    endpoints = {r["endpoint_identity"] for r in _records() if not r["hit"]}
    assert endpoints == {f"ep-{i}" for i in range(6)}


# ── endpoint: shadow-hit threading (S0-1 interaction) ───────────────────

def test_endpoint_threads_through_shadow_hit_record():
    with endpoint_ctx("vllm-pool"):
        @tokeymeter.cache(model="m", shadow=True)
        def ask(p):
            set_reported_usage(10, 20)
            return "r"
        ask("p")
        ask("p")   # shadow hit + real call
    recs = _records()
    shadow_hits = [r for r in recs if r["hit"] and r.get("shadow")]
    assert len(shadow_hits) == 1
    assert shadow_hits[0]["endpoint_identity"] == "vllm-pool"
    # and the paired real-call record too
    assert recs[-1]["endpoint_identity"] == "vllm-pool"


# ── queue_wait: semantics ───────────────────────────────────────────────

def test_queue_wait_lands_on_miss_record():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_queue_wait_ms(42.5)
        return "r"
    ask("p")
    assert _last()["queue_wait_ms"] == 42.5


def test_queue_wait_absent_is_none_never_zero():
    @tokeymeter.cache(model="m")
    def ask(p):
        return "r"   # nothing reports queue wait
    ask("p")
    assert _last()["queue_wait_ms"] is None


def test_hit_carries_no_queue_wait():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_queue_wait_ms(99.0)
        return "r"
    ask("p")     # miss: 99.0
    ask("p")     # real hit: nothing queued
    recs = _records()
    assert recs[0]["queue_wait_ms"] == 99.0
    assert recs[1]["hit"] is True
    assert recs[1]["queue_wait_ms"] is None


def test_queue_wait_consume_once_no_bleed():
    # set on the first call; it must attach to THAT record and not the next.
    @tokeymeter.cache(model="m")
    def first(p):
        set_queue_wait_ms(55.0)
        return "a"

    @tokeymeter.cache(model="m")
    def second(p):
        return "b"   # sets nothing

    first("x")
    second("y")
    recs = _records()
    assert recs[0]["queue_wait_ms"] == 55.0
    assert recs[1]["queue_wait_ms"] is None


def test_queue_wait_rejects_negative_and_garbage():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_queue_wait_ms(-5)       # ignored
        set_queue_wait_ms("nope")   # ignored
        return "r"
    ask("p")
    assert _last()["queue_wait_ms"] is None


def test_shadow_hit_carries_no_queue_wait_but_miss_does():
    @tokeymeter.cache(model="m", shadow=True)
    def ask(p):
        set_queue_wait_ms(33.0)
        return "r"
    ask("p")
    ask("p")   # shadow hit (None) + real call (33.0)
    recs = _records()
    shadow_hits = [r for r in recs if r["hit"] and r.get("shadow")]
    assert shadow_hits[0]["queue_wait_ms"] is None      # hits never queued
    assert recs[-1]["queue_wait_ms"] == 33.0            # real call carries it


# ── queue_wait: extractor schema tolerance ──────────────────────────────

class _Resp:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_extractor_reads_ms_field_on_usage():
    resp = _Resp(usage=_Resp(queue_time_ms=17.0))
    assert extract_queue_wait_ms(resp) == 17.0


def test_extractor_reads_from_metrics_dict():
    resp = _Resp(metrics={"time_in_queue_ms": 8.0})
    assert extract_queue_wait_ms(resp) == 8.0


def test_extractor_converts_seconds_to_ms():
    resp = _Resp(usage=_Resp(queue_time=0.25))   # seconds field
    assert extract_queue_wait_ms(resp) == 250.0


def test_extractor_returns_none_when_absent():
    resp = _Resp(usage=_Resp(prompt_tokens=10))
    assert extract_queue_wait_ms(resp) is None


def test_extractor_ignores_negative_and_nonnumeric():
    assert extract_queue_wait_ms(_Resp(usage=_Resp(queue_time_ms=-1))) is None
    assert extract_queue_wait_ms(_Resp(usage=_Resp(queue_time_ms="x"))) is None


def test_extractor_never_raises_on_odd_input():
    for bad in (None, object(), 42, "string", _Resp()):
        assert extract_queue_wait_ms(bad) is None


# ── serialization: fields round-trip through the ledger ─────────────────

def test_fields_serialize_to_ledger_json():
    with endpoint_ctx("vllm-pool"):
        @tokeymeter.cache(model="m")
        def ask(p):
            set_queue_wait_ms(12.0)
            return "r"
        ask("p")
    import json
    from tokeymeter import savings as sv
    # asdict round-trip is what _iter_records exercises; assert both keys exist
    rec = _last()
    blob = json.dumps(rec)
    assert "endpoint_identity" in blob and "queue_wait_ms" in blob
    assert rec["endpoint_identity"] == "vllm-pool"
    assert rec["queue_wait_ms"] == 12.0
