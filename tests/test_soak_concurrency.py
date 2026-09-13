"""Bounded concurrency / soak gates (#4).

A real soak runs for hours; CI can't. These are *bounded* soaks: enough churn and
concurrency to surface leaks, races, and outage-recovery bugs, but iteration-
bounded so they finish in seconds and assert INVARIANTS (bounded memory, zero
errors, correctness) rather than flaky timing. Set TOKEYMETER_SOAK_SCALE=N to
crank the same tests up for a nightly deep run without touching code.

Covered: sustained memory churn, thread-leak, concurrent correctness/no-bleed,
mixed sync+async under load, backend outage -> fail-open -> recovery, high
session churn, and audit durability under sustained burst.
"""
import asyncio
import concurrent.futures as cf
import gc
import os
import threading
import time

import pytest

import tokeymeter
import tokeymeter.decorator as dec
from tokeymeter import degraded
from tokeymeter.storage import MemoryStore

SCALE = max(1, int(os.environ.get("TOKEYMETER_SOAK_SCALE", "1")))

pytestmark = pytest.mark.soak


def _fresh(store=None):
    # NOTE: MemoryStore defines __len__, so an empty store is *falsy* — never use
    # `store or MemoryStore()` (it would discard a passed-in empty store). Use an
    # explicit None check, exactly as the library does internally.
    tokeymeter.set_default_store(store if store is not None else MemoryStore())
    tokeymeter.reset()


# ---------- 1. memory bounded under sustained concurrent churn ----------
def test_memory_bounded_under_sustained_churn():
    store = MemoryStore(max_entries=1000)
    _fresh(store)

    @tokeymeter.cache(model="gpt-4o-mini")
    def handler(p):
        return "r"

    n = 40_000 * SCALE
    errors = {"n": 0}

    def worker(base):
        for i in range(n // 8):
            try:
                handler(f"unique-{base}-{i}")
            except Exception:
                errors["n"] += 1

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(8)))

    assert errors["n"] == 0
    assert len(store) <= 1000, "cache must stay bounded under sustained churn"
    # single-flight in-memory maps must also stay bounded
    assert len(getattr(store, "_recent", {})) <= 1024
    st = store.stats()
    assert st["evictions"] > 0  # eviction actually exercised and counted


# ---------- 2. no thread leak under heavy decorate + call ----------
def test_no_thread_leak_under_load():
    _fresh()
    # warm up so any lazy singletons spin their threads before we baseline
    @tokeymeter.cache(model="gpt-4o-mini")
    def warm(p):
        return "x"
    warm("warm")
    gc.collect()
    baseline = threading.active_count()

    funcs = []
    for i in range(200 * SCALE):
        @tokeymeter.cache(model="gpt-4o-mini")
        def f(p, _i=i):
            return _i
        funcs.append(f)
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda fn: [fn("p") for _ in range(20)], funcs))
    gc.collect()
    time.sleep(0.1)
    # allow a tiny delta for transient pool threads, but no per-call growth
    assert threading.active_count() - baseline <= 2


# ---------- 3. concurrent correctness: no cross-function/tenant bleed ----------
def test_concurrent_correctness_no_bleed():
    _fresh()
    n_funcs = 60 * SCALE
    fns = []
    for i in range(n_funcs):
        @tokeymeter.cache(model="gpt-4o")
        def fn(p, _i=i):
            return f"F{_i}:{p}"
        fns.append((i, fn))

    errors = {"n": 0}

    def hammer(arg):
        idx, fn = arg
        for _ in range(30):
            if fn("identical prompt for everyone") != f"F{idx}:identical prompt for everyone":
                errors["n"] += 1

    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(hammer, fns))
    assert errors["n"] == 0


# ---------- 4. mixed sync + async under concurrent load ----------
def test_mixed_sync_async_under_load():
    _fresh()
    sync_calls = {"n": 0}
    async_calls = {"n": 0}

    @tokeymeter.cache(model="gpt-4o", single_flight=True)
    def sync_fn(p):
        sync_calls["n"] += 1
        return f"S:{p}"

    @tokeymeter.cache(model="gpt-4o", single_flight=True)
    async def async_fn(p):
        async_calls["n"] += 1
        await asyncio.sleep(0)
        return f"A:{p}"

    errors = {"n": 0}

    def sync_worker():
        for i in range(200 * SCALE):
            if sync_fn(f"k{i % 50}") != f"S:k{i % 50}":
                errors["n"] += 1

    async def async_worker():
        for i in range(200 * SCALE):
            r = await async_fn(f"k{i % 50}")
            if r != f"A:k{i % 50}":
                errors["n"] += 1

    def run_async():
        async def _amain():
            await asyncio.gather(*[async_worker() for _ in range(4)])
        asyncio.run(_amain())

    threads = [threading.Thread(target=sync_worker) for _ in range(4)] + \
              [threading.Thread(target=run_async) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors["n"] == 0
    # caching worked: far fewer underlying computations than calls
    assert sync_calls["n"] <= 50
    assert async_calls["n"] <= 50


# ---------- 5. backend outage -> fail-open -> recovery ----------
def test_backend_outage_then_recovery():
    class FlakyStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.down = False
        def get(self, k):
            if self.down:
                raise ConnectionError("backend down")
            return super().get(k)
        def set(self, k, v):
            if self.down:
                raise ConnectionError("backend down")
            return super().set(k, v)

    store = FlakyStore()
    _fresh(store)
    degraded.clear_subscribers()

    calls = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        calls["n"] += 1
        return f"r:{p}"

    # healthy: populate + hit
    assert ask("q") == "r:q"
    assert ask("q") == "r:q"
    healthy_calls = calls["n"]

    # OUTAGE: every call must still succeed (fail-open), never raise
    store.down = True
    errors = {"n": 0}
    for i in range(500 * SCALE):
        try:
            if ask(f"q{i % 10}") != f"r:q{i % 10}":
                errors["n"] += 1
        except Exception:
            errors["n"] += 1
    assert errors["n"] == 0, "calls must never raise during a backend outage"

    # RECOVERY: caching resumes
    store.down = False
    calls["n"] = 0
    ask("recovery-probe")
    ask("recovery-probe")
    assert calls["n"] == 1, "caching must resume after recovery (2nd call hits)"


# ---------- 6. high session churn: bounded locks, enforced cap ----------
def test_high_session_churn_bounded():
    from tokeymeter.memory import ConversationMemory, InMemoryMemoryStore
    mem = ConversationMemory(max_turns=10, store=InMemoryMemoryStore())
    mem._max_session_locks = 500

    async def run():
        n_sessions = 5000 * SCALE
        for i in range(n_sessions):
            sid = f"sess{i}"
            # a few turns per session, exceeding max_turns for some
            for t in range(3):
                await mem.add_turn(sid, f"u{t}", f"a{t}")
        # locks bounded despite many sessions
        assert len(mem._session_locks) <= 500
        # cap enforced on a heavily-used session
        await mem.add_turn("hot", "u", "a")
        for t in range(50):
            await mem.add_turn("hot", f"u{t}", f"a{t}")
        turns = await asyncio.to_thread(mem._store.get_turns, "hot")
        assert len(turns) == 10

    asyncio.run(run())


# ---------- 7. audit durability under sustained burst ----------
def test_audit_durability_soak():
    import tempfile
    from tokeymeter.audit.log import AuditLog
    d = tempfile.mkdtemp()
    log = AuditLog(
        path=os.path.join(d, "a.db"),
        install_secret_path=os.path.join(d, "s"),
        signing_key_path=os.path.join(d, "k"),
        queue_max_size=50,
        flush_interval_seconds=0.05,
        durable=True,
    )
    total = 3000 * SCALE
    errors = {"n": 0}

    def writer(base):
        for i in range(total // 6):
            try:
                log.append(decision_type="cache_hit", prompt_text=f"p{base}-{i}", model="gpt-4o")
            except Exception:
                errors["n"] += 1

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(writer, range(6)))
    log.flush(timeout=20.0)
    time.sleep(0.3)
    log.flush(timeout=20.0)

    s = log.stats()
    assert errors["n"] == 0
    assert s["entries_dropped_queue_full"] == 0, "durable mode must not drop under sustained burst"
    result = log.verify_chain()
    assert getattr(result, "ok", getattr(result, "valid", None)) is True
