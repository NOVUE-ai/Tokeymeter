"""Regression tests for two concurrency bugs found in the v0.11 audit:

  1. single_flight=True was a silent no-op on the DEFAULT (SQLite) store —
     SQLiteStore lacked the compute-lock, so a burst paid for every call.
  2. _get_default_store() had a lazy-init race — a concurrent first-access
     burst built one store instance PER THREAD (each with its own in-memory
     single-flight state), so the very first stampede never collapsed.

Both are exercised here against the default store specifically, which the
pre-existing single-flight tests did not do (they used MemoryStore).
"""
import threading
import time

import pytest

import tokeymeter
from tokeymeter import decorator as _dec
from tokeymeter.storage import SQLiteStore, MemoryStore, _InProcessSingleFlight


@pytest.fixture(autouse=True)
def _isolate_default_store():
    """Snapshot/restore the module-global default store around each test."""
    saved = _dec._default_store
    tokeymeter.reset()
    yield
    _dec._default_store = saved
    tokeymeter.reset()


def test_default_sqlite_store_supports_single_flight():
    # The bug: SQLiteStore (the default) had no acquire_compute_lock at all.
    assert issubclass(SQLiteStore, _InProcessSingleFlight)
    assert hasattr(SQLiteStore, "acquire_compute_lock")
    assert hasattr(SQLiteStore, "wait_for_result")
    assert hasattr(SQLiteStore, "release_compute_lock")


@pytest.mark.parametrize("store_factory", [
    lambda p: SQLiteStore(path=str(p / "sf.db")),
    lambda p: MemoryStore(),
])
def test_single_flight_collapses_burst(tmp_path, store_factory):
    """A simultaneous burst of identical calls collapses to ~1 computation on
    BOTH the persistent default store and the in-memory store."""
    tokeymeter.set_default_store(store_factory(tmp_path))
    N = 64
    underlying = {"n": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(N)

    @tokeymeter.cache(model="gpt-4o", single_flight=True)
    def slow(prompt: str) -> str:
        with lock:
            underlying["n"] += 1
        time.sleep(0.03)  # simulate a model call
        return "result"

    def worker(_):
        barrier.wait()
        return slow("identical prompt")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Before the fix this was N (no collapse). After, it must be ~1.
    assert underlying["n"] <= 2, (
        f"single-flight failed to collapse a {N}-way burst: "
        f"{underlying['n']} underlying calls"
    )


def test_lazy_default_store_init_is_thread_safe():
    """Concurrent first-access must yield ONE shared store, not one per thread.

    The race made every thread in the first burst construct its own store
    (separate in-memory single-flight state), so the stampede never collapsed.
    """
    _dec._default_store = None  # force the lazy-init path
    seen_ids = []
    seen_lock = threading.Lock()
    N = 32
    barrier = threading.Barrier(N)

    def grab():
        barrier.wait()
        store = _dec._get_default_store()
        with seen_lock:
            seen_ids.append(id(store))

    threads = [threading.Thread(target=grab) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen_ids)) == 1, (
        f"lazy init created {len(set(seen_ids))} distinct default stores under "
        "concurrency; must be exactly 1"
    )


def test_single_flight_results_are_correct_under_burst(tmp_path):
    """Collapsing must never serve a wrong answer: every caller gets the leader's
    result for the same key, and distinct keys stay distinct."""
    tokeymeter.set_default_store(SQLiteStore(path=str(tmp_path / "sf2.db")))

    @tokeymeter.cache(model="gpt-4o-mini", single_flight=True)
    def handler(prompt: str) -> str:
        return f"answer-for-{prompt}"

    N = 50
    results = {}
    res_lock = threading.Lock()
    barrier = threading.Barrier(N)

    def worker(i):
        key = f"prompt-{i % 5}"  # 5 distinct keys, 10 callers each
        barrier.wait()
        out = handler(key)
        with res_lock:
            results[(i, key)] = out

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for (i, key), out in results.items():
        assert out == f"answer-for-{key}", f"wrong result for {key}: {out!r}"
