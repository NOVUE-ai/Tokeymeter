"""Locks the in-process single-flight fix found by the stress harness:
concurrent identical calls must collapse to one computation with the
in-memory store (not just the Redis backend)."""
import threading, time
import concurrent.futures as cf
import tokeymeter
import tokeymeter as tk


def test_inprocess_single_flight_collapses_concurrent_identical():
    tokeymeter.set_default_store(tokeymeter.storage.MemoryStore())
    calls = {"n": 0}; lock = threading.Lock()

    @tk.meter(model="gpt-4o-mini", single_flight=True)
    def ask(p):
        with lock: calls["n"] += 1
        time.sleep(0.05)
        return "ans-" + p

    with cf.ThreadPoolExecutor(max_workers=30) as ex:
        res = list(ex.map(lambda _: ask("identical prompt"), range(30)))

    assert calls["n"] <= 3, f"single-flight did not collapse: {calls['n']} calls"
    assert len(set(res)) == 1, "answers diverged under single-flight"


def test_memorystore_compute_lock_interface():
    ms = tokeymeter.storage.MemoryStore()
    assert hasattr(ms, "acquire_compute_lock")
    assert hasattr(ms, "wait_for_result")
    assert hasattr(ms, "release_compute_lock")
    tok = ms.acquire_compute_lock("k")
    assert tok is not None                     # leader
    assert ms.acquire_compute_lock("k") is None  # follower while in-flight
    ms.set("k", "value")
    ms.release_compute_lock("k", tok)
    assert ms.wait_for_result("k") == "value"
