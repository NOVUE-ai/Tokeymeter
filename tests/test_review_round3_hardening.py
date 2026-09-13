"""Regression tests for three issues from independent review:

  #1 admin._evict_expired_memory mutated MemoryStore._data without the store
     lock (race vs concurrent get/set). It now delegates to the lock-safe
     MemoryStore.evict_expired().
  #2 RedisStore.set surfaced serialize/encrypt failures but the FINAL write
     failure only flipped health — no redis_write degraded event (less visible
     than sqlite_write). It now emits redis_write too.
  #3 SemanticCache.clear() only cleared semantic_vec when _use_vec was still
     true; if vec mode self-disabled after inserts, vector rows were orphaned.
     clear() now loads vec0 for maintenance (force_vec) and always clears it.
"""
import os
import tempfile
import threading
import time

import pytest

from tokeymeter.storage import MemoryStore
from tokeymeter.envelope import wrap
from tokeymeter import admin, degraded


# ---- #1: concurrent eviction is race-free ----
def test_evict_expired_memory_is_thread_safe():
    store = MemoryStore(max_entries=100_000)
    stop = {"s": False}
    errors = {"n": 0}

    def setter():
        from tokeymeter.envelope import _ENVELOPE_KEY
        i = 0
        while not stop["s"]:
            try:
                # genuinely-expired (past expires_at) vs live, so the evictor
                # actually removes entries while setters keep adding — exercising
                # the eviction-vs-mutation race, not a no-op sweep.
                env = ({_ENVELOPE_KEY: ["v", time.time() - 100]} if i % 2
                       else {_ENVELOPE_KEY: ["v", time.time() + 9999]})
                store.set(f"k{i % 4000}", env)
                i += 1
            except Exception:
                errors["n"] += 1

    def evictor():
        while not stop["s"]:
            try:
                admin._evict_expired_memory(store)
            except Exception:
                errors["n"] += 1

    def reader():
        while not stop["s"]:
            try:
                _ = len(store)
                store.get("k1")
            except Exception:
                errors["n"] += 1

    ts = ([threading.Thread(target=setter) for _ in range(4)] +
          [threading.Thread(target=evictor) for _ in range(2)] +
          [threading.Thread(target=reader) for _ in range(2)])
    for t in ts:
        t.start()
    time.sleep(1.0)
    stop["s"] = True
    for t in ts:
        t.join()
    assert errors["n"] == 0
    assert len(store) <= 100_000


# ---- #2: redis write failure is observable ----
def test_redis_write_failure_emits_degraded_event():
    from tokeymeter.backends.redis_store import RedisStore
    from tokeymeter.backends.cipher import FernetCipher
    from cryptography.fernet import Fernet

    class BoomClient:
        def set(self, *a, **k):
            raise ConnectionError("redis down")
        def get(self, *a, **k):
            return None
        def scan(self, *a, **k):
            return (0, [])

    degraded.clear_subscribers()
    seen = []
    degraded.on_degraded(seen.append)
    try:
        store = RedisStore(client=BoomClient(), namespace="t",
                           cipher=FernetCipher(Fernet.generate_key()))
        store.set("k", {"v": 1})
        assert any(e.source == "redis_write" for e in seen)
        assert store._healthy is False  # health also tracked
    finally:
        degraded.clear_subscribers()


# ---- #3: clear() reaps vec rows even after vec mode self-disables ----
def test_semantic_clear_reaps_orphaned_vec_rows_after_self_disable():
    from tokeymeter.semantic import SemanticCache, is_vec_index_available
    if not is_vec_index_available():
        pytest.skip("sqlite-vec not available in this environment")
    import numpy as np

    d = tempfile.mkdtemp()
    c = SemanticCache(path=os.path.join(d, "sem.db"),
                      encoder=lambda t: (lambda v: v / np.linalg.norm(v))(
                          np.array([float((hash(t) % 97) + 1), 1.0, 2.0], dtype="float32")),
                      dim=3)
    for i in range(8):
        c.store(f"prompt number {i} with text", f"resp{i}")

    def vec_count():
        with c._connect(force_vec=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM semantic_vec").fetchone()[0]

    assert vec_count() == 8
    c._use_vec = False          # runtime self-disable AFTER inserts
    c.clear()
    assert vec_count() == 0     # orphans reaped despite the disabled flag


def test_evict_uses_public_api_only_no_private_reach_in():
    """admin._evict_expired_memory must use ONLY the public evict_expired API.
    A store that exposes get/set but NOT evict_expired must be skipped cleanly —
    admin must never touch a store's private state (._data/._lock)."""
    import time
    from tokeymeter.envelope import _ENVELOPE_KEY

    # 1) public-API store: eviction works
    store = MemoryStore(max_entries=1000)
    for i in range(8):
        store.set(f"exp{i}", {_ENVELOPE_KEY: ["v", time.time() - 100]})
    for i in range(3):
        store.set(f"live{i}", {_ENVELOPE_KEY: ["v", time.time() + 9999]})
    assert admin._evict_expired_memory(store) == 8
    assert len(store) == 3

    # 2) custom store WITHOUT evict_expired and WITHOUT _data/_lock: must be
    #    skipped cleanly (returns 0, no AttributeError from private reach-in).
    class MinimalStore:
        def __init__(self):
            self.kv = {}
        def get(self, k):
            return self.kv.get(k)
        def set(self, k, v):
            self.kv[k] = v
        # intentionally NO evict_expired, NO _data, NO _lock

    minimal = MinimalStore()
    minimal.set("x", {_ENVELOPE_KEY: ["v", time.time() - 100]})
    assert admin._evict_expired_memory(minimal) == 0   # skipped, not crashed
    assert minimal.get("x") is not None                # untouched (no private access)


def test_eviction_cleans_vec_rows_after_self_disable_no_orphans():
    """Maintenance-drift regression: if vec mode is active, rows exist in BOTH
    semantic_cache and semantic_vec, and the vec backend later self-disables,
    subsequent LRU eviction must still clean the matching semantic_vec rows
    instead of orphaning them."""
    import tempfile, os, sqlite3
    import pytest
    from tokeymeter.semantic import SemanticCache, is_vec_index_available
    if not is_vec_index_available():
        pytest.skip("sqlite-vec not available")
    import numpy as np
    import sqlite_vec

    path = os.path.join(tempfile.mkdtemp(), "sem.db")
    c = SemanticCache(path=path, encoder=lambda t: np.array([1., 2., 3.], dtype="float32"),
                      dim=3, max_entries=5, use_vec_index=True)
    for i in range(5):
        c.store_by_embedding(f"p{i}", np.array([float(i), 2., 3.], dtype="float32"), f"r{i}")

    c._use_vec = False  # runtime self-disable AFTER rows were inserted with vec
    for i in range(5, 12):  # trigger LRU eviction of the old (vec-backed) rows
        c.store_by_embedding(f"p{i}", np.array([float(i), 2., 3.], dtype="float32"), f"r{i}")

    cc = sqlite3.connect(path)
    cc.enable_load_extension(True)
    sqlite_vec.load(cc)
    orphans = cc.execute(
        "SELECT COUNT(*) FROM semantic_vec v WHERE NOT EXISTS "
        "(SELECT 1 FROM semantic_cache s WHERE s.id = v.rowid)").fetchone()[0]
    cc.close()
    assert orphans == 0, f"eviction orphaned {orphans} vec rows after self-disable"


def test_sync_wrapper_warns_once_when_blocking_event_loop():
    """The sync _run_sync path, when reached from a running loop, blocks the loop
    thread. It must (a) still work, (b) warn exactly once advising the async
    wrapper, and (c) not warn when there is no running loop."""
    import asyncio
    import warnings
    import tokeymeter.memory as m

    m._WARNED_SYNC_IN_LOOP = False

    async def in_loop():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a = m._run_sync(asyncio.sleep(0, result="A"))
            b = m._run_sync(asyncio.sleep(0, result="B"))
        rt = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        return a, b, rt

    a, b, rt = asyncio.run(in_loop())
    assert (a, b) == ("A", "B")            # still works
    assert len(rt) == 1                     # warned exactly once
    assert "BLOCKS the loop" in str(rt[0].message)
    assert "async def" in str(rt[0].message)

    # no running loop -> asyncio.run path -> no warning
    m._WARNED_SYNC_IN_LOOP = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = m._run_sync(asyncio.sleep(0, result="C"))
    assert c == "C"
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
