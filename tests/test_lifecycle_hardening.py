"""Regression tests for five lifecycle / correctness / observability fixes
surfaced by independent review:

  #5 ConversationMemory.max_turns was not enforced (sessions grew unbounded).
  #1 clear_session leaked the per-session asyncio.Lock; the lock map was unbounded.
  #4 Store .set() silently skipped on serialize/encrypt failure (looked like a miss).
  #3 import_cache() did not restore created_at (no faithful round-trip).
  #2 MemoryStore.__len__ read without the lock.

Each test encodes the original probe so the regression cannot return.
"""
import asyncio
import os
import sqlite3
import tempfile
import threading

import tokeymeter
from tokeymeter import admin, degraded
from tokeymeter.memory import ConversationMemory, InMemoryMemoryStore, SQLiteMemoryStore
from tokeymeter.storage import SQLiteStore, MemoryStore


# ---- #5: max_turns is actually enforced, on both stores ----
def _run_cap(store):
    mem = ConversationMemory(recent_window=2, summary_threshold=3, max_turns=5, store=store)

    async def go():
        for i in range(50):
            await mem.add_turn("s1", f"u{i}", f"a{i}")
        turns = await asyncio.to_thread(store.get_turns, "s1")
        return turns

    return asyncio.run(go())


def test_max_turns_enforced_in_memory():
    turns = _run_cap(InMemoryMemoryStore())
    assert len(turns) == 5
    assert turns[-1].user == "u49"   # newest retained
    assert turns[0].user == "u45"    # oldest beyond cap pruned


def test_max_turns_enforced_sqlite():
    d = tempfile.mkdtemp()
    turns = _run_cap(SQLiteMemoryStore(path=os.path.join(d, "m.db")))
    assert len(turns) == 5
    assert turns[-1].user == "u49"
    assert turns[0].user == "u45"


# ---- #1: session locks are cleaned and bounded ----
def test_clear_session_removes_lock():
    mem = ConversationMemory(store=InMemoryMemoryStore())

    async def go():
        for i in range(100):
            await mem.add_turn(f"s{i}", "u", "a")
        before = len(mem._session_locks)
        for i in range(100):
            await mem.clear_session(f"s{i}")
        return before, len(mem._session_locks)

    before, after = asyncio.run(go())
    assert before == 100
    assert after == 0


def test_session_lock_map_is_lru_bounded():
    mem = ConversationMemory(store=InMemoryMemoryStore())
    mem._max_session_locks = 200

    async def go():
        for i in range(5000):  # many idle sessions, never cleared
            await mem.add_turn(f"sess{i}", "u", "a")
        return len(mem._session_locks)

    assert asyncio.run(go()) <= 200


# ---- #4: store-set failures are observable, not silent ----
def test_sqlite_set_failure_emits_degraded_event():
    seen = []
    degraded.on_degraded(seen.append)
    before = degraded.degraded_event_count()

    d = tempfile.mkdtemp()
    s = SQLiteStore(path=os.path.join(d, "c.db"))
    s.set("bad", {(1, 2): "tuple-key-unserializable"})  # json.dumps -> TypeError

    assert degraded.degraded_event_count() > before
    assert any(e.source == "sqlite_serialize" for e in seen)
    assert s.get("bad") is None  # still a miss for the reader, but now visible


# ---- #3: backup/restore preserves created_at ----
def test_import_cache_preserves_created_at():
    d = tempfile.mkdtemp()
    dbp = os.path.join(d, "c.db")
    s = SQLiteStore(path=dbp)
    tokeymeter.set_default_store(s)
    s.set("k1", {"v": 1})

    def created_at():
        with sqlite3.connect(dbp) as c:
            return c.execute("SELECT created_at FROM cache WHERE key='k1'").fetchone()[0]

    original = created_at()
    expf = os.path.join(d, "e.jsonl")
    admin.export_cache(expf)
    import time
    time.sleep(0.2)
    admin.import_cache(expf)

    assert abs(created_at() - original) < 0.001  # timestamp preserved


def test_sqlite_set_explicit_created_at():
    d = tempfile.mkdtemp()
    dbp = os.path.join(d, "c.db")
    s = SQLiteStore(path=dbp)
    s.set("k", {"v": 1}, created_at=1234567.0)
    with sqlite3.connect(dbp) as c:
        ts = c.execute("SELECT created_at FROM cache WHERE key='k'").fetchone()[0]
    assert ts == 1234567.0


# ---- #2: __len__ is thread-safe ----
def test_memory_store_len_is_thread_safe():
    store = MemoryStore(max_entries=10000)
    stop = {"s": False}
    errors = {"n": 0}

    def mutate():
        i = 0
        while not stop["s"]:
            store.set(f"k{i % 5000}", i)
            i += 1

    def measure():
        while not stop["s"]:
            try:
                _ = len(store)
            except Exception:
                errors["n"] += 1

    ts = [threading.Thread(target=mutate) for _ in range(4)] + \
         [threading.Thread(target=measure) for _ in range(4)]
    for t in ts:
        t.start()
    import time
    time.sleep(0.8)
    stop["s"] = True
    for t in ts:
        t.join()
    assert errors["n"] == 0
