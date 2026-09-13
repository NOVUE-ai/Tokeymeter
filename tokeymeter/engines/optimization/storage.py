"""
Storage backends for the cache.

Two backends ship by default:
  - MemoryStore: in-process dict with LRU eviction. Fast, no persistence.
  - SQLiteStore: SQLite-backed, persistent across processes. Default.

Both implement the same minimal interface: get(key) -> Any | None, set(key, value) -> None.
Custom backends (Redis, S3, KMS-encrypted, etc.) can implement the same interface.

Backends should NEVER raise on get/set in normal operation — the decorator
catches exceptions defensively, but a well-behaved backend returns None on
miss and silently no-ops on full-disk / network blip / etc.
"""
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional


_SF_MISSING = object()  # sentinel: "no value supplied" (distinct from a real None)


class _InProcessSingleFlight:
    """Shared in-process single-flight coordination (thundering-herd collapse).

    Mixed into every store that implements get(key)/set(key, value). Concurrent
    identical calls *within one process* collapse to a single computation: the
    first caller (leader) computes; everyone else (followers, and late arrivers)
    receives the leader's result.

    Coordination is fully in-memory and never relies on the backing store's
    read-after-write visibility. That distinction matters: an earlier version
    keyed the decision off get(key), which made collapse perfect on the memory
    store but leaky on SQLite (a late caller whose get() hadn't yet seen the
    leader's just-committed row would wrongly elect itself a second leader). We
    therefore (a) hold the leader's value in the in-flight holder for waiting
    followers, and (b) retain it briefly in a bounded `_recent` map so a caller
    arriving just after release still finds it in memory instead of recomputing.

    Scope is deliberately in-process — what the documented single-flight
    contract promises, with no external dependency. Cross-machine collapse needs
    a shared lock (the Redis backend provides that). One implementation, shared
    by the in-memory and the default SQLite store, so behavior cannot drift.

    Fail-safe: a follower whose wait times out falls back to the store; if still
    empty it recomputes — occasionally redundant, never wrong.
    """

    _RECENT_MAX = 1024  # bounded; covers the post-release visibility window

    def _init_single_flight(self) -> None:
        self._inflight: dict = {}          # key -> holder {"event", "value"}
        self._recent: "OrderedDict[str, Any]" = OrderedDict()  # just-computed
        self._inflight_lock = threading.Lock()

    def acquire_compute_lock(self, key: str):
        """Return a token if this caller should compute (leader), else None.

        Purely in-memory and lock-protected, so it stays O(1) and race-free even
        when the backing store's reads are slow (cold SQLite under a burst). The
        decorator only calls this AFTER its exact-cache lookup has already missed,
        so we don't re-read the store here — that read was both redundant and the
        source of a cold-start window where slow get()s let extra leaders slip in.
        """
        with self._inflight_lock:
            if key in self._recent:
                return None  # a peer just computed it (value held in memory)
            if key in self._inflight:
                return None  # a peer is computing now -> follower
            holder = {"event": threading.Event(), "value": _SF_MISSING}
            self._inflight[key] = holder
            return holder  # token (truthy) — passed back to release

    def wait_for_result(self, key: str, timeout: float = 30.0):
        """Follower waits for the leader, then returns the in-memory result."""
        with self._inflight_lock:
            holder = self._inflight.get(key)
            if holder is None:
                if key in self._recent:
                    return self._recent[key]
        if holder is None:
            return self.get(key)
        holder["event"].wait(timeout)
        if holder["value"] is not _SF_MISSING:
            return holder["value"]      # in-memory: no store re-read, no race
        return self.get(key)            # leader stored but published no value

    def release_compute_lock(self, key: str, token, value=_SF_MISSING) -> None:
        """Leader releases: publish the value in-memory, then wake followers.

        The value is recorded in the in-flight holder (for threads already
        waiting) and in the bounded `_recent` map (for threads that arrive just
        after, before the store read would see it). Both are in-memory, so the
        collapse is independent of the backing store's read latency.
        """
        with self._inflight_lock:
            holder = self._inflight.pop(key, None)
            if value is not _SF_MISSING:
                self._recent[key] = value
                self._recent.move_to_end(key)
                while len(self._recent) > self._RECENT_MAX:
                    self._recent.popitem(last=False)
        if holder is not None:
            if value is not _SF_MISSING:
                holder["value"] = value
            holder["event"].set()

    def _clear_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight.clear()
            self._recent.clear()


class MemoryStore(_InProcessSingleFlight):
    """In-process LRU cache. Lost on process restart. Fast (<1ms get).

    Supports in-process single-flight: concurrent identical calls collapse to
    one computation via acquire_compute_lock / wait_for_result, so a single
    process gets the same thundering-herd protection the Redis backend provides
    across machines.
    """

    def __init__(self, max_entries: int = 10_000):
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self._evictions = 0  # cumulative LRU evictions (operational counter)
        self._init_single_flight()  # acquire/wait/release inherited from mixin

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            # Move to end = mark as most recently used
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max:
                # popitem(last=False) evicts the OLDEST entry
                self._data.popitem(last=False)
                self._evictions += 1

    def stats(self) -> dict:
        """Operational counters for this store. LRU eviction is routine (not a
        degraded condition), so it is surfaced as a counter rather than per-event
        noise — operators can watch the rate to size the cache."""
        with self._lock:
            return {"size": len(self._data), "max_entries": self._max,
                    "evictions": self._evictions}

    # ---- single-flight: acquire_compute_lock / wait_for_result /
    #      release_compute_lock are inherited from _InProcessSingleFlight ----

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
        self._clear_inflight()

    def evict_expired(self, is_expired: Callable[[Any], bool]) -> int:
        """Drop entries for which is_expired(value) is True. Performed under the
        store lock so it is safe against concurrent get/set, and so callers never
        need to reach into private state. Never raises into the caller."""
        evicted = 0
        with self._lock:
            for k in list(self._data.keys()):  # snapshot under lock
                try:
                    v = self._data.get(k)
                    if v is not None and is_expired(v):
                        self._data.pop(k, None)
                        evicted += 1
                except Exception:
                    continue  # a bad predicate on one entry must not abort the sweep
        return evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class SQLiteStore(_InProcessSingleFlight):
    """SQLite-backed persistent cache.

    Stores values as JSON. For non-JSON-serializable responses (e.g. an
    OpenAI ChatCompletion object), the decorator should be wrapped around
    a function that returns serializable data (a string, a dict, etc.).

    Single-flight: in-process thundering-herd collapse is provided by the
    _InProcessSingleFlight mixin, so single_flight=True works on the default
    store with no external dependency. (Persistence is shared across processes
    via the file; single-flight coordination is per-process, which matches the
    documented in-process contract.)
    """

    def __init__(self, path: Optional[str] = None):
        from tokeymeter import paths as _paths
        self._path = os.path.expanduser(
            path if path is not None else _paths.state_path("cache.db"))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_single_flight()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path, timeout=2.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)"
            )
            # WAL = better concurrency, especially for read-heavy workloads.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        try:
            with sqlite3.connect(self._path, timeout=1.0) as conn:
                row = conn.execute(
                    "SELECT value FROM cache WHERE key = ?", (key,)
                ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError, OSError):
            # Fail-open: a broken backend should not crash the caller.
            return None

    def set(self, key: str, value: Any, created_at: Optional[float] = None) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError) as e:
            # Not silently lost: a write that can't be serialized is surfaced as
            # a degraded event so it shows in metrics, not as a phantom miss.
            self._emit_store_degraded("sqlite_serialize", e)
            return
        ts = float(created_at) if created_at is not None else time.time()
        try:
            with sqlite3.connect(self._path, timeout=1.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache(key, value, created_at) "
                    "VALUES (?, ?, ?)",
                    (key, payload, ts),
                )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            self._emit_store_degraded("sqlite_write", e)
            return

    @staticmethod
    def _emit_store_degraded(source: str, error: BaseException) -> None:
        try:
            from tokeymeter.engines.reliability.degraded import emit_degraded
            emit_degraded(source, error)
        except Exception:
            pass

    def clear(self) -> None:
        try:
            with sqlite3.connect(self._path, timeout=1.0) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        except (sqlite3.Error, OSError):
            pass
        self._clear_inflight()

    def __len__(self) -> int:
        try:
            with sqlite3.connect(self._path, timeout=1.0) as conn:
                row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
                return int(row[0]) if row else 0
        except (sqlite3.Error, OSError):
            return 0
