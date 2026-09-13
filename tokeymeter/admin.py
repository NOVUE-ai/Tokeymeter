"""
Admin and inspection tools for ops teams.

These are the operations a real ops team needs to feel comfortable
deploying Tokeymeter to production:

  - cache_info():         what backends are in use, paths, versions
  - cache_stats():        sizes, distribution, top-N
  - clear_cache():        wipe both tiers
  - evict_expired():      proactive cleanup of TTL'd entries
  - export_cache():       JSONL dump for backup
  - import_cache():       restore from a dump

All admin operations are fail-safe: they never raise, they return
status dicts. If a backend doesn't support an operation, the result
includes a clear "supported": false note.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .decorator import _get_default_semantic_cache, _get_default_store
from .envelope import is_expired
from .storage import MemoryStore, SQLiteStore

log = logging.getLogger("tokeymeter.admin")


# ============================ INFO ============================

def cache_info() -> dict:
    """Return diagnostic info about the active caches.

    Useful as a /health or /info endpoint in a production app.
    """
    info = {
        "version": _get_version(),
        "exact_cache": _exact_info(),
        "semantic_cache": _semantic_info(),
        "subscribers": _subscriber_info(),
    }
    return info


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def _exact_info() -> dict:
    store = _get_default_store()
    info: Dict[str, Any] = {
        "backend": type(store).__name__,
    }
    if isinstance(store, SQLiteStore):
        info["path"] = store._path
        try:
            info["size_bytes"] = os.path.getsize(store._path)
        except OSError:
            info["size_bytes"] = None
    info["entries"] = len(store) if hasattr(store, "__len__") else None
    return info


def _semantic_info() -> dict:
    sem = _get_default_semantic_cache()
    if sem is None:
        return {"enabled": False, "reason": "no encoder / deps not installed"}
    info: Dict[str, Any] = {
        "enabled": True,
        "backend": sem.backend,
        "threshold": sem._threshold,
        "max_entries": sem._max,
        "path": sem._path,
        "entries": len(sem),
        "is_functional": sem.is_functional,
    }
    try:
        info["size_bytes"] = os.path.getsize(sem._path)
    except OSError:
        info["size_bytes"] = None
    return info


def _subscriber_info() -> dict:
    try:
        from .events import subscriber_count
        return {"event_subscribers": subscriber_count()}
    except Exception:
        return {"event_subscribers": 0}


# ============================ STATS ============================

def cache_stats(*, top_n: int = 10) -> dict:
    """Aggregate stats: sizes, oldest entries, top-N most-referenced."""
    store = _get_default_store()
    stats: Dict[str, Any] = {
        "exact": {
            "entries": len(store) if hasattr(store, "__len__") else None,
        },
    }

    if isinstance(store, SQLiteStore):
        try:
            with sqlite3.connect(store._path, timeout=1.0) as conn:
                # Oldest entry
                row = conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM cache"
                ).fetchone()
                if row and row[0] is not None:
                    stats["exact"]["oldest_age_seconds"] = round(time.time() - row[0], 1)
                    stats["exact"]["newest_age_seconds"] = round(time.time() - row[1], 1)
        except sqlite3.Error:
            pass

    sem = _get_default_semantic_cache()
    if sem is not None:
        stats["semantic"] = {
            "backend": sem.backend,
            "entries": len(sem),
            "threshold": sem._threshold,
        }
        try:
            with sqlite3.connect(sem._path, timeout=1.0) as conn:
                row = conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM semantic_cache"
                ).fetchone()
                if row and row[0] is not None:
                    stats["semantic"]["oldest_age_seconds"] = round(time.time() - row[0], 1)
                    stats["semantic"]["newest_age_seconds"] = round(time.time() - row[1], 1)
        except sqlite3.Error:
            pass

    return stats


# ============================ EVICTION ============================

def clear_cache() -> dict:
    """Wipe both exact and semantic caches. Returns a status dict."""
    result = {"exact_cleared": False, "semantic_cleared": False}
    store = _get_default_store()
    if hasattr(store, "clear"):
        try:
            store.clear()
            result["exact_cleared"] = True
        except Exception as e:
            log.debug("admin: exact clear failed: %s", e)

    sem = _get_default_semantic_cache()
    if sem is not None and hasattr(sem, "clear"):
        try:
            sem.clear()
            result["semantic_cleared"] = True
        except Exception as e:
            log.debug("admin: semantic clear failed: %s", e)

    return result


def evict_expired() -> dict:
    """Proactively delete TTL-expired entries from both cache tiers.

    Normal lookups already treat expired entries as misses, so they're
    harmless. But over time expired entries accumulate and eat disk
    space. Call this periodically (e.g. nightly cron) to reclaim space.

    Returns a count of evicted entries per backend.
    """
    result = {"exact_evicted": 0, "semantic_evicted": 0}

    # Exact cache: iterate, find expired, delete
    store = _get_default_store()
    if isinstance(store, SQLiteStore):
        result["exact_evicted"] = _evict_expired_sqlite(store._path, "cache", "key")
    elif isinstance(store, MemoryStore):
        # Best-effort for in-memory store
        result["exact_evicted"] = _evict_expired_memory(store)

    # Semantic cache: similar
    sem = _get_default_semantic_cache()
    if sem is not None:
        result["semantic_evicted"] = _evict_expired_sqlite(
            sem._path, "semantic_cache", "id", also_clean_vec=True
        )

    return result


def _evict_expired_sqlite(
    path: str,
    table: str,
    key_col: str,
    also_clean_vec: bool = False,
) -> int:
    """Walk a SQLite store, delete envelope-expired entries."""
    # Defense-in-depth: identifiers are interpolated into SQL below, so pin
    # them to the only values internal callers use.
    if table not in ("cache", "semantic_cache") or key_col not in ("key", "id"):
        raise ValueError(f"unexpected identifier: {table!r}/{key_col!r}")
    count = 0
    try:
        with sqlite3.connect(path, timeout=2.0) as conn:
            # We can't use SQL to check expiry because values are JSON-encoded
            # envelopes; we have to parse them in Python. For small caches
            # this is fine; for huge caches (millions of rows) consider an
            # index on a separate exp column in future versions.
            rows = conn.execute(f"SELECT {key_col}, value FROM {table}"
                                if table == "cache"
                                else f"SELECT {key_col}, response FROM {table}").fetchall()
            expired_keys = []
            for key, blob in rows:
                try:
                    val = json.loads(blob)
                except (TypeError, ValueError):
                    continue
                if is_expired(val):
                    expired_keys.append(key)
            if expired_keys:
                qmarks = ",".join("?" for _ in expired_keys)
                conn.execute(
                    f"DELETE FROM {table} WHERE {key_col} IN ({qmarks})",
                    expired_keys,
                )
                if also_clean_vec:
                    try:
                        conn.execute(
                            f"DELETE FROM semantic_vec WHERE rowid IN ({qmarks})",
                            expired_keys,
                        )
                    except sqlite3.Error:
                        pass
                conn.commit()
                count = len(expired_keys)
    except sqlite3.Error as e:
        log.debug("admin: SQL eviction failed: %s", e)
    return count


def _evict_expired_memory(store: MemoryStore) -> int:
    """Drop expired entries from an in-memory store via its PUBLIC maintenance
    API only.

    `evict_expired(predicate)` performs the sweep under the store's own lock, so
    this is safe against concurrent get/set AND requires no access to private
    state. A store that does not expose that public API is simply skipped — admin
    never reaches into a store's internals (`._data`, `._lock`, …). Custom stores
    that want admin-driven eviction implement `evict_expired(predicate) -> int`.
    """
    evict = getattr(store, "evict_expired", None)
    if not callable(evict):
        log.debug("admin: store %s exposes no public evict_expired(); skipping "
                  "memory eviction.", type(store).__name__)
        return 0
    try:
        return evict(is_expired)
    except Exception as e:
        log.debug("admin: memory eviction failed: %s", e)
        return 0


# ============================ EXPORT / IMPORT ============================
#
# Backup format (v1), JSONL with an integrity header:
#   line 1: {"__tokeymeter_export__": {"version":1,"tool_version":...,
#            "created_at":...,"record_count":N,"sha256":"<hash of record lines>"}}
#   lines 2..N+1: {"key":..., "value":<envelope>, "created_at":...}
# The header's sha256 covers the exact bytes of all record lines, so a truncated,
# corrupted, or tampered backup is detected on import BEFORE any state is touched.
# Legacy headerless backups still import (in merge mode) with integrity unverified.

_EXPORT_FORMAT_VERSION = 1
_HEADER_KEY = "__tokeymeter_export__"


def _tool_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def export_cache(path: str) -> dict:
    """Write the current exact cache to an integrity-checked JSONL backup.

    The file begins with a header line carrying a SHA-256 over all record lines
    plus the record count, so import_cache can verify the backup is intact and
    untampered before restoring. Written atomically (temp file + os.replace).

    Returns: {"supported": True, "exported": N, "path": ..., "sha256": ...,
              "format_version": 1}.
    """
    store = _get_default_store()
    if not isinstance(store, SQLiteStore):
        return {"supported": False, "reason": "only SQLiteStore is exportable"}

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_records = f"{path}.records.tmp"
    tmp_final = f"{path}.tmp"
    n = 0
    hasher = hashlib.sha256()
    try:
        with sqlite3.connect(store._path, timeout=2.0) as conn, open(tmp_records, "wb") as tf:
            for key, value, created_at in conn.execute(
                "SELECT key, value, created_at FROM cache"
            ):
                try:
                    parsed = json.loads(value)  # validate stored value
                    line = (json.dumps({
                        "key": key, "value": parsed, "created_at": created_at,
                    }) + "\n").encode("utf-8")
                except (TypeError, ValueError):
                    continue
                tf.write(line)
                hasher.update(line)
                n += 1
        digest = hasher.hexdigest()
        header = (json.dumps({_HEADER_KEY: {
            "version": _EXPORT_FORMAT_VERSION,
            "tool_version": _tool_version(),
            "created_at": time.time(),
            "record_count": n,
            "sha256": digest,
        }}) + "\n").encode("utf-8")
        with open(tmp_final, "wb") as out, open(tmp_records, "rb") as tf:
            out.write(header)
            shutil.copyfileobj(tf, out)
        os.replace(tmp_final, path)  # atomic publish
    except (sqlite3.Error, OSError) as e:
        for t in (tmp_records, tmp_final):
            try:
                os.remove(t)
            except OSError:
                pass
        return {"supported": True, "exported": n, "error": str(e), "path": path}
    finally:
        try:
            os.remove(tmp_records)
        except OSError:
            pass
    return {"supported": True, "exported": n, "path": path,
            "sha256": digest, "format_version": _EXPORT_FORMAT_VERSION}


def _read_export_header(path: str) -> Optional[dict]:
    """Return the export header dict if the file is the v1 format, else None."""
    try:
        with open(path, "rb") as f:
            first = f.readline()
    except OSError:
        return None
    try:
        obj = json.loads(first.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict) and _HEADER_KEY in obj:
        return obj[_HEADER_KEY]
    return None


def _iter_records(path: str, skip_header: bool):
    """Stream (record_dict | None, error_bool) over a backup's record lines."""
    with open(path, "r", encoding="utf-8") as f:
        if skip_header:
            f.readline()  # discard header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "key" not in rec:
                    raise KeyError("key")
                yield rec, False
            except (KeyError, json.JSONDecodeError):
                yield None, True


def import_cache(path: str, mode: str = "merge",
                 allow_integrity_mismatch: bool = False) -> dict:
    """Restore the exact cache from a backup produced by export_cache.

    Restore is *deliberate*: the backup's integrity is verified before any state
    is touched, and a corrupt/tampered backup is refused (so it can never wipe a
    live cache).

    mode:
      "merge"    (default) — set each record into the store, overwriting same keys.
      "replace"  — verify integrity, then CLEAR the store, then import. The clear
                   happens only AFTER integrity passes, so a bad backup never
                   destroys the existing cache.
      "validate" — dry run: verify integrity and parse every record, mutating
                   NOTHING. Use this to check a backup before trusting it.

    allow_integrity_mismatch: if True, proceed despite a failed checksum/count
      (e.g. to salvage a partially-corrupted backup). Default False = refuse.

    Returns a structured report including the integrity verdict and counts.
    """
    if mode not in ("merge", "replace", "validate"):
        raise ValueError(f"mode must be 'merge', 'replace', or 'validate', got {mode!r}")

    store = _get_default_store()
    header = _read_export_header(path)
    has_header = header is not None

    if not os.path.exists(path):
        return {"imported": 0, "skipped": 0, "mode": mode, "error": "file not found"}

    # ---- integrity pass (streaming, O(1) memory): verify BEFORE mutating ----
    integrity = "unverified_legacy"
    actual_count = 0
    actual_sha = None
    if has_header:
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                f.readline()  # skip header
                for raw in f:
                    hasher.update(raw)
                    if raw.strip():
                        actual_count += 1
        except OSError as e:
            return {"imported": 0, "skipped": 0, "mode": mode, "error": str(e)}
        actual_sha = hasher.hexdigest()
        exp_sha = header.get("sha256")
        exp_count = header.get("record_count")
        ok = (exp_sha == actual_sha) and (exp_count is None or exp_count == actual_count)
        integrity = "ok" if ok else "mismatch"
        if not ok and not allow_integrity_mismatch:
            return {
                "imported": 0, "skipped": 0, "mode": mode, "integrity": "mismatch",
                "aborted": True,
                "expected_sha256": exp_sha, "actual_sha256": actual_sha,
                "expected_count": exp_count, "actual_count": actual_count,
                "error": "integrity check failed; refusing to restore "
                         "(pass allow_integrity_mismatch=True to override)",
                "path": path,
            }

    # ---- validate (dry run): parse-check records, mutate nothing ----
    if mode == "validate":
        valid = invalid = 0
        for rec, err in _iter_records(path, has_header):
            if err:
                invalid += 1
            else:
                valid += 1
        return {
            "mode": "validate", "integrity": integrity, "aborted": False,
            "valid_records": valid, "invalid_records": invalid,
            "would_import": valid, "path": path,
            **({"format_version": header.get("version")} if has_header else {}),
        }

    # ---- replace: clear only AFTER integrity has been verified ----
    if mode == "replace":
        try:
            store.clear()
        except Exception as e:
            log.debug("admin: import_cache replace-clear failed: %s", e)

    # ---- apply pass (streaming) ----
    n = errors = 0
    for rec, err in _iter_records(path, has_header):
        if err:
            errors += 1
            continue
        key = rec["key"]
        value = rec.get("value")
        created_at = rec.get("created_at")
        try:
            if created_at is not None:
                try:
                    store.set(key, value, created_at=created_at)
                except TypeError:
                    store.set(key, value)
            else:
                store.set(key, value)
            n += 1
        except Exception:
            errors += 1
    return {"imported": n, "skipped": errors, "mode": mode,
            "integrity": integrity, "aborted": False, "path": path}
