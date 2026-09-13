"""Regression tests for admin/restore safety (#7).

export_cache now writes an integrity header (SHA-256 over record lines + count),
and import_cache verifies it BEFORE touching state, supports explicit restore
modes, and — critically — refuses a corrupt/tampered backup so it can never wipe
a live cache.
"""
import os
import sqlite3
import tempfile
import time

import tokeymeter
from tokeymeter import admin
from tokeymeter.storage import SQLiteStore


def _store_with(n=20):
    d = tempfile.mkdtemp()
    store = SQLiteStore(path=os.path.join(d, "cache.db"))
    tokeymeter.set_default_store(store)
    for i in range(n):
        store.set(f"k{i}", {"v": i})
    return store, os.path.join(d, "backup.jsonl")


def test_export_writes_integrity_header():
    store, exp = _store_with()
    r = admin.export_cache(exp)
    assert r["exported"] == 20
    assert r["format_version"] == 1
    assert "sha256" in r
    with open(exp) as f:
        assert "__tokeymeter_export__" in f.readline()


def test_validate_mode_mutates_nothing():
    store, exp = _store_with()
    admin.export_cache(exp)
    store.set("sentinel", {"keep": True})
    r = admin.import_cache(exp, mode="validate")
    assert r["mode"] == "validate"
    assert r["integrity"] == "ok"
    assert r["valid_records"] == 20
    assert r["aborted"] is False
    assert store.get("sentinel") is not None  # nothing was touched


def test_merge_preserves_created_at_and_integrity_ok():
    store, exp = _store_with()
    admin.export_cache(exp)

    def created_at(key):
        with sqlite3.connect(store._path) as c:
            row = c.execute("SELECT created_at FROM cache WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    orig = created_at("k0")
    time.sleep(0.15)
    r = admin.import_cache(exp, mode="merge")
    assert r["imported"] == 20
    assert r["integrity"] == "ok"
    assert abs(created_at("k0") - orig) < 0.001


def test_replace_clears_then_imports():
    store, exp = _store_with()
    admin.export_cache(exp)
    store.set("stale_key", {"old": True})
    r = admin.import_cache(exp, mode="replace")
    assert r["integrity"] == "ok"
    assert r["aborted"] is False
    assert store.get("stale_key") is None   # cleared
    assert store.get("k0") is not None       # restored


def test_tampered_backup_aborts_and_does_not_wipe():
    """The decisive safety property: a corrupt/tampered backup must be refused,
    and a `replace` restore must NOT clear the live cache when it aborts."""
    store, exp = _store_with()
    admin.export_cache(exp)
    with open(exp, "a") as f:
        f.write('{"key":"injected","value":"evil"}\n')  # breaks sha + count

    before = len(store)
    r = admin.import_cache(exp, mode="replace")
    assert r["integrity"] == "mismatch"
    assert r["aborted"] is True
    assert len(store) == before              # NOT wiped
    assert store.get("injected") is None     # tampered record not applied


def test_integrity_override_allows_salvage():
    store, exp = _store_with()
    admin.export_cache(exp)
    with open(exp, "a") as f:
        f.write('{"key":"extra","value":{"v":99}}\n')
    r = admin.import_cache(exp, mode="merge", allow_integrity_mismatch=True)
    assert r["aborted"] is False
    assert r["integrity"] == "mismatch"
    assert r["imported"] == 21


def test_legacy_headerless_backup_still_imports():
    tokeymeter.set_default_store(SQLiteStore(path=os.path.join(tempfile.mkdtemp(), "c.db")))
    legacy = os.path.join(tempfile.mkdtemp(), "legacy.jsonl")
    with open(legacy, "w") as f:
        f.write('{"key":"a","value":{"x":1}}\n{"key":"b","value":{"x":2}}\n')
    r = admin.import_cache(legacy, mode="merge")
    assert r["imported"] == 2
    assert r["integrity"] == "unverified_legacy"
    assert r["aborted"] is False


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        admin.import_cache("/nonexistent", mode="bogus")
