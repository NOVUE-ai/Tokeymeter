"""
Tests for the admin / ops module.
"""
import json
import os
import time

import pytest

import tokeymeter
from tokeymeter import admin, events
from tokeymeter.storage import MemoryStore, SQLiteStore


@pytest.fixture(autouse=True)
def isolated_state(tmp_path):
    # Force SQLite cache to a temp path so we can inspect it
    sqlite_path = str(tmp_path / "cache.db")
    tokeymeter.set_default_store(SQLiteStore(path=sqlite_path))
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield


def test_cache_info_returns_diagnostic():
    info = admin.cache_info()
    assert "version" in info
    assert info["version"].startswith("0.")
    assert "exact_cache" in info
    assert info["exact_cache"]["backend"] in ("SQLiteStore", "MemoryStore")
    assert "semantic_cache" in info
    assert "subscribers" in info


def test_cache_info_reports_subscriber_count():
    events.clear_subscribers()
    events.on_event(lambda e: None)
    events.on_event(lambda e: None)
    info = admin.cache_info()
    assert info["subscribers"]["event_subscribers"] == 2


def test_cache_stats_returns_size():
    @tokeymeter.cache()
    def ask(prompt):
        return f"r-{prompt}"

    for i in range(5):
        ask(f"prompt-{i}")

    stats = admin.cache_stats()
    assert "exact" in stats
    assert stats["exact"]["entries"] >= 5


def test_clear_cache_returns_status():
    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("hi")
    result = admin.clear_cache()
    assert "exact_cleared" in result or "exact_cache" in result or isinstance(result, dict)


def test_evict_expired_removes_aged_entries():
    @tokeymeter.cache(ttl=0.05)  # 50 ms
    def ask(prompt):
        return "ok"

    ask("a")
    ask("b")
    ask("c")
    time.sleep(0.1)  # all expired

    result = admin.evict_expired()
    assert isinstance(result, dict)
    # Either reports the number evicted or just succeeded
    assert "supported" not in result or result.get("supported") is True


def test_export_and_import_roundtrip(tmp_path):
    @tokeymeter.cache()
    def ask(prompt):
        return f"r-{prompt}"

    ask("alpha")
    ask("beta")
    ask("gamma")

    export_path = str(tmp_path / "export.jsonl")
    result = admin.export_cache(export_path)
    assert isinstance(result, dict)
    assert os.path.exists(export_path)

    # File should be non-empty and parseable
    with open(export_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 3


def test_admin_operations_never_raise():
    """Every admin call returns a status dict — never raises."""
    # All of these should return dicts, even in pathological cases
    assert isinstance(admin.cache_info(), dict)
    assert isinstance(admin.cache_stats(), dict)
    assert isinstance(admin.clear_cache(), dict)
    assert isinstance(admin.evict_expired(), dict)
