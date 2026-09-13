"""
P0.1 — configurable Tokeymeter home + ledger health.

The savings ledger must never silently fail. In CI, containers, and locked-down
machines the default ~/.tokeymeter may be unwritable; the ledger must degrade
visibly (health + event), never crash the caller, and never return zero without
explaining why. Paths must be overridable by env var and programmatically.

    PYTHONPATH=. python -m pytest tests/test_savings_ledger_health.py -q
"""
import os
import tempfile
import time

import tokeymeter as tk
from tokeymeter import degraded, paths
from tokeymeter.savings import SavingsTracker, CallRecord


def _rec(cost=0.01, hit=True):
    return CallRecord(time.time(), "gpt-4o", hit, "exact" if hit else None,
                      100, 50, cost, 5.0)


def test_custom_writable_path_records_metrics():
    d = tempfile.mkdtemp()
    t = SavingsTracker(path=os.path.join(d, "nested", "savings.jsonl"))  # dir auto-created
    t.record(_rec())
    h = t.ledger_health()
    assert h["records_written"] == 1
    assert h["writable"] is True and h["degraded"] is False
    assert t.report()["estimated_saved_usd"] > 0


def test_report_detail_exposes_ledger_health():
    d = tempfile.mkdtemp()
    tk.set_savings_path(os.path.join(d, "savings.jsonl"))
    try:
        rep = tk.report(detail=True)
        assert "ledger_health" in rep
        assert rep["ledger_health"]["path"].endswith("savings.jsonl")
        assert rep["ledger_health"]["writable"] is True
    finally:
        tk.set_savings_path(None)


def test_env_var_resolution():
    d = tempfile.mkdtemp()
    prior = os.environ.get("TOKEYMETER_SAVINGS_PATH")
    os.environ["TOKEYMETER_SAVINGS_PATH"] = os.path.join(d, "viaenv.jsonl")
    try:
        tk.set_savings_path(None)  # clear explicit override → env wins
        assert paths.savings_path().endswith("viaenv.jsonl")
    finally:
        if prior is None:
            os.environ.pop("TOKEYMETER_SAVINGS_PATH", None)
        else:
            os.environ["TOKEYMETER_SAVINGS_PATH"] = prior


def test_home_env_var_resolution():
    d = tempfile.mkdtemp()
    prior = os.environ.get("TOKEYMETER_HOME")
    os.environ["TOKEYMETER_HOME"] = d
    try:
        tk.set_home(None)
        tk.set_savings_path(None)
        assert paths.home() == d
        assert paths.savings_path() == os.path.join(d, "savings.jsonl")
    finally:
        if prior is None:
            os.environ.pop("TOKEYMETER_HOME", None)
        else:
            os.environ["TOKEYMETER_HOME"] = prior
        tk.set_home(None)


def test_unwritable_home_emits_degraded_event_without_crashing():
    # file-as-parent forces OSError on mkdir/open for ANY uid (incl. root) —
    # a path cannot be nested under a regular file.
    f = tempfile.mktemp()
    with open(f, "w") as fh:
        fh.write("x")

    events = []
    degraded.clear_subscribers()
    degraded.on_degraded(lambda e: events.append(e))
    try:
        # construction must NOT raise even though the parent dir can't be made
        t = SavingsTracker(path=os.path.join(f, "savings.jsonl"))
        assert t.ledger_health()["degraded"] is True

        # a write must NOT raise into the caller, and must be counted
        t.record(_rec(hit=False))
        h = t.ledger_health()
        assert h["writable"] is False
        assert h["write_failures"] >= 1
        assert h["last_error_type"] is not None

        # at least one degraded event was emitted
        assert len(events) >= 1
        assert any("savings" in e.source for e in events)
    finally:
        degraded.clear_subscribers()


def test_unwritable_home_falls_back_to_memory_so_metrics_survive():
    # When the disk is unwritable, the ledger keeps metrics in RAM (durability is
    # impossible there anyway) so report() still reflects the session, rather than
    # silently dropping records. The degraded event still fires.
    f = tempfile.mktemp()
    with open(f, "w") as fh:
        fh.write("x")
    events = []
    degraded.clear_subscribers()
    degraded.on_degraded(lambda e: events.append(e))
    try:
        t = SavingsTracker(path=os.path.join(f, "savings.jsonl"))
        for _ in range(10):
            t.record(_rec(cost=0.01))
        rep = t.report()
        h = t.ledger_health()
        assert rep["total_calls"] == 10            # metrics survived in memory
        assert h["records_written"] == 10
        assert h["writable"] is False and h["degraded"] is True
        assert h["in_memory"] is True              # auto-fell-back
        assert len(events) >= 1                     # operator was told
        assert not os.path.exists(os.path.join(f, "savings.jsonl"))  # nothing on disk
    finally:
        degraded.clear_subscribers()


def test_report_never_returns_silent_zero_on_failure():
    # an unwritable ledger returns a valid (zero) report whose health explains it
    f = tempfile.mktemp()
    with open(f, "w") as fh:
        fh.write("x")
    t = SavingsTracker(path=os.path.join(f, "savings.jsonl"))
    rep = t.report()
    assert rep["total_calls"] == 0          # zero, but...
    assert t.ledger_health()["degraded"] is True  # ...health says why
