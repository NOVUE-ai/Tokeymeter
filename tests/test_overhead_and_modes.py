"""
P0.3 — cache-hit latency: instrumentation, doctor, and write modes.

Verifies the meter's own overhead is measured and near-invisible (p50 < 1ms),
and that the buffered and in-memory write modes are correct and durable: buffered
batches writes off the hot path and never loses records across a flush/exit, and
in-memory keeps everything in RAM (no disk) while report() still works.

    PYTHONPATH=. python -m pytest tests/test_overhead_and_modes.py -q
"""
import os
import tempfile
import time

import tokeymeter as tk
from tokeymeter import overhead
import tokeymeter.savings as sv
from tokeymeter.savings import CallRecord


def _rec(cost=0.001, hit=True):
    return CallRecord(time.time(), "gpt-4o", hit, "exact" if hit else None,
                      5, 2, cost, 0.1)


def _restore_sync():
    tk.set_in_memory_savings(False)
    tk.set_buffered_savings(False)


def test_doctor_reports_cache_hit_overhead_under_1ms():
    overhead.reset()
    tk.set_savings_path(os.path.join(tempfile.mkdtemp(), "s.jsonl"))
    _restore_sync()

    @tk.cache(model="gpt-4o-mini", prompt_arg="p")
    def ask(p):
        return "answer"

    ask(p="hi")                       # prime
    for _ in range(3000):             # cache hits → overhead samples
        ask(p="hi")

    doc = tk.doctor()
    oh = doc["cache_hit_overhead"]
    assert oh["samples"] >= 2000
    assert oh["p50_ms"] is not None
    # the meter's own cache-hit overhead must be well under 1ms on normal hardware
    assert oh["p50_ms"] < 1.0, f"p50 overhead {oh['p50_ms']}ms exceeds 1ms"
    assert doc["savings_mode"] == "sync"


def test_buffered_mode_batches_and_loses_nothing():
    tk.set_savings_path(os.path.join(tempfile.mkdtemp(), "buf.jsonl"))
    try:
        tk.set_buffered_savings(True, buffer_size=1000)   # large buffer: stays pending
        path = tk.savings_ledger_health()["path"]
        for _ in range(50):
            sv._tracker.record(_rec())
        before = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        assert before == 0                                 # nothing written yet (buffered)
        assert tk.savings_ledger_health()["buffer_pending"] == 50
        tk.flush_savings()                                 # == what atexit does on exit
        after = sum(1 for _ in open(path))
        assert after == 50                                 # all 50 persisted in one batch
        assert tk.savings_ledger_health()["records_written"] == 50
    finally:
        _restore_sync()


def test_buffered_mode_auto_flushes_at_threshold():
    tk.set_savings_path(os.path.join(tempfile.mkdtemp(), "buf2.jsonl"))
    try:
        tk.set_buffered_savings(True, buffer_size=10)      # flush every 10
        path = tk.savings_ledger_health()["path"]
        for _ in range(25):
            sv._tracker.record(_rec())
        # 25 records, threshold 10 → 20 auto-flushed, 5 pending
        on_disk = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        assert on_disk == 20
        assert tk.savings_ledger_health()["buffer_pending"] == 5
    finally:
        _restore_sync()


def test_in_memory_mode_uses_no_disk_but_report_works():
    p = os.path.join(tempfile.mkdtemp(), "mem.jsonl")
    tk.set_savings_path(p)
    try:
        tk.set_in_memory_savings(True)
        for _ in range(30):
            sv._tracker.record(_rec(cost=0.002))
        rep = tk.report()
        assert rep["total_calls"] == 30
        assert round(rep["estimated_saved_usd"], 4) == round(30 * 0.002, 4)
        # nothing was written to disk
        assert not os.path.exists(tk.savings_ledger_health()["path"])
        assert tk.savings_ledger_health()["in_memory"] is True
    finally:
        _restore_sync()


def test_write_mode_survives_path_change():
    tk.set_savings_path(os.path.join(tempfile.mkdtemp(), "a.jsonl"))
    try:
        tk.set_buffered_savings(True, buffer_size=1000)
        # change the path AFTER enabling buffered — mode must persist
        tk.set_savings_path(os.path.join(tempfile.mkdtemp(), "b.jsonl"))
        assert tk.savings_ledger_health()["buffered"] is True
    finally:
        _restore_sync()


def test_overhead_module_records_and_percentiles():
    overhead.reset()
    for ms in [0.1, 0.2, 0.3, 0.4, 0.5]:
        overhead.record(ms)
    p = overhead.percentiles()
    assert p["samples"] == 5
    assert p["p50_ms"] is not None and p["p99_ms"] is not None
    assert p["p50_ms"] <= p["p99_ms"]


def test_doctor_cli_subcommand_runs():
    # `python -m tokeymeter doctor` must exist and emit machine-readable JSON
    import subprocess
    import sys
    import json
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    r = subprocess.run([sys.executable, "-m", "tokeymeter", "doctor"],
                       capture_output=True, encoding="utf-8", env=env, timeout=30)
    assert r.returncode == 0
    assert "cache-hit overhead" in r.stdout
    # last line is JSON
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert "cache_hit_overhead" in payload and "savings_mode" in payload


def test_buffered_mode_no_loss_on_clean_exit():
    # buffered records must survive a clean process exit (atexit flush)
    import subprocess
    import sys
    import tempfile as _tf
    ledger = os.path.join(_tf.mkdtemp(), "s.jsonl")
    child = (
        "import os\n"
        f"os.environ['TOKEYMETER_SAVINGS_PATH']={ledger!r}\n"
        "import tokeymeter as tk\n"
        "tk.set_buffered_savings(True, buffer_size=10000)\n"
        "@tk.cache(model='gpt-4o-mini', prompt_arg='p')\n"
        "def ask(p): return 'a'\n"
        "ask(p='x')\n"
        "[ask(p='x') for _ in range(40)]\n"
        # rely only on atexit
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    subprocess.run([sys.executable, "-c", child], env=env, check=True, timeout=30)
    n = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
    assert n >= 40, f"buffered records lost on clean exit: {n}"
