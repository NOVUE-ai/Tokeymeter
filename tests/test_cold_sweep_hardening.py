"""Regression tests for findings from the full cold sweep.

  - SavingsTracker wrote ~/.tokeymeter/savings.jsonl append-only with no cap, so
    a high-volume service would grow it without bound and fill the disk. It is
    now trimmed to a rolling window bounded by max_bytes.
  - degraded.on_degraded had no per-callback removal (only clear_subscribers),
    so dynamic subscription could grow the list without bound. off_degraded()
    now mirrors events.off_event().
"""
import dataclasses
import os
import tempfile

import pytest

from tokeymeter.savings import SavingsTracker, CallRecord
from tokeymeter import degraded


def _mkrec(i):
    vals = {}
    for f in dataclasses.fields(CallRecord):
        t = str(f.type)
        vals[f.name] = i if "int" in t else (0.001 if "float" in t else
                       (False if "bool" in t else f"val{i}"))
    return CallRecord(**vals)


@pytest.mark.io_heavy
def test_savings_log_is_size_bounded():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "savings.jsonl")
    log = SavingsTracker(path=p, max_bytes=200_000)
    for i in range(60_000):
        log.record(_mkrec(i))
    size = os.path.getsize(p)
    # bounded to roughly the cap (trim keeps most-recent half; ~10% overshoot)
    assert size <= int(200_000 * 1.2), f"savings log not bounded: {size} bytes"


@pytest.mark.io_heavy
def test_savings_report_works_after_trim():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "savings.jsonl")
    log = SavingsTracker(path=p, max_bytes=100_000)
    for i in range(40_000):
        log.record(_mkrec(i))
    r = log.report()
    assert isinstance(r, dict)  # report still parses the trimmed rolling window


def test_savings_record_never_raises_into_caller():
    d = tempfile.mkdtemp()
    log = SavingsTracker(path=os.path.join(d, "s.jsonl"), max_bytes=10_000)
    # an unserializable field must be swallowed, not raised
    class Weird:
        pass
    rec = _mkrec(1)
    object.__setattr__(rec, "model", Weird())  # force a json failure
    log.record(rec)  # must not raise


def test_off_degraded_removes_subscriber():
    degraded.clear_subscribers()
    seen = []
    cb = seen.append
    degraded.on_degraded(cb)
    degraded.emit_degraded("test_source", RuntimeError("x"))
    assert len(seen) == 1
    degraded.off_degraded(cb)
    degraded.emit_degraded("test_source", RuntimeError("y"))
    assert len(seen) == 1  # no longer receiving after removal
    degraded.off_degraded(cb)  # idempotent, no raise
    degraded.clear_subscribers()
