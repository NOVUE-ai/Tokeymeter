"""Ledger crash consistency: torn-tail repair (found in enterprise stress pass).

A writer killed mid-write (SIGKILL, OOM, power loss) leaves a torn final line
with no trailing newline. Before the repair, the NEXT process's first append
concatenated onto that fragment — corrupting both lines and silently losing a
record: one crash cost one record, forever. The repair: each tracker instance,
before its first disk append, writes a newline if the file ends without one,
isolating the fragment on its own line (which the tolerant reader skips).

Pinned:
  1. sync path: post-crash appends land intact; zero records lost
  2. buffered path: same guarantee through _flush_locked
  3. clean files are untouched (no spurious blank lines)
  4. the reader tolerates the isolated fragment and reports correct totals
  5. an empty/missing file is a no-op (never raises)
"""
import json
import os
import tempfile

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.usage import set_reported_usage


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    home = tempfile.mkdtemp()
    ledger = os.path.join(home, "savings.jsonl")
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.set_savings_path(ledger)
    tokeymeter.reset_savings()
    yield ledger
    tokeymeter.set_buffered_savings(False)
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()


def _ask_factory():
    @tokeymeter.cache(model="m")
    def ask(p):
        set_reported_usage(100, 50)
        return "v"
    return ask


def _tear(ledger):
    """Simulate SIGKILL mid-write: truncated JSON, no trailing newline."""
    with open(ledger, "a", encoding="utf-8") as f:
        f.write('{"timestamp": 1.0, "model": "m", "hit": fal')


def _parse(ledger):
    good, torn = [], 0
    for line in open(ledger, encoding="utf-8").read().splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return good, torn


def test_sync_append_after_torn_tail_loses_nothing(_clean):
    ledger = _clean
    ask = _ask_factory()
    for i in range(10):
        ask(f"p{i}")
    _tear(ledger)
    # a crash means a NEW process: rebind → fresh tracker instance whose
    # first append must run the repair
    tokeymeter.set_savings_path(ledger)
    ask = _ask_factory()
    ask("post-1")
    ask("post-2")
    good, torn = _parse(ledger)
    assert torn == 1                      # fragment isolated, not compounded
    assert len(good) == 12                # ZERO records lost
    assert tokeymeter.savings_report()["total_calls"] == 12


def test_buffered_flush_after_torn_tail_loses_nothing(_clean):
    ledger = _clean
    tokeymeter.set_buffered_savings(True)
    ask = _ask_factory()
    for i in range(5):
        ask(f"b{i}")
    tokeymeter.flush_savings()
    _tear(ledger)
    tokeymeter.set_savings_path(ledger)   # fresh instance
    tokeymeter.set_buffered_savings(True)
    ask = _ask_factory()
    ask("post-1")
    ask("post-2")
    tokeymeter.flush_savings()
    good, torn = _parse(ledger)
    assert torn == 1
    assert len(good) == 7


def test_clean_file_gets_no_spurious_blank_line(_clean):
    ledger = _clean
    ask = _ask_factory()
    ask("a")
    tokeymeter.set_savings_path(ledger)   # fresh instance over a CLEAN file
    ask = _ask_factory()
    ask("b")
    raw = open(ledger, encoding="utf-8").read()
    assert "\n\n" not in raw              # repair must be a no-op on clean tails
    good, torn = _parse(ledger)
    assert (len(good), torn) == (2, 0)


def test_repair_is_noop_on_missing_and_empty_file(_clean):
    ledger = _clean
    # missing file: first-ever append creates it normally
    ask = _ask_factory()
    ask("first")
    good, torn = _parse(ledger)
    assert (len(good), torn) == (1, 0)
    # empty file: truncate, rebind, append — still clean
    open(ledger, "w").close()
    tokeymeter.set_savings_path(ledger)
    ask = _ask_factory()
    ask("after-empty")
    good, torn = _parse(ledger)
    assert (len(good), torn) == (1, 0)


def test_reader_reports_correctly_with_isolated_fragment(_clean):
    ledger = _clean
    ask = _ask_factory()
    for i in range(4):
        ask(f"p{i}")
    _tear(ledger)
    tokeymeter.set_savings_path(ledger)
    # no new writes: reader over (4 valid + 1 torn-without-newline)
    rep = tokeymeter.savings_report()
    assert rep["total_calls"] == 4        # fragment skipped, never raises
    cap = tokeymeter.capacity_recovery_report(measured_tokens_per_second=1400)
    assert cap["total_recovered"]["recovered_tokens_total"] == 0
