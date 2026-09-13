"""Regression tests for P1: audit ledger dropped entries under queue pressure.

Audit finding (Codex enterprise probe): with a small queue and a rapid burst,
the ledger appended some entries and DROPPED the rest (put_nowait + queue.Full).
Drops were counted/warned, but "every AI action is a provable decision" was not
guaranteed under saturation.

Fix: an opt-in durability mode (SecurityPolicy.require_audit_durability, or
AuditLog(durable=True)) that never silently drops — it applies bounded
backpressure (block-with-timeout) and, if the async queue is still saturated
(stalled flusher), writes the entry through SYNCHRONOUSLY via the same chain-safe
path. The non-blocking mode remains the performance default.
"""
import os
import tempfile
import time

import tokeymeter
from tokeymeter.audit.log import AuditLog
from tokeymeter.policy import set_security_policy, get_security_policy


def _mklog(durable, queue_max_size=5, timeout=5.0):
    d = tempfile.mkdtemp()
    return AuditLog(
        path=os.path.join(d, "audit.db"),
        install_secret_path=os.path.join(d, "secret"),
        signing_key_path=os.path.join(d, "key"),
        queue_max_size=queue_max_size,
        flush_interval_seconds=0.05,
        durable=durable,
        durability_timeout_seconds=timeout,
    )


def _burst(log, n=500):
    for i in range(n):
        log.append(decision_type="cache_hit", prompt_text=f"p{i}", model="gpt-4o")
    log.flush(timeout=10.0)
    time.sleep(0.3)
    log.flush(timeout=10.0)


def test_nondurable_default_can_drop_under_burst():
    """Documents the default (performance) behavior: under extreme pressure with
    a tiny queue, entries may drop and are counted."""
    log = _mklog(durable=False)
    _burst(log, 500)
    s = log.stats()
    # The whole point of the probe: without durability, drops happen and are visible.
    assert s["entries_dropped_queue_full"] > 0
    assert len(log.get_entries()) < 500


def test_durable_mode_never_drops_under_burst():
    """The fix: durability guarantees every entry is persisted under the same
    burst that dropped 400+ without it."""
    log = _mklog(durable=True)
    _burst(log, 500)
    s = log.stats()
    assert s["entries_dropped_queue_full"] == 0, "durable mode must not drop"
    assert len(log.get_entries()) == 500, "every decision must be recorded"


def test_durable_chain_still_verifies():
    """Durability must not corrupt the hash chain (synchronous and async writes
    are serialized on the same lock)."""
    log = _mklog(durable=True)
    _burst(log, 500)
    result = log.verify_chain()
    ok = getattr(result, "ok", getattr(result, "valid", None))
    assert ok is True
    assert len(log.get_entries()) == 500


def test_synchronous_write_through_when_flusher_stalls():
    """Safety net: if the async flusher cannot drain (simulated stall), overflow
    entries are written through synchronously rather than dropped."""
    log = _mklog(durable=True, queue_max_size=2, timeout=0.2)
    log._stop.set()           # simulate a dead/stalled flusher
    time.sleep(0.1)
    for i in range(10):
        log.append(decision_type="cache_hit", prompt_text=f"q{i}", model="gpt-4o")
    s = log.stats()
    assert s["entries_dropped_queue_full"] == 0
    assert s["entries_written_synchronously"] >= 8  # overflow persisted, not lost


def test_durability_resolved_from_security_policy():
    """Durability can be required globally via SecurityPolicy, without passing
    durable= to each AuditLog (so enterprise_defaults() reaches the ledger)."""
    set_security_policy(require_audit_durability=True)
    assert get_security_policy().require_audit_durability is True
    log = _mklog(durable=None)  # defer to policy
    _burst(log, 300)
    assert log.stats()["entries_dropped_queue_full"] == 0
    assert len(log.get_entries()) == 300


def test_enterprise_defaults_enables_audit_durability():
    """enterprise_defaults(require_audit_durability=True) flips the policy flag."""
    tokeymeter.enterprise_defaults(require_audit_durability=True)
    assert get_security_policy().require_audit_durability is True
