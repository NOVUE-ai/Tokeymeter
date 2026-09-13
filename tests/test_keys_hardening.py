"""Keys hardening (Phase-1, item 2) — billing logic under time and threads.

Budget-enforced keys are the enforcement valve of the Tokenomics engine:
money-adjacent logic where a lost update, a stuck month, or a warn storm is
an enterprise incident. Pinned here:

  2.1  UTC month rollover: spend/calls/warned-bands reset, cap re-arms —
       through EVERY public path (check, accrual, status), across the
       Dec→Jan year boundary, and end-to-end (blocked in July → succeeds
       on Aug 1) through the real decorator.
  2.2  32-thread concurrent accrual: call count exactly correct (lost-update
       detector), spend equal to the arithmetic sum.
  2.3  Enforcement contract (circuit-breaker, per keys.py docstring):
       sequential overshoot ≤ one call; K-way concurrent overshoot ≤ K
       in-flight calls; from the first check AFTER breach, every call is
       refused (breach latch), including under a 32-thread stampede.
  2.4  Soft-warn exactly once even when 32 threads cross the band together.
  2.5  Re-register mid-month preserves accrual and warned-bands; a new cap
       takes effect immediately; rollover afterward clears everything.

Plus one regression this file exists to guard forever: the soft-warn is
emitted OUTSIDE the module lock, so a degraded-bus subscriber that calls
back into keys APIs (key_status) must not deadlock.
"""
import threading
import time

import pytest

import tokeymeter
from tokeymeter import degraded as dg
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    K.clear_keys()
    dg.clear_subscribers()
    yield
    dg.clear_subscribers()
    K.clear_keys()
    tokeymeter.reset_savings()
    tokeymeter.set_in_memory_savings(False)


def _info(name):
    with K._lock:
        return K._KEYS[name]


def _age_to_past_month(name, month="2020-01"):
    """Simulate a key whose last activity was in a past UTC month."""
    with K._lock:
        K._KEYS[name].month = month


# ── 2.1 rollover ─────────────────────────────────────────────────────────
def test_rollover_primitive_resets_all_state_incl_year_boundary():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=10)
    info = _info("k")
    # Dec 31 23:59:59 UTC 2025 → Jan 1 00:00:01 UTC 2026 (year boundary)
    dec = time.mktime(time.strptime("2025-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"))
    jan = dec + 2
    K._roll_month(info, now=dec)
    with K._lock:
        info.spent_usd, info.calls = 9.0, 7
        info._warned_bands.add("soft")
    K._roll_month(info, now=dec)                 # same month → untouched
    assert (info.spent_usd, info.calls) == (9.0, 7)
    K._roll_month(info, now=jan)                 # year boundary → full reset
    assert (info.spent_usd, info.calls) == (0.0, 0)
    assert info._warned_bands == set()
    assert info.month == time.strftime("%Y-%m", time.gmtime(jan))


@pytest.mark.parametrize("path", ["check", "accrual", "status"])
def test_rollover_triggers_through_every_public_path(path):
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=10)
    with K._lock:
        K._KEYS["k"].spent_usd, K._KEYS["k"].calls = 9.5, 3
        K._KEYS["k"]._warned_bands.add("soft")
    _age_to_past_month("k")
    tok = K._CURRENT.set("k")
    try:
        if path == "check":
            K.check_current()                    # must roll, not raise-stale
        elif path == "accrual":
            K.on_spend("k", 0.25, hit=False, shadow=False)
        else:
            K.key_status("k")
    finally:
        K._CURRENT.reset(tok)
    st = K.key_status("k")
    assert st["month"] == time.strftime("%Y-%m", time.gmtime())
    expected = 0.25 if path == "accrual" else 0.0
    assert st["spent_usd"] == expected
    assert _info("k")._warned_bands == set()


def test_blocked_last_month_succeeds_this_month_end_to_end():
    """The user-visible promise: a key exhausted in July works on Aug 1 —
    through the real decorator, not module internals."""
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=0.001)
    computes = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        computes["n"] += 1
        return "x" * 4000
    with pytest.raises(tokeymeter.KeyBudgetExceeded):
        with tokeymeter.key("k"):
            for i in range(5):
                ask(f"july {i} " * 100)
    blocked_at = computes["n"]
    _age_to_past_month("k", "2020-07")           # ...a month passes
    with tokeymeter.key("k"):
        ask("august fresh " * 100)               # cap re-armed → succeeds
    assert computes["n"] == blocked_at + 1
    st = K.key_status("k")
    assert st["calls"] == 1 and st["spent_usd"] > 0


# ── 2.2 concurrent accrual exactness ─────────────────────────────────────
def test_32_thread_accrual_no_lost_updates():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=1e9)
    THREADS, PER, COST = 32, 250, 0.003
    barrier = threading.Barrier(THREADS)

    def worker():
        barrier.wait()                            # maximize contention
        for _ in range(PER):
            K.on_spend("k", COST, hit=False, shadow=False)
    ts = [threading.Thread(target=worker) for _ in range(THREADS)]
    [t.start() for t in ts]
    [t.join(timeout=30) for t in ts]
    assert not any(t.is_alive() for t in ts)
    st = K.key_status("k")
    assert st["calls"] == THREADS * PER           # ints: exact, no excuses
    assert st["spent_usd"] == pytest.approx(THREADS * PER * COST, rel=1e-9)


# ── 2.3 enforcement contract: circuit-breaker bounds + breach latch ──────
def test_sequential_overshoot_is_at_most_one_call():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=1.0)
    COST = 0.4
    tok = K._CURRENT.set("k")
    completed = 0
    try:
        with pytest.raises(K.KeyBudgetExceeded):
            for _ in range(10):
                K.check_current()
                K.on_spend("k", COST, hit=False, shadow=False)
                completed += 1
    finally:
        K._CURRENT.reset(tok)
    st = K.key_status("k")
    # spends 0.4, 0.8, 1.2(breach) → 4th check refuses: overshoot ≤ one call
    assert completed == 3
    assert st["spent_usd"] - 1.0 <= COST + 1e-9


def test_concurrent_overshoot_bounded_by_inflight_calls():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=1.0)
    THREADS, COST = 8, 0.5
    barrier = threading.Barrier(THREADS)
    outcomes = []
    lock = threading.Lock()

    def caller():
        tok = K._CURRENT.set("k")
        try:
            barrier.wait()
            K.check_current()                     # all may pass pre-breach
            time.sleep(0.01)                      # in-flight window
            K.on_spend("k", COST, hit=False, shadow=False)
            with lock:
                outcomes.append("spent")
        except K.KeyBudgetExceeded:
            with lock:
                outcomes.append("refused")
        finally:
            K._CURRENT.reset(tok)
    ts = [threading.Thread(target=caller) for _ in range(THREADS)]
    [t.start() for t in ts]
    [t.join(timeout=15) for t in ts]
    st = K.key_status("k")
    # documented bound: overshoot ≤ (in-flight at breach) × one call's cost
    assert st["spent_usd"] - 1.0 <= THREADS * COST + 1e-9
    assert len(outcomes) == THREADS
    # and the latch: once breached, fresh checks refuse — all of them
    errors = []
    def late():
        tok = K._CURRENT.set("k")
        try:
            K.check_current()
            errors.append("passed-after-breach")
        except K.KeyBudgetExceeded:
            pass
        finally:
            K._CURRENT.reset(tok)
    lts = [threading.Thread(target=late) for _ in range(32)]
    [t.start() for t in lts]
    [t.join(timeout=15) for t in lts]
    assert errors == [], errors


# ── 2.4 soft-warn exactly once under a thread storm ──────────────────────
def test_soft_warn_exactly_once_under_32_threads():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=100.0, soft_pct=0.5)
    K.on_spend("k", 60.0, hit=False, shadow=False)   # 60% ≥ soft band
    warns = []
    cb = dg.on_degraded(lambda e: warns.append(e.source)
                        if e.source == "keys" else None)
    try:
        barrier = threading.Barrier(32)
        def checker():
            tok = K._CURRENT.set("k")
            try:
                barrier.wait()
                K.check_current()
            finally:
                K._CURRENT.reset(tok)
        ts = [threading.Thread(target=checker) for _ in range(32)]
        [t.start() for t in ts]
        [t.join(timeout=15) for t in ts]
    finally:
        dg.off_degraded(cb)
    assert warns.count("keys") == 1, warns


def test_degraded_subscriber_calling_keys_api_does_not_deadlock():
    """Regression guard for the lock-held-across-callback hazard: the warn
    is emitted OUTSIDE the module lock, so a subscriber may safely call
    back into keys APIs."""
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=10.0, soft_pct=0.5)
    K.on_spend("k", 6.0, hit=False, shadow=False)
    seen = {}
    def nosy(e):
        if e.source == "keys":
            seen["status"] = K.key_status("k")    # would deadlock pre-fix
    cb = dg.on_degraded(nosy)
    try:
        done = threading.Event()
        def check():
            tok = K._CURRENT.set("k")
            try:
                K.check_current()
                done.set()
            finally:
                K._CURRENT.reset(tok)
        t = threading.Thread(target=check)
        t.start()
        t.join(timeout=5)
        assert done.is_set(), "check_current deadlocked in subscriber"
    finally:
        dg.off_degraded(cb)
    assert seen["status"]["spent_usd"] == 6.0


# ── 2.5 re-register semantics ────────────────────────────────────────────
def test_reregister_preserves_accrual_and_bands_then_rollover_clears():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=100.0, soft_pct=0.5)
    K.on_spend("k", 60.0, hit=False, shadow=False)
    tok = K._CURRENT.set("k")
    try:
        K.check_current()                          # trips the soft band
    finally:
        K._CURRENT.reset(tok)
    assert "soft" in _info("k")._warned_bands
    K.register_key("k", monthly_cap_usd=200.0)     # mid-month cap change
    st = K.key_status("k")
    assert st["spent_usd"] == 60.0 and st["calls"] == 1     # accrual kept
    assert "soft" in _info("k")._warned_bands               # bands kept
    assert st["monthly_cap_usd"] == 200.0
    _age_to_past_month("k")
    st2 = K.key_status("k")                        # rollover clears all
    assert st2["spent_usd"] == 0.0 and st2["calls"] == 0
    assert _info("k")._warned_bands == set()


def test_reregister_new_cap_takes_effect_immediately():
    K.register_key("k", "val-abcdefgh", monthly_cap_usd=1.0)
    K.on_spend("k", 5.0, hit=False, shadow=False)  # far over old cap
    tok = K._CURRENT.set("k")
    try:
        with pytest.raises(K.KeyBudgetExceeded):
            K.check_current()                      # old cap refuses
        K.register_key("k", monthly_cap_usd=50.0)  # raise the cap
        K.check_current()                          # new cap admits
        K.register_key("k", monthly_cap_usd=2.0)   # lower below spend
        with pytest.raises(K.KeyBudgetExceeded):
            K.check_current()                      # refuses again
    finally:
        K._CURRENT.reset(tok)
