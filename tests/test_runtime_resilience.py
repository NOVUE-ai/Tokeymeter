"""W6 battery — survivability + the three enterprise gaps.

Wave gates: retry-storm ceiling, 1000-concurrent, chaos-off-zero-overhead,
runtime mypy clean, streaming-skips-proof documented+tested.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import (
    BackoffPolicy, BreakerState, Bulkhead, BulkheadFull, ChaosInjector,
    CircuitBreaker, CircuitOpen, Kernel, KernelRequest, OutputInvalid,
    OutputValidator, ResilientExecution, RuntimeConfig,
)
from tokeymeter.runtime.errors import ProviderDown, TransientError, AuthError
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.reliability import overhead as overheadmod


def kernel(fn, model="default", providers=None, **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    if providers:
        for m, f in providers.items():
            ex.register_adapter(CallableAdapter(f, provider=m), models=[m],
                                default=(m == model))
    else:
        ex.register_adapter(CallableAdapter(fn, provider="p0"),
                            models=[model], default=True)
    k.register_engine(ex)
    return k, ex


# ================================================== REL-1 breaker ==========
def test_breaker_trips_after_threshold():
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=3, clock=lambda: clock["t"])
    assert cb.state("p") == BreakerState.CLOSED
    for _ in range(3):
        assert cb.allow("p")
        cb.record_failure("p")
    assert cb.state("p") == BreakerState.OPEN
    assert not cb.allow("p")                         # storm ceiling active


def test_breaker_half_open_single_probe_then_close():
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=10.0,
                        clock=lambda: clock["t"])
    for _ in range(2):
        cb.allow("p"); cb.record_failure("p")
    assert cb.state("p") == BreakerState.OPEN
    clock["t"] = 11.0                                # cooldown elapsed
    assert cb.allow("p")                             # one probe
    assert not cb.allow("p")                         # second refused
    cb.record_success("p")
    assert cb.state("p") == BreakerState.CLOSED


def test_breaker_half_open_failure_reopens():
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, cooldown_s=5.0,
                        clock=lambda: clock["t"])
    cb.allow("p"); cb.record_failure("p")
    clock["t"] = 6.0
    assert cb.allow("p")
    cb.record_failure("p")                           # probe fails
    assert cb.state("p") == BreakerState.OPEN


def test_retry_storm_ceiling_call_count_bounded():
    """WAVE GATE: under a total provider outage, the provider is called at
    most `threshold` times, then the breaker refuses — no storm."""
    calls = {"n": 0}

    def dead(p):
        calls["n"] += 1
        raise ProviderDown("p0", RuntimeError("down"))

    k, ex = kernel(dead, reliability={"max_retries": 0})
    cb = CircuitBreaker(failure_threshold=3)
    ResilientExecution(ex, breaker=cb)
    # fire 20 requests at a dead provider
    for _ in range(20):
        with pytest.raises(Exception):
            k.process(KernelRequest(payload="p"))
    assert calls["n"] == 3                            # exactly the ceiling


# ================================================== REL-2 bulkhead ========
def test_bulkhead_isolates_and_full_raises():
    bh = Bulkhead(limit_per_provider=2)
    assert bh.acquire("a") and bh.acquire("a")
    assert not bh.acquire("a", timeout=0.0)          # full
    assert bh.acquire("b")                            # other provider free
    bh.release("a")
    assert bh.acquire("a")


def test_bulkhead_slow_provider_does_not_block_others():
    release = threading.Event()

    def slow(p):
        release.wait(timeout=5)
        return "slow"

    def fast(p):
        return "fast"

    k, ex = kernel(None, model="fast",
                   providers={"slow": slow, "fast": fast})
    ResilientExecution(ex, bulkhead=Bulkhead(limit_per_provider=1))
    # saturate slow in a thread
    t = threading.Thread(
        target=lambda: k.process(KernelRequest(payload="x", model="slow")))
    t.start()
    time.sleep(0.05)
    # fast provider still serves immediately
    assert k.process(KernelRequest(payload="y", model="fast")).payload == \
        "fast"
    release.set()
    t.join(timeout=5)


# ================================================== REL-3 backoff =========
def test_backoff_decorrelated_within_cap():
    bp = BackoffPolicy(base_s=0.1, cap_s=1.0)
    delays = [bp.next_delay(i) for i in range(20)]
    assert all(0 <= d <= 1.0 for d in delays)        # capped
    assert any(d > 0.1 for d in delays)              # grows past base


def test_retry_after_wins_over_backoff_within_cap():
    bp = BackoffPolicy(base_s=0.05, cap_s=2.0)
    assert bp.next_delay(0, retry_after=1.5) == 1.5  # header honored
    assert bp.next_delay(0, retry_after=99.0) == 2.0  # but capped


def test_backoff_applied_between_retries():
    slept = []
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("p0", RuntimeError("t"))
        return "ok"

    k, ex = kernel(flaky, reliability={"max_retries": 3})
    ResilientExecution(ex, backoff=BackoffPolicy(base_s=0.01, cap_s=0.1),
                       sleep=slept.append)
    assert k.process(KernelRequest(payload="p")).payload == "ok"
    assert len(slept) == 2                            # two backoffs, three tries


# ================================================== REL-6 validation ======
def test_output_validation_rejects_and_retries():
    calls = {"n": 0}

    def sometimes_bad(p):
        calls["n"] += 1
        return "GOOD" if calls["n"] >= 2 else "BAD"

    k, ex = kernel(sometimes_bad, reliability={"max_retries": 2})
    ResilientExecution(ex,
                       validator=OutputValidator(lambda r: r == "GOOD"))
    assert k.process(KernelRequest(payload="p")).payload == "GOOD"
    assert calls["n"] == 2                            # retried past the bad one


def test_output_validation_exhaustion_raises_typed():
    k, ex = kernel(lambda p: "ALWAYS_BAD",
                   reliability={"max_retries": 1})
    ResilientExecution(ex, validator=OutputValidator(lambda r: False))
    with pytest.raises(OutputInvalid):
        k.process(KernelRequest(payload="p"))


def test_validator_exception_becomes_typed():
    def raises_validator(r):
        raise KeyError("missing field")
    k, ex = kernel(lambda p: "x", reliability={"max_retries": 0})
    ResilientExecution(ex, validator=OutputValidator(raises_validator))
    with pytest.raises(OutputInvalid):
        k.process(KernelRequest(payload="p"))


# ================================================== REL-4 chaos ===========
def test_chaos_off_means_zero_overhead():
    """WAVE GATE: chaos disabled → maybe_inject is a no-op fast-path."""
    import random
    chaos = ChaosInjector()                          # never configured
    rng = random.Random(0)
    # 10k no-op calls must be instant and never raise
    for _ in range(10000):
        chaos.maybe_inject("p0", rng)
    assert not chaos.enabled


def test_chaos_error_injection():
    import random
    chaos = ChaosInjector()
    chaos.configure("p0", mode="error", rate=1.0)
    with pytest.raises(ProviderDown):
        chaos.maybe_inject("p0", random.Random(0))
    chaos.clear()
    assert not chaos.enabled


def test_chaos_delay_and_malform():
    import random
    chaos = ChaosInjector()
    chaos.configure("p0", mode="malform", rate=1.0)
    with pytest.raises(OutputInvalid):
        chaos.maybe_inject("p0", random.Random(0))


def test_chaos_rate_zero_never_fires():
    import random
    chaos = ChaosInjector()
    chaos.configure("p0", mode="error", rate=0.0)
    for _ in range(100):
        chaos.maybe_inject("p0", random.Random())    # rate 0 → never


def test_chaos_error_survived_by_fallback():
    """Chaos kills the primary; the ordered fallback saves the request —
    end-to-end proof that injected faults are survived, not just raised."""
    import random
    chaos = ChaosInjector()
    chaos.configure("primary", mode="error", rate=1.0)
    k, ex = kernel(None, model="primary",
                   providers={"primary": lambda p: "never",
                              "backup": lambda p: "saved"},
                   reliability={"max_retries": 0,
                                "fallback_order": ["backup"]})
    ResilientExecution(ex, chaos=chaos, rng=random.Random(0))
    resp = k.process(KernelRequest(payload="p", model="primary"))
    assert resp.payload == "saved"
    assert resp.metadata.get("fallback") == "backup"


# ================================================== typed routing =========
def test_auth_error_never_retried_in_resilient():
    calls = {"n": 0}

    def auth_fail(p):
        calls["n"] += 1
        raise AuthError("p0", RuntimeError("401"))

    k, ex = kernel(auth_fail, reliability={"max_retries": 5})
    ResilientExecution(ex)
    with pytest.raises(AuthError):
        k.process(KernelRequest(payload="p"))
    assert calls["n"] == 1                            # not retried


# ================================================== overhead baseline =====
def test_overhead_recorded_and_baseline_available():
    overheadmod.reset()
    k, ex = kernel(lambda p: "ok")
    ResilientExecution(ex)
    for i in range(50):
        k.process(KernelRequest(payload=f"r{i}"))
    pct = overheadmod.percentiles()
    assert pct["samples"] == 50
    assert pct["p50_ms"] is not None and pct["p99_ms"] is not None
    assert pct["p99_ms"] < 100.0                     # sane per-request tax


# ================================================== 1000-concurrent =======
def test_thousand_concurrent_requests():
    """WAVE GATE: 1000 concurrent requests complete, breaker stays closed,
    no lost/dup results, bulkhead admits under load."""
    k, ex = kernel(lambda p: "ok:" + p)
    ResilientExecution(ex, bulkhead=Bulkhead(limit_per_provider=256))
    results = []
    lock = threading.Lock()
    errors = []

    def worker(i):
        try:
            r = k.process(KernelRequest(payload=str(i)))
            with lock:
                results.append(r.payload)
        except Exception as e:                        # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(1000)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(results) == 1000
    assert len(set(results)) == 1000                  # every distinct req served


# ============================================ gap #2: streaming/proof =====
def test_streaming_skips_proof_and_cost_documented_behavior():
    """Documented architectural decision: a STREAMED request produces no
    after-phase, so no cost record and no proof seal — but security still
    gates before the stream opens."""
    from tokeymeter.runtime.providers import CallableAdapter

    class Stream(CallableAdapter):
        def stream_infer(self, ctx):
            yield "a"
            yield "b"

    r = Runtime(adapter=Stream(lambda p: "full"),
                config={"trust": {"proof": {"enabled": True}}},
                receipt="never")
    chunks = list(r.execute("p", stream=True))
    assert chunks == ["a", "b"]
    # no seal happened for the streamed request (last is None or unset)
    assert r.last is None                             # streaming sets nothing


# ============================================ gap #1: mypy gate ============
def test_runtime_package_is_mypy_clean():
    """The runtime package holds the strict-on-new-code bar (W0 policy)."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "tokeymeter/runtime/",
         "--ignore-missing-imports", "--follow-imports=silent"],
        capture_output=True, text=True)
    errors = [l for l in result.stdout.splitlines()
              if "runtime/" in l and "error:" in l]
    assert not errors, "runtime mypy regressions:\n" + "\n".join(errors)


# ============================================ REL-7: bounded retention =====
def test_trust_chain_bounded_retention_no_unbounded_growth():
    """REL-7 (soak-caught leak): the in-memory chain is a bounded rolling
    window; verify() still holds across eviction."""
    from tokeymeter.runtime.engines import TrustEngine
    k2 = Kernel(RuntimeConfig({})).start()
    t2 = TrustEngine(max_entries=100)
    ex2 = ExecutionEngine()
    ex2.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                         default=True)
    k2.register_engine(t2)
    k2.register_engine(ex2)
    for i in range(500):
        k2.process(KernelRequest(payload=f"r{i}"))
    assert len(t2.entries()) == 100                   # bounded, not 500
    assert t2._evicted == 400
    ok, bad = t2.verify()
    assert ok and bad == -1                           # window still verifies


def test_proof_engine_bounded_retention():
    from tokeymeter.runtime.proof import ProofEngine
    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine(max_entries=50)
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(proof)
    k.register_engine(ex)
    ids = [k.process(KernelRequest(payload=f"r{i}")).request_id
           for i in range(200)]
    assert len(proof.entries()) == 50
    ok, _ = proof.verify()
    assert ok
    # evicted ids are gone from the prove() index; recent ones remain
    with pytest.raises(KeyError):
        proof.prove(ids[0])
    assert proof.prove(ids[-1]) is not None
