"""RUNTIME TRAFFIC & RESOURCE SAFETY (W0-W7) — SRE-grade.

Correctness under load is necessary but not sufficient; the runtime must also
stay smooth and bounded. These tests exercise sustained traffic through the
full stack and assert operational properties: no unbounded growth, no thread
or fd leak, latency overhead measured and bounded, clean drain on shutdown,
and stable behavior across a provider outage-and-recovery cycle.
"""
from __future__ import annotations

import gc
import os
import threading
import time

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import Kernel, KernelRequest, KernelStopped, RuntimeConfig
from tokeymeter.runtime.engines import TrustEngine
from tokeymeter.runtime.proof import ProofEngine
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.reliability import overhead as overheadmod


def _rss_kb():
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _fds():
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


def runtime_stack(fn=None, **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok:" + p[:16])),
                        models=["default"], default=True)
    k.register_engine(ex)
    return k, ex


# =====================================================================
# Throughput smoothness — no stalls, consistent completion
# =====================================================================
def test_sustained_throughput_completes_smoothly():
    k, _ = runtime_stack()
    n = 5000
    latencies = []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        k.process(KernelRequest(payload=f"req {i}"))
        latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - t_start
    rate = n / elapsed
    latencies.sort()
    p50 = latencies[n // 2]
    p99 = latencies[int(n * 0.99)]
    # smoothness: p99 must not be a wild multiple of p50 (no periodic stalls)
    assert rate > 500, f"throughput too low: {rate:.0f}/s"
    assert p99 < max(50.0, p50 * 50), f"p99 {p99:.2f} vs p50 {p50:.4f}"


def test_overhead_baseline_recorded_and_bounded():
    from tokeymeter.runtime.resilience import ResilientExecution
    overheadmod.reset()
    k, ex = runtime_stack()
    ResilientExecution(ex)
    for i in range(500):
        k.process(KernelRequest(payload=f"r{i}"))
    pct = overheadmod.percentiles()
    assert pct["samples"] == 500
    assert pct["p50_ms"] is not None and pct["p99_ms"] < 100.0


# =====================================================================
# Resource bounds — memory, threads, fds flat at steady state
# =====================================================================
def test_memory_bounded_under_sustained_full_stack_load():
    trust = TrustEngine(max_entries=2000)
    k = Kernel(RuntimeConfig({})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(trust)
    k.register_engine(ex)
    # warm past the retention cap so we measure steady state, not fill
    for i in range(4000):
        k.process(KernelRequest(payload=f"warm {i}"))
    gc.collect()
    base = _rss_kb()
    for i in range(10000):
        k.process(KernelRequest(payload=f"steady {i}"))
    gc.collect()
    end = _rss_kb()
    growth = end - base
    assert len(trust.entries()) == 2000                   # bounded, not 14000
    assert growth < 8000, f"RSS grew {growth}KB at steady state"  # ~<8MB drift


def test_no_thread_leak_across_many_requests():
    base_threads = threading.active_count()
    k, _ = runtime_stack()
    for i in range(2000):
        k.process(KernelRequest(payload=f"r{i}"))
    gc.collect()
    assert threading.active_count() <= base_threads + 2    # no per-request thread


def test_no_fd_leak_with_proof_sink(tmp_path):
    from tokeymeter.runtime.proof import FileAuditSink
    sink_path = str(tmp_path / "audit.ndjson")
    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine(sink=FileAuditSink(sink_path))
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(proof)
    k.register_engine(ex)
    base_fds = _fds()
    for i in range(1000):
        k.process(KernelRequest(payload=f"r{i}"))
    # the WORM sink opens/flushes/closes per append — no accumulating fds
    assert _fds() <= base_fds + 3


# =====================================================================
# Graceful drain — shutdown never severs in-flight work
# =====================================================================
def test_graceful_drain_completes_inflight_refuses_new():
    release = threading.Event()
    started = threading.Event()

    def slow(p):
        started.set()
        release.wait(timeout=5)
        return "done"

    k, _ = runtime_stack(fn=slow, kernel={"drain_timeout_s": 5.0})
    result = {}

    def inflight():
        result["r"] = k.process(KernelRequest(payload="inflight")).payload

    t = threading.Thread(target=inflight)
    t.start()
    started.wait(timeout=5)
    stopper = threading.Thread(target=k.shutdown)
    stopper.start()
    time.sleep(0.1)
    # new work refused during drain
    with pytest.raises(KernelStopped):
        k.process(KernelRequest(payload="new"))
    release.set()
    t.join(timeout=5)
    stopper.join(timeout=5)
    assert result["r"] == "done"                          # in-flight completed


# =====================================================================
# Outage and recovery — smooth degradation, clean recovery
# =====================================================================
def test_provider_outage_then_recovery_is_smooth():
    from tokeymeter.runtime.errors import ProviderDown
    from tokeymeter.runtime.resilience import CircuitBreaker, ResilientExecution
    state = {"down": True}

    def toggling(p):
        if state["down"]:
            raise ProviderDown("p0", RuntimeError("down"))
        return "recovered"

    clock = {"t": 0.0}
    k, ex = runtime_stack(fn=toggling, reliability={"max_retries": 0})
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=5.0,
                        clock=lambda: clock["t"])
    ResilientExecution(ex, breaker=cb)
    # during outage: breaker opens, requests fail fast (not hang)
    for _ in range(10):
        with pytest.raises(Exception):
            k.process(KernelRequest(payload="p"))
    from tokeymeter.runtime.resilience import BreakerState
    assert cb.state("callable") == BreakerState.OPEN
    # provider recovers; after cooldown the half-open probe succeeds
    state["down"] = False
    clock["t"] = 6.0
    resp = k.process(KernelRequest(payload="p"))
    assert resp.payload == "recovered"
    assert cb.state("callable") == BreakerState.CLOSED    # clean recovery


# =====================================================================
# Facade under concurrent traffic — the product surface holds
# =====================================================================
def test_facade_concurrent_traffic_stable():
    r = Runtime(call=lambda p: "ok:" + p, receipt="never")
    results = []
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            out = r.execute(f"msg {i}")
            with lock:
                results.append(out)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(300)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(results) == 300 and len(set(results)) == 300
