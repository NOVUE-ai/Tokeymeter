"""HIGH-WORKLOAD STRESS (W0-W9) — the full stack under extreme load.

These are the tests an SRE team runs before a production launch: not "does it
work" but "does it hold at volume, under contention, with hostile inputs
mixed in, without leaking or tearing." Every test drives the ASSEMBLED runtime
(security + optimization + economics + proof) at scale.
"""
from __future__ import annotations

import gc
import os
import random
import string
import threading
import time

import pytest

import tokeymeter
from tokeymeter.runtime import (
    Kernel, KernelRequest, RuntimeConfig, SecretBlocked,
)
from tokeymeter.runtime.economics import EconomicsEngine
from tokeymeter.runtime.enforcement import (
    AccessEngine, RateLimitEngine, SecurityEngine)
from tokeymeter.runtime.optimization import OptimizationEngine
from tokeymeter.runtime.proof import ProofEngine
from tokeymeter.runtime.engines import TrustEngine
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.trust.audit.signers import Ed25519Signer


def _rss_kb():
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def full_stack(fn, *, model="s-model", max_entries=5000):
    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine(signer=Ed25519Signer.generate(),
                        max_entries=max_entries)
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn), models=[model], default=True)
    k.register_engine(proof)                              # first, unwinds last
    k.register_engine(SecurityEngine(secrets_mode="block", pii=True))
    k.register_engine(OptimizationEngine(compress=True, tier="structural"))
    k.register_engine(EconomicsEngine())
    k.register_engine(ex)
    return k, proof


# =====================================================================
# 1. MASSIVE CONCURRENCY — 2000 threads through the full stack
# =====================================================================
def test_stress_2000_concurrent_full_stack():
    k, proof = full_stack(lambda p: "ok", max_entries=5000)
    results = []
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            resp = k.process(KernelRequest(payload=f"request {i} content",
                                           model="s-model"))
            with lock:
                results.append(resp.request_id)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(2000)]
    t0 = time.perf_counter()
    [t.start() for t in threads]
    [t.join() for t in threads]
    elapsed = time.perf_counter() - t0

    assert not errors, f"{len(errors)} errors under load: {errors[:2]}"
    assert len(results) == 2000
    assert len(set(results)) == 2000                     # every request unique
    assert proof.verify()[0]                             # chain intact at scale
    rate = 2000 / elapsed
    assert rate > 200, f"throughput collapsed under load: {rate:.0f}/s"


# =====================================================================
# 2. MIXED ADVERSARIAL TRAFFIC — good + hostile requests interleaved
# =====================================================================
def test_stress_mixed_adversarial_traffic():
    """A realistic hostile mix: clean requests, secret-bearing requests, giant
    payloads, and unicode garbage — all at once. Clean ones succeed, hostile
    ones are handled, nothing crashes, proof chain stays valid."""
    k, proof = full_stack(lambda p: "ok")
    key = "sk-ant-api03-" + "A" * 88
    good = {"n": 0}
    blocked = {"n": 0}
    errored = {"n": 0}
    lock = threading.Lock()

    def clean(i):
        try:
            k.process(KernelRequest(payload=f"clean request {i}",
                                    model="s-model"))
            with lock:
                good["n"] += 1
        except Exception:
            with lock:
                errored["n"] += 1

    def hostile_secret(i):
        try:
            k.process(KernelRequest(payload=f"leak {key} now {i}",
                                    model="s-model"))
        except SecretBlocked:
            with lock:
                blocked["n"] += 1
        except Exception:
            with lock:
                errored["n"] += 1

    def hostile_giant(i):
        try:
            k.process(KernelRequest(payload="x" * 500_000, model="s-model"))
            with lock:
                good["n"] += 1
        except Exception:
            with lock:
                errored["n"] += 1

    def hostile_unicode(i):
        try:
            k.process(KernelRequest(
                payload=("\u200b\ufeff" * 500 + "💥" * 200 + str(i)),
                model="s-model"))
            with lock:
                good["n"] += 1
        except Exception:
            with lock:
                errored["n"] += 1

    workers = []
    for i in range(150):
        workers.append(threading.Thread(target=clean, args=(i,)))
        workers.append(threading.Thread(target=hostile_secret, args=(i,)))
        if i % 10 == 0:
            workers.append(threading.Thread(target=hostile_giant, args=(i,)))
            workers.append(threading.Thread(target=hostile_unicode, args=(i,)))
    random.shuffle(workers)
    [t.start() for t in workers]
    [t.join() for t in workers]

    assert blocked["n"] == 150                           # every secret blocked
    assert errored["n"] == 0                             # nothing crashed
    assert proof.verify()[0]                             # chain valid post-chaos


# =====================================================================
# 3. SUSTAINED VOLUME — memory bounded across 50k requests
# =====================================================================
def test_stress_50k_requests_memory_bounded():
    k, proof = full_stack(lambda p: "ok", max_entries=3000)
    # warm past the retention cap
    for i in range(4000):
        k.process(KernelRequest(payload=f"warm {i}", model="s-model"))
    gc.collect()
    base = _rss_kb()
    for i in range(50000):
        k.process(KernelRequest(payload=f"vol {i}", model="s-model"))
    gc.collect()
    end = _rss_kb()
    growth = end - base
    assert len(proof.entries()) == 3000                  # bounded, not 54000
    assert proof.verify()[0]
    # 50k requests must not grow RSS more than a few MB at steady state
    assert growth < 15000, f"RSS grew {growth}KB over 50k requests"


# =====================================================================
# 4. RATE LIMIT PRECISION UNDER EXTREME CONTENTION
# =====================================================================
def test_stress_rate_limit_exact_at_1000_threads():
    clock = {"t": 0.0}
    k = Kernel(RuntimeConfig({})).start()
    rl = RateLimitEngine(requests_per_min=100, clock=lambda: clock["t"])
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(rl)
    k.register_engine(ex)
    ok = []
    blocked = []
    lock = threading.Lock()

    def worker():
        try:
            k.process(KernelRequest(payload="p"))
            with lock:
                ok.append(1)
        except Exception:
            with lock:
                blocked.append(1)

    threads = [threading.Thread(target=worker) for _ in range(1000)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 100                                # EXACT cap at 1000 threads
    assert len(blocked) == 900


# =====================================================================
# 5. PROOF INTEGRITY UNDER CONCURRENT SEALING
# =====================================================================
def test_stress_proof_chain_integrity_under_concurrency():
    """500 threads sealing concurrently — the hash chain must remain valid and
    every request independently provable."""
    signer = Ed25519Signer.generate()
    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine(signer=signer, max_entries=10000)
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(proof)
    k.register_engine(ex)
    ids = []
    lock = threading.Lock()

    def worker(i):
        resp = k.process(KernelRequest(payload=f"seal {i}"))
        with lock:
            ids.append(resp.request_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(500)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert len(set(ids)) == 500
    assert proof.verify()[0]                             # chain valid
    # a sample of requests are independently provable
    from tokeymeter.runtime import verify_proof_packet
    for rid in random.sample(ids, 20):
        packet = proof.prove(rid).to_dict()
        assert verify_proof_packet(packet)[0]


# =====================================================================
# 6. BUDGET ENFORCEMENT UNDER CONCURRENT SPEND
# =====================================================================
def test_stress_budget_no_overspend_at_scale():
    from tokeymeter.engines.economics import keys as keysmod
    keysmod.clear_keys()
    try:
        keysmod.register_key("stress", "sk-x", monthly_cap_usd=50.0)
        ok = []
        blocked = []
        lock = threading.Lock()

        def worker():
            try:
                with keysmod.key("stress"):
                    keysmod.check_current(estimated_cost=1.0)
                    keysmod.on_spend("stress", 1.0, hit=False, shadow=False)
                with lock:
                    ok.append(1)
            except keysmod.KeyBudgetExceeded:
                with lock:
                    blocked.append(1)

        threads = [threading.Thread(target=worker) for _ in range(500)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        spent = keysmod.key_status("stress")["spent_usd"]
        # breaker doctrine: bounded overshoot, never unbounded
        assert spent <= 50.0 + 5.0                       # cap + small in-flight margin
        assert len(ok) + len(blocked) == 500
    finally:
        keysmod.clear_keys()


# =====================================================================
# 7. NO THREAD/FD LEAK AT SUSTAINED VOLUME
# =====================================================================
def test_stress_no_resource_leak_over_volume():
    base_threads = threading.active_count()
    k, _ = full_stack(lambda p: "ok")
    for i in range(10000):
        k.process(KernelRequest(payload=f"r{i}", model="s-model"))
    gc.collect()
    assert threading.active_count() <= base_threads + 3
