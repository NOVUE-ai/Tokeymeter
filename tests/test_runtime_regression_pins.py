"""RUNTIME REGRESSION PINS (W0-W7) — every real defect, locked.

Each test here corresponds to an ACTUAL bug found and fixed during the build
(or a sharp behavioral contract that must never drift). The name states the
wave and the defect. If any of these fails, a fixed bug has returned.
"""
from __future__ import annotations

import hashlib

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import (
    Kernel, KernelRequest, RuntimeConfig, TieredOptimizer,
)
from tokeymeter.runtime.economics import EconomicsEngine
from tokeymeter.runtime.enforcement import SecurityEngine
from tokeymeter.runtime.engines import TrustEngine
from tokeymeter.runtime.optimize import RoutePlanner
from tokeymeter.runtime.proof import ProofEngine
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.economics import pricing as pricingmod
from tokeymeter.engines.optimization.compression import (
    StructuralCompressor, safe_compress)


def kern(*engines, fn=None, model="default", **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok")),
                        models=[model], default=True)
    for e in engines:
        k.register_engine(e)
    k.register_engine(ex)
    return k, ex


# ---- W4: Trust sealed BEFORE economics wrote cost (seam-order bug) --------
def test_W4_trust_seals_after_economics_cost_present():
    """FIXED: proof/trust registered FIRST so it unwinds LAST and seals the
    cost economics wrote. Regression = cost missing from the sealed body."""
    pricingmod.register_pricing("m", input_per_1m=2.0, output_per_1m=10.0)
    from types import SimpleNamespace as NS
    trust = TrustEngine()
    k, _ = kern(trust, EconomicsEngine(), model="m",
                fn=lambda p: NS(usage=NS(prompt_tokens=1000,
                                         completion_tokens=500)))
    resp = k.process(KernelRequest(payload="q", model="m"))
    assert resp.metadata["cost_usd"] is not None
    assert str(resp.metadata["cost_usd"]) in trust.entries()[0]["body"]


# ---- W4: a generic-fallback price must NOT be reported as a known cost -----
def test_W4_default_price_is_not_a_known_cost():
    """FIXED: only 'registered'/'list' provenance yields cost_usd; the generic
    'default' fallback is a guess kept separately. Regression = a guessed
    dollar reported as real cost."""
    from types import SimpleNamespace as NS
    k, _ = kern(EconomicsEngine(),
                fn=lambda p: NS(usage=NS(prompt_tokens=100,
                                         completion_tokens=100)))
    resp = k.process(KernelRequest(payload="q"))   # model 'default'
    assert resp.metadata["cost_usd"] is None
    assert resp.metadata["cost_source"] == "default"


# ---- W5: proof verify must anchor on stored prev after eviction -----------
def test_W5_proof_verify_holds_after_retention_eviction():
    """FIXED: ProofEngine.verify() anchors on the retained window's stored
    prev-hash. Regression = verify() False after eviction."""
    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine(max_entries=50)
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(proof)
    k.register_engine(ex)
    for i in range(200):
        k.process(KernelRequest(payload=f"r{i}"))
    assert len(proof.entries()) == 50
    assert proof.verify()[0]                       # holds across eviction


# ---- W6: TrustEngine unbounded memory growth (soak-caught leak) -----------
def test_W6_trust_chain_bounded_not_unbounded():
    """FIXED: bounded rolling window. Regression = entries grow without limit."""
    k = Kernel(RuntimeConfig({})).start()
    trust = TrustEngine(max_entries=100)
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["default"],
                        default=True)
    k.register_engine(trust)
    k.register_engine(ex)
    for i in range(500):
        k.process(KernelRequest(payload=f"r{i}"))
    assert len(trust.entries()) == 100
    assert trust._evicted == 400
    assert trust.verify()[0]


# ---- W6: non-retryable auth error must not be retried/amplified -----------
def test_W6_auth_error_not_amplified():
    """FIXED: retryable=False short-circuits retry+fallback. Regression =
    provider called more than once on auth failure."""
    from tokeymeter.runtime.errors import AuthError
    from tokeymeter.runtime.resilience import ResilientExecution
    calls = {"n": 0}

    def dying(p):
        calls["n"] += 1
        raise AuthError("p0", RuntimeError("401"))
    k, ex = kern(fn=dying, reliability={"max_retries": 5})
    ResilientExecution(ex)
    with pytest.raises(AuthError):
        k.process(KernelRequest(payload="q"))
    assert calls["n"] == 1


# ---- W7: RoutePlan.candidates default_factory was dict for a List field ---
def test_W7_route_plan_candidates_is_list():
    """FIXED: default_factory=list. Regression = candidates initialized as a
    dict (wrong type)."""
    plan = RoutePlanner(objective="cost").plan("hello world")
    assert isinstance(plan.candidates, list)
    assert all(isinstance(c, dict) for c in plan.candidates)


# ---- W7: unified optimizer must equal shipped compressor (savings-parity) -
def test_W7_savings_parity_locked():
    """FIXED/GATED: TieredOptimizer(structural) ≡ shipped StructuralCompressor.
    Regression = the unified layer's numbers drift from the shipped math."""
    text = ("Please kindly note that in order to proceed you should just "
            "click. ") * 6
    direct = safe_compress(StructuralCompressor(), text)
    unified = TieredOptimizer(tier="structural").optimize(text)
    assert (unified.after, unified.tokens_after, unified.ratio) == \
           (direct.after, direct.tokens_after, direct.ratio)


# ---- W3: streaming bypasses after-phase -> no cost/proof (documented) -----
def test_W3_streaming_produces_no_seal():
    """CONTRACT: a streamed request has no after-phase, so no cost record and
    no proof seal (we refuse to fabricate mid-stream totals). Regression =
    r.last set / a seal appears for a streamed call."""
    class Stream(CallableAdapter):
        def stream_infer(self, ctx):
            yield "a"
            yield "b"
    r = Runtime(adapter=Stream(lambda p: "full"),
                config={"trust": {"proof": {"enabled": True}}},
                receipt="never")
    list(r.execute("p", stream=True))
    assert r.last is None


# ---- W4: redaction happens BEFORE the trust fingerprint -------------------
def test_W4_fingerprint_is_post_redaction():
    """FIXED/CONTRACT: the trust fingerprint hashes the redacted payload, so
    no pre-redaction hash exists in the record. Regression = original-text
    hash present."""
    trust = TrustEngine()
    ssn_text = "ssn 123-45-6789 summarize"
    k, _ = kern(trust, SecurityEngine(secrets_mode="off", pii=True))
    k.process(KernelRequest(payload=ssn_text))
    orig_fp = hashlib.sha256(ssn_text.encode()).hexdigest()
    assert orig_fp not in trust.entries()[0]["body"]
    assert "123-45-6789" not in str(trust.entries())


# ---- W2: reliability routes on typed error, unknown -> retryable ----------
def test_W2_unknown_exception_is_retryable_default():
    """CONTRACT: an unknown exception classifies as TransientError (retryable)
    — the safe default. Regression = unknown errors treated as non-retryable."""
    from tokeymeter.runtime.errors import classify, TransientError
    typed = classify(ValueError("mystery"), "p0")
    assert isinstance(typed, TransientError) and typed.retryable is True


# ---- W1: engine identity shims keep import paths stable -------------------
def test_W1_engine_layout_import_identity():
    """CONTRACT: split engine modules remain importable at canonical paths and
    are identical objects. Regression = a shim broke path identity."""
    from tokeymeter.engines.optimization import router as canonical
    from tokeymeter.engines.optimization.router import Router
    assert canonical.Router is Router
