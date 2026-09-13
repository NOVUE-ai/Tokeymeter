"""RUNTIME INTEGRATION (W0-W7) — every engine active together.

Unit tests prove each engine alone; these prove the ASSEMBLY. A real request
in production runs security + access + rate-limit + optimization + economics
+ reliability + proof at once, in order, and the invariants must hold across
all of them simultaneously. Each test wires the FULL stack and asserts a
whole-pipeline property.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

import tokeymeter
from tokeymeter import Runtime
from tokeymeter.runtime import (
    Kernel, KernelRequest, RuntimeConfig, SecretBlocked,
)
from tokeymeter.runtime.economics import EconomicsEngine
from tokeymeter.runtime.enforcement import (
    AccessEngine, RateLimitEngine, SecurityEngine)
from tokeymeter.runtime.optimization import OptimizationEngine
from tokeymeter.runtime.proof import ProofEngine, verify_proof_packet
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.economics import keys as keysmod
from tokeymeter.engines.economics import pricing as pricingmod
from tokeymeter.engines.trust.audit.signers import Ed25519Signer

VERBOSE = ("Please kindly note that you should basically just proceed with "
           "the task at this point in time without delay. ") * 5
ANTHROPIC_KEY = "sk-ant-api03-" + "".join(
    __import__("secrets").choice("abcdefghijklmnopqrstuvwxyz0123456789")
    for _ in range(88))


def usage_resp(pin=1000, pout=500, text="done"):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None),
                          finish_reason="stop")],
              usage=NS(prompt_tokens=pin, completion_tokens=pout),
              model="m", id="i")


@pytest.fixture(autouse=True)
def _clean():
    keysmod.clear_keys()
    yield
    keysmod.clear_keys()


def full_stack(fn, *, model="w-model", principal_role=True, **overrides):
    """The whole pipeline, seam-order correct: proof FIRST (unwinds last),
    then security, access, rate-limit, optimization, economics."""
    k = Kernel(RuntimeConfig(overrides or {})).start()
    proof = ProofEngine(signer=Ed25519Signer.generate())
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn), models=[model], default=True)
    k.register_engine(proof)                              # FIRST
    k.register_engine(SecurityEngine(secrets_mode="block", pii=True))
    if principal_role:
        k.register_engine(AccessEngine(
            roles={"eng": {"models": [model]}},
            principals={"dev": "eng"}))
    k.register_engine(RateLimitEngine(requests_per_min=1000))
    k.register_engine(OptimizationEngine(compress=True, tier="structural"))
    k.register_engine(EconomicsEngine())
    k.register_engine(ex)
    return k, proof, ex


# =====================================================================
# 1. The happy path through EVERY engine, all invariants at once
# =====================================================================
def test_full_stack_happy_path_all_invariants():
    pricingmod.register_pricing("w-model", input_per_1m=2.0, output_per_1m=10.0)
    seen = {}
    k, proof, _ = full_stack(
        lambda p: seen.setdefault("p", p) or usage_resp(), model="w-model")
    with tokeymeter.principal("dev"):
        resp = k.process(KernelRequest(payload=VERBOSE, model="w-model"))

    # every engine ran, in order (VERBOSE carries no secret/PII, so it flows)
    engines = [t["engine"] for t in resp.trace if t["phase"] == "before_request"]
    assert engines == ["proof", "security", "access", "rate_limit",
                       "optimization", "economics", "execution"]
    # optimization actually compressed what the provider saw
    assert len(seen["p"]) < len(VERBOSE)
    # economics produced a real, provenanced cost
    assert resp.metadata["cost_usd"] is not None
    assert resp.metadata["cost_source"] == "registered"
    # verdict ledger recorded allow for each policy
    verdicts = resp.metadata["policy_verdicts"]
    policies = [v["policy"] for v in verdicts]
    assert "secrets" in policies and "rbac" in policies and "pii" in policies
    # proof: the request is sealed and independently verifiable
    packet = proof.prove(resp.request_id).to_dict()
    assert verify_proof_packet(packet)[0]
    # content-blind: nothing from the verbose payload in the proof
    assert "kindly" not in json.dumps(packet)


def test_full_stack_via_facade_end_to_end():
    pricingmod.register_pricing("gpt-x", input_per_1m=1.0, output_per_1m=3.0)
    r = Runtime(
        config={
            "governance": {"security": {"enabled": True, "secrets_mode": "block"}},
            "optimization": {"enabled": True, "compress": True},
            "trust": {"proof": {"enabled": True}},
        },
        call=lambda p: usage_resp(), model="gpt-x", receipt="never",
        signer=Ed25519Signer.generate())
    r.execute(VERBOSE)
    assert r.last.metadata.get("tokens_saved", 0) >= 0
    packet = r.prove(r.last.request_id).to_dict()
    assert verify_proof_packet(packet)[0]


# =====================================================================
# 2. Cross-engine ORDERING invariants — the subtle correctness
# =====================================================================
def test_security_redacts_before_optimization_compresses():
    """Redaction must happen before compression, so the compressor never sees
    PII and cannot carry it forward. NOTE: SSN is a BLOCKING secret in the
    firewall (defense in depth); to isolate the PII-redact -> compress
    ordering we run secrets OFF with PII ON, so redaction (not blocking) is
    the active path."""
    seen = {}
    k = Kernel(RuntimeConfig({})).start()
    ex = ExecutionEngine()
    ex.register_adapter(
        CallableAdapter(lambda p: seen.setdefault("p", p) or usage_resp()),
        models=["default"], default=True)
    k.register_engine(SecurityEngine(secrets_mode="off", pii=True))
    k.register_engine(OptimizationEngine(compress=True, tier="structural"))
    k.register_engine(ex)
    k.process(KernelRequest(payload="ssn 123-45-6789 " + VERBOSE))
    assert "123-45-6789" not in seen["p"]                 # redacted before provider
    # and the SECRET firewall independently blocks SSN in block mode:
    kb = Kernel(RuntimeConfig({})).start()
    exb = ExecutionEngine()
    exb.register_adapter(CallableAdapter(lambda p: "x"),
                         models=["default"], default=True)
    kb.register_engine(SecurityEngine(secrets_mode="block"))
    kb.register_engine(exb)
    with pytest.raises(SecretBlocked):
        kb.process(KernelRequest(payload="ssn 123-45-6789 here"))


def test_secret_block_prevents_all_downstream_engines():
    """A secret block in security must stop optimization, economics, and the
    provider — but proof must STILL seal the blocked attempt."""
    ran = {"opt": 0, "econ": 0, "provider": 0}

    class SpyOpt(OptimizationEngine):
        def before_request(self, ctx):
            ran["opt"] += 1

    class SpyEcon(EconomicsEngine):
        def before_request(self, ctx):
            ran["econ"] += 1

    k = Kernel(RuntimeConfig({})).start()
    proof = ProofEngine()
    ex = ExecutionEngine()
    ex.register_adapter(
        CallableAdapter(lambda p: ran.__setitem__("provider", 1) or "x"),
        models=["default"], default=True)
    k.register_engine(proof)
    k.register_engine(SecurityEngine(secrets_mode="block"))
    k.register_engine(SpyOpt(compress=True))
    k.register_engine(SpyEcon())
    k.register_engine(ex)
    with pytest.raises(SecretBlocked):
        k.process(KernelRequest(payload=f"leak {ANTHROPIC_KEY}"))
    assert ran == {"opt": 0, "econ": 0, "provider": 0}    # all stopped
    assert len(proof.entries()) == 1                      # but proof sealed


def test_economics_cost_uses_optimized_token_count():
    """Cost must be computed on what was actually sent (post-compression),
    via the provider's reported usage — the money reflects the optimization."""
    pricingmod.register_pricing("w-model", input_per_1m=2.0, output_per_1m=10.0)
    k, _, _ = full_stack(lambda p: usage_resp(pin=100, pout=50),
                         model="w-model")
    with tokeymeter.principal("dev"):
        resp = k.process(KernelRequest(payload=VERBOSE, model="w-model"))
    # reported usage is authoritative
    assert resp.metadata["usage_source"] == "response_usage"
    assert resp.metadata["tokens_in"] == 100


# =====================================================================
# 3. Failure propagation across the stack
# =====================================================================
def test_provider_failure_still_seals_and_attributes():
    from tokeymeter.runtime.errors import ProviderDown
    k = Kernel(RuntimeConfig({"reliability": {"max_retries": 0}})).start()
    proof = ProofEngine()
    ex = ExecutionEngine()

    def dead(p):
        raise ProviderDown("p0", RuntimeError("down"))
    ex.register_adapter(CallableAdapter(dead), models=["default"], default=True)
    k.register_engine(proof)
    k.register_engine(SecurityEngine(secrets_mode="off", pii=False))
    k.register_engine(ex)
    with pytest.raises(Exception):
        k.process(KernelRequest(payload="q"))
    entries = proof.entries()
    assert len(entries) == 1 and entries[0]["outcome"].startswith("error:")
    assert proof.verify()[0]


def test_reliability_recovers_and_full_record_still_correct():
    pricingmod.register_pricing("w-model", input_per_1m=1.0, output_per_1m=1.0)
    from tokeymeter.runtime.resilience import ResilientExecution
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] < 2:
            from tokeymeter.runtime.errors import TransientError
            raise TransientError("p0", RuntimeError("t"))
        return usage_resp(pin=10, pout=10)

    k = Kernel(RuntimeConfig({"reliability": {"max_retries": 3}})).start()
    proof = ProofEngine()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(flaky), models=["w-model"],
                        default=True)
    k.register_engine(proof)
    k.register_engine(EconomicsEngine())
    ResilientExecution(ex)
    k.register_engine(ex)
    resp = k.process(KernelRequest(payload="q", model="w-model"))
    assert resp.metadata["cost_usd"] is not None          # recovered + priced
    assert proof.verify()[0]                              # sealed once, valid


# =====================================================================
# 4. State isolation across concurrent full-stack requests
# =====================================================================
def test_concurrent_full_stack_no_cross_contamination():
    import threading
    pricingmod.register_pricing("w-model", input_per_1m=1.0, output_per_1m=1.0)
    k, proof, _ = full_stack(
        lambda p: usage_resp(pin=len(p), pout=5, text="ok"), model="w-model")
    results = {}
    lock = threading.Lock()

    def worker(i):
        with tokeymeter.principal("dev"):
            resp = k.process(KernelRequest(
                payload=f"request number {i} " + "x" * i, model="w-model"))
        with lock:
            results[i] = resp.request_id

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # every request got a unique id and sealed independently
    assert len(set(results.values())) == 50
    assert proof.verify()[0]                              # chain intact
    assert len(proof.entries()) == 50
