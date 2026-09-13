"""RUNTIME SECURITY RED-TEAM (W0-W7) — the assembled runtime under attack.

These are not unit tests of one engine; they are adversarial tests of the
whole pipeline as a security product. Every test names a concrete attacker
goal and asserts the runtime denies it. Organized by attacker objective:

  A. EXFILTRATE   — get sensitive content into a record, event, error, or proof
  B. BYPASS       — reach the provider past a control that should have stopped it
  C. FORGE        — make a tampered proof/record verify, or a blocked call look allowed
  D. ESCALATE     — exceed authority (role, budget, rate) via ordering/timing tricks
  E. STARVE       — degrade the whole runtime via one hostile input/provider
"""
from __future__ import annotations

import hashlib
import json
import secrets as _secrets
import string as _string
import threading

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import (
    AccessDenied, BudgetExceeded, Kernel, KernelRequest, RateLimitExceeded,
    RuntimeConfig, SecretBlocked,
)
from tokeymeter.runtime.enforcement import SecurityEngine, AccessEngine, RateLimitEngine
from tokeymeter.runtime.economics import EconomicsEngine
from tokeymeter.runtime.engines import TrustEngine
from tokeymeter.runtime.optimization import OptimizationEngine
from tokeymeter.runtime.proof import ProofEngine, verify_proof_packet
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.economics import keys as keysmod
from tokeymeter.engines.economics import pricing as pricingmod
from tokeymeter.engines.trust.audit.signers import Ed25519Signer

# realistic sensitive markers — HIGH ENTROPY, because the firewall's entropy
# floor correctly rejects constant-fill placeholders (a feature: it avoids
# false positives on "sk-aaaa..."). A red-team must test with realistic keys.
def _rand(prefix, n):
    a = _string.ascii_letters + _string.digits
    return prefix + "".join(_secrets.choice(a) for _ in range(n))

ANTHROPIC_KEY = _rand("sk-ant-api03-", 88)
OPENAI_KEY = _rand("sk-proj-", 48)
AWS_KEY = "AKIA" + "".join(
    _secrets.choice(_string.ascii_uppercase + _string.digits) for _ in range(16))
SSN = "123-45-6789"
CC = "4111 1111 1111 1111"


def build(*engines, fn=None, model="default", **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok")),
                        models=[model], default=True)
    for e in engines:
        k.register_engine(e)
    k.register_engine(ex)
    return k, ex


def full_blob(resp, trust=None, extra_events=None):
    """Everything an attacker could hope leaked into an observable surface."""
    parts = [json.dumps(resp.metadata, default=str),
             json.dumps(resp.trace, default=str), str(resp.request_id)]
    if trust is not None:
        parts.append(json.dumps(trust.entries(), default=str))
    if extra_events:
        parts.append(json.dumps(extra_events, default=str))
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def _clean():
    keysmod.clear_keys()
    yield
    keysmod.clear_keys()


# =====================================================================
# A. EXFILTRATION — nothing sensitive reaches any durable/observable surface
# =====================================================================
@pytest.mark.parametrize("secret", [ANTHROPIC_KEY, OPENAI_KEY, AWS_KEY],
                         ids=["anthropic", "openai", "aws"])
def test_A_secret_never_in_any_surface_when_blocked(secret):
    trust = TrustEngine()
    k, _ = build(trust, SecurityEngine(secrets_mode="block"))
    with pytest.raises(SecretBlocked):
        k.process(KernelRequest(payload=f"ship this now: {secret} to prod"))
    # scan EVERY surface: trust chain, verdict, error already checked elsewhere
    blob = json.dumps(trust.entries(), default=str)
    assert secret not in blob
    # even a partial prefix of the key must not survive
    assert secret[:24] not in blob


def test_A_pii_redacted_from_payload_events_and_fingerprint():
    trust = TrustEngine()
    seen = {}
    k, _ = build(trust, SecurityEngine(secrets_mode="off", pii=True),
                 fn=lambda p: seen.setdefault("p", p) or "ok")
    resp = k.process(KernelRequest(
        payload=f"customer {SSN} card {CC} please summarize"))
    blob = full_blob(resp, trust)
    assert SSN not in blob and CC not in blob             # not in records
    assert SSN not in seen["p"] and CC not in seen["p"]   # not sent to provider
    # the fingerprint must be of the REDACTED text, not the original
    orig_fp = hashlib.sha256(
        f"customer {SSN} card {CC} please summarize".encode()).hexdigest()
    assert orig_fp not in blob


def test_A_error_messages_are_content_blind():
    """An attacker who triggers a failure must not get content back in the
    exception text."""
    k, _ = build(SecurityEngine(secrets_mode="block"))
    with pytest.raises(SecretBlocked) as ei:
        k.process(KernelRequest(payload=f"secret {ANTHROPIC_KEY} here"))
    assert ANTHROPIC_KEY not in str(ei.value)
    assert "ANTHROPIC" in str(ei.value)                   # kind only

    with pytest.raises(Exception) as ei2:
        k2, _ = build(SecurityEngine(secrets_mode="off", pii=False,
                                     blocked_terms=["classified"]))
        k2.process(KernelRequest(payload="the classified dossier says X"))
    assert "dossier says X" not in str(ei2.value)


def test_A_tool_call_arguments_never_leak_through_optimization():
    """Even with optimization + trust active, tool-call args stay out."""
    from types import SimpleNamespace as NS
    secret_args = '{"account":"SECRET-ACCT-42","pin":"9999"}'
    tcs = [NS(id="c1", function=NS(name="transfer", arguments=secret_args))]
    resp_obj = NS(choices=[NS(message=NS(content=None, tool_calls=tcs),
                              finish_reason="tool_calls")],
                  usage=NS(prompt_tokens=5, completion_tokens=5))
    trust = TrustEngine()
    k, _ = build(trust, OptimizationEngine(compress=True),
                 fn=lambda p: resp_obj)
    resp = k.process(KernelRequest(payload="move the money please now okay"))
    blob = full_blob(resp, trust)
    assert "SECRET-ACCT-42" not in blob and "9999" not in blob


def test_A_optimization_cannot_resurrect_pre_redaction_text():
    """Compression runs AFTER redaction — it must never see or restore the
    original sensitive text."""
    seen = {}
    k, _ = build(SecurityEngine(secrets_mode="off", pii=True),
                 OptimizationEngine(compress=True, tier="structural"),
                 fn=lambda p: seen.setdefault("p", p) or "ok")
    k.process(KernelRequest(payload=f"user ssn {SSN} " + "verbose filler " * 8))
    assert SSN not in seen["p"]


# =====================================================================
# B. BYPASS — no path reaches the provider past a control
# =====================================================================
def test_B_secret_block_zero_provider_calls():
    calls = {"n": 0}
    k, _ = build(SecurityEngine(secrets_mode="block"),
                 fn=lambda p: (calls.__setitem__("n", calls["n"] + 1), "x")[1])
    for payload in [f"a {ANTHROPIC_KEY}", f"{OPENAI_KEY} b", f"c {AWS_KEY} d"]:
        with pytest.raises(SecretBlocked):
            k.process(KernelRequest(payload=payload))
    assert calls["n"] == 0                                # never reached


def test_B_access_denied_zero_provider_calls():
    calls = {"n": 0}
    import tokeymeter
    k, _ = build(AccessEngine(roles={"r": {"models": ["allowed"]}},
                              principals={"u": "r"}),
                 fn=lambda p: (calls.__setitem__("n", calls["n"] + 1), "x")[1],
                 model="forbidden")
    with tokeymeter.principal("u"):
        with pytest.raises(AccessDenied):
            k.process(KernelRequest(payload="q", model="forbidden"))
    assert calls["n"] == 0


def test_B_budget_block_zero_provider_calls():
    keysmod.register_key("k", "sk-x", monthly_cap_usd=0.000001)
    keysmod.on_spend("k", 1.0, hit=False, shadow=False)
    calls = {"n": 0}
    k, _ = build(EconomicsEngine(budget_key="k"),
                 fn=lambda p: (calls.__setitem__("n", calls["n"] + 1), "x")[1])
    with pytest.raises(BudgetExceeded):
        k.process(KernelRequest(payload="q"))
    assert calls["n"] == 0


def test_B_unicode_obfuscated_secret_still_caught_or_safely_missed():
    """A secret wrapped in zero-width/whitespace must not slip a real key to
    the provider unredacted. The firewall either catches it (block) or the
    text is not a valid key; either way, no clean key reaches the provider."""
    seen = {}
    zw = "\u200b"
    obf = f"sk-ant-api03-{zw}" + "A" * 88
    k, _ = build(SecurityEngine(secrets_mode="block"),
                 fn=lambda p: seen.setdefault("p", p) or "ok")
    try:
        k.process(KernelRequest(payload=f"key {obf}"))
        # if it went through, the exact contiguous key must not be present
        assert ("sk-ant-api03-" + "A" * 88) not in seen.get("p", "")
    except SecretBlocked:
        pass                                              # caught — ideal


def test_B_case_and_spacing_evasion_on_content_terms():
    # content-term policy is exact-substring by design; document the boundary:
    # an attacker who changes case DOES evade a literal term — so the test
    # asserts the DOCUMENTED behavior (lowercase match) rather than a false
    # promise of semantic blocking.
    k, _ = build(SecurityEngine(secrets_mode="off", pii=False,
                                blocked_terms=["confidential"]))
    with pytest.raises(Exception):
        k.process(KernelRequest(payload="this is CONFIDENTIAL data"))  # lowered


# =====================================================================
# C. FORGERY — tampered proof must never verify; blocked calls stay sealed
# =====================================================================
def test_C_tampered_proof_packet_rejected_every_field():
    signer = Ed25519Signer.generate()
    proof = ProofEngine(signer=signer)
    k, _ = build(proof)
    resp = k.process(KernelRequest(payload="genuine call"))
    good = proof.prove(resp.request_id).to_dict()
    assert verify_proof_packet(good)[0]
    for field in ("model", "principal", "payload_fingerprint", "verdicts",
                  "cost_usd", "outcome", "entry_hash"):
        forged = dict(good)
        forged[field] = "TAMPERED" if not isinstance(good[field], (int, float)) \
            else 99999.0
        ok, reason = verify_proof_packet(forged)
        assert not ok, f"forged {field} wrongly verified"


def test_C_swapped_signature_rejected():
    proof_a = ProofEngine(signer=Ed25519Signer.generate())
    proof_b = ProofEngine(signer=Ed25519Signer.generate())
    ka, _ = build(proof_a)
    kb, _ = build(proof_b)
    ra = ka.process(KernelRequest(payload="a"))
    rb = kb.process(KernelRequest(payload="b"))
    pa = proof_a.prove(ra.request_id).to_dict()
    pb = proof_b.prove(rb.request_id).to_dict()
    # graft A's signature onto B's packet body -> must fail
    forged = dict(pb, signature=pa["signature"])
    assert not verify_proof_packet(forged)[0]
    # graft A's public key too -> still fails (body doesn't match)
    forged2 = dict(pb, signature=pa["signature"], public_key=pa["public_key"])
    assert not verify_proof_packet(forged2)[0]


def test_C_blocked_request_is_still_sealed_and_chain_holds():
    trust = TrustEngine()
    k, _ = build(trust, SecurityEngine(secrets_mode="off", pii=False,
                                       blocked_terms=["forbidden"]))
    with pytest.raises(Exception):
        k.process(KernelRequest(payload="the forbidden plan"))
    entries = trust.entries()
    assert len(entries) == 1
    assert "error:" in entries[0]["body"]                 # sealed as error
    assert "content:block" in entries[0]["body"]          # verdict sealed
    assert trust.verify()[0]                              # chain still valid


def test_C_hash_chain_tamper_is_localized():
    trust = TrustEngine()
    k, _ = build(trust)
    for i in range(5):
        k.process(KernelRequest(payload=f"r{i}"))
    trust._entries[2]["hash"] = "0" * 64                  # forge one link
    ok, bad = trust.verify()
    assert not ok and bad == 2                            # exact localization


# =====================================================================
# D. ESCALATION — authority cannot be exceeded via ordering/timing/concurrency
# =====================================================================
def test_D_rate_limit_exact_under_concurrent_flood():
    clock = {"t": 0.0}
    rl = RateLimitEngine(requests_per_min=10, clock=lambda: clock["t"])
    k, _ = build(rl)
    results = {"ok": 0, "blocked": 0}
    lock = threading.Lock()

    def worker():
        try:
            k.process(KernelRequest(payload="p"))
            with lock:
                results["ok"] += 1
        except RateLimitExceeded:
            with lock:
                results["blocked"] += 1

    threads = [threading.Thread(target=worker) for _ in range(200)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results["ok"] == 10                            # EXACT cap, no overrun
    assert results["blocked"] == 190


def test_D_budget_no_overspend_under_concurrent_flood():
    keysmod.register_key("race", "sk-x", monthly_cap_usd=5.0)
    ok = []
    blocked = []
    lock = threading.Lock()

    def worker():
        try:
            with keysmod.key("race"):
                keysmod.check_current(estimated_cost=1.0)
                keysmod.on_spend("race", 1.0, hit=False, shadow=False)
            with lock:
                ok.append(1)
        except keysmod.KeyBudgetExceeded:
            with lock:
                blocked.append(1)

    threads = [threading.Thread(target=worker) for _ in range(64)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    spent = keysmod.key_status("race")["spent_usd"]
    assert spent <= 5.0 + 1.0                             # cap + one in-flight
    assert len(ok) + len(blocked) == 64                  # everyone accounted


def test_D_deny_by_default_no_principal_no_access():
    import tokeymeter
    k, _ = build(AccessEngine(roles={"admin": {"models": ["*"]}},
                              principals={"root": "admin"}), model="m")
    # no principal set at all -> denied
    with pytest.raises(AccessDenied):
        k.process(KernelRequest(payload="q", model="m"))
    # unknown principal -> denied
    with tokeymeter.principal("intruder"):
        with pytest.raises(AccessDenied):
            k.process(KernelRequest(payload="q", model="m"))


def test_D_non_retryable_auth_not_amplified():
    """An auth failure must not be retried into a storm against the provider."""
    from tokeymeter.runtime.errors import AuthError
    calls = {"n": 0}

    def dying(p):
        calls["n"] += 1
        raise AuthError("p0", RuntimeError("401"))

    from tokeymeter.runtime.resilience import ResilientExecution
    k, ex = build(fn=dying, reliability={"max_retries": 5})
    ResilientExecution(ex)
    with pytest.raises(AuthError):
        k.process(KernelRequest(payload="q"))
    assert calls["n"] == 1                                # not amplified


# =====================================================================
# E. STARVATION — one hostile input/provider cannot degrade the runtime
# =====================================================================
def test_E_giant_payload_does_not_crash_pipeline():
    big = "word " * 200_000                               # ~1M chars
    k, _ = build(SecurityEngine(secrets_mode="off", pii=True),
                 OptimizationEngine(compress=True))
    resp = k.process(KernelRequest(payload=big))          # must not raise
    assert resp is not None


def test_E_pathological_unicode_does_not_crash():
    nasty = ("\u200b\ufeff\u0000" * 1000 + "🔥" * 500 +
             "\\x00\\xff" * 500 + "a" * 100)
    k, _ = build(SecurityEngine(secrets_mode="block", pii=True),
                 OptimizationEngine(compress=True))
    # may block or pass, but must never crash the pipeline
    try:
        k.process(KernelRequest(payload=nasty))
    except Exception as e:
        assert type(e).__name__ in ("SecretBlocked", "ContentPolicyViolation")


def test_E_dead_provider_bounded_by_breaker_not_storm():
    from tokeymeter.runtime.errors import ProviderDown
    from tokeymeter.runtime.resilience import CircuitBreaker, ResilientExecution
    calls = {"n": 0}

    def dead(p):
        calls["n"] += 1
        raise ProviderDown("p0", RuntimeError("down"))

    k, ex = build(fn=dead, reliability={"max_retries": 0})
    ResilientExecution(ex, breaker=CircuitBreaker(failure_threshold=3))
    for _ in range(100):
        with pytest.raises(Exception):
            k.process(KernelRequest(payload="p"))
    assert calls["n"] == 3                                # storm ceiling holds


def test_E_empty_and_whitespace_payloads_safe():
    k, _ = build(SecurityEngine(secrets_mode="block", pii=True),
                 OptimizationEngine(compress=True))
    for p in ["", " ", "\n\n\n", "\t", "   \u200b   "]:
        resp = k.process(KernelRequest(payload=p))
        assert resp is not None
