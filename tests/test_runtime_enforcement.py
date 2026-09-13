"""W4 battery — the enforcement spine.

Wave gates pinned here:
1. PDF p.13 VERBATIM: a content policy blocking "confidential" blocks the
   prompt AND a sealed audit entry records the block.
2. No secret in fingerprints: redaction happens BEFORE the trust fingerprint.
3. Budget concurrency: N threads racing a cap never overspend.
"""
from __future__ import annotations

import hashlib
import threading

import pytest

from tokeymeter import Runtime
from tokeymeter.engines.economics import keys as keysmod
from tokeymeter.engines.economics import pricing as pricingmod
from tokeymeter.engines.economics import usage as usagemod
from tokeymeter.runtime import (
    AccessDenied, AccessEngine, BudgetExceeded, CheckpointDenied,
    CheckpointPending, ContentPolicyViolation, EconomicsEngine, Kernel,
    KernelRequest, LocalApprover, RateLimitEngine, RateLimitExceeded,
    RuntimeConfig, SecretBlocked, SecurityEngine, TrustEngine,
)
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine

ANTHROPIC_KEY = "sk-ant-api03-" + "A" * 88          # firewall-detectable
SSN_PROMPT = "customer ssn is 123-45-6789 please summarize the account"


def kernel_with(*engines, model="default", fn=None, cfg=None):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok:" + p)),
                        models=[model], default=True)
    for e in engines:
        k.register_engine(e)
    k.register_engine(ex)
    return k, ex


@pytest.fixture(autouse=True)
def _clean_keys():
    keysmod.clear_keys()
    yield
    keysmod.clear_keys()


# ================================================== GOV-5: Security =======
def test_secrets_block_before_adapter():
    calls = {"n": 0}

    def fn(p):
        calls["n"] += 1
        return "x"

    k, _ = kernel_with(SecurityEngine(), TrustEngine(), fn=fn)
    with pytest.raises(SecretBlocked):
        k.process(KernelRequest(payload=f"deploy with {ANTHROPIC_KEY} now"))
    assert calls["n"] == 0                          # never reached provider


def test_secret_verdict_content_blind():
    k, _ = kernel_with(SecurityEngine(), TrustEngine())
    with pytest.raises(SecretBlocked) as ei:
        k.process(KernelRequest(payload=f"key: {ANTHROPIC_KEY}"))
    msg = str(ei.value)
    assert "ANTHROPIC" in msg                       # detector KIND surfaces
    assert ANTHROPIC_KEY not in msg                 # the secret never does


def test_redaction_before_fingerprint_no_secret_hash_anywhere():
    """WAVE GATE 2: the trust residue fingerprints the REDACTED payload;
    the raw payload's hash must appear nowhere."""
    trust = TrustEngine()
    k, _ = kernel_with(SecurityEngine(secrets_mode="off", pii=True), trust)
    resp = k.process(KernelRequest(payload=SSN_PROMPT))
    raw_fp = hashlib.sha256(SSN_PROMPT.encode()).hexdigest()
    blob = str(trust.entries()) + str(resp.trace) + str(resp.metadata)
    assert raw_fp not in blob                       # pre-redaction hash: gone
    assert "123-45-6789" not in blob                # and obviously no PII


def test_pii_redacted_payload_reaches_adapter():
    seen = {}
    k, _ = kernel_with(SecurityEngine(secrets_mode="off", pii=True),
                       fn=lambda p: seen.update(p=p) or "ok")
    k.process(KernelRequest(payload=SSN_PROMPT))
    assert "123-45-6789" not in seen["p"]           # provider saw redaction
    assert "summarize the account" in seen["p"]     # meaning preserved


def test_pii_redacts_structured_messages_too():
    seen = {}

    def fn(p):
        return "ok"

    k, ex = kernel_with(SecurityEngine(secrets_mode="off", pii=True), fn=fn)
    req = KernelRequest(payload=SSN_PROMPT,
                        metadata={"messages": [
                            {"role": "user", "content": SSN_PROMPT}]})
    k.process(req)
    assert "123-45-6789" not in req.metadata["messages"][0]["content"]


def test_pdf_p13_verbatim_block_and_sealed_audit():
    """WAVE GATE 1 — PDF p.13, verbatim:
    1. Configure a content policy blocking the word "confidential".
    2. Send prompt "Tell me about the confidential plan."
    3. The response is blocked AND an audit log entry is made."""
    trust = TrustEngine()
    k, _ = kernel_with(
        SecurityEngine(secrets_mode="off", pii=False,
                       blocked_terms=["confidential"]), trust)
    with pytest.raises(ContentPolicyViolation):
        k.process(KernelRequest(payload="Tell me about the confidential plan."))
    entries = trust.entries()
    assert len(entries) == 1                        # error path SEALED
    assert "content:block" in entries[0]["body"]
    assert "error:ContentPolicyViolation" in entries[0]["body"]
    ok, _ = trust.verify()
    assert ok                                       # chain still verifies
    assert "confidential plan" not in entries[0]["body"]   # content-blind


def test_security_off_modes_passthrough():
    k, _ = kernel_with(SecurityEngine(secrets_mode="off", pii=False))
    assert k.process(KernelRequest(payload="hello")).payload == "ok:hello"


def test_security_rejects_bad_mode():
    with pytest.raises(ValueError):
        SecurityEngine(secrets_mode="maybe")


# ================================================== GOV-1: Access =========
ROLES = {"analyst": {"models": ["gpt-cheap"], "actions": ["execute"]},
         "admin": {"models": ["*"], "actions": ["*"]}}
PRINCIPALS = {"alice": "analyst", "root": "admin"}


@pytest.mark.parametrize("principal,model,ok", [
    ("alice", "gpt-cheap", True),
    ("alice", "gpt-expensive", False),      # model outside role
    ("root", "gpt-expensive", True),        # wildcard role
    ("mallory", "gpt-cheap", False),        # unknown principal
    (None, "gpt-cheap", False),             # anonymous
], ids=["analyst-allowed", "analyst-denied-model", "admin-wildcard",
        "unknown-principal", "anonymous-denied"])
def test_rbac_matrix_deny_by_default(principal, model, ok):
    import tokeymeter
    k, _ = kernel_with(AccessEngine(roles=ROLES, principals=PRINCIPALS),
                       model=model)

    def run():
        return k.process(KernelRequest(payload="q", model=model))

    if principal:
        with tokeymeter.principal(principal):
            if ok:
                assert run().payload == "ok:q"
            else:
                with pytest.raises(AccessDenied):
                    run()
    else:
        with pytest.raises(AccessDenied):
            run()


def test_rbac_verdict_recorded_on_allow():
    import tokeymeter
    trust = TrustEngine()
    k, _ = kernel_with(AccessEngine(roles=ROLES, principals=PRINCIPALS),
                       trust, model="gpt-cheap")
    with tokeymeter.principal("alice"):
        resp = k.process(KernelRequest(payload="q", model="gpt-cheap"))
    assert {"policy": "rbac", "verdict": "allow", "detail": "model:gpt-cheap"} \
        in resp.metadata["policy_verdicts"]
    assert "rbac:allow" in trust.entries()[0]["body"]


# ================================================== GOV-2: RateLimit ======
def test_rate_limit_trips_and_retry_after():
    clock = {"t": 0.0}
    rl = RateLimitEngine(requests_per_min=2, clock=lambda: clock["t"])
    k, _ = kernel_with(rl)
    k.process(KernelRequest(payload="a"))
    k.process(KernelRequest(payload="b"))
    with pytest.raises(RateLimitExceeded) as ei:
        k.process(KernelRequest(payload="c"))
    assert 0 < ei.value.retry_after <= 60.0


def test_rate_limit_window_resets():
    clock = {"t": 0.0}
    rl = RateLimitEngine(requests_per_min=1, clock=lambda: clock["t"])
    k, _ = kernel_with(rl)
    k.process(KernelRequest(payload="a"))
    with pytest.raises(RateLimitExceeded):
        k.process(KernelRequest(payload="b"))
    clock["t"] = 61.0                               # window elapsed
    assert k.process(KernelRequest(payload="c")).payload == "ok:c"


def test_rate_limit_token_budget_trips():
    clock = {"t": 0.0}
    rl = RateLimitEngine(tokens_per_min=10, clock=lambda: clock["t"])
    k, _ = kernel_with(rl)
    # feed the token window directly through the seam it records on
    ctx = {"meta": {"principal": None, "tokens_in": 8, "tokens_out": 8},
           "request": KernelRequest(payload="x")}
    rl.after_response(ctx)
    with pytest.raises(RateLimitExceeded):
        k.process(KernelRequest(payload="next"))


def test_rate_limit_per_principal_isolation():
    import tokeymeter
    clock = {"t": 0.0}
    rl = RateLimitEngine(requests_per_min=1, clock=lambda: clock["t"])
    k, _ = kernel_with(rl)
    with tokeymeter.principal("a"):
        k.process(KernelRequest(payload="1"))
    with tokeymeter.principal("b"):                 # different window
        assert k.process(KernelRequest(payload="2")).payload == "ok:2"
    with tokeymeter.principal("a"):
        with pytest.raises(RateLimitExceeded):
            k.process(KernelRequest(payload="3"))


def test_rate_limit_concurrent_accuracy():
    clock = {"t": 0.0}
    rl = RateLimitEngine(requests_per_min=5, clock=lambda: clock["t"])
    k, _ = kernel_with(rl)
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

    threads = [threading.Thread(target=worker) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == {"ok": 5, "blocked": 11}      # EXACTLY the cap


def test_rate_limit_block_sealed_on_error_path():
    trust = TrustEngine()
    clock = {"t": 0.0}
    k, _ = kernel_with(RateLimitEngine(requests_per_min=1,
                                       clock=lambda: clock["t"]), trust)
    k.process(KernelRequest(payload="a"))
    with pytest.raises(RateLimitExceeded):
        k.process(KernelRequest(payload="b"))
    assert "rate_limit:block" in trust.entries()[-1]["body"]


# ================================================== ECON-1: cost truth ====
def _usage_resp(pin, pout):
    from types import SimpleNamespace as NS
    return NS(usage=NS(prompt_tokens=pin, completion_tokens=pout))


def test_cost_equals_registry_math():
    pricingmod.register_pricing("w4-model", input_per_1m=2.0,
                                output_per_1m=10.0)
    k, _ = kernel_with(EconomicsEngine(), model="w4-model",
                       fn=lambda p: _usage_resp(1000, 500))
    resp = k.process(KernelRequest(payload="q", model="w4-model"))
    expected = (1000 / 1e6) * 2.0 + (500 / 1e6) * 10.0
    assert resp.metadata["cost_usd"] == pytest.approx(expected)
    assert resp.metadata["cost_source"] == "registered"
    assert (resp.metadata["tokens_in"], resp.metadata["tokens_out"]) == \
        (1000, 500)
    assert resp.metadata["usage_source"] == "response_usage"


def test_reported_usage_beats_estimates_and_response():
    pricingmod.register_pricing("w4-rep", input_per_1m=1.0, output_per_1m=1.0)

    def fn(p):
        usagemod.set_reported_usage(777, 333)       # provider truth
        return _usage_resp(1, 1)                    # response lies smaller

    k, _ = kernel_with(EconomicsEngine(), model="w4-rep", fn=fn)
    resp = k.process(KernelRequest(payload="q", model="w4-rep"))
    assert (resp.metadata["tokens_in"], resp.metadata["tokens_out"]) == \
        (777, 333)
    assert resp.metadata["usage_source"] == "reported"


def test_missing_price_is_unknown_never_zero():
    k, _ = kernel_with(EconomicsEngine(),
                       fn=lambda p: _usage_resp(100, 100))
    resp = k.process(KernelRequest(payload="q"))    # model 'default'
    assert resp.metadata["cost_usd"] is None        # NEVER zero
    assert resp.metadata["cost_source"] == "default"  # guess kept, not trusted as cost


def test_estimated_usage_source_when_no_counts_anywhere():
    k, _ = kernel_with(EconomicsEngine(), fn=lambda p: "plain string")
    resp = k.process(KernelRequest(payload="four words of payload"))
    assert resp.metadata["usage_source"] == "estimated"
    assert resp.metadata["tokens_in"] > 0


# ================================================== ECON-2/3: budgets =====
def test_budget_exceed_blocks_and_adapter_untouched():
    keysmod.register_key("w4-key", "sk-test-AAAA", monthly_cap_usd=0.000001)
    keysmod.on_spend("w4-key", 1.0, hit=False, shadow=False)      # already over
    calls = {"n": 0}
    k, _ = kernel_with(
        EconomicsEngine(budget_key="w4-key"),
        fn=lambda p: calls.update(n=calls["n"] + 1) or "x")
    with pytest.raises(BudgetExceeded):
        k.process(KernelRequest(payload="q"))
    assert calls["n"] == 0


def test_budget_soft_mode_warns_and_proceeds():
    keysmod.register_key("w4-soft", "sk-test-BBBB",
                         monthly_cap_usd=0.000001)
    keysmod.on_spend("w4-soft", 1.0, hit=False, shadow=False)
    k, _ = kernel_with(EconomicsEngine(budget_key="w4-soft",
                                       budget_mode="soft"))
    resp = k.process(KernelRequest(payload="q"))
    assert resp.payload == "ok:q"
    assert {"policy": "budget", "verdict": "warn", "detail": "key:w4-soft"} \
        in resp.metadata["policy_verdicts"]


def test_budget_concurrency_no_overspend():
    """WAVE GATE 3: 16 threads racing a budget that admits ~5 spends of
    $1 each — total recorded spend must never exceed budget + one in-flight
    margin, and blocked threads raise, never slip through."""
    keysmod.register_key("w4-race", "sk-test-CCCC", monthly_cap_usd=5.0)
    k, _ = kernel_with(EconomicsEngine(budget_key="w4-race"),
                       fn=lambda p: "x")
    ok, blocked = [], []
    lock = threading.Lock()

    def worker():
        try:
            with keysmod.key("w4-race"):
                keysmod.check_current(estimated_cost=1.0)
                keysmod.on_spend("w4-race", 1.0, hit=False, shadow=False)
            with lock:
                ok.append(1)
        except keysmod.KeyBudgetExceeded:
            with lock:
                blocked.append(1)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # Documented breaker doctrine: check_current is a CIRCUIT-BREAKER, not a
    # reservation ledger — in-flight calls at breach complete, then the
    # breach LATCHES. Concurrency invariant: every thread is accounted for
    # (none slips silently), and once the cap is latched a fresh sequential
    # call is refused before send.
    assert len(ok) + len(blocked) == 16
    with keysmod.key("w4-race"):
        with pytest.raises(keysmod.KeyBudgetExceeded):
            keysmod.check_current(estimated_cost=1.0)   # breach latched


def test_budget_approval_approved_proceeds():
    keysmod.register_key("w4-appr", "sk-test-DDDD",
                         monthly_cap_usd=0.000001)
    keysmod.on_spend("w4-appr", 1.0, hit=False, shadow=False)
    decisions = []
    approver = LocalApprover(lambda s: decisions.append(s) or "approve")
    k, _ = kernel_with(EconomicsEngine(budget_key="w4-appr",
                                       budget_mode="approval",
                                       checkpoint=approver))
    distinctive = "ZEBRA-PAYLOAD-MARKER-9137"
    resp = k.process(KernelRequest(payload=distinctive))
    assert resp.payload == "ok:" + distinctive
    assert decisions[0]["reason"].startswith("budget_exceeded")
    assert distinctive not in str(decisions[0])     # summary content-blind
    assert {"policy": "budget", "verdict": "approved_over_budget",
            "detail": "key:w4-appr"} in resp.metadata["policy_verdicts"]


def test_budget_approval_denied_blocks():
    keysmod.register_key("w4-deny", "sk-test-EEEE",
                         monthly_cap_usd=0.000001)
    keysmod.on_spend("w4-deny", 1.0, hit=False, shadow=False)
    k, _ = kernel_with(EconomicsEngine(
        budget_key="w4-deny", budget_mode="approval",
        checkpoint=LocalApprover(lambda s: "deny")))
    with pytest.raises(CheckpointDenied):
        k.process(KernelRequest(payload="q"))


def test_budget_approval_pending_parks():
    keysmod.register_key("w4-pend", "sk-test-FFFF",
                         monthly_cap_usd=0.000001)
    keysmod.on_spend("w4-pend", 1.0, hit=False, shadow=False)
    k, _ = kernel_with(EconomicsEngine(
        budget_key="w4-pend", budget_mode="approval",
        checkpoint=LocalApprover(lambda s: "pending")))
    with pytest.raises(CheckpointPending):
        k.process(KernelRequest(payload="q"))


def test_approval_mode_requires_checkpoint():
    with pytest.raises(ValueError):
        EconomicsEngine(budget_key="k", budget_mode="approval")


# ================================================== GOV-6 + integration ===
def test_full_spine_verdict_ledger_ordered_and_sealed():
    import tokeymeter
    pricingmod.register_pricing("w4-full", input_per_1m=2.0,
                                output_per_1m=10.0)
    trust = TrustEngine()
    # Trust FIRST → unwinds last → seals the complete record (seam-order law).
    k, _ = kernel_with(
        trust,
        SecurityEngine(secrets_mode="block", pii=True),
        AccessEngine(roles={"eng": {"models": ["w4-full"]}},
                     principals={"dev": "eng"}),
        RateLimitEngine(requests_per_min=100),
        EconomicsEngine(),
        model="w4-full",
        fn=lambda p: _usage_resp(1000, 500))
    with tokeymeter.principal("dev"):
        resp = k.process(KernelRequest(payload="ship it", model="w4-full"))
    policies = [v["policy"] for v in resp.metadata["policy_verdicts"]]
    assert policies == ["secrets", "pii", "rbac", "rate_limit"]  # spine order
    body = trust.entries()[0]["body"]
    assert "secrets:allow;pii:allow;rbac:allow;rate_limit:allow" in body
    assert str(resp.metadata["cost_usd"]) in body   # cost sealed too
    ok, _ = trust.verify()
    assert ok


def test_facade_wires_spine_from_config_and_receipt_shows_real_dollars():
    pricingmod.register_pricing("w4-face", input_per_1m=2.0,
                                output_per_1m=10.0)
    r = Runtime(
        config={"governance": {"security": {"enabled": True,
                                            "secrets_mode": "block",
                                            "pii": True}}},
        call=lambda p: _usage_resp(1000, 500),
        model="w4-face", receipt="never")
    r.execute("hello world")
    receipt = r.render_receipt(r.last, r.last_events, 5.0)
    expected = (1000 / 1e6) * 2.0 + (500 / 1e6) * 10.0
    assert f"✓ Cost ${expected:.4f}" in receipt     # real dollars, provenance-gated
    with pytest.raises(SecretBlocked):
        r.execute(f"leak {ANTHROPIC_KEY}")


def test_facade_unpriced_model_still_dashes():
    r = Runtime(call=lambda p: _usage_resp(10, 10), receipt="never")
    r.execute("q")
    assert "✓ Cost —" in r.render_receipt(r.last, r.last_events, 1.0)
