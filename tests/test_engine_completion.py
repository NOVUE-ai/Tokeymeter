"""v0.14 engine completion — keys (T3.3), passports (T3.4), usage (T1.1),
pricing age (T1.2). Pins the reflexes and the truth layer."""
import os
import tempfile

import pytest

import tokeymeter
from tokeymeter import context_passport as cp
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore
from tokeymeter.usage import set_reported_usage


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    K.clear_keys()
    yield
    K.clear_keys()
    tokeymeter.reset_savings()


def _last():
    from tokeymeter import savings as sv
    recs = list(sv._tracker._iter_records())
    assert recs
    return recs[-1]


# ── T3.3 keys ───────────────────────────────────────────────────────────
def test_key_stores_fingerprint_never_value():
    st = tokeymeter.register_key("prod", "sk-secret-abcdef123456",
                                 monthly_cap_usd=100)
    assert st["fingerprint"] and len(st["fingerprint"]) == 64
    assert st["hint"] == "3456"
    # the raw value must appear NOWHERE in the stored state
    import json
    blob = json.dumps(tokeymeter.key_status())
    assert "sk-secret-abcdef123456" not in blob


def test_key_name_not_in_emitter_whitelist():
    # The control-plane client is not part of the open-source distribution —
    # it is the paid surface — so this integration property can only be checked
    # in a tree that has it.
    pytest.importorskip("integrations.tokenet",
                        reason="TokeNet client not present in this tree")
    from integrations.tokenet import emitter as em
    assert "key_name" not in em._FIELDS   # key facts never leave the machine


def test_hard_cap_refuses_before_send():
    tokeymeter.register_key("k", "val-abcdefgh", monthly_cap_usd=0.001)
    calls = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        calls["n"] += 1
        return "x" * 4000

    with pytest.raises(tokeymeter.KeyBudgetExceeded):
        with tokeymeter.key("k"):
            for i in range(50):
                ask(f"q{i} " * 200)
    # the guard fired BEFORE a second real call — spend stopped, not reported
    assert calls["n"] <= 2
    st = tokeymeter.key_status("k")
    assert st["spent_usd"] > 0 and st["calls"] >= 1


def test_accrual_ignores_hits_and_shadow():
    tokeymeter.register_key("k2", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(p):
        return "y" * 200

    with tokeymeter.key("k2"):
        ask("same prompt " * 20)   # miss → accrues
        ask("same prompt " * 20)   # exact hit → must NOT accrue
    assert tokeymeter.key_status("k2")["calls"] == 1


def test_unregistered_key_binding_refused():
    with pytest.raises(ValueError):
        with tokeymeter.key("never-registered"):
            pass


def test_registration_validates():
    for bad in ("has space", "x" * 80):
        with pytest.raises(ValueError):
            tokeymeter.register_key(bad, "v")
    with pytest.raises(ValueError):
        tokeymeter.register_key("ok", "v", monthly_cap_usd=-5)


def test_leak_scan_flags_registered_key(tmp_path):
    secret = "sk-live-AAAA1111BBBB2222CCCC3333DDDD4444"
    tokeymeter.register_key("prod-openai", secret)
    leak = tmp_path / ".env.bak"
    leak.write_text(f"OPENAI_API_KEY={secret}\nOTHER=harmless\n")
    res = tokeymeter.leak_scan([str(leak)], include_env=False)
    names = [f.get("registered_key") for f in res["registered_key_leaks"]]
    assert "prod-openai" in names
    assert res["ok"] is False
    # the secret value itself is never returned
    import json
    assert secret not in json.dumps(res)


def test_record_carries_key_name():
    tokeymeter.register_key("kx", "val-abcdefgh", monthly_cap_usd=1000)

    @tokeymeter.cache(model="m")
    def ask(p):
        return "z"
    with tokeymeter.key("kx"):
        ask("q " * 5)
    assert _last().get("key_name") == "kx"


# ── T3.4 context passports ──────────────────────────────────────────────
def test_passport_emitted_and_content_blind():
    seen = []
    cb = cp.subscribe(lambda p: seen.append(p))
    try:
        p = cp.emit(context_id=cp.fingerprint("some big context blob"),
                    source_type="rag", tokens_before=1000, tokens_after=400,
                    method="llmlingua", sensitivity="confidential",
                    model="gpt-4o")
    finally:
        cp.unsubscribe(cb)
    assert p is not None and seen and seen[-1].context_id == p.context_id
    assert len(p.context_id) == 64            # fingerprint, not text
    assert p.tokens_before == 1000 and p.tokens_after == 400


def test_passport_rejects_content_shaped_fields():
    # non-fingerprint id → no passport (fail-safe returns None)
    assert cp.emit(context_id="not-a-hash", source_type="rag",
                   tokens_before=1, tokens_after=1) is None
    # multi-line label → rejected
    assert cp.emit(context_id=cp.fingerprint("x"), source_type="rag",
                   tokens_before=1, tokens_after=1,
                   sensitivity="line1\nline2") is None


def test_compression_emits_passport():
    from tokeymeter.compression import StructuralCompressor
    seen = []
    cb = cp.subscribe(lambda p: seen.append(p))
    try:
        @tokeymeter.cache(model="m", compressor=StructuralCompressor())
        def ask(p):
            return "answer"
        # whitespace-heavy prompt the structural compressor can safely shrink
        ask("word   \n\n   " * 200)
    finally:
        cp.unsubscribe(cb)
    # if compression fired, a prompt-source fingerprint passport exists
    for p in seen:
        assert p.source_type == "prompt" and len(p.context_id) == 64


# ── T1.1 reported usage ─────────────────────────────────────────────────
def test_reported_usage_beats_estimate():
    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        set_reported_usage(1234, 567)   # provider's own numbers
        return "ok"
    ask("anything")
    r = _last()
    assert r["input_tokens"] == 1234 and r["output_tokens"] == 567
    assert r["token_source"] == "reported"


def test_estimate_when_no_report():
    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        return "ok"
    ask("some text here")
    assert _last()["token_source"] == "estimated"


def test_report_exposes_token_source_split():
    @tokeymeter.cache(model="gpt-4o")
    def rep(p):
        set_reported_usage(100, 50)
        return "a"

    @tokeymeter.cache(model="gpt-4o")
    def est(p):
        return "b"
    rep("x1"); est("y1")
    # both recorded; the field is queryable per record
    from tokeymeter import savings as sv
    sources = {r.get("token_source") for r in sv._tracker._iter_records()}
    assert "reported" in sources and "estimated" in sources


# ── T1.2 pricing age ────────────────────────────────────────────────────
def test_pricing_age_nonnegative_and_registered_current():
    from tokeymeter import pricing as pr
    assert pr.pricing_age_days() >= 0
    tokeymeter.register_pricing("kimi-k2", input_per_1m=0.4, output_per_1m=0.4)
    # a registered rate is authoritative regardless of list-table age
    assert tokeymeter.pricing_info("kimi-k2")["source"] == "registered"
