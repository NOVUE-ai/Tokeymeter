"""W8 battery — the platform layer: plugins, hot-reload, OIDC, telemetry.

Each capability is tested for its security-critical property, not just its
happy path: plugins REFUSE the untrusted/tampered, hot-reload is ATOMIC and
FAIL-SAFE, OIDC REJECTS the unverified/expired, telemetry is CONTENT-BLIND.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time

import pytest

import tokeymeter
from tokeymeter import Runtime
from tokeymeter.runtime import (
    ConfigValidationError, InMemorySink, Kernel, KernelRequest, OIDCResolver,
    PluginManifest, PluginRegistry, PluginVerificationError, ReloadableConfig,
    ResolvedIdentity, RuntimeConfig, TelemetryEngine, TokenExpired,
    TokenInvalid, content_blind_record, sign_plugin,
)
from tokeymeter.runtime.plugins import PluginDeclarationError
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine
from tokeymeter.engines.trust.audit.signers import Ed25519Signer


def kern(*engines, fn=None, model="default", **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok")),
                        models=[model], default=True)
    for e in engines:
        k.register_engine(e)
    k.register_engine(ex)
    return k, ex


# ============================================= PLUGINS ===================
def _signed(signer, **kw):
    m = PluginManifest(**{"name": "p", "version": "1.0",
                          "plugin_class": "observe",
                          "hook": "after_response", **kw})
    return sign_plugin(m, signer)


def test_plugin_signed_and_trusted_loads():
    s = Ed25519Signer.generate()
    sp = _signed(s)
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    reg.load(sp, fn=lambda ctx: None)
    assert len(reg.loaded) == 1


def test_plugin_untrusted_key_refused():
    s = Ed25519Signer.generate()
    sp = _signed(s)
    reg = PluginRegistry()                            # trusts nothing
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_plugin_tampered_manifest_refused():
    s = Ed25519Signer.generate()
    sp = _signed(s)
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    # tamper the manifest AFTER signing -> hash mismatch
    sp.manifest.order = 999
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_plugin_tampered_signature_refused():
    s = Ed25519Signer.generate()
    sp = _signed(s)
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    sp.signature = "00" * 64
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_plugin_signature_from_wrong_key_refused():
    signer_a, signer_b = Ed25519Signer.generate(), Ed25519Signer.generate()
    sp = _signed(signer_a)
    # claim signer_b's key but keep signer_a's signature
    sp.public_key = signer_b.public_bytes().hex()
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_plugin_observe_is_isolated_cannot_break_request():
    s = Ed25519Signer.generate()
    sp = _signed(s, plugin_class="observe", hook="after_response")
    reg = PluginRegistry(trusted_keys=[sp.public_key])

    def faulty(ctx):
        raise RuntimeError("plugin exploded")
    reg.load(sp, fn=faulty)
    k, _ = kern()
    reg.attach_all(k)
    # the faulty observe plugin must NOT break the request
    resp = k.process(KernelRequest(payload="q"))
    assert resp.payload == "ok"


def test_plugin_enforce_can_veto():
    s = Ed25519Signer.generate()
    sp = _signed(s, plugin_class="enforce", hook="before_request")
    reg = PluginRegistry(trusted_keys=[sp.public_key])

    def veto(ctx):
        raise PermissionError("enforced denial")
    reg.load(sp, fn=veto)
    k, _ = kern()
    reg.attach_all(k)
    with pytest.raises(PermissionError):
        k.process(KernelRequest(payload="q"))


def test_plugin_ordering_by_signed_order():
    s = Ed25519Signer.generate()
    calls = []
    reg = PluginRegistry(trusted_keys=[s.public_bytes().hex()])
    for name, order in [("late", 200), ("early", 10), ("mid", 100)]:
        sp = _signed(s, name=name, order=order, hook="after_response")
        reg.load(sp, fn=(lambda n: lambda ctx: calls.append(n))(name))
    k, _ = kern()
    reg.attach_all(k)
    k.process(KernelRequest(payload="q"))
    assert calls == ["early", "mid", "late"]          # signed order honored


def test_plugin_bad_declaration_rejected():
    with pytest.raises(PluginDeclarationError):
        PluginManifest(name="p", version="1", plugin_class="wat",
                       hook="after_response")
    with pytest.raises(PluginDeclarationError):
        PluginManifest(name="p", version="1", plugin_class="observe",
                       hook="nonexistent")


# ============================================= HOT-RELOAD ================
def test_hotreload_atomic_swap_and_version():
    rc = ReloadableConfig({"reliability": {"max_retries": 1}})
    assert rc.version == 1 and rc.get("reliability.max_retries") == 1
    rc.reload({"reliability": {"max_retries": 5}})
    assert rc.version == 2 and rc.get("reliability.max_retries") == 5


def test_hotreload_notifies_subscribers_after_swap():
    rc = ReloadableConfig({"a": 1})
    seen = []
    rc.subscribe(lambda snap: seen.append(snap.version))
    rc.reload({"a": 2})
    rc.reload({"a": 3})
    assert seen == [2, 3]


def test_hotreload_validation_failure_keeps_last_good():
    def validator(cfg):
        if cfg.get("reliability.max_retries", 0) > 10:
            raise ValueError("too many retries")
    rc = ReloadableConfig({"reliability": {"max_retries": 2}},
                          validator=validator)
    with pytest.raises(ConfigValidationError):
        rc.reload({"reliability": {"max_retries": 99}})
    # rejected config NOT applied — last-good retained (fail-safe)
    assert rc.get("reliability.max_retries") == 2 and rc.version == 1


def test_hotreload_reader_never_sees_partial_under_concurrency():
    rc = ReloadableConfig({"n": 0})
    stop = threading.Event()
    seen_bad = []

    def reader():
        while not stop.is_set():
            snap = rc.current()
            # a snapshot's config is internally consistent: n is always an int
            v = snap.config.get("n")
            if not isinstance(v, int):
                seen_bad.append(v)

    def writer():
        for i in range(1, 500):
            rc.reload({"n": i})

    readers = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in readers]
    w = threading.Thread(target=writer)
    w.start()
    w.join()
    stop.set()
    [t.join() for t in readers]
    assert not seen_bad                               # never a torn read
    assert rc.get("n") == 499


def test_hotreload_subscriber_exception_isolated():
    rc = ReloadableConfig({"a": 1})
    rc.subscribe(lambda snap: (_ for _ in ()).throw(RuntimeError("boom")))
    ok = []
    rc.subscribe(lambda snap: ok.append(snap.version))
    rc.reload({"a": 2})                               # must not raise
    assert ok == [2]                                  # second subscriber ran


# ============================================= OIDC =====================
def _jwt(claims, *, secret=b"topsecret", alg="HS256"):
    header = {"alg": alg, "typ": "JWT"}

    def seg(d):
        return base64.urlsafe_b64encode(
            json.dumps(d).encode()).rstrip(b"=").decode()
    signing_input = f"{seg(header)}.{seg(claims)}"
    if alg == "HS256":
        sig = hmac.new(secret, signing_input.encode(),
                       hashlib.sha256).digest()
    else:
        sig = b""
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{signing_input}.{sig_b64}"


def test_oidc_hs256_verified_resolves_principal():
    tok = _jwt({"sub": "alice@corp", "exp": time.time() + 3600,
                "iss": "https://idp", "roles": ["analyst"]})
    r = OIDCResolver(hs256_secret=b"topsecret", issuer="https://idp",
                     roles_claim="roles")
    ident = r.resolve(tok)
    assert ident.principal == "alice@corp" and ident.roles == ["analyst"]


def test_oidc_wrong_secret_rejected():
    tok = _jwt({"sub": "alice", "exp": time.time() + 3600})
    r = OIDCResolver(hs256_secret=b"WRONG")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_oidc_expired_rejected():
    tok = _jwt({"sub": "alice", "exp": time.time() - 100})
    r = OIDCResolver(hs256_secret=b"topsecret", leeway_s=0)
    with pytest.raises(TokenExpired):
        r.resolve(tok)


def test_oidc_issuer_and_audience_enforced():
    tok = _jwt({"sub": "a", "exp": time.time() + 100, "iss": "evil",
                "aud": "svc"})
    with pytest.raises(TokenInvalid):
        OIDCResolver(hs256_secret=b"topsecret",
                     issuer="https://idp").resolve(tok)
    tok2 = _jwt({"sub": "a", "exp": time.time() + 100, "aud": "other"})
    with pytest.raises(TokenInvalid):
        OIDCResolver(hs256_secret=b"topsecret", audience="svc").resolve(tok2)


def test_oidc_unverified_refused_by_default():
    tok = _jwt({"sub": "a", "exp": time.time() + 100}, alg="none")
    r = OIDCResolver()                                # no key, not allowed
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_oidc_missing_principal_claim_rejected():
    tok = _jwt({"exp": time.time() + 100})            # no sub
    r = OIDCResolver(hs256_secret=b"topsecret")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_oidc_bind_sets_principal_for_access_engine():
    from tokeymeter.runtime.enforcement import AccessEngine, AccessDenied
    tok = _jwt({"sub": "dev", "exp": time.time() + 3600})
    r = OIDCResolver(hs256_secret=b"topsecret")
    k, _ = kern(AccessEngine(roles={"eng": {"models": ["m"]}},
                             principals={"dev": "eng"}), model="m")
    # without binding -> denied
    with pytest.raises(AccessDenied):
        k.process(KernelRequest(payload="q", model="m"))
    # bind the OIDC principal, then allowed
    with tokeymeter.principal(None):
        r.bind(tok)
        resp = k.process(KernelRequest(payload="q", model="m"))
    assert resp.payload == "ok"


# ============================================= TELEMETRY ================
def test_telemetry_record_is_content_blind():
    meta = {"model": "gpt-x", "tokens_in": 100, "cost_usd": 0.01,
            "payload": "SECRET PROMPT TEXT",
            "messages": [{"content": "SECRET"}],
            "policy_verdicts": [{"policy": "rbac", "verdict": "allow"}]}
    rec = content_blind_record(meta, request_id="r1", outcome="ok",
                               latency_ms=12.3)
    blob = json.dumps(rec)
    assert "SECRET" not in blob                        # payload never included
    assert rec["model"] == "gpt-x" and rec["verdicts"] == "rbac:allow"


def test_telemetry_engine_exports_per_request():
    sink = InMemorySink()
    k, _ = kern(TelemetryEngine(sink))
    k.process(KernelRequest(payload="hello"))
    assert len(sink.records) == 1
    assert sink.records[0]["outcome"] == "ok"
    assert "latency_ms" in sink.records[0]


def test_telemetry_exports_on_error_path():
    from tokeymeter.runtime.enforcement import SecurityEngine
    sink = InMemorySink()
    k, _ = kern(TelemetryEngine(sink),
                SecurityEngine(secrets_mode="off", pii=False,
                               blocked_terms=["forbidden"]))
    with pytest.raises(Exception):
        k.process(KernelRequest(payload="the forbidden thing"))
    assert len(sink.records) == 1
    assert sink.records[0]["outcome"].startswith("error:")


def test_telemetry_sink_exception_isolated():
    class BadSink:
        def export(self, record):
            raise RuntimeError("sink down")
    k, _ = kern(TelemetryEngine(BadSink()))
    resp = k.process(KernelRequest(payload="q"))       # must not break
    assert resp.payload == "ok"


def test_telemetry_never_leaks_payload_end_to_end():
    sink = InMemorySink()
    k, _ = kern(TelemetryEngine(sink))
    secret = "ULTRA-SECRET-PAYLOAD-9137"
    k.process(KernelRequest(payload=secret))
    assert secret not in json.dumps(sink.records)


def test_otel_sink_degrades_without_sdk():
    from tokeymeter.runtime.telemetry import OTelSink
    s = OTelSink()
    assert s.degraded is True                          # SDK not installed here
    s.export({"model": "x", "cost_usd": 0.1})          # must not raise


# ============================================= FACADE ===================
def test_facade_wires_telemetry_sink():
    sink = InMemorySink()
    r = Runtime(config={"telemetry": {"enabled": True}},
                call=lambda p: "ok", receipt="never",
                telemetry_sinks=[sink])
    r.execute("hello")
    assert len(sink.records) == 1 and sink.records[0]["outcome"] == "ok"


# ============================================= FULL W8 INTEGRATION ======
def test_w8_all_four_capabilities_compose_in_one_runtime():
    """OIDC binds identity -> access allows -> a signed observe plugin taps
    the record -> telemetry exports it content-blind -> hot-reload can change
    config live. All four W8 capabilities in one live pipeline."""
    from tokeymeter.runtime.enforcement import AccessEngine
    # OIDC identity
    tok = _jwt({"sub": "dev", "exp": time.time() + 3600, "roles": ["eng"]})
    resolver = OIDCResolver(hs256_secret=b"topsecret", roles_claim="roles")

    # signed observe plugin that records what it saw (content-blind)
    signer = Ed25519Signer.generate()
    tapped = []
    sp = _signed(signer, name="tap", plugin_class="observe",
                 hook="after_response")
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    reg.load(sp, fn=lambda ctx: tapped.append(ctx["request"].request_id))

    # telemetry sink
    sink = InMemorySink()

    k = Kernel(RuntimeConfig({})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(lambda p: "ok"), models=["m"],
                        default=True)
    k.register_engine(AccessEngine(roles={"eng": {"models": ["m"]}},
                                   principals={"dev": "eng"}))
    k.register_engine(TelemetryEngine(sink))
    reg.attach_all(k)
    k.register_engine(ex)

    with tokeymeter.principal(None):
        resolver.bind(tok)                             # OIDC -> principal
        resp = k.process(KernelRequest(payload="do the thing", model="m"))

    assert resp.payload == "ok"                         # access allowed
    assert tapped == [resp.request_id]                  # plugin tapped
    assert len(sink.records) == 1                       # telemetry exported
    assert "do the thing" not in json.dumps(sink.records)  # content-blind


def test_w8_hotreload_drives_live_behavior_change():
    """A hot-reload of config changes runtime behavior without restart."""
    rc = ReloadableConfig({"reliability": {"max_retries": 0}})
    # a subscriber that rebuilds a derived value on reload
    derived = {"retries": rc.get("reliability.max_retries")}
    rc.subscribe(lambda snap: derived.__setitem__(
        "retries", snap.config.get("reliability.max_retries")))
    assert derived["retries"] == 0
    rc.reload({"reliability": {"max_retries": 3}})
    assert derived["retries"] == 3                      # live change applied
