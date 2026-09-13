"""W8/W9 DEEP ADVERSARIAL RED-TEAM — the platform & universality surface
under attack by a hostile security assessor.

Functional tests prove these features work; THESE prove they cannot be
subverted. Every test names a concrete attack from the real threat model for
identity systems, plugin loaders, and multi-provider fabrics:

  J. JWT ATTACKS       — alg-confusion, none-injection, signature stripping,
                         claim smuggling, expiry/nbf edge, key confusion
  P. PLUGIN FORGERY    — deep signature/trust attacks on the loader
  F. FRAMEWORK INJECT  — malicious content through the framework wrap points
  C. CATALOG ISOLATION — one provider's config cannot poison another
  H. HOTRELOAD RACE    — config swap under adversarial concurrency
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time

import pytest

from tokeymeter.runtime import (
    ConfigValidationError, OIDCResolver, PluginManifest, PluginRegistry,
    PluginVerificationError, ReloadableConfig, TokenExpired, TokenInvalid,
    adapter_for, check_adapter, get_spec, governed_tool, sign_plugin,
    wrap_callable,
)
from tokeymeter.runtime.conformance import ConformanceClient
from tokeymeter.engines.trust.audit.signers import Ed25519Signer


def _b64(d):
    return base64.urlsafe_b64encode(
        json.dumps(d).encode()).rstrip(b"=").decode()


def _jwt(claims, *, secret=b"realsecret", alg="HS256", sig_override=None):
    header = {"alg": alg, "typ": "JWT"}
    signing_input = f"{_b64(header)}.{_b64(claims)}"
    if sig_override is not None:
        sig = sig_override
    elif alg == "HS256":
        sig = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    else:
        sig = b""
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{signing_input}.{sig_b64}"


# =====================================================================
# J. JWT ATTACKS — the classic identity-system exploit class
# =====================================================================
def test_J_alg_none_injection_refused():
    """Attacker sets alg=none to bypass signature verification entirely."""
    tok = _jwt({"sub": "admin", "exp": time.time() + 999}, alg="none")
    r = OIDCResolver(hs256_secret=b"realsecret")       # expects HS256
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_signature_stripping_refused():
    """Attacker removes the signature but keeps a valid-looking token."""
    tok = _jwt({"sub": "admin", "exp": time.time() + 999})
    stripped = tok.rsplit(".", 1)[0] + "."          # empty signature
    r = OIDCResolver(hs256_secret=b"realsecret")
    with pytest.raises(TokenInvalid):
        r.resolve(stripped)


def test_J_forged_signature_refused():
    """Attacker fabricates a signature without knowing the secret."""
    tok = _jwt({"sub": "admin", "exp": time.time() + 999},
               sig_override=b"\x00" * 32)
    r = OIDCResolver(hs256_secret=b"realsecret")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_wrong_secret_cannot_verify():
    """A token signed with a different secret must not verify."""
    tok = _jwt({"sub": "admin", "exp": time.time() + 999},
               secret=b"attacker-secret")
    r = OIDCResolver(hs256_secret=b"realsecret")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_alg_confusion_eddsa_token_to_hs256_verifier():
    """Attacker submits an EdDSA-alg token to an HS256 verifier hoping the
    verifier mishandles the algorithm. Must refuse, not verify."""
    tok = _jwt({"sub": "admin", "exp": time.time() + 999}, alg="EdDSA",
               sig_override=b"\x01" * 64)
    r = OIDCResolver(hs256_secret=b"realsecret")       # only HS256 configured
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_expired_token_not_accepted_even_one_second():
    tok = _jwt({"sub": "u", "exp": time.time() - 1})
    r = OIDCResolver(hs256_secret=b"realsecret", leeway_s=0)
    with pytest.raises(TokenExpired):
        r.resolve(tok)


def test_J_not_before_in_future_refused():
    tok = _jwt({"sub": "u", "exp": time.time() + 999,
                "nbf": time.time() + 500})
    r = OIDCResolver(hs256_secret=b"realsecret", leeway_s=0)
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_issuer_spoofing_refused():
    tok = _jwt({"sub": "u", "exp": time.time() + 999, "iss": "https://evil"})
    r = OIDCResolver(hs256_secret=b"realsecret", issuer="https://trusted")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_audience_confusion_refused():
    tok = _jwt({"sub": "u", "exp": time.time() + 999, "aud": "other-service"})
    r = OIDCResolver(hs256_secret=b"realsecret", audience="my-service")
    with pytest.raises(TokenInvalid):
        r.resolve(tok)


def test_J_malformed_tokens_never_crash():
    r = OIDCResolver(hs256_secret=b"realsecret")
    for bad in ["", "notajwt", "a.b", "a.b.c.d", "...", "x." * 3,
                base64.b64encode(b"garbage").decode()]:
        with pytest.raises(TokenInvalid):
            r.resolve(bad)


def test_J_principal_claim_cannot_be_empty_or_missing():
    for claims in [{"exp": time.time() + 99},        # no sub
                   {"sub": "", "exp": time.time() + 99},   # empty sub
                   {"sub": None, "exp": time.time() + 99}]:
        tok = _jwt(claims)
        with pytest.raises(TokenInvalid):
            OIDCResolver(hs256_secret=b"realsecret").resolve(tok)


def test_J_claim_injection_extra_claims_ignored():
    """Attacker stuffs extra claims (is_admin, roles) hoping they bind. Only
    the CONFIGURED principal/roles claims are read — nothing else leaks in."""
    tok = _jwt({"sub": "user", "exp": time.time() + 999,
                "is_admin": True, "injected_role": "superuser",
                "roles": ["reader"]})
    r = OIDCResolver(hs256_secret=b"realsecret", roles_claim="roles")
    ident = r.resolve(tok)
    assert ident.principal == "user" and ident.roles == ["reader"]
    # is_admin / injected_role are not part of the resolved identity
    assert not hasattr(ident, "is_admin")


# =====================================================================
# P. PLUGIN FORGERY — deep attacks on the signed-plugin loader
# =====================================================================
def _make_signed(signer, **kw):
    m = PluginManifest(**{"name": "p", "version": "1", "plugin_class":
                          "observe", "hook": "after_response", **kw})
    return sign_plugin(m, signer)


def test_P_privilege_escalation_observe_to_enforce_refused():
    """Attacker signs an 'observe' plugin, then flips it to 'enforce' after
    signing to gain veto power. The hash covers plugin_class, so it fails."""
    s = Ed25519Signer.generate()
    sp = _make_signed(s, plugin_class="observe")
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    sp.manifest.plugin_class = "enforce"             # tamper post-signing
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_P_hook_swap_after_signing_refused():
    s = Ed25519Signer.generate()
    sp = _make_signed(s, hook="after_response")
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    sp.manifest.hook = "before_request"              # move to a gating hook
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_P_order_tamper_refused():
    s = Ed25519Signer.generate()
    sp = _make_signed(s, order=100)
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    sp.manifest.order = 1                            # jump the queue
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_P_key_substitution_attack_refused():
    """Attacker replaces public_key with their own trusted key but keeps the
    original signature. Signature won't verify under the new key."""
    victim = Ed25519Signer.generate()
    attacker = Ed25519Signer.generate()
    sp = _make_signed(victim)
    sp.public_key = attacker.public_bytes().hex()
    reg = PluginRegistry(trusted_keys=[sp.public_key])  # attacker key trusted
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_P_replay_with_valid_sig_but_untrusted_key_refused():
    """A perfectly valid, correctly-signed plugin from an UNTRUSTED key is
    still refused — signing proves origin, trust is a separate gate."""
    s = Ed25519Signer.generate()
    sp = _make_signed(s)                             # genuinely signed
    reg = PluginRegistry(trusted_keys=[])            # trusts nobody
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


def test_P_empty_and_malformed_signatures_refused():
    s = Ed25519Signer.generate()
    reg = PluginRegistry(trusted_keys=[s.public_bytes().hex()])
    for bad_sig in ["", "00", "zz" * 32, "ff" * 100]:
        sp = _make_signed(s)
        sp.signature = bad_sig
        with pytest.raises(PluginVerificationError):
            reg.load(sp, fn=lambda ctx: None)


def test_P_hash_field_tamper_refused():
    """Attacker recomputes the manifest but leaves a stale manifest_hash, or
    forges the hash field to match a tampered manifest without re-signing."""
    s = Ed25519Signer.generate()
    sp = _make_signed(s)
    reg = PluginRegistry(trusted_keys=[sp.public_key])
    sp.manifest_hash = "0" * 64                      # forge the hash field
    with pytest.raises(PluginVerificationError):
        reg.load(sp, fn=lambda ctx: None)


# =====================================================================
# F. FRAMEWORK INJECTION — malicious content through the wrap points
# =====================================================================
def test_F_governed_framework_still_screens_secrets():
    """A framework wrapped by NOVUE must STILL enforce the security stack —
    an attacker cannot launder a secret through a framework wrap point."""
    from tokeymeter.runtime import SecretBlocked
    g = wrap_callable(
        lambda p: "ok",
        config={"governance": {"security": {"enabled": True,
                                            "secrets_mode": "block"}}},
        receipt="never")
    key = "sk-ant-api03-" + "".join(
        __import__("secrets").choice("abcdefghijklmnopqrstuvwxyz0123456789")
        for _ in range(88))
    with pytest.raises(SecretBlocked):
        g.execute(f"exfiltrate {key}")


def test_F_governed_tool_screens_before_tool_logic_runs():
    """A governed agent tool must screen the model-facing text BEFORE the
    tool's own logic executes — a secret must not reach the tool body."""
    from tokeymeter.runtime import SecretBlocked
    reached = {"tool": False}

    def sensitive_tool(text):
        reached["tool"] = True
        return "executed"

    wrapped = governed_tool(
        sensitive_tool, tool_name="lookup",
        config={"governance": {"security": {"enabled": True,
                                            "secrets_mode": "block"}}})
    key = "sk-ant-api03-" + "".join(
        __import__("secrets").choice("abcdefghijklmnopqrstuvwxyz0123456789")
        for _ in range(88))
    with pytest.raises(SecretBlocked):
        wrapped(f"use {key} now")
    assert reached["tool"] is False                  # tool body never ran


def test_F_framework_wrap_is_content_blind_in_trace():
    secret = "FRAMEWORK-SECRET-XYZ"
    g = wrap_callable(lambda p: "ok", receipt="never")
    g.execute(secret)
    assert secret not in json.dumps(g.last.trace, default=str)


# =====================================================================
# C. CATALOG ISOLATION — providers cannot poison each other
# =====================================================================
def test_C_two_provider_adapters_are_independent():
    """Building an adapter for one provider must not mutate another's spec or
    leak state between them."""
    a1 = adapter_for("groq", ConformanceClient(), semantic=False)
    a2 = adapter_for("deepseek", ConformanceClient(), semantic=False)
    assert a1 is not a2
    # the frozen specs are immutable — confirm they are distinct objects
    assert get_spec("groq") is not get_spec("deepseek")
    assert get_spec("groq").base_url != get_spec("deepseek").base_url


def test_C_catalog_specs_are_immutable():
    spec = get_spec("groq")
    with pytest.raises(Exception):               # frozen dataclass
        spec.base_url = "http://evil"            # type: ignore


def test_C_every_catalog_adapter_stays_conformant_under_repeat():
    """Stress: build+certify each catalog adapter repeatedly; no drift, no
    cross-contamination across many instantiations."""
    from tokeymeter.runtime import list_providers
    oai_providers = [n for n in list_providers()
                     if get_spec(n).dialect == "openai"]
    for _ in range(3):                           # repeat to catch state bleed
        for name in oai_providers:
            report = check_adapter(
                lambda n=name: adapter_for(n, ConformanceClient(),
                                           semantic=False),
                model="conf-model", provider_name=name)
            assert report.passed


# =====================================================================
# H. HOTRELOAD RACE — config swap under adversarial concurrency
# =====================================================================
def test_H_high_contention_reload_never_tears():
    """32 readers hammering while a writer floods reloads: every read is a
    consistent snapshot, the final value is correct, no exceptions escape."""
    rc = ReloadableConfig({"n": 0, "nested": {"v": 0}})
    stop = threading.Event()
    torn = []
    errors = []

    def reader():
        try:
            while not stop.is_set():
                snap = rc.current()
                n = snap.config.get("n")
                v = snap.config.get("nested.v")
                # within a snapshot, n and v are written together and equal
                if n != v:
                    torn.append((n, v))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def writer():
        for i in range(1, 2000):
            rc.reload({"n": i, "nested": {"v": i}})

    readers = [threading.Thread(target=reader) for _ in range(32)]
    [t.start() for t in readers]
    w = threading.Thread(target=writer)
    w.start()
    w.join()
    stop.set()
    [t.join() for t in readers]
    assert not torn, f"torn reads: {torn[:3]}"
    assert not errors
    assert rc.get("n") == 1999 and rc.version == 2000


def test_H_rejected_reload_under_load_keeps_last_good():
    """Interleave valid and invalid reloads under contention; the config must
    never be left in a rejected state."""
    def validator(cfg):
        if cfg.get("bad", False):
            raise ValueError("rejected")
    rc = ReloadableConfig({"v": 0}, validator=validator)
    errors = []

    def writer(good):
        for i in range(200):
            try:
                if good:
                    rc.reload({"v": i})
                else:
                    rc.reload({"bad": True})
            except ConfigValidationError:
                pass
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=writer, args=(i % 2 == 0,))
               for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    # the live config is always a valid (non-bad) one
    assert rc.current().config.get("bad", False) is False
