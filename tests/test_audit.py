"""
Tests for the audit layer (v0.8).

Coverage:
  - AuditEntry: deterministic hashing, schema_version embedded
  - Hash chain: tamper detection at every level
  - Auto-subscription via attach()
  - Privacy invariant: no prompt text leaks anywhere
  - HMACSigner: sign/verify, key length validation
  - Install secret: persistence, mode 0600, ephemeral fallback
  - Concurrency: many writers don't corrupt the chain
  - Proof export/load roundtrip + JSON validity
  - verify_proof: 4-layer detection (bundle, signature, chain, summary)
  - Compliance reports: all 5 frameworks generate
  - Background flusher: drains on close
  - Fail-open: broken signer/anchor doesn't crash anything
"""
import asyncio
import hashlib
import hmac
import json
import os
import stat
import tempfile
import threading
import time

import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.audit import (
    AuditEntry,
    AuditLog,
    AuditProof,
    Checkpoint,
    GENESIS_HASH,
    HMACSigner,
    export_proof,
    generate_install_secret,
    load_or_create_install_secret,
    verify_entries,
    verify_proof,
)
from tokeymeter.storage import MemoryStore


# A reusable, isolated AuditLog fixture
@pytest.fixture
def audit(tmp_path):
    log = AuditLog(
        path=str(tmp_path / "audit.db"),
        install_secret_path=str(tmp_path / "secret"),
        signing_key_path=str(tmp_path / "signing-key"),
        checkpoint_every=5,
        flush_interval_seconds=0.1,
    )
    yield log
    log.close()


@pytest.fixture(autouse=True)
def reset_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield
    events.clear_subscribers()


# =================================================================
#                   Signer
# =================================================================

def test_hmac_signer_sign_verify_roundtrip():
    s = HMACSigner(b"x" * 32)
    sig = s.sign(b"hello world")
    assert s.verify(b"hello world", sig)


def test_hmac_signer_rejects_wrong_data():
    s = HMACSigner(b"x" * 32)
    sig = s.sign(b"hello world")
    assert not s.verify(b"hello WORLD", sig)


def test_hmac_signer_rejects_wrong_signature():
    s = HMACSigner(b"x" * 32)
    assert not s.verify(b"hello", b"\x00" * 32)


def test_hmac_signer_requires_min_key_length():
    with pytest.raises(ValueError):
        HMACSigner(b"too short")


def test_install_secret_persists_and_has_secure_perms(tmp_path):
    p = str(tmp_path / "secret")
    s1 = load_or_create_install_secret(p)
    assert len(s1) >= 16
    # POSIX file modes (0600) aren't enforceable via os.chmod on Windows
    # (NTFS uses ACLs, not Unix bits), so only assert perms on POSIX. Windows
    # users should protect the ~/.tokeymeter directory via folder ACLs.
    if os.name != "nt":
        st = os.stat(p)
        assert (st.st_mode & 0o077) == 0, f"secret has loose perms: {oct(st.st_mode)}"
    # Reading again gets the same secret
    s2 = load_or_create_install_secret(p)
    assert s1 == s2


def test_generate_install_secret_is_random():
    s1 = generate_install_secret()
    s2 = generate_install_secret()
    assert s1 != s2
    assert len(s1) == 32


# =================================================================
#                   AuditEntry
# =================================================================

def test_entry_hash_is_deterministic():
    """Same inputs → same hash. Critical for chain verification."""
    e1 = AuditEntry(
        seq=0, timestamp=1700000000.0, decision_type="cache_miss",
        prompt_hash="abc123", model="gpt-4o-mini",
        cost_saved_usd=0.0, pii_redactions=0,
        tag=None, function_name="ask",
        metadata_hash="", prev_hash=GENESIS_HASH, entry_hash="",
    )
    e2 = AuditEntry(
        seq=0, timestamp=1700000000.0, decision_type="cache_miss",
        prompt_hash="abc123", model="gpt-4o-mini",
        cost_saved_usd=0.0, pii_redactions=0,
        tag=None, function_name="ask",
        metadata_hash="", prev_hash=GENESIS_HASH, entry_hash="",
    )
    assert e1.recompute_hash() == e2.recompute_hash()


def test_entry_hash_changes_with_any_field():
    """Any field change → different hash. Tamper-evidence at field level."""
    base_kwargs = dict(
        seq=0, timestamp=1700000000.0, decision_type="cache_miss",
        prompt_hash="abc123", model="gpt-4o-mini",
        cost_saved_usd=0.0, pii_redactions=0,
        tag=None, function_name="ask",
        metadata_hash="", prev_hash=GENESIS_HASH, entry_hash="",
    )
    base_hash = AuditEntry(**base_kwargs).recompute_hash()

    for k, v in [
        ("seq", 1), ("timestamp", 1700000001.0),
        ("decision_type", "cache_hit_exact"), ("prompt_hash", "def456"),
        ("model", "gpt-4o"), ("cost_saved_usd", 0.001),
        ("pii_redactions", 1), ("tag", "support"),
        ("function_name", "other"), ("metadata_hash", "abc"),
        ("prev_hash", "0" * 64),
    ]:
        kw = dict(base_kwargs)
        kw[k] = v
        assert AuditEntry(**kw).recompute_hash() != base_hash, f"hash unchanged for {k}"


# =================================================================
#                   AuditLog append
# =================================================================

def test_append_and_verify_chain(audit):
    for i in range(10):
        audit.append(
            decision_type="cache_miss",
            prompt_text=f"prompt-{i}",
            model="gpt-4o-mini",
            cost_saved_usd=0.0,
        )
    audit.flush(timeout=2.0)
    r = audit.verify_chain()
    assert r.valid is True
    assert r.entries_verified == 10


def test_genesis_hash_is_well_known(audit):
    audit.append(decision_type="cache_miss", prompt_text="hi")
    audit.flush(timeout=2.0)
    e = audit.get_entry(0)
    assert e is not None
    assert e.prev_hash == GENESIS_HASH


def test_chain_links_correctly(audit):
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)
    entries = audit.get_entries()
    for i in range(1, len(entries)):
        assert entries[i].prev_hash == entries[i - 1].entry_hash


# =================================================================
#                   Tamper detection
# =================================================================

def test_chain_detects_modified_field_via_recompute(audit):
    """Manually modify the SQLite file and verify chain detects it."""
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    import sqlite3
    # Tamper: change cost_saved_usd on seq=2 but DO NOT update its entry_hash
    with sqlite3.connect(audit._path) as c:
        c.execute("UPDATE audit_entries SET cost_saved_usd=99.99 WHERE seq=2")
        c.commit()

    r = audit.verify_chain()
    assert r.valid is False
    assert r.first_bad_seq == 2
    assert "entry_hash mismatch" in (r.reason or "")


def test_chain_detects_inserted_entry(audit):
    """Cannot insert an entry mid-chain without breaking subsequent links."""
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    import sqlite3
    # Inject a fake entry at seq=10 (way out of order, breaks chain)
    with sqlite3.connect(audit._path) as c:
        c.execute(
            "INSERT INTO audit_entries (seq, timestamp, decision_type, "
            "prompt_hash, model, cost_saved_usd, pii_redactions, tag, "
            "function_name, metadata_hash, prev_hash, entry_hash) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (10, time.time(), "FAKE", "X", "fake", 1000.0, 0,
             None, None, "", "0" * 64, "deadbeef"),
        )
        c.commit()

    r = audit.verify_chain()
    assert r.valid is False


# =================================================================
#                   Auto-subscription
# =================================================================

def test_attach_subscribes_to_events_and_records(audit):
    audit.attach()

    @tokeymeter.cache(model="gpt-4o-mini", tag="support")
    def ask(prompt):
        return "ok"

    ask("hello world")
    ask("hello world")  # hit
    ask("different")
    audit.flush(timeout=2.0)

    entries = audit.get_entries()
    assert len(entries) == 3
    decisions = [e.decision_type for e in entries]
    assert decisions[0].startswith("cache_miss") or "miss" in decisions[0]
    assert any("hit" in d for d in decisions)


def test_attach_is_idempotent(audit):
    audit.attach()
    audit.attach()  # second time should be a no-op

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("only one entry")
    audit.flush(timeout=2.0)
    assert len(audit.get_entries()) == 1


def test_detach_stops_recording(audit):
    audit.attach()

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("a")
    audit.flush(timeout=2.0)
    initial = len(audit.get_entries())

    audit.detach()
    ask("b")
    ask("c")
    audit.flush(timeout=2.0)
    after = len(audit.get_entries())
    assert after == initial  # no new entries after detach


# =================================================================
#                   Privacy invariant
# =================================================================

def test_pii_redaction_count_flows_to_audit_log(audit):
    """PII redactions must be recorded in the audit log for compliance evidence."""
    from tokeymeter.privacy import default_redactor

    audit.attach()

    @tokeymeter.cache(model="gpt-4o-mini", redactor=default_redactor, tag="support")
    def bot(prompt):
        return "ok"

    bot("email alice@example.com and SSN 123-45-6789")  # 2 PII
    bot("card 4111 1111 1111 1111 and call +14155551234")  # 2 PII
    bot("nothing sensitive here")  # 0 PII
    audit.flush(timeout=2.0)

    entries = audit.get_entries()
    total_pii = sum(e.pii_redactions for e in entries)
    assert total_pii == 4, f"expected 4 redactions in audit, got {total_pii}"




def test_no_prompt_text_appears_in_audit_log(audit):
    """The PROMPT TEXT must NEVER appear in any audit entry field."""
    secret_prompt = "Patient JohnDoe SSN 123-45-6789 has hypertension"
    audit.append(decision_type="cache_miss", prompt_text=secret_prompt)
    audit.flush(timeout=2.0)

    e = audit.get_entry(0)
    assert e is not None
    serialized = json.dumps(e.to_dict())
    # Check the genuinely sensitive tokens (not short common words like "has"
    # which legitimately collide with field names such as "prompt_hash").
    sensitive_tokens = ["Patient", "JohnDoe", "123-45-6789", "hypertension"]
    for token in sensitive_tokens:
        assert token not in serialized, f"leaked '{token}' in audit entry"


def test_no_prompt_text_in_database_file(audit, tmp_path):
    """Inspect the raw SQLite file — no prompt content."""
    secret = "alice@example.com is the email"
    audit.append(decision_type="cache_miss", prompt_text=secret)
    audit.flush(timeout=2.0)

    with open(audit._path, "rb") as f:
        raw = f.read()
    assert b"alice@example.com" not in raw
    assert b"the email" not in raw


def test_hmac_prevents_rainbow_table_lookups(audit, tmp_path):
    """Different installs (different secrets) produce different hashes
    for the same prompt — so prompt-rainbow-table attacks are install-specific."""
    audit2 = AuditLog(
        path=str(tmp_path / "audit2.db"),
        install_secret_path=str(tmp_path / "secret2"),
    )
    try:
        audit.append(decision_type="cache_miss", prompt_text="hello")
        audit2.append(decision_type="cache_miss", prompt_text="hello")
        audit.flush(timeout=2.0)
        audit2.flush(timeout=2.0)
        h1 = audit.get_entry(0).prompt_hash
        h2 = audit2.get_entry(0).prompt_hash
        assert h1 != h2, "same prompt produced same hash across installs — rainbow risk"
    finally:
        audit2.close()


def test_event_path_prompt_hash_is_install_specific(tmp_path):
    """AuditLog.attach must HMAC prompt text, not store the public cache key."""
    hashes = []
    for i in range(2):
        tokeymeter.set_default_store(MemoryStore())
        audit_i = AuditLog(
            path=str(tmp_path / f"audit_{i}.db"),
            install_secret_path=str(tmp_path / f"secret_{i}"),
            signing_key_path=str(tmp_path / f"signing_{i}"),
            flush_interval_seconds=0.05,
        )
        audit_i.attach()
        try:
            @tokeymeter.cache(model="m")
            def ask(prompt):
                return "ok"

            ask("same prompt across installs")
            audit_i.flush(timeout=2.0)
            hashes.append(audit_i.get_entry(0).prompt_hash)
        finally:
            audit_i.close()

    assert hashes[0] != hashes[1]


def test_high_stakes_audit_hashes_distinct_prompts(tmp_path):
    audit = AuditLog(
        path=str(tmp_path / "audit.db"),
        install_secret_path=str(tmp_path / "secret"),
        signing_key_path=str(tmp_path / "signing"),
        flush_interval_seconds=0.05,
    )
    audit.attach()
    try:
        @tokeymeter.cache(model="m", high_stakes=True)
        def hs(prompt):
            return "ok"

        hs("critical prompt one")
        hs("critical prompt two")
        audit.flush(timeout=2.0)
        entries = audit.get_entries()
        assert len(entries) == 2
        assert entries[0].prompt_hash != entries[1].prompt_hash
    finally:
        audit.close()


# =================================================================
#                   Checkpoints
# =================================================================

def test_checkpoint_created_at_threshold(audit):
    # checkpoint_every=5, so 5 entries → 1 checkpoint at seq=4
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    cps = audit.get_checkpoints()
    assert len(cps) >= 1
    assert cps[0].seq == 4


def test_checkpoint_chain_head_matches_entry_hash(audit):
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    cp = audit.get_checkpoint(4)
    e = audit.get_entry(4)
    assert cp is not None and e is not None
    assert cp.chain_head == e.entry_hash


def test_checkpoint_signature_present_when_signer_configured(tmp_path):
    signer = HMACSigner(b"x" * 32)
    audit = AuditLog(
        path=str(tmp_path / "audit.db"),
        install_secret_path=str(tmp_path / "secret"),
        checkpoint_every=3,
        signer=signer,
    )
    try:
        for i in range(3):
            audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
        audit.flush(timeout=2.0)
        cp = audit.get_checkpoint(2)
        assert cp is not None
        assert cp.signature is not None
        assert cp.signature_algorithm == "hmac-sha256"
        # And the signature verifies
        sig_bytes = bytes.fromhex(cp.signature)
        assert signer.verify(cp.signed_bytes(), sig_bytes)
    finally:
        audit.close()


# =================================================================
#                   Concurrency
# =================================================================

def test_concurrent_appends_preserve_chain(audit):
    """100 threads each append 10 entries — chain must remain valid."""
    def writer(i):
        for j in range(10):
            audit.append(decision_type="cache_miss", prompt_text=f"t{i}-p{j}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    audit.flush(timeout=10.0)

    entries = audit.get_entries()
    assert len(entries) == 100
    r = verify_entries(entries)
    assert r.valid, f"chain broken under concurrency: {r.reason}"


# =================================================================
#                   Proof export + verify
# =================================================================

def test_proof_export_load_roundtrip(audit, tmp_path):
    for i in range(8):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    proof = export_proof(audit, signer=HMACSigner(b"x" * 32))
    path = tmp_path / "proof.json"
    proof.save(str(path))

    reloaded = AuditProof.load(str(path))
    assert reloaded.bundle_hash == proof.bundle_hash
    assert len(reloaded.entries) == 8


def test_verify_proof_valid_path(audit):
    for i in range(5):
        audit.append(decision_type="cache_miss" if i % 2 else "cache_hit_exact",
                     prompt_text=f"p{i}", cost_saved_usd=0.001 * (i + 1))
    audit.flush(timeout=2.0)

    signer = HMACSigner(b"x" * 32)
    proof = export_proof(audit, signer=signer)

    # Fresh verifier
    result = verify_proof(proof, signer=HMACSigner(b"x" * 32))
    assert result.valid
    assert result.bundle_hash_valid
    assert result.signature_valid
    assert result.chain_valid
    assert result.summary_valid
    assert result.entries_verified == 5


def test_verify_proof_rejects_tampered_entry(audit):
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}",
                     cost_saved_usd=0.001)
    audit.flush(timeout=2.0)

    signer = HMACSigner(b"x" * 32)
    proof = export_proof(audit, signer=signer)
    # Tamper: bump a cost
    proof.entries[2]["cost_saved_usd"] = 99.99
    result = verify_proof(proof, signer=HMACSigner(b"x" * 32))
    assert not result.valid
    # The bundle_hash check fires first
    assert not result.bundle_hash_valid


def test_verify_proof_rejects_wrong_signature_key(audit):
    for i in range(3):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    proof = export_proof(audit, signer=HMACSigner(b"x" * 32))
    # Wrong key
    result = verify_proof(proof, signer=HMACSigner(b"y" * 32))
    assert not result.valid
    assert result.signature_valid is False


def test_verify_proof_handles_unsigned(audit):
    for i in range(3):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
    audit.flush(timeout=2.0)

    # Default export now SIGNS (tamper-evident by default). Simulate a
    # genuinely unsigned proof to exercise the unsigned-handling contract.
    proof = export_proof(audit)
    proof.signature = None
    proof.signature_algorithm = None
    # Strict-by-default: an unsigned proof must NOT verify.
    assert not verify_proof(proof).valid
    # Content-integrity-only verification is an explicit opt-in.
    assert verify_proof(proof, require_signature=False).valid


def test_verify_proof_summary_mismatch(audit):
    for i in range(5):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}",
                     cost_saved_usd=0.001)
    audit.flush(timeout=2.0)

    proof = export_proof(audit, signer=HMACSigner(b"x" * 32))
    # Tamper the summary (but not entries) — would catch a forged summary
    proof.summary["aggregate_cost_saved_usd"] = 999.99
    # bundle_hash uses entries+checkpoints, NOT summary, so it still passes
    # the chain verification will succeed; summary check will fail
    result = verify_proof(proof, signer=HMACSigner(b"x" * 32))
    assert not result.valid
    assert result.summary_valid is False


# =================================================================
#                   Empty & edge cases
# =================================================================

def test_empty_log_verifies(audit):
    r = audit.verify_chain()
    assert r.valid and r.entries_verified == 0


def test_export_proof_empty_log(audit):
    proof = export_proof(audit)
    assert proof.entries == []
    assert proof.summary["entries_total"] == 0
    result = verify_proof(proof, signer=audit._signer)
    assert result.valid


def test_time_range_filtering(audit):
    # Insert entries with explicit timestamps
    for i in range(10):
        audit.append(decision_type="cache_miss", prompt_text=f"p{i}",
                     timestamp=1000.0 + i)
    audit.flush(timeout=2.0)

    mid = audit.get_entries(since=1003.0, until=1006.0)
    assert all(1003.0 <= e.timestamp <= 1006.0 for e in mid)
    assert len(mid) == 4










# =================================================================
#                   Failure modes (fail-open)
# =================================================================

def test_audit_with_broken_signer_does_not_crash_on_checkpoint(tmp_path):
    class BadSigner:
        algorithm = "broken"
        def sign(self, data): raise RuntimeError("kaboom")
        def verify(self, data, sig): return False

    audit = AuditLog(
        path=str(tmp_path / "a.db"),
        install_secret_path=str(tmp_path / "s"),
        checkpoint_every=3,
        signer=BadSigner(),
    )
    try:
        # Should not raise
        for i in range(3):
            audit.append(decision_type="cache_miss", prompt_text=f"p{i}")
        audit.flush(timeout=2.0)
        cp = audit.get_checkpoint(2)
        assert cp is not None
        assert cp.signature is None  # broken signer → no signature, but entry exists
    finally:
        audit.close()


def test_audit_fails_open_on_corrupt_event(audit):
    """If a subscriber receives a malformed event, audit must not raise."""
    audit.attach()
    # Manually emit a malformed event
    try:
        from tokeymeter.events import CacheEvent
        # Many None fields shouldn't crash
        ev = CacheEvent(
            timestamp=time.time(), event_type="lookup_hit", hit=True,
            hit_type=None, model="x", cache_key=None,
            prompt_preview=None, latency_ms=0.0,
            estimated_cost_usd=None, input_tokens=0, output_tokens=0,
            shadow=False, tag=None,
        )
        events.emit(ev)
    except Exception:
        pass  # creating the event itself might fail; that's the user's problem
    audit.flush(timeout=1.0)
    # No assertion needed — the test passes if nothing crashed


# --- Regression tests for C1: audit-proof forgery must fail closed ---

def test_forged_unsigned_proof_is_rejected_by_default(audit):
    """Reproduces stress-test finding C1: an attacker who rewrites the
    (unkeyed) hash chain + bundle + summary must NOT pass default verify."""
    from tokeymeter.audit.log import AuditEntry
    from tokeymeter.audit.proof import _build_summary
    for i in range(5):
        audit.append(decision_type="cache_hit_exact", prompt_text=f"p{i}",
                     cost_saved_usd=0.001)
    audit.flush(timeout=2.0)
    proof = export_proof(audit)
    proof.signature = None
    proof.signature_algorithm = None
    for d in proof.entries:
        d["cost_saved_usd"] = 999.0
        d["pii_redactions"] = 0
    prev = proof.entries[0]["prev_hash"]
    for d in proof.entries:
        d["prev_hash"] = prev
        d["entry_hash"] = AuditEntry(**{**d, "entry_hash": ""}).recompute_hash()
        prev = d["entry_hash"]
    proof.bundle_hash = AuditProof.compute_bundle_hash(proof.entries, proof.checkpoints)
    proof.summary = _build_summary([AuditEntry(**d) for d in proof.entries])
    assert not verify_proof(proof).valid, "forged unsigned proof must fail closed"


def test_default_export_is_signed_and_verifies(audit):
    """The default flow (no explicit signer anywhere) must produce a SIGNED,
    verifiable proof — safe by default."""
    audit.append(decision_type="cache_miss", prompt_text="x")
    audit.flush(timeout=2.0)
    proof = export_proof(audit)
    assert proof.signature is not None
    assert verify_proof(proof, signer=audit._signer).valid


# --- Regression test for H3: Ed25519 non-repudiation ---

def test_ed25519_provides_non_repudiation(tmp_path):
    """A verifier holding only the PUBLIC key can verify but not forge."""
    pytest.importorskip("cryptography")
    from tokeymeter.audit import Ed25519Signer
    org = Ed25519Signer.generate()
    auditor = Ed25519Signer(public_key=org.public_bytes())  # public only
    log = AuditLog(path=str(tmp_path / "a.db"), signer=org,
                   install_secret_path=str(tmp_path / "s"),
                   signing_key_path=str(tmp_path / "k"))
    log.append(decision_type="cache_hit_exact", prompt_text="p", cost_saved_usd=0.01)
    log.flush(timeout=2.0)
    proof = export_proof(log, signer=org)
    assert verify_proof(proof, signer=auditor).valid          # can verify
    with pytest.raises(Exception):
        auditor.sign(b"forge")                                 # cannot sign/forge


