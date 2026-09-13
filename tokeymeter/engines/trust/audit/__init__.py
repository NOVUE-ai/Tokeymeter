"""
Tokeymeter audit layer (v0.8).

The provable decision ledger. Every cache hit, every PII redaction,
every compression and memory summarization becomes a tamper-evident,
hash-chained entry in an append-only log.

Key properties:
  - **Zero-content**: entries contain only hashes (HMAC-SHA256 with a
    per-installation secret), integer counts, timestamps, tags, and
    cost numbers. NO prompt or response text. The log can be made
    public without leaking a single prompt.
  - **Tamper-evident**: each entry's hash includes the previous entry's
    hash. Modifying any past entry breaks the chain.
  - **Non-repudiable**: optional signed checkpoints, externally
    anchorable (timestamping services, blockchain, S3).
  - **Externally verifiable**: an auditor receives a JSON proof bundle
    and verifies it with one function call — no Tokeymeter runtime needed.
  - **Compliance-ready**: ships report generators for SOC 2 Type II,
    EU AI Act, GDPR, India DPDP, and HIPAA.
  - **Fail-open**: a broken audit subsystem never breaks the user's
    primary call.

Quickstart:

    import tokeymeter
    from tokeymeter.engines.trust.audit import AuditLog, export_proof

    audit = AuditLog()
    audit.attach()    # auto-subscribes to every Tokeymeter event

    # ... your normal application code, fully Tokeymeter-instrumented ...

    # Export a portable proof for an auditor
    proof = export_proof(audit, since=quarter_start, until=quarter_end)
    proof.save("q1_audit.json")

    # The auditor runs (no Tokeymeter needed beyond this module):
    #     from tokeymeter.engines.trust.audit import AuditProof, verify_proof
    #     proof = AuditProof.load("q1_audit.json")
    #     result = verify_proof(proof)
    #     print(result.valid, result.entries_verified)
"""
from tokeymeter.engines.trust.audit.log import (
    AuditEntry,
    AuditLog,
    Checkpoint,
    GENESIS_HASH,
    SCHEMA_VERSION,
    VerificationResult,
    verify_entries,
    is_attached,
)
from tokeymeter.engines.trust.audit.proof import (
    AuditProof,
    ProofVerificationResult,
    export_proof,
    verify_proof,
)
from tokeymeter.engines.trust.audit.signers import (
    HMACSigner,
    Ed25519Signer,
    Signer,
    generate_install_secret,
    load_or_create_install_secret,
)

__all__ = [
    "is_attached",
    # Core
    "AuditLog",
    "AuditEntry",
    "Checkpoint",
    "VerificationResult",
    "verify_entries",
    "GENESIS_HASH",
    "SCHEMA_VERSION",
    # Proof
    "AuditProof",
    "ProofVerificationResult",
    "export_proof",
    "verify_proof",
    # Signing
    "Signer",
    "HMACSigner",
    "Ed25519Signer",
    "generate_install_secret",
    "load_or_create_install_secret",
    # Compliance
]
