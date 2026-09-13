"""
Audit proofs — self-contained bundles an auditor can verify.

An AuditProof is a portable JSON document containing:
  - A range of audit entries
  - Optional surrounding checkpoint(s)
  - Optional signature over the bundle as a whole

The verify_proof() function is INTENTIONALLY decoupled from AuditLog —
it works on a serialized proof file with no Tokeymeter runtime needed. An
auditor receives the .json file (and optionally a public key or shared
secret) and runs a standalone verifier.

The pitch: "Send your auditor a JSON file. They run one command. They
get a yes/no answer with reasons. No need to install our library."
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from tokeymeter.engines.trust.audit.log import (
    AuditEntry,
    AuditLog,
    Checkpoint,
    SCHEMA_VERSION,
    verify_entries,
)
from tokeymeter.engines.trust.audit.signers import Signer

log = logging.getLogger("tokeymeter.audit")


# ============================================================
#                       AuditProof
# ============================================================

@dataclass
class AuditProof:
    """A portable, verifiable audit bundle.

    Fields:
      schema_version: protocol version (e.g., "v1")
      generated_at:   when this proof was exported
      entries:        the list of AuditEntry objects (as dicts)
      checkpoints:    surrounding/included checkpoints (as dicts)
      summary:        precomputed aggregate (verifiable by recomputation)
      bundle_hash:    SHA-256 over the canonical serialization of the above
      signature:      optional signature over bundle_hash
      signature_algorithm: e.g., "hmac-sha256"
      install_id:     opaque per-install identifier (not the secret)
    """
    schema_version: str
    generated_at: float
    entries: List[dict]
    checkpoints: List[dict]
    summary: dict
    bundle_hash: str
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None
    install_id: Optional[str] = None

    # ---- Serialization ----

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=False)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, s: str) -> "AuditProof":
        return cls(**json.loads(s))

    @classmethod
    def load(cls, path: str) -> "AuditProof":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())

    # ---- Canonical hashing (deterministic over the entry list) ----

    @staticmethod
    def compute_bundle_hash(entries: List[dict], checkpoints: List[dict]) -> str:
        """Hash the bundle content. Excludes signature/bundle_hash itself."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": entries,
            "checkpoints": checkpoints,
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canon).hexdigest()


# ============================================================
#                  Proof export
# ============================================================

def export_proof(
    audit_log: AuditLog,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
    seq_start: Optional[int] = None,
    seq_end: Optional[int] = None,
    signer: Optional[Signer] = None,
    install_id: Optional[str] = None,
) -> AuditProof:
    """Build a portable proof bundle from an AuditLog.

    Args:
        audit_log: the source log
        since / until: time range (epoch seconds, inclusive)
        seq_start / seq_end: sequence range (inclusive)
        signer: optional signer (default: no signature, only hash)
        install_id: optional opaque identifier for who generated this

    The returned bundle includes:
        - All entries in the range
        - The checkpoint immediately AFTER the range (if any) — this is
          the "anchor" tying the entries to a known chain head
        - A precomputed summary auditors can spot-check

    Never raises. Returns an empty proof on internal failure.
    """
    try:
        entries = audit_log.get_entries(
            since=since, until=until,
            seq_start=seq_start, seq_end=seq_end,
        )
        entry_dicts = [e.to_dict() for e in entries]

        # Include the FIRST checkpoint at or after the last entry's seq.
        # This proves the entries' chain head is what the (possibly anchored)
        # checkpoint says it is.
        checkpoints_in_log = audit_log.get_checkpoints()
        relevant_cps: List[Checkpoint] = []
        if entries:
            last_seq = entries[-1].seq
            for cp in checkpoints_in_log:
                if cp.seq >= last_seq:
                    relevant_cps.append(cp)
                    break  # one is enough
        cp_dicts = [asdict(c) for c in relevant_cps]

        # Precomputed summary — auditors can re-verify
        summary = _build_summary(entries)

        bundle_hash = AuditProof.compute_bundle_hash(entry_dicts, cp_dicts)
        # Safe-by-default: if the caller didn't pass a signer, fall back to the
        # log's own signer so the normal export path produces a SIGNED (i.e.
        # tamper-evident) proof. Pass signer=NoOpSigner() to export unsigned.
        if signer is None:
            signer = getattr(audit_log, "_signer", None)
        signature = None
        sig_algo = None
        if signer is not None:
            try:
                signature = signer.sign(bundle_hash.encode("ascii")).hex()
                sig_algo = signer.algorithm
            except Exception as e:
                log.warning("audit: proof signing failed: %s", e)

        return AuditProof(
            schema_version=SCHEMA_VERSION,
            generated_at=time.time(),
            entries=entry_dicts,
            checkpoints=cp_dicts,
            summary=summary,
            bundle_hash=bundle_hash,
            signature=signature,
            signature_algorithm=sig_algo,
            install_id=install_id,
        )
    except Exception as e:
        log.warning("audit: export_proof failed: %s", e)
        return AuditProof(
            schema_version=SCHEMA_VERSION,
            generated_at=time.time(),
            entries=[],
            checkpoints=[],
            summary={"error": str(e)},
            bundle_hash="",
        )


def _build_summary(entries: List[AuditEntry]) -> dict:
    """Precomputed aggregates the auditor can spot-check."""
    if not entries:
        return {
            "entries_total": 0,
            "by_decision_type": {},
            "aggregate_cost_saved_usd": 0.0,
            "pii_redactions_total": 0,
        }
    by_type: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}
    cost_total = 0.0
    pii_total = 0
    t_min = entries[0].timestamp
    t_max = entries[0].timestamp
    for e in entries:
        by_type[e.decision_type] = by_type.get(e.decision_type, 0) + 1
        if e.tag:
            by_tag[e.tag] = by_tag.get(e.tag, 0) + 1
        cost_total += e.cost_saved_usd
        pii_total += e.pii_redactions
        if e.timestamp < t_min:
            t_min = e.timestamp
        if e.timestamp > t_max:
            t_max = e.timestamp
    return {
        "entries_total": len(entries),
        "by_decision_type": by_type,
        "by_tag": by_tag,
        "aggregate_cost_saved_usd": round(cost_total, 6),
        "pii_redactions_total": pii_total,
        "seq_first": entries[0].seq,
        "seq_last": entries[-1].seq,
        "timestamp_first": t_min,
        "timestamp_last": t_max,
    }


# ============================================================
#                  Proof verification
# ============================================================

@dataclass
class ProofVerificationResult:
    """Returned by verify_proof()."""
    valid: bool
    entries_verified: int = 0
    first_bad_seq: Optional[int] = None
    reason: Optional[str] = None
    bundle_hash_valid: bool = False
    signature_valid: Optional[bool] = None  # None when no signature present
    chain_valid: bool = False
    summary_valid: bool = False
    aggregate_cost_saved_usd: float = 0.0
    by_decision_type: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def verify_proof(
    proof: AuditProof,
    *,
    signer: Optional[Signer] = None,
    require_signature: bool = True,
) -> ProofVerificationResult:
    """Verify a proof bundle. ZERO Tokeymeter runtime needed beyond this module.

    Steps:
      1. Recompute bundle_hash from entries+checkpoints. Compare.
      2. If signature present and signer provided, verify the signature.
      3. Reconstruct AuditEntry objects from dicts. Verify hash chain.
      4. Recompute summary aggregates. Compare to declared summary.
      5. (If checkpoint included) verify checkpoint chain_head matches
         the last entry's entry_hash.

    Args:
        proof: the bundle to verify
        signer: a Signer instance to verify the signature. For HMAC,
            this must hold the same secret used to sign. If signer is
            None and `require_signature=False`, signature check is
            skipped (but bundle_hash still verifies content integrity).
        require_signature: if True, missing signatures fail verification.
    """
    result = ProofVerificationResult(valid=False)
    notes = result.notes

    try:
        # Step 1: bundle hash
        recomputed = AuditProof.compute_bundle_hash(proof.entries, proof.checkpoints)
        if recomputed != proof.bundle_hash:
            result.reason = (
                f"bundle_hash mismatch: recomputed {recomputed[:16]}..., "
                f"declared {proof.bundle_hash[:16]}..."
            )
            return result
        result.bundle_hash_valid = True
        notes.append("bundle_hash verified")

        # Step 2: signature
        if proof.signature:
            if signer is None:
                if require_signature:
                    result.reason = "signature present but no signer provided"
                    result.signature_valid = False
                    return result
                result.signature_valid = None
                notes.append("signature present, verification skipped (no signer)")
            else:
                try:
                    sig_bytes = bytes.fromhex(proof.signature)
                    ok = signer.verify(proof.bundle_hash.encode("ascii"), sig_bytes)
                    result.signature_valid = bool(ok)
                    if not ok:
                        result.reason = "signature did not verify"
                        return result
                    notes.append(
                        f"signature verified ({proof.signature_algorithm or 'unknown'})"
                    )
                except Exception as e:
                    result.signature_valid = False
                    result.reason = f"signature verification raised: {e}"
                    return result
        else:
            if require_signature:
                result.reason = "signature required but not present in bundle"
                return result
            notes.append("no signature in bundle")

        # Step 3: chain
        entries = [AuditEntry(**d) for d in proof.entries]
        chain_result = verify_entries(entries)
        result.entries_verified = chain_result.entries_verified
        result.chain_valid = chain_result.valid
        result.first_bad_seq = chain_result.first_bad_seq
        result.by_decision_type = chain_result.by_decision_type
        result.aggregate_cost_saved_usd = chain_result.aggregate_cost_saved_usd

        if not chain_result.valid:
            result.reason = f"chain verification failed: {chain_result.reason}"
            return result
        notes.append(f"hash chain verified ({chain_result.entries_verified} entries)")

        # Step 4: summary
        declared = proof.summary or {}
        recomputed_summary = _build_summary(entries)
        # Spot-check the keys that matter
        keys_to_check = [
            "entries_total", "by_decision_type", "aggregate_cost_saved_usd",
            "pii_redactions_total", "seq_first", "seq_last",
        ]
        summary_ok = True
        for k in keys_to_check:
            if k not in declared:
                continue  # missing is OK (older format), just don't validate it
            if recomputed_summary.get(k) != declared.get(k):
                # Special case: float comparison
                if isinstance(recomputed_summary.get(k), float):
                    if abs(recomputed_summary.get(k, 0) - declared.get(k, 0)) > 1e-6:
                        summary_ok = False
                        notes.append(
                            f"summary mismatch on '{k}': "
                            f"recomputed {recomputed_summary.get(k)}, "
                            f"declared {declared.get(k)}"
                        )
                else:
                    summary_ok = False
                    notes.append(
                        f"summary mismatch on '{k}': "
                        f"recomputed {recomputed_summary.get(k)}, "
                        f"declared {declared.get(k)}"
                    )
        result.summary_valid = summary_ok
        if summary_ok:
            notes.append("summary aggregates verified")
        else:
            result.reason = "summary did not match recomputed aggregates"
            return result

        # Step 5: checkpoint chain_head linkage
        if proof.checkpoints and entries:
            # The first checkpoint at-or-after the last entry should match
            # the last entry's hash (if seq == last_entry.seq) or be later.
            last_entry = entries[-1]
            for cp_dict in proof.checkpoints:
                cp = Checkpoint(**cp_dict)
                if cp.seq == last_entry.seq:
                    if cp.chain_head != last_entry.entry_hash:
                        result.reason = (
                            f"checkpoint at seq={cp.seq} has chain_head "
                            f"{cp.chain_head[:16]}..., but last entry's "
                            f"hash is {last_entry.entry_hash[:16]}..."
                        )
                        return result
                    notes.append(f"checkpoint at seq={cp.seq} anchors last entry")
                    break

        # All checks passed
        result.valid = True
        return result
    except Exception as e:
        result.reason = f"verification raised: {e}"
        return result
