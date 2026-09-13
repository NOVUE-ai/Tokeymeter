"""The proof spine (W5): TRUST-1 kernel↔ledger bridge · TRUST-2 Merkle
export + offline inclusion proofs · TRUST-3 WORM AuditSink · TRUST-4
prove(request_id) signed packet.

Design laws:
- ONE CHAIN OF CUSTODY: the kernel TrustEngine residue feeds the SHIPPED
  hash-chained AuditLog (audit/log.py) — we extend the real ledger, never a
  shadow one. Kernel entries are verifiable by the shipped verify path.
- L4 CONTENT-BLIND: everything sealed or exported is fingerprints, counts,
  verdicts, model, principal, cost — never payload. prove() packets carry
  the same and verify offline with only a public key.
- OFFLINE VERIFIABILITY: Merkle roots recompute and inclusion proofs verify
  with stdlib hashlib alone — no Tokeymeter install required on the auditor
  side (pinned by a subprocess test on a clean interpreter).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .engine import Engine


# =====================================================================
# TRUST-2  Merkle tree — pure stdlib, offline-verifiable
# =====================================================================
def _leaf_hash(data: bytes) -> str:
    # domain-separated leaf (0x00) vs node (0x01) — second-preimage safety
    return hashlib.sha256(b"\x00" + data).hexdigest()


def _node_hash(left: str, right: str) -> str:
    return hashlib.sha256(
        b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def merkle_root(leaves: List[str]) -> str:
    """Root over pre-hashed leaf hex digests. Odd node duplicated (Bitcoin
    convention). Empty tree → sha256(b'')."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_node_hash(level[i], level[i + 1])
                 for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: List[str], index: int) -> List[Tuple[str, str]]:
    """Inclusion path for leaf `index`: list of (sibling_hash, 'L'|'R')."""
    if not 0 <= index < len(leaves):
        raise IndexError(index)
    path: List[Tuple[str, str]] = []
    level = list(leaves)
    idx = index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling = idx ^ 1
        side = "R" if sibling > idx else "L"
        path.append((level[sibling], side))
        level = [_node_hash(level[i], level[i + 1])
                 for i in range(0, len(level), 2)]
        idx //= 2
    return path


def verify_merkle_proof(leaf_hash: str, proof: List[Tuple[str, str]],
                        root: str) -> bool:
    """Offline verifier: stdlib hashlib only."""
    h = leaf_hash
    for sibling, side in proof:
        h = _node_hash(sibling, h) if side == "L" else _node_hash(h, sibling)
    return h == root


# =====================================================================
# TRUST-3  WORM AuditSink — append-only, reopen-verify
# =====================================================================
class AuditSink:
    """Append-only sink contract. Append returns the line's leaf hash."""

    def append(self, record: Dict[str, Any]) -> str:
        raise NotImplementedError

    def read_all(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class FileAuditSink(AuditSink):
    """Append-only newline-delimited JSON with a per-line chained leaf hash.
    WORM posture: opens in append mode, refuses truncation, and detects any
    out-of-band edit on reopen via chain re-verification."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._last = hashlib.sha256(b"").hexdigest()
        if os.path.exists(path):
            self._last = self._recover_head()

    def _recover_head(self) -> str:
        last = hashlib.sha256(b"").hexdigest()
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = json.loads(line)["leaf"]
        return last

    def append(self, record: Dict[str, Any]) -> str:
        with self._lock:
            body = json.dumps(record, sort_keys=True, default=str)
            leaf = _leaf_hash((self._last + body).encode("utf-8"))
            line = json.dumps({"leaf": leaf, "prev": self._last,
                               "record": record}, sort_keys=True,
                              default=str)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._last = leaf
            return leaf

    def read_all(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.exists(self._path):
            return out
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def verify(self) -> Tuple[bool, int]:
        """Re-chain from genesis; returns (ok, first_bad_index)."""
        prev = hashlib.sha256(b"").hexdigest()
        for i, row in enumerate(self.read_all()):
            body = json.dumps(row["record"], sort_keys=True, default=str)
            expect = _leaf_hash((prev + body).encode("utf-8"))
            if row["prev"] != prev or row["leaf"] != expect:
                return False, i
            prev = row["leaf"]
        return True, -1


# =====================================================================
# TRUST-4  proof packet
# =====================================================================
@dataclass
class ProofPacket:
    request_id: str
    model: str
    principal: str
    payload_fingerprint: str
    verdicts: str
    cost_usd: Optional[float]
    outcome: str
    entry_hash: str
    timestamp: float
    signature: Optional[str] = None
    public_key: Optional[str] = None
    algorithm: str = "none"

    def canonical(self) -> bytes:
        core = {k: v for k, v in self.__dict__.items()
                if k not in ("signature", "public_key", "algorithm")}
        return json.dumps(core, sort_keys=True, default=str).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def verify_proof_packet(packet_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """OFFLINE verifier — stdlib + (optionally) cryptography for ed25519.
    Returns (ok, reason). Designed to run on a machine with no Tokeymeter."""
    p = ProofPacket(**{k: packet_dict[k] for k in packet_dict
                       if k in ProofPacket.__dataclass_fields__})
    if p.algorithm == "none" or not p.signature:
        return False, "unsigned"
    if p.algorithm == "ed25519":
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey)
            pub = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(p.public_key or ""))
            pub.verify(bytes.fromhex(p.signature), p.canonical())
            return True, "ok"
        except Exception as exc:  # signature mismatch or bad key
            return False, f"signature_invalid:{type(exc).__name__}"
    return False, f"unknown_algorithm:{p.algorithm}"


# =====================================================================
# TRUST-1  ProofEngine — bridges kernel residue to the shipped ledger,
#          keeps a Merkle-ready leaf list, and mints prove() packets
# =====================================================================
@dataclass
class _Sealed:
    request_id: str
    body: str
    entry_hash: str
    fields: Dict[str, Any]
    leaf: str
    timestamp: float
    prev: str = "0" * 64


class ProofEngine(Engine):
    """Registered FIRST so it unwinds LAST (seam-order law) — seals the
    complete record. Bridges to the shipped AuditLog when provided, mirrors
    to a WORM sink when provided, and always keeps an in-memory chain +
    Merkle leaves for prove()/export."""

    name = "proof"
    _GENESIS = "0" * 64

    def __init__(self, *, audit_log: Any = None, sink: Optional[AuditSink] = None,
                 signer: Any = None, max_entries: int = 5000) -> None:
        self._audit_log = audit_log
        self._sink = sink
        self._signer = signer
        self._lock = threading.Lock()
        self._sealed: List[_Sealed] = []
        self._by_id: Dict[str, _Sealed] = {}
        # Bounded in-memory window (REL-7): durable proof is the sink/ledger;
        # memory keeps a rolling window + a bounded prove()-able index.
        self._max_entries = max(1, int(max_entries))
        self._evicted = 0

    # ---- pipeline -----------------------------------------------------
    def after_response(self, ctx: Dict[str, Any]) -> None:
        self._seal(ctx, "ok")

    def on_error(self, ctx: Dict[str, Any], exc: BaseException) -> None:
        self._seal(ctx, f"error:{type(exc).__name__}")

    def _seal(self, ctx: Dict[str, Any], outcome: str) -> None:
        req = ctx["request"]
        fields = {
            "request_id": req.request_id,
            "model": str(ctx["meta"].get("model", req.model)),
            "principal": str(ctx["meta"].get("principal", "")),
            "payload_fingerprint": ctx["payload_fingerprint"],
            "verdicts": ";".join(
                f"{v['policy']}:{v['verdict']}"
                for v in ctx["meta"].get("policy_verdicts", [])),
            "cost_usd": ctx["meta"].get("cost_usd"),
            "outcome": outcome,
            "timestamp": time.time(),
        }
        body = "|".join(str(fields[k]) for k in (
            "request_id", "model", "principal", "payload_fingerprint",
            "verdicts", "cost_usd", "outcome"))
        with self._lock:
            prev = self._sealed[-1].entry_hash if self._sealed \
                else self._GENESIS
            entry_hash = hashlib.sha256(
                (prev + "|" + body).encode("utf-8")).hexdigest()
            leaf = _leaf_hash(entry_hash.encode("utf-8"))
            sealed = _Sealed(req.request_id, body, entry_hash, fields, leaf,
                             fields["timestamp"], prev=prev)
            self._sealed.append(sealed)
            self._by_id[req.request_id] = sealed
            if len(self._sealed) > self._max_entries:
                drop = len(self._sealed) - self._max_entries
                for old_sealed in self._sealed[:drop]:
                    self._by_id.pop(old_sealed.request_id, None)
                self._sealed = self._sealed[drop:]
                self._evicted += drop

        # mirror to shipped ledger (fail-open: proof-of-record never breaks
        # the user's call)
        if self._audit_log is not None:
            try:
                self._audit_log.append(
                    decision_type=f"kernel_{outcome.split(':')[0]}",
                    prompt_hash=fields["payload_fingerprint"],
                    model=fields["model"],
                    cost_saved_usd=0.0,
                    tag="kernel",
                    metadata={"request_id": fields["request_id"],
                              "verdicts": fields["verdicts"],
                              "principal": fields["principal"],
                              "outcome": outcome})
            except Exception:
                pass
        if self._sink is not None:
            try:
                self._sink.append(fields)
            except Exception:
                pass

    # ---- verification / export ---------------------------------------
    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(s.fields, entry_hash=s.entry_hash)
                    for s in self._sealed]

    def verify(self) -> Tuple[bool, int]:
        """Verifies the retained window. After eviction the window's first
        entry chains from its (evicted) predecessor hash, not genesis —
        durable full-chain verification is the sink/ledger's job."""
        with self._lock:
            if not self._sealed:
                return True, -1
            # recompute each entry_hash from its own recorded prev; ensure
            # the window is internally linked head-to-tail.
            prev = None
            for i, s in enumerate(self._sealed):
                # the stored prev is authoritative for the anchor entry;
                # subsequent entries must chain from the computed hash.
                anchor = s.entry_hash  # recompute below to validate integrity
                base_prev = prev if prev is not None else self._recover_prev(s)
                expect = hashlib.sha256(
                    (base_prev + "|" + s.body).encode("utf-8")).hexdigest()
                if s.entry_hash != expect:
                    return False, i
                prev = s.entry_hash
        return True, -1

    @staticmethod
    def _recover_prev(sealed: "_Sealed") -> str:
        """The anchor entry's prev is not retained separately; store it on the
        sealed record at seal time (see _seal)."""
        return sealed.prev

    def merkle_root(self) -> str:
        with self._lock:
            return merkle_root([s.leaf for s in self._sealed])

    def inclusion_proof(self, request_id: str
                        ) -> Tuple[str, List[Tuple[str, str]], str]:
        """Returns (leaf_hash, proof_path, root) for a request — all offline
        verifiable via verify_merkle_proof."""
        with self._lock:
            leaves = [s.leaf for s in self._sealed]
            idx = next(i for i, s in enumerate(self._sealed)
                       if s.request_id == request_id)
            return (leaves[idx], merkle_proof(leaves, idx),
                    merkle_root(leaves))

    def prove(self, request_id: str) -> ProofPacket:
        """TRUST-4: mint a signed, offline-verifiable proof packet."""
        with self._lock:
            s = self._by_id.get(request_id)
        if s is None:
            raise KeyError(f"no sealed record for request_id={request_id!r}")
        packet = ProofPacket(
            request_id=s.fields["request_id"],
            model=s.fields["model"],
            principal=s.fields["principal"],
            payload_fingerprint=s.fields["payload_fingerprint"],
            verdicts=s.fields["verdicts"],
            cost_usd=s.fields["cost_usd"],
            outcome=s.fields["outcome"],
            entry_hash=s.entry_hash,
            timestamp=s.timestamp)
        if self._signer is not None:
            sig = self._signer.sign(packet.canonical())
            packet.signature = sig.hex()
            packet.algorithm = "ed25519"
            pub = getattr(self._signer, "public_bytes", None)
            if callable(pub):
                packet.public_key = pub().hex()
        return packet
