"""
The append-only audit log with hash chain + periodic checkpoints.

Every Tokeymeter decision (cache hit/miss, redaction, compression, memory
summarization) becomes one immutable AuditEntry. Each entry's hash
includes the previous entry's hash — tampering with any entry breaks
the chain at that point and at all subsequent points.

Periodic checkpoints (every N entries) capture the chain head at a
specific time. Checkpoints can be signed and externally anchored
(timestamping service, blockchain, S3) to provide non-repudiation
even if the log file itself is later modified.

Privacy contract:
  Entries contain ONLY:
    - Hashes (HMAC-SHA256 with per-install secret — not rainbow-tableable)
    - Decision types (categorical strings)
    - Timestamps
    - Cost numbers
    - Counters (e.g., pii_redactions)
    - User-chosen tags
  Entries contain NO:
    - Prompt or response text
    - PII values
    - Plain cache keys
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tokeymeter.engines.trust.audit.signers import HMACSigner, Signer, load_or_create_install_secret

log = logging.getLogger("tokeymeter.audit")

# Number of AuditLog instances currently attached to the event bus. Lets the
# decision-record layer honestly report whether a decision is being recorded to
# a provable ledger.
_ATTACHED_COUNT = 0


def is_attached() -> bool:
    """True if at least one AuditLog is currently recording decisions."""
    return _ATTACHED_COUNT > 0


_PRIVATE_SUBSCRIBERS_LOCK = threading.Lock()
_PRIVATE_SUBSCRIBERS: List[Callable[[Any, Optional[str]], None]] = []


def _subscribe_private(callback: Callable[[Any, Optional[str]], None]) -> Callable:
    """Register an internal audit subscriber.

    This bus is intentionally not public: it carries full prompt text only to
    attached AuditLog instances so they can derive per-install HMAC hashes.
    Public observability events continue to receive only prompt_preview.
    """
    with _PRIVATE_SUBSCRIBERS_LOCK:
        if callback not in _PRIVATE_SUBSCRIBERS:
            _PRIVATE_SUBSCRIBERS.append(callback)
    return callback


def _unsubscribe_private(callback: Callable[[Any, Optional[str]], None]) -> None:
    with _PRIVATE_SUBSCRIBERS_LOCK:
        try:
            _PRIVATE_SUBSCRIBERS.remove(callback)
        except ValueError:
            pass


def _emit_private(event: Any, prompt_text: Optional[str]) -> None:
    """Dispatch to attached audit logs. Never raises."""
    try:
        with _PRIVATE_SUBSCRIBERS_LOCK:
            snapshot = list(_PRIVATE_SUBSCRIBERS)
        for sub in snapshot:
            try:
                sub(event, prompt_text)
            except Exception as e:
                log.debug("audit: private subscriber error: %s", e)
    except Exception:
        pass


# ============================================================
#                        Data model
# ============================================================

# Genesis hash: well-known constant for the FIRST entry's prev_hash.
GENESIS_HASH = hashlib.sha256(b"tokeymeter-audit-genesis-v1").hexdigest()

# Schema version embedded in each entry's hash — protects against future
# format changes silently affecting old logs.
SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class AuditEntry:
    """One immutable decision record.

    seq:              monotonic sequence number (0-indexed)
    timestamp:        unix epoch seconds (float, microsecond precision)
    decision_type:    e.g. "cache_hit_exact", "cache_miss", "redaction_applied"
    prompt_hash:      HMAC(install_secret, prompt_text)  — irreversible
    model:            LLM identifier (e.g., "gpt-4o-mini"). Used for cost math.
    cost_saved_usd:   savings claim. Auditable.
    pii_redactions:   count of PII patterns matched (NOT the values).
    tag:              optional workload label.
    function_name:    optional function identifier.
    metadata_hash:    HMAC of any user-provided structured metadata (or empty).
    prev_hash:        hash of previous entry (genesis if seq=0).
    entry_hash:       SHA-256(canonical-json of all above + schema_version).
    """
    seq: int
    timestamp: float
    decision_type: str
    prompt_hash: str
    model: str
    cost_saved_usd: float
    pii_redactions: int
    tag: Optional[str]
    function_name: Optional[str]
    metadata_hash: str
    prev_hash: str
    entry_hash: str

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization for hashing/signing.

        Excludes entry_hash itself (that's what we're computing). Uses
        sorted keys, no whitespace — bit-for-bit reproducible across
        Python versions and JSON libraries.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": self.seq,
            "timestamp": round(self.timestamp, 6),  # microsecond precision
            "decision_type": self.decision_type,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "cost_saved_usd": round(self.cost_saved_usd, 9),
            "pii_redactions": self.pii_redactions,
            "tag": self.tag,
            "function_name": self.function_name,
            "metadata_hash": self.metadata_hash,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def recompute_hash(self) -> str:
        """Recompute entry_hash from the canonical payload."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Checkpoint:
    """A signed (optionally) snapshot of the chain head at a moment in time.

    Checkpoints are produced every N entries (configurable). They can be
    externally anchored (timestamping service, S3, etc.) to provide
    non-repudiation: even if the local log file is later modified, the
    externally anchored checkpoint proves what the chain looked like at
    that time.
    """
    seq: int                  # the seq of the last entry covered
    timestamp: float
    chain_head: str           # the entry_hash of entry[seq]
    entries_covered: int      # how many entries are in this checkpoint window
    signature: Optional[str]  # hex-encoded signature, or None
    signature_algorithm: Optional[str]
    anchor_ref: Optional[str] = None  # opaque reference to external anchor

    def signed_bytes(self) -> bytes:
        """What gets signed: seq + timestamp + chain_head + entries_covered.

        Deterministic encoding."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": self.seq,
            "timestamp": round(self.timestamp, 6),
            "chain_head": self.chain_head,
            "entries_covered": self.entries_covered,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ============================================================
#                  Verification result
# ============================================================

@dataclass
class VerificationResult:
    """Returned by verify_chain() and verify_proof()."""
    valid: bool
    entries_verified: int
    first_bad_seq: Optional[int] = None
    reason: Optional[str] = None
    checkpoint_signature_valid: Optional[bool] = None
    aggregate_cost_saved_usd: float = 0.0
    by_decision_type: Dict[str, int] = field(default_factory=dict)


# ============================================================
#                  Audit subscriber
# ============================================================

# Maps event hit/hit_type to audit decision_type
def _decision_type_from_event(event) -> str:
    """Translate a CacheEvent to an audit decision_type."""
    if event is None:
        return "unknown"
    if event.hit:
        if event.hit_type and event.hit_type.startswith("shadow_"):
            return f"shadow_{event.hit_type.replace('shadow_', '')}"
        return f"cache_hit_{event.hit_type or 'unknown'}"
    if event.shadow:
        return "shadow_miss"
    return "cache_miss"


# ============================================================
#                       AuditLog
# ============================================================

class AuditLog:
    """The append-only audit log.

    Storage: SQLite at `path` (default ~/.tokeymeter/audit.db).
    Concurrency: writes go through a thread-safe queue and a background
        flusher. The hot path (event emission) is non-blocking.
    Privacy: the install_secret_path stores a 32-byte HMAC key generated
        on first use, mode 0600. Used to derive non-rainbow-tableable
        prompt_hash values.

    The log is "fail-open at every layer": any internal error is logged
    but does not raise to the caller.
    """

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        install_secret_path: Optional[str] = None,
        signing_key_path: Optional[str] = None,
        checkpoint_every: int = 1000,
        signer: Optional[Signer] = None,
        flush_interval_seconds: float = 0.5,
        queue_max_size: int = 100_000,
        durable: Optional[bool] = None,
        durability_timeout_seconds: float = 5.0,
    ):
        # Resolve through tokeymeter.paths so TOKEYMETER_HOME / set_home() are
        # honored. Lazy (None sentinel) because the home may be configured
        # after this module was imported.
        from tokeymeter import paths as _paths
        self._path = os.path.expanduser(
            path if path is not None else _paths.state_path("audit.db"))
        self._install_secret_path = os.path.expanduser(
            install_secret_path if install_secret_path is not None
            else _paths.state_path("install-secret"))
        self._checkpoint_every = max(1, int(checkpoint_every))
        # Safe-by-default: if no signer is supplied, auto-create one from a
        # persisted per-install signing key so exported proofs are signed and
        # therefore tamper-evident out of the box. An unsigned ledger's hash
        # chain is NOT tamper-evident (the hash is unkeyed and recomputable),
        # so signing must be the default, not an opt-in. To run intentionally
        # unsigned, pass an explicit NoOpSigner.
        if signer is None:
            signing_key = load_or_create_install_secret(
                os.path.expanduser(
                    signing_key_path if signing_key_path is not None
                    else _paths.state_path("audit-signing-key"))
            )
            signer = HMACSigner(signing_key)
        self._signer = signer
        # --- Enforce active security policy (opt-in; default permissive) ---
        try:
            from tokeymeter.engines.governance.policy import get_security_policy, SecurityPolicyError
            _pol = get_security_policy()
        except Exception:
            _pol = None
        if _pol is not None and _pol.require_nonrepudiable_audit:
            if isinstance(self._signer, HMACSigner):
                raise SecurityPolicyError(
                    "SecurityPolicy.require_nonrepudiable_audit is enabled: the audit "
                    "ledger must use an ASYMMETRIC signer (e.g. Ed25519Signer) for true "
                    "non-repudiation. HMAC provides integrity but the key holder can "
                    "forge, so it is refused. Pass signer=Ed25519Signer(...).")
        self._flush_interval = max(0.05, float(flush_interval_seconds))
        # Durability: when required, audit appends are never silently dropped.
        # `durable` may be None (defer to the active SecurityPolicy at append
        # time) or an explicit bool override.
        self._durable = durable
        self._durability_timeout = max(0.0, float(durability_timeout_seconds))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        # Per-install HMAC key for prompt hashing
        self._install_secret = load_or_create_install_secret(self._install_secret_path)

        # SQLite (WAL mode)
        self._sqlite_lock = threading.Lock()
        self._init_db()

        # Background flusher
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=queue_max_size)
        self._stop = threading.Event()
        self._flusher = threading.Thread(
            target=self._flusher_loop, daemon=True, name="tokeymeter-audit-flusher"
        )
        self._flusher.start()

        # Event subscription handle (set by attach())
        self._subscribed_handler: Optional[Callable] = None

        # Cache of the latest seq/hash for fast appends.
        # Initialized from disk on construction.
        self._latest_seq = -1
        self._latest_hash = GENESIS_HASH
        self._refresh_latest_from_disk()

        # Stats
        self._stats_lock = threading.Lock()
        self._stats = {
            "entries_appended": 0,
            "entries_dropped_queue_full": 0,
            "entries_written_synchronously": 0,
            "checkpoints_created": 0,
            "subscriber_errors": 0,
        }

    # ---- DB ----

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._sqlite_lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS audit_entries (
                    seq             INTEGER PRIMARY KEY,
                    timestamp       REAL NOT NULL,
                    decision_type   TEXT NOT NULL,
                    prompt_hash     TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    cost_saved_usd  REAL NOT NULL,
                    pii_redactions  INTEGER NOT NULL,
                    tag             TEXT,
                    function_name   TEXT,
                    metadata_hash   TEXT NOT NULL,
                    prev_hash       TEXT NOT NULL,
                    entry_hash      TEXT NOT NULL
                )
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_entries_timestamp
                ON audit_entries(timestamp)
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS audit_checkpoints (
                    seq                  INTEGER PRIMARY KEY,
                    timestamp            REAL NOT NULL,
                    chain_head           TEXT NOT NULL,
                    entries_covered      INTEGER NOT NULL,
                    signature            TEXT,
                    signature_algorithm  TEXT,
                    anchor_ref           TEXT
                )
            """)

    def _refresh_latest_from_disk(self) -> None:
        try:
            with self._sqlite_lock, self._conn() as c:
                row = c.execute(
                    "SELECT seq, entry_hash FROM audit_entries "
                    "ORDER BY seq DESC LIMIT 1"
                ).fetchone()
            if row:
                self._latest_seq, self._latest_hash = row
        except sqlite3.Error as e:
            log.debug("audit: refresh_latest failed: %s", e)

    # ---- Internal hashing helpers ----

    def _hmac(self, value: Any) -> str:
        """HMAC-SHA256(install_secret, str(value)) → hex string."""
        if value is None:
            return ""
        if isinstance(value, str):
            data = value.encode("utf-8", "replace")
        elif isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode("utf-8", "replace")
        return hmac.new(self._install_secret, data, hashlib.sha256).hexdigest()

    # ---- Public append API ----

    def append(
        self,
        *,
        decision_type: str,
        prompt_text: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        model: str = "_default",
        cost_saved_usd: float = 0.0,
        pii_redactions: int = 0,
        tag: Optional[str] = None,
        function_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Queue an entry for the background flusher. Never raises.

        Either provide raw `prompt_text` (we'll HMAC it) or a pre-computed
        `prompt_hash`. Providing both is allowed; we'll prefer prompt_hash.
        """
        try:
            if prompt_hash is None:
                prompt_hash = self._hmac(prompt_text or "")
            if metadata:
                # Hash all metadata values together to avoid leaking
                meta_json = json.dumps(metadata, sort_keys=True, default=str)
                metadata_hash = self._hmac(meta_json)
            else:
                metadata_hash = ""

            record = {
                "timestamp": float(timestamp) if timestamp is not None else time.time(),
                "decision_type": str(decision_type),
                "prompt_hash": str(prompt_hash),
                "model": str(model),
                "cost_saved_usd": float(cost_saved_usd),
                "pii_redactions": int(pii_redactions),
                "tag": tag,
                "function_name": function_name,
                "metadata_hash": metadata_hash,
            }
            try:
                if self._durability_required():
                    self._durable_enqueue(record)
                else:
                    self._queue.put_nowait(record)
            except queue.Full:
                with self._stats_lock:
                    self._stats["entries_dropped_queue_full"] += 1
                log.warning("tokeymeter.audit: queue full, dropping entry. "
                            "Consider increasing queue_max_size or flush_interval.")
        except Exception as e:
            log.debug("tokeymeter.audit: append failed: %s", e)

    def _durability_required(self) -> bool:
        """Whether audit appends must be guaranteed (no silent drop).

        Explicit constructor override wins; otherwise defer to the live
        SecurityPolicy so enterprise_defaults()/set_security_policy() take effect
        regardless of when the ledger was constructed.
        """
        if self._durable is not None:
            return self._durable
        try:
            from tokeymeter.engines.governance.policy import get_security_policy
            return bool(get_security_policy().require_audit_durability)
        except Exception:
            return False

    def _durable_enqueue(self, record: dict) -> None:
        """Enqueue with bounded backpressure; never drop.

        Step 1: block up to durability_timeout for a queue slot — absorbs all
        realistic bursts (the flusher drains continuously).
        Step 2: if the queue is STILL full (a stalled/dead flusher), write the
        entry through SYNCHRONOUSLY via the same chain-safe path the flusher
        uses (serialized on the sqlite lock), so the decision is persisted rather
        than lost. This keeps the observer from ever breaking the observed (no
        raise into the caller) while guaranteeing the record survives.
        """
        try:
            self._queue.put(record, timeout=self._durability_timeout)
            return
        except queue.Full:
            pass
        # Backpressure exhausted -> persist synchronously instead of dropping.
        try:
            self._write_batch([record])
            with self._stats_lock:
                self._stats["entries_written_synchronously"] += 1
            log.warning("tokeymeter.audit: queue saturated under durability "
                        "requirement; wrote entry synchronously (flusher may be "
                        "stalled).")
        except Exception as e:
            with self._stats_lock:
                self._stats["entries_dropped_queue_full"] += 1
            log.error("tokeymeter.audit: durable synchronous write failed, "
                      "entry lost: %s", e)

    # ---- Background flusher ----

    def _flusher_loop(self) -> None:
        """Drains the queue periodically, writing batched entries."""
        buf: List[dict] = []
        while not self._stop.is_set():
            try:
                # Block briefly for the first item, then drain anything else
                try:
                    first = self._queue.get(timeout=self._flush_interval)
                    buf.append(first)
                except queue.Empty:
                    continue

                # Drain whatever else is sitting in the queue right now
                while True:
                    try:
                        buf.append(self._queue.get_nowait())
                    except queue.Empty:
                        break

                if buf:
                    self._write_batch(buf)
                    buf.clear()
            except Exception as e:
                log.warning("tokeymeter.audit: flusher loop error: %s", e)
                buf.clear()  # don't get stuck on a bad batch
                time.sleep(self._flush_interval)

        # On shutdown, drain remaining
        try:
            while True:
                try:
                    buf.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if buf:
                self._write_batch(buf)
        except Exception as e:
            log.debug("tokeymeter.audit: shutdown drain failed: %s", e)

    def _write_batch(self, batch: List[dict]) -> None:
        """Atomically append a batch of entries to the chain."""
        try:
            with self._sqlite_lock, self._conn() as c:
                # Begin tx
                c.execute("BEGIN")
                try:
                    prev_hash = self._latest_hash
                    next_seq = self._latest_seq + 1
                    rows: List[Tuple] = []
                    checkpoints_to_create: List[Tuple[int, str, float]] = []

                    for rec in batch:
                        entry = AuditEntry(
                            seq=next_seq,
                            timestamp=rec["timestamp"],
                            decision_type=rec["decision_type"],
                            prompt_hash=rec["prompt_hash"],
                            model=rec["model"],
                            cost_saved_usd=rec["cost_saved_usd"],
                            pii_redactions=rec["pii_redactions"],
                            tag=rec["tag"],
                            function_name=rec["function_name"],
                            metadata_hash=rec["metadata_hash"],
                            prev_hash=prev_hash,
                            entry_hash="",
                        )
                        entry_hash = entry.recompute_hash()
                        # rebuild with the hash filled in
                        rows.append((
                            entry.seq, entry.timestamp, entry.decision_type,
                            entry.prompt_hash, entry.model, entry.cost_saved_usd,
                            entry.pii_redactions, entry.tag, entry.function_name,
                            entry.metadata_hash, entry.prev_hash, entry_hash,
                        ))

                        # Possibly create a checkpoint at this seq
                        if (next_seq + 1) % self._checkpoint_every == 0:
                            checkpoints_to_create.append(
                                (next_seq, entry_hash, rec["timestamp"])
                            )

                        prev_hash = entry_hash
                        next_seq += 1

                    c.executemany(
                        "INSERT INTO audit_entries (seq, timestamp, decision_type, "
                        "prompt_hash, model, cost_saved_usd, pii_redactions, tag, "
                        "function_name, metadata_hash, prev_hash, entry_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )

                    # Create any checkpoints in the same transaction
                    for cp_seq, cp_hash, cp_ts in checkpoints_to_create:
                        cp = self._build_checkpoint(cp_seq, cp_hash, cp_ts)
                        c.execute(
                            "INSERT OR REPLACE INTO audit_checkpoints "
                            "(seq, timestamp, chain_head, entries_covered, "
                            "signature, signature_algorithm, anchor_ref) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (cp.seq, cp.timestamp, cp.chain_head, cp.entries_covered,
                             cp.signature, cp.signature_algorithm, cp.anchor_ref),
                        )
                        with self._stats_lock:
                            self._stats["checkpoints_created"] += 1

                    c.execute("COMMIT")
                except Exception:
                    c.execute("ROLLBACK")
                    raise

            # Update in-memory cache
            self._latest_seq = next_seq - 1
            self._latest_hash = prev_hash
            with self._stats_lock:
                self._stats["entries_appended"] += len(batch)
        except Exception as e:
            log.warning("tokeymeter.audit: batch write failed: %s", e)

    def _build_checkpoint(
        self, seq: int, chain_head: str, timestamp: float
    ) -> Checkpoint:
        entries_covered = self._checkpoint_every
        cp = Checkpoint(
            seq=seq, timestamp=timestamp, chain_head=chain_head,
            entries_covered=entries_covered,
            signature=None, signature_algorithm=None,
        )
        if self._signer is not None:
            try:
                sig = self._signer.sign(cp.signed_bytes())
                cp = Checkpoint(
                    seq=cp.seq, timestamp=cp.timestamp, chain_head=cp.chain_head,
                    entries_covered=cp.entries_covered,
                    signature=sig.hex(),
                    signature_algorithm=self._signer.algorithm,
                )
            except Exception as e:
                log.debug("audit: checkpoint signing failed: %s", e)
        return cp

    # ---- Subscriber wiring ----

    def attach(self) -> None:
        """Subscribe to Tokeymeter decisions through the private audit bus.

        The private bus receives raw prompt text in-process so the ledger can
        store HMAC(prompt) without putting plaintext on public observability
        events. Idempotent.
        """

        if self._subscribed_handler is not None:
            return  # already attached

        def handler(event, prompt_text=None):
            try:
                # Translate event → audit entry, fail-open on any error
                cost_saved = (
                    float(event.estimated_cost_usd)
                    if event.hit and event.estimated_cost_usd
                    else 0.0
                )
                extra = event.extra or {}

                self.append(
                    decision_type=_decision_type_from_event(event),
                    prompt_text=prompt_text or "",
                    model=event.model or "_default",
                    cost_saved_usd=cost_saved,
                    pii_redactions=int(extra.get("pii_redactions", 0) or 0),
                    tag=event.tag,
                    function_name=event.function_name,
                    timestamp=event.timestamp,
                    metadata={
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "latency_ms": event.latency_ms,
                        "compression_ratio": extra.get("compression_ratio"),
                        "compression_method": extra.get("compression_method"),
                    },
                )
            except Exception as e:
                with self._stats_lock:
                    self._stats["subscriber_errors"] += 1
                log.debug("audit: subscriber error: %s", e)

        _subscribe_private(handler)
        self._subscribed_handler = handler
        global _ATTACHED_COUNT
        _ATTACHED_COUNT += 1

    def detach(self) -> None:
        """Unsubscribe from the private audit bus."""

        if self._subscribed_handler is not None:
            _unsubscribe_private(self._subscribed_handler)
            self._subscribed_handler = None
            global _ATTACHED_COUNT
            _ATTACHED_COUNT = max(0, _ATTACHED_COUNT - 1)

    # ---- Read API ----

    def get_entry(self, seq: int) -> Optional[AuditEntry]:
        """Fetch one entry by seq number."""
        try:
            with self._sqlite_lock, self._conn() as c:
                row = c.execute(
                    "SELECT seq, timestamp, decision_type, prompt_hash, model, "
                    "cost_saved_usd, pii_redactions, tag, function_name, "
                    "metadata_hash, prev_hash, entry_hash "
                    "FROM audit_entries WHERE seq = ?",
                    (seq,),
                ).fetchone()
            return AuditEntry(*row) if row else None
        except sqlite3.Error:
            return None

    def get_entries(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        seq_start: Optional[int] = None,
        seq_end: Optional[int] = None,
    ) -> List[AuditEntry]:
        """Fetch entries by time range or sequence range."""
        conditions = []
        params: list = []
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(float(since))
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(float(until))
        if seq_start is not None:
            conditions.append("seq >= ?")
            params.append(int(seq_start))
        if seq_end is not None:
            conditions.append("seq <= ?")
            params.append(int(seq_end))

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        try:
            with self._sqlite_lock, self._conn() as c:
                rows = c.execute(
                    "SELECT seq, timestamp, decision_type, prompt_hash, model, "
                    "cost_saved_usd, pii_redactions, tag, function_name, "
                    "metadata_hash, prev_hash, entry_hash "
                    f"FROM audit_entries{where} ORDER BY seq ASC",
                    params,
                ).fetchall()
            return [AuditEntry(*r) for r in rows]
        except sqlite3.Error:
            return []

    def get_checkpoint(self, seq: int) -> Optional[Checkpoint]:
        try:
            with self._sqlite_lock, self._conn() as c:
                row = c.execute(
                    "SELECT seq, timestamp, chain_head, entries_covered, "
                    "signature, signature_algorithm, anchor_ref "
                    "FROM audit_checkpoints WHERE seq = ?",
                    (seq,),
                ).fetchone()
            return Checkpoint(*row) if row else None
        except sqlite3.Error:
            return None

    def get_checkpoints(self) -> List[Checkpoint]:
        try:
            with self._sqlite_lock, self._conn() as c:
                rows = c.execute(
                    "SELECT seq, timestamp, chain_head, entries_covered, "
                    "signature, signature_algorithm, anchor_ref "
                    "FROM audit_checkpoints ORDER BY seq ASC"
                ).fetchall()
            return [Checkpoint(*r) for r in rows]
        except sqlite3.Error:
            return []

    # ---- Verification ----

    def verify_chain(
        self,
        *,
        seq_start: int = 0,
        seq_end: Optional[int] = None,
    ) -> VerificationResult:
        """Walk the chain and verify every entry's hash + prev_hash linkage.

        Returns a VerificationResult with details. NEVER raises.
        """
        try:
            entries = self.get_entries(seq_start=seq_start, seq_end=seq_end)
            if not entries:
                return VerificationResult(valid=True, entries_verified=0)
            return verify_entries(entries)
        except Exception as e:
            return VerificationResult(
                valid=False, entries_verified=0,
                reason=f"verify_chain raised: {e}",
            )

    # ---- Stats / lifecycle ----

    def stats(self) -> dict:
        """Process-local audit stats."""
        with self._stats_lock:
            base = dict(self._stats)
        base["latest_seq"] = self._latest_seq
        base["latest_hash"] = self._latest_hash
        return base

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue is drained (best-effort)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.empty():
                # Give the flusher a moment to finalize
                time.sleep(0.05)
                if self._queue.empty():
                    return
            time.sleep(0.02)

    def close(self) -> None:
        """Stop the background flusher and drain remaining entries."""
        self._stop.set()
        try:
            self._flusher.join(timeout=5.0)
        except RuntimeError:
            pass
        self.detach()


# ============================================================
#               Stateless entry verification
# ============================================================

def verify_entries(entries: List[AuditEntry]) -> VerificationResult:
    """Verify a list of entries forms a valid hash chain.

    Stateless: no Tokeymeter dependency needed beyond this file. Anyone with
    a list of AuditEntry objects (or equivalent dicts) can call this.

    Checks:
      1. Each entry's recomputed hash matches the stored entry_hash.
      2. Each entry's prev_hash matches the previous entry's entry_hash.
      3. Sequence numbers are monotonically increasing by 1.
      4. First entry's prev_hash is GENESIS_HASH (if seq_start=0) OR
         we don't verify that (caller verifies via checkpoint).
    """
    if not entries:
        return VerificationResult(valid=True, entries_verified=0)

    by_type: Dict[str, int] = {}
    cost_total = 0.0
    expected_prev = None
    expected_seq = entries[0].seq

    for i, entry in enumerate(entries):
        # 1. Seq increments by 1
        if entry.seq != expected_seq:
            return VerificationResult(
                valid=False, entries_verified=i,
                first_bad_seq=entry.seq,
                reason=f"seq gap: expected {expected_seq}, got {entry.seq}",
            )
        expected_seq += 1

        # 2. Entry hash is correctly computed
        recomputed = entry.recompute_hash()
        if recomputed != entry.entry_hash:
            return VerificationResult(
                valid=False, entries_verified=i,
                first_bad_seq=entry.seq,
                reason=f"entry_hash mismatch at seq={entry.seq}: "
                       f"recomputed {recomputed[:16]}... vs stored {entry.entry_hash[:16]}...",
            )

        # 3. prev_hash links to previous (only after the first entry in this batch)
        if expected_prev is not None and entry.prev_hash != expected_prev:
            return VerificationResult(
                valid=False, entries_verified=i,
                first_bad_seq=entry.seq,
                reason=f"prev_hash broken at seq={entry.seq}: "
                       f"expected {expected_prev[:16]}..., got {entry.prev_hash[:16]}...",
            )

        # 4. Genesis check (only if this batch starts at seq 0)
        if i == 0 and entry.seq == 0 and entry.prev_hash != GENESIS_HASH:
            return VerificationResult(
                valid=False, entries_verified=0,
                first_bad_seq=0,
                reason=f"genesis mismatch: prev_hash={entry.prev_hash[:16]}... "
                       f"vs GENESIS_HASH={GENESIS_HASH[:16]}...",
            )

        by_type[entry.decision_type] = by_type.get(entry.decision_type, 0) + 1
        cost_total += entry.cost_saved_usd
        expected_prev = entry.entry_hash

    return VerificationResult(
        valid=True,
        entries_verified=len(entries),
        aggregate_cost_saved_usd=round(cost_total, 6),
        by_decision_type=by_type,
    )
