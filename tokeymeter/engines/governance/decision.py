"""DecisionRecord — the first-class record of one AI decision.

This is the control-plane primitive. Where `CacheEvent` is the low-level
observability event and the audit ledger is the signed, hash-chained proof
store, `DecisionRecord` is the single, public, human- and machine-readable
object that answers, for one decision:

  * what decision happened        -> .decision, .hit_type
  * why it happened               -> .reason
  * what it cost                  -> .cost_usd, .cost_saved_usd, tokens
  * what was redacted             -> .pii_redactions
  * whether it was shadowed       -> .shadowed
  * whether cached or routed      -> .cached, .hit_type, .model, .routed_model
  * how it can be replayed/proven -> .cache_key, .provable, proof_reference()

DecisionRecords are derived from CacheEvents (no new data is invented). Subscribe
with `tokeymeter.on_decision(cb)` to receive one per call — that stream IS the control
plane: feed it to a dashboard, a ledger export, a policy engine, or storage.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class DecisionRecord:
    # ---- what ----
    decision: str                      # e.g. "cache_hit", "cache_miss", "high_stakes", "shadow", "error"
    timestamp: float
    model: str
    tag: Optional[str] = None
    function_name: Optional[str] = None
    # ---- why ----
    reason: str = ""                   # human-readable explanation of the decision path
    # ---- cost ----
    cost_usd: float = 0.0              # what this call SPENT (0 on a hit)
    cost_saved_usd: float = 0.0        # what this call SAVED (the avoided cost, on a hit)
    input_tokens: int = 0
    output_tokens: int = 0
    # ---- transformations ----
    cached: bool = False
    hit_type: Optional[str] = None     # "exact" | "semantic" | "single_flight" | "high_stakes"
    shadowed: bool = False
    compressed: bool = False
    compression_ratio: Optional[float] = None
    tokens_saved_via_compression: int = 0
    pii_redactions: int = 0
    high_stakes: bool = False
    lineage: Optional[str] = None
    routed_model: Optional[str] = None  # reserved for provable routing (v0.11)
    # ---- proof / replay ----
    principal: Optional[str] = None     # v0.14 identity binding (who ran this)
    token_source: Optional[str] = None  # v0.14 T1.1: reported|estimated
    cache_key: Optional[str] = None     # links to the signed audit entry (prompt_hash)
    provable: bool = False              # an audit ledger was recording when this happened
    latency_ms: float = 0.0
    error: Optional[str] = None

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_event(cls, event: Any, provable: Optional[bool] = None) -> "DecisionRecord":
        """Derive a DecisionRecord from a tokeymeter.events.CacheEvent."""
        extra = getattr(event, "extra", None) or {}
        hit = bool(getattr(event, "hit", False))
        hit_type = getattr(event, "hit_type", None)
        est = float(getattr(event, "estimated_cost_usd", 0.0) or 0.0)
        comp_ratio = extra.get("compression_ratio")
        is_high_stakes = (hit_type == "high_stakes")
        err = getattr(event, "error", None)

        if provable is None:
            try:
                from tokeymeter.engines.trust.audit import is_attached
                provable = bool(is_attached())
            except Exception:
                provable = False

        decision = cls._decision_label(event, hit, hit_type, err)
        reason = cls._reason(decision, hit_type, comp_ratio,
                             int(extra.get("pii_redactions", 0) or 0),
                             bool(getattr(event, "shadow", False)), err)

        return cls(
            principal=getattr(event, "principal", None),
            token_source=getattr(event, "token_source", None),
            decision=decision,
            timestamp=float(getattr(event, "timestamp", 0.0) or 0.0),
            model=getattr(event, "model", "_default") or "_default",
            tag=getattr(event, "tag", None),
            function_name=getattr(event, "function_name", None),
            reason=reason,
            cost_usd=0.0 if hit else est,
            cost_saved_usd=est if hit else 0.0,
            input_tokens=int(getattr(event, "input_tokens", 0) or 0),
            output_tokens=int(getattr(event, "output_tokens", 0) or 0),
            cached=hit,
            hit_type=hit_type,
            shadowed=bool(getattr(event, "shadow", False)),
            compressed=comp_ratio is not None,
            compression_ratio=comp_ratio,
            tokens_saved_via_compression=int(extra.get("tokens_saved_via_compression", 0) or 0),
            pii_redactions=int(extra.get("pii_redactions", 0) or 0),
            high_stakes=is_high_stakes,
            lineage=extra.get("lineage"),
            routed_model=extra.get("routed_model"),
            cache_key=getattr(event, "cache_key", None),
            provable=bool(provable),
            latency_ms=float(getattr(event, "latency_ms", 0.0) or 0.0),
            error=err,
        )

    @staticmethod
    def _decision_label(event, hit, hit_type, err) -> str:
        if err:
            return "error"
        if getattr(event, "shadow", False):
            return "shadow"
        if hit_type == "high_stakes":
            return "high_stakes"
        if hit:
            return "cache_hit"
        return "cache_miss"

    @staticmethod
    def _reason(decision, hit_type, comp_ratio, pii, shadowed, err) -> str:
        if decision == "error":
            return f"Call errored: {err}"
        if decision == "shadow":
            return "Shadow comparison; result computed for comparison, not served."
        if decision == "high_stakes":
            return "Computed fresh; ALL optimization bypassed (high-stakes); not cached."
        if decision == "cache_hit":
            kind = {"exact": "exact", "semantic": "semantic",
                    "single_flight": "single-flight coalesced"}.get(hit_type, hit_type or "exact")
            return f"Served from {kind} cache; no model call."
        # cache_miss
        bits = ["Computed fresh (cache miss)"]
        if comp_ratio is not None:
            bits.append(f"prompt compressed (ratio {comp_ratio:.2f})")
        if pii:
            bits.append(f"{pii} PII span(s) redacted")
        return "; ".join(bits) + "."

    # ---------------------------------------------------------------- surfaces
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def explain(self) -> str:
        """One-line human summary."""
        money = (f"saved ${self.cost_saved_usd:.6f}" if self.cached
                 else f"spent ${self.cost_usd:.6f}")
        prov = "provable" if self.provable else "unrecorded"
        return f"[{self.decision}] {self.reason} ({money}; {prov})"

    def proof_reference(self) -> Optional[str]:
        """Handle to locate this decision's signed entry in the audit ledger
        (the ledger stores it as prompt_hash). None if not provable / no key."""
        return self.cache_key if self.provable else None

    def replayable(self) -> bool:
        """A decision is replayable if it was not a shadow/error: re-invoking the
        wrapped function with the same inputs reproduces the same routing."""
        return self.decision not in ("error", "shadow")


# --------------------------------------------------------------- subscription
_lock = threading.Lock()
_subscribers: List[Callable[[DecisionRecord], None]] = []


def on_decision(callback: Callable[[DecisionRecord], None]) -> Callable[[DecisionRecord], None]:
    """Register a callback to receive a DecisionRecord for every decision.

    Returns the callback (so it can be used as a decorator) and is the public
    entry point to the control plane. Never let a subscriber raise into Tokeymeter —
    dispatch is wrapped and fail-open."""
    with _lock:
        if callback not in _subscribers:
            _subscribers.append(callback)
    return callback


def remove_decision_subscriber(callback: Callable[[DecisionRecord], None]) -> None:
    with _lock:
        if callback in _subscribers:
            _subscribers.remove(callback)


def clear_decision_subscribers() -> None:
    with _lock:
        _subscribers.clear()


def _dispatch(record: DecisionRecord) -> None:
    """Internal: deliver a record to all subscribers. Fail-open per subscriber."""
    with _lock:
        snapshot = list(_subscribers)
    for cb in snapshot:
        try:
            cb(record)
        except Exception:
            pass
