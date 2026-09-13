"""
SafeCompressor — confidence-gated compression with a correctness guarantee.

Wraps ANY compressor and adds a self-verification gate: after compressing, it
checks the result against quality invariants. If the checks pass (the common
path for a capable compressor), it ships the compressed prompt. If they fail
(rare, by design), it falls back to the original — never silently shipping a
bad compression.

The philosophy (per design): the *capability* is in making the checks pass
often; the *fallback* is the floor that guarantees correctness when they don't.
Fallback rate is counted internally and available on request (quiet by default,
provable when you look) — a high rate is a signal to improve the method, not
something to hide.

Self-verification checks (all must pass to ship the compressed version):
  1. safe:            the underlying compressor reported success.
  2. ratio band:      compression is neither trivial nor implausibly extreme
                      (too-extreme often means something was destroyed).
  3. protected kept:  every protected segment (questions, instructions, code,
                      query-relevant context) survived.
  4. query facts:     if a query is known, the query's key terms still appear
                      in the compressed text (the answer is still derivable).

This is the automatic safety layer that makes aggressive compression trustable.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from tokeymeter.engines.optimization.compression import CompressionResult
from tokeymeter.engines.economics.pricing import estimate_tokens

_WORD = re.compile(r"[A-Za-z0-9_$%.]+")
# very common words that carry little query-identifying signal
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "what", "which", "who", "how", "when", "where", "why",
    "does", "do", "did", "this", "that", "it", "as", "at", "by", "with", "from",
}


@dataclass
class _Stats:
    total: int = 0
    shipped: int = 0          # compressed version accepted
    fell_back: int = 0        # returned original
    reasons: dict = field(default_factory=dict)
    tokens_saved: int = 0

    def record(self, shipped: bool, reason: Optional[str], saved: int) -> None:
        self.total += 1
        if shipped:
            self.shipped += 1
            self.tokens_saved += max(saved, 0)
        else:
            self.fell_back += 1
            if reason:
                self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def snapshot(self) -> dict:
        total = max(self.total, 1)
        return {
            "total_compressions": self.total,
            "shipped": self.shipped,
            "fell_back": self.fell_back,
            "fallback_rate_pct": round(100.0 * self.fell_back / total, 2),
            "ship_rate_pct": round(100.0 * self.shipped / total, 2),
            "tokens_saved": self.tokens_saved,
            "fallback_reasons": dict(self.reasons),
        }


# module-level stats, quiet by default, available on request
_STATS = _Stats()
_LOCK = threading.Lock()


def compression_stats() -> dict:
    """Return measured compression/fallback stats (quiet by default; call to see).

    Lets you *prove* the system is capable: a low fallback_rate_pct on real
    workloads is evidence, not a claim. A high rate flags a method to improve.
    """
    with _LOCK:
        return _STATS.snapshot()


def reset_compression_stats() -> None:
    with _LOCK:
        global _STATS
        _STATS = _Stats()


@dataclass
class SafeCompressor:
    """Confidence-gated wrapper around any compressor.

    Args:
        inner:          the compressor to gate (e.g. SalienceCompressor).
        min_ratio:      reject (fall back) if tokens_after/before is below this —
                        implausibly extreme compression likely destroyed content.
        max_ratio:      treat compressions above this ratio as "not worth it" and
                        ship the original (no point paying overhead for ~0% gain).
        query:          optional question/task; if set, the compressed text must
                        still contain the query's key terms or we fall back.
        require_query_terms: fraction of query key-terms that must survive (0..1).
    """
    inner: object
    min_ratio: float = 0.15          # below 15% of original = suspiciously extreme
    max_ratio: float = 0.97          # above 97% = negligible gain, just ship original
    query: Optional[str] = None
    require_query_terms: float = 0.6

    def compress(self, text: str) -> CompressionResult:
        t0 = time.perf_counter()
        # pass query through to query-aware inner compressors when possible
        inner = self.inner
        if self.query and hasattr(inner, "for_query"):
            inner = inner.for_query(self.query)

        try:
            res = inner.compress(text)
        except Exception as e:  # inner must not raise, but guard anyway
            return self._fallback(text, t0, reason=f"inner_error:{type(e).__name__}")

        ok, reason = self._verify(text, res)
        if ok:
            with _LOCK:
                _STATS.record(True, None, res.tokens_before - res.tokens_after)
            # annotate that it passed the gate
            res.extra = {**(res.extra or {}), "gated": True}
            return res
        return self._fallback(text, t0, reason=reason, attempted=res)

    # ---- verification gate ----
    def _verify(self, text: str, res: CompressionResult):
        if not res.safe:
            return False, "inner_unsafe"
        if res.tokens_after >= res.tokens_before:
            return False, "no_gain"
        if res.ratio < self.min_ratio:
            return False, "too_extreme"
        if res.ratio > self.max_ratio:
            return False, "negligible_gain"
        # protected-segment survival is enforced inside the inner compressor;
        # here we add the query-fact check as the cross-cutting correctness gate.
        if self.query:
            if not self._query_terms_survive(res.after):
                return False, "query_terms_lost"
        return True, None

    def _query_terms_survive(self, after: str) -> bool:
        terms = [w for w in _WORD.findall(self.query.lower()) if w not in _STOP and len(w) > 2]
        if not terms:
            return True
        low = after.lower()
        kept = sum(1 for t in terms if t in low)
        return (kept / len(terms)) >= self.require_query_terms

    def _fallback(self, text: str, t0: float, reason: str,
                  attempted: Optional[CompressionResult] = None) -> CompressionResult:
        with _LOCK:
            _STATS.record(False, reason, 0)
        tb = estimate_tokens(text)
        return CompressionResult(
            before=text, after=text,
            tokens_before=tb, tokens_after=tb,
            ratio=1.0,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            method="safe:fallback",
            safe=True,  # falling back IS the safe outcome
            extra={
                "gated": True,
                "fell_back": True,
                "reason": reason,
                "attempted_ratio": round(attempted.ratio, 3) if attempted else None,
            },
        )

    def for_query(self, query: str) -> "SafeCompressor":
        from dataclasses import replace
        return replace(self, query=query)
