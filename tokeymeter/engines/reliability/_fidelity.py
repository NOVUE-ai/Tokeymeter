"""Compression fidelity controls — extracted from decorator.py.

Self-contained, state-owning unit: the conservative reduction cap and the
measured-fidelity circuit breaker, with their module-level state. Readers in
decorator.py reach the live state via get_breaker() / get_max_reduction() so
the setters' rebinds are always observed (never a stale alias).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("tokeymeter")


# Conservative compression: fall back to the original prompt when a compressor
# would remove more than this fraction of tokens (default 80%). This makes Tokeymeter
# a quality-preserving optimizer by refusing pathological/over-aggressive
# compression. Set to 1.0 to disable (allow any reduction).
_COMPRESSION_MAX_REDUCTION: float = 0.8


def set_compression_max_reduction(fraction: float) -> None:
    """Set the conservative compression fidelity floor (0.0–1.0).

    A compressor that would remove more than `fraction` of the prompt's tokens
    is treated as too lossy and the ORIGINAL prompt is used instead. 1.0
    disables the guard (any reduction allowed); lower is more conservative.
    """
    global _COMPRESSION_MAX_REDUCTION
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0.0, 1.0]")
    _COMPRESSION_MAX_REDUCTION = float(fraction)


class _CompressionFidelityBreaker:
    """Per-workload circuit breaker driven by MEASURED compression fidelity.

    It consumes the similarity scores produced by the verify_rate sampler
    (compressed-output vs original-output) and, when recent measured fidelity
    for a workload falls below a threshold, OPENS — causing Tokeymeter to withhold
    compression for that workload (fall back to the original prompt) until a
    cooldown elapses and fidelity is re-probed.

    Important honesty property: this reacts ONLY to measured fidelity. It never
    predicts or guesses quality. With verify_rate == 0 there are no samples, so
    the breaker stays CLOSED and never trips (compression behaves as before).

    States per workload key:
      CLOSED    -> compression allowed; samples accumulate
      OPEN      -> compression withheld; after `cooldown_s` -> HALF_OPEN
      HALF_OPEN -> compression allowed again (probe); recovery if fidelity
                   climbs back to >= close_threshold, else re-OPEN

    Thread-safe. Fail-open: any internal error degrades to "allow compression".
    """

    def __init__(self, *, enabled: bool = True, open_threshold: float = 0.80,
                 close_threshold: float = 0.85, min_samples: int = 5,
                 window: int = 20, cooldown_s: float = 300.0):
        self.enabled = enabled
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.min_samples = max(1, int(min_samples))
        self.window = max(self.min_samples, int(window))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._lock = __import__("threading").Lock()
        # key -> dict(state, samples[list], opened_at)
        self._wk: dict = {}

    def _entry(self, key: str) -> dict:
        e = self._wk.get(key)
        if e is None:
            e = {"state": "closed", "samples": [], "opened_at": 0.0}
            self._wk[key] = e
        return e

    def should_compress(self, key: str) -> bool:
        """Return True if compression is currently allowed for this workload."""
        if not self.enabled:
            return True
        try:
            with self._lock:
                e = self._entry(key)
                if e["state"] == "open":
                    # Transition to half-open after cooldown to re-probe. Clear
                    # stale samples so recovery is judged ONLY on fresh probe
                    # data (otherwise old low scores would re-trip immediately).
                    if (time.time() - e["opened_at"]) >= self.cooldown_s:
                        e["state"] = "half_open"
                        e["samples"] = []
                        return True
                    return False
                return True  # closed or half_open
        except Exception:
            return True  # fail-open

    def record(self, key: str, similarity: float) -> None:
        """Feed a MEASURED fidelity sample and update the breaker state."""
        if not self.enabled:
            return
        try:
            sim = float(similarity)
        except (TypeError, ValueError):
            return
        try:
            with self._lock:
                e = self._entry(key)
                e["samples"].append(sim)
                if len(e["samples"]) > self.window:
                    del e["samples"][: len(e["samples"]) - self.window]
                n = len(e["samples"])
                if n < self.min_samples:
                    return
                mean = sum(e["samples"]) / n
                if e["state"] == "half_open":
                    # Recovery decision on fresh probe samples (hysteresis).
                    if mean >= self.close_threshold:
                        e["state"] = "closed"
                        log.info("tokeymeter.compression: fidelity breaker CLOSED for %r "
                                 "(recovered to %.3f).", key, mean)
                    elif mean < self.open_threshold:
                        e["state"] = "open"
                        e["opened_at"] = time.time()
                        log.warning("tokeymeter.compression: fidelity breaker RE-OPEN for %r "
                                    "(probe %.3f still < %.3f).", key, mean, self.open_threshold)
                elif e["state"] == "closed":
                    if mean < self.open_threshold:
                        e["state"] = "open"
                        e["opened_at"] = time.time()
                        log.warning(
                            "tokeymeter.compression: fidelity breaker OPEN for %r "
                            "(mean similarity %.3f < %.3f over %d samples); "
                            "withholding compression for this workload.",
                            key, mean, self.open_threshold, n)
        except Exception:
            pass

    def state(self, key: str) -> dict:
        with self._lock:
            e = self._entry(key)
            n = len(e["samples"])
            return {
                "state": e["state"],
                "samples": n,
                "mean_similarity": (sum(e["samples"]) / n) if n else None,
                "opened_at": e["opened_at"] or None,
            }

    def reset(self) -> None:
        with self._lock:
            self._wk.clear()


# Module-level singleton. Conservative defaults; tune via the setter below.
_compression_breaker = _CompressionFidelityBreaker()


def set_fidelity_circuit_breaker(*, enabled: bool = True, open_threshold: float = 0.80,
                                 close_threshold: float = 0.85, min_samples: int = 5,
                                 window: int = 20, cooldown_s: float = 300.0) -> None:
    """Configure the measured-fidelity compression circuit breaker.

    The breaker withholds compression for a workload once its MEASURED fidelity
    (from verify_rate sampling) drops below `open_threshold`, and recovers after
    `cooldown_s` if fidelity climbs back to `close_threshold` (hysteresis avoids
    flapping). Requires verify_rate > 0 to have data; otherwise it never trips.
    """
    global _compression_breaker
    if not (0.0 <= open_threshold <= 1.0 and 0.0 <= close_threshold <= 1.0):
        raise ValueError("thresholds must be in [0.0, 1.0]")
    if close_threshold < open_threshold:
        raise ValueError("close_threshold must be >= open_threshold (hysteresis)")
    _compression_breaker = _CompressionFidelityBreaker(
        enabled=enabled, open_threshold=open_threshold, close_threshold=close_threshold,
        min_samples=min_samples, window=window, cooldown_s=cooldown_s)


def compression_breaker_state(function_name: str, tag: Optional[str] = None) -> dict:
    """Inspect the breaker state for a workload (function_name + optional tag)."""
    return _compression_breaker.state(f"{function_name}::{tag}")


def get_breaker() -> "_CompressionFidelityBreaker":
    """Live accessor: always returns the CURRENT breaker (set_fidelity_circuit_breaker rebinds it)."""
    return _compression_breaker


def get_max_reduction() -> float:
    """Live accessor: always returns the CURRENT max-reduction cap."""
    return _COMPRESSION_MAX_REDUCTION
