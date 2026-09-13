"""
Lightweight, always-on instrumentation of the meter's own per-call overhead.

It records the wrapper overhead of cache HITS — where no wrapped function runs, so
the measured time is pure meter overhead — into a bounded ring, and exposes
p50/p95/p99 so operators can prove the overhead is near-invisible and catch
regressions over time.

The recording path is a single bounded-deque append (atomic under the GIL); it must
never add meaningful cost to the very thing it measures.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

_MAXLEN = 4096
_samples: Deque[float] = deque(maxlen=_MAXLEN)
_count = 0


def record(overhead_ms: float) -> None:
    """Record one cache-hit overhead sample (milliseconds). Hot-path safe."""
    global _count
    _samples.append(overhead_ms)   # deque.append is atomic under the GIL — no lock
    _count += 1                     # a benign race here is fine; it isn't load-bearing


def reset() -> None:
    global _count
    _samples.clear()
    _count = 0


def _pct(values, q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * q))], 4)


def percentiles() -> dict:
    """p50/p95/p99 of recent cache-hit overhead, plus sample counts."""
    snap = list(_samples)
    return {
        "p50_ms": _pct(snap, 0.50),
        "p95_ms": _pct(snap, 0.95),
        "p99_ms": _pct(snap, 0.99),
        "samples": len(snap),
        "total_recorded": _count,
    }
