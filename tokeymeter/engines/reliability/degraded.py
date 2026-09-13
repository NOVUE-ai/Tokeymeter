"""
tokeymeter.degraded — fail-open observability bus.

Tokeymeter fails open by design — when a redactor, cache backend, or cipher
errors, the wrapped call still succeeds with degraded behavior. By
default this is silent (the call works, the user sees nothing wrong).

This module adds an *optional* observability layer: subscribers can
register to be notified when any fail-open path fires, with enough
context to investigate without breaking the fail-open contract.

Design contract:
  - Emitting a degraded event MUST NOT raise. If a subscriber errors,
    its error is swallowed. The fail-open behavior is sacred.
  - Subscribers run synchronously in the same thread that emitted.
    They should be cheap (a queue push, a log line). Slow subscribers
    will slow the host call — they have been warned in the docstring.
  - No information about the wrapped call's prompt or response leaks
    through this bus. Only the failure source, the exception type,
    and an exception message capped at 200 chars.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional


@dataclass(frozen=True)
class DegradedEvent:
    """A single fail-open event. Frozen so subscribers can't mutate it."""
    timestamp: float
    source: str            # "redactor" | "cipher" | "store" | other
    error_type: str        # exception class name
    error_message: str     # capped at 200 chars
    function_name: Optional[str] = None
    tag: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


_subscribers: List[Callable[[DegradedEvent], None]] = []
_lock = threading.Lock()
_counter = 0  # total degraded events since process start
_counts_by_source: "dict[str, int]" = {}  # per-source breakdown (bounded: fixed enum)
_counter_lock = threading.Lock()


def on_degraded(callback: Callable[[DegradedEvent], None]) -> Callable:
    """Register a subscriber. Returns the callback for decorator-style use."""
    with _lock:
        _subscribers.append(callback)
    return callback


def clear_subscribers() -> None:
    """For tests only — remove all subscribers and reset counters."""
    global _counter
    with _lock:
        _subscribers.clear()
    with _counter_lock:
        _counter = 0
        _counts_by_source.clear()


def off_degraded(callback: Callable[[DegradedEvent], None]) -> None:
    """Remove a previously-registered subscriber (symmetry with on_degraded, so
    callers that subscribe dynamically don't grow the list without bound)."""
    with _lock:
        try:
            _subscribers.remove(callback)
        except ValueError:
            pass


def degraded_event_count() -> int:
    """Total degraded events since process start. For panel display."""
    with _counter_lock:
        return _counter


def degraded_counts() -> "dict[str, int]":
    """Per-source degraded-event counts since process start, e.g.
    {"sqlite_write": 3, "redis_unhealthy": 1, "compression_fallback": 27}.
    Lets operators see not just THAT something degraded but exactly HOW and HOW
    OFTEN — by failure type. The source set is a fixed enumeration, so this map
    is bounded regardless of traffic volume."""
    with _counter_lock:
        return dict(_counts_by_source)


def emit_degraded(
    source: str,
    error: BaseException,
    function_name: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """Emit a degraded event. NEVER raises — fail-open is sacred."""
    global _counter
    try:
        msg = str(error)
        if len(msg) > 200:
            msg = msg[:197] + "..."
        event = DegradedEvent(
            timestamp=time.time(),
            source=source,
            error_type=type(error).__name__,
            error_message=msg,
            function_name=function_name,
            tag=tag,
        )
        with _counter_lock:
            _counter += 1
            _counts_by_source[source] = _counts_by_source.get(source, 0) + 1
        with _lock:
            subs = list(_subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass  # subscriber bug must not break fail-open
    except Exception:
        pass  # we tried; fail-open is sacred