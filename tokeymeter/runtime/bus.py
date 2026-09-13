"""Kernel hook bus — the plugin attachment surface.

Hooks are string-named lifecycle points ("before_request", "after_response",
"on_error", plus any custom point an engine exposes). Multiple subscribers per
hook run in explicit priority order (lower runs first; ties keep subscription
order — deterministic, per the doc's plugin-governance rule).

SAFE-FAILOVER: a subscriber that raises is reported on Tokeymeter's existing
degraded bus (reason="runtime_hook_error") and skipped. A bad plugin can never
crash the core.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Tuple

try:  # degraded bus is the shipped failure-reporting channel
    from tokeymeter.degraded import emit_degraded as _emit_degraded  # type: ignore
except Exception:  # pragma: no cover - degraded module always present in-tree
    def _emit_degraded(**kwargs: Any) -> None:
        pass

Hook = Callable[[Dict[str, Any]], None]


class HookBus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._seq = 0
        # hook -> list of (priority, seq, fn)
        self._subs: Dict[str, List[Tuple[int, int, Hook]]] = {}
        self._error_count = 0

    def subscribe(self, hook: str, fn: Hook, *, priority: int = 100) -> Hook:
        with self._lock:
            self._seq += 1
            self._subs.setdefault(hook, []).append((priority, self._seq, fn))
            self._subs[hook].sort(key=lambda t: (t[0], t[1]))
        return fn

    def unsubscribe(self, hook: str, fn: Hook) -> None:
        with self._lock:
            subs = self._subs.get(hook, [])
            self._subs[hook] = [t for t in subs if t[2] is not fn]

    def subscriber_count(self, hook: str) -> int:
        with self._lock:
            return len(self._subs.get(hook, []))

    @property
    def error_count(self) -> int:
        return self._error_count

    def emit(self, hook: str, ctx: Dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs.get(hook, []))
        for _prio, _seq, fn in subs:
            try:
                fn(ctx)
            except Exception as exc:  # safe-failover: report, never crash
                self._error_count += 1
                try:
                    _emit_degraded(
                        reason="runtime_hook_error",
                        detail=f"hook={hook} subscriber={getattr(fn, '__name__', repr(fn))} "
                               f"error={type(exc).__name__}",
                    )
                except Exception:
                    pass

    def emit_strict(self, hook: str, ctx: Dict[str, Any]) -> None:
        """ENFORCEMENT semantics: exceptions PROPAGATE (fail closed).

        Observability plugins go through emit() and can never crash the
        core; policy/enforcement hooks go through emit_strict() and a veto
        (raise) stops the request. Running enforcement through safe-failover
        would silently fail OPEN — a real defect this split exists to
        prevent.
        """
        with self._lock:
            subs = list(self._subs.get(hook, []))
        for _prio, _seq, fn in subs:
            fn(ctx)
