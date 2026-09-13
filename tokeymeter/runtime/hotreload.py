"""Config hot-reload (W8) — swap policy/config without restarting the runtime.

An enterprise cannot restart every service to change a rate limit or a budget.
The ReloadableConfig holds a versioned snapshot behind an atomic pointer:
readers always see a consistent snapshot (never a half-applied change), a
reload swaps the pointer in one operation, and subscribers are notified after
the swap so engines can refresh derived state.

Safety laws:
- ATOMIC: a reader in flight either sees the old snapshot or the new one,
  never a mix. The swap is a single reference assignment under a lock.
- VERSIONED: every snapshot carries a monotonically increasing version, so a
  subscriber can detect and order changes.
- VALIDATED: a reload runs an optional validator; a rejected config is NOT
  applied — the runtime keeps the last-good snapshot (fail-safe, not
  fail-open to a broken config).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import RuntimeConfig


@dataclass
class ConfigSnapshot:
    version: int
    config: RuntimeConfig


class ConfigValidationError(Exception):
    pass


class ReloadableConfig:
    """Atomic, versioned, validated config with reload notification."""

    def __init__(self, initial: Dict[str, Any],
                 *, validator: Optional[Callable[[RuntimeConfig], None]] = None
                 ) -> None:
        self._lock = threading.RLock()
        self._validator = validator
        self._version = 1
        self._snapshot = ConfigSnapshot(1, RuntimeConfig(initial))
        self._subscribers: List[Callable[[ConfigSnapshot], None]] = []

    def current(self) -> ConfigSnapshot:
        # single atomic read of the reference — no lock needed for readers,
        # because reference assignment is atomic in CPython and we never
        # mutate a published snapshot in place.
        return self._snapshot

    def get(self, dotted: str, default: Any = None) -> Any:
        return self._snapshot.config.get(dotted, default)

    def subscribe(self, fn: Callable[[ConfigSnapshot], None]) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def reload(self, new_data: Dict[str, Any]) -> ConfigSnapshot:
        """Validate, swap atomically, notify. On validation failure the old
        snapshot is retained and the error is raised — never applied."""
        candidate = RuntimeConfig(new_data)
        if self._validator is not None:
            try:
                self._validator(candidate)
            except Exception as exc:
                raise ConfigValidationError(
                    f"rejected config: {exc}") from exc
        with self._lock:
            self._version += 1
            snap = ConfigSnapshot(self._version, candidate)
            self._snapshot = snap                    # ATOMIC swap
            subscribers = list(self._subscribers)
        # notify OUTSIDE the lock so a slow subscriber cannot block readers;
        # a subscriber that raises is isolated (reload already succeeded).
        for fn in subscribers:
            try:
                fn(snap)
            except Exception:
                pass
        return snap

    @property
    def version(self) -> int:
        return self._snapshot.version
