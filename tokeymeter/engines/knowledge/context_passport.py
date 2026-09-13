"""Context Passports (T3.4, engine emission half) — provenance for context.

A passport is METADATA ABOUT knowledge, never knowledge: a fingerprint,
where it came from, how big it was before/after optimization, and how it may
be used. This module is the engine-side emission bus; the queryable registry
and sensitivity-flow policies are the plane half (N9.2).

Content-blind by construction: `context_id` is a SHA-256 fingerprint; no
field may carry text (validated). Emission is fail-open — a passport bus
problem can never touch the call path.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_MAX_LABEL = 80


@dataclass(frozen=True)
class ContextPassport:
    context_id: str                 # sha256 hex fingerprint of the context
    source_type: str                # "prompt" | "memory" | "rag" | "custom"
    tokens_before: int
    tokens_after: int
    method: Optional[str] = None    # optimization method, if any
    sensitivity: Optional[str] = None   # label only (e.g. "confidential")
    model: Optional[str] = None
    ts: float = field(default_factory=time.time)


_subs: List[Callable[[ContextPassport], None]] = []
_lock = threading.Lock()
_recent: List[ContextPassport] = []
_RECENT_MAX = 256


def fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def subscribe(cb: Callable[[ContextPassport], None]) -> Callable:
    with _lock:
        _subs.append(cb)
    return cb


def unsubscribe(cb) -> None:
    with _lock:
        if cb in _subs:
            _subs.remove(cb)


def recent(n: int = 50) -> List[ContextPassport]:
    with _lock:
        return list(_recent[-n:])


def emit(*, context_id: str, source_type: str, tokens_before: int,
         tokens_after: int, method: Optional[str] = None,
         sensitivity: Optional[str] = None,
         model: Optional[str] = None) -> Optional[ContextPassport]:
    """Emit a passport. Validates content-blindness; never raises."""
    try:
        for name, v in (("source_type", source_type), ("method", method),
                        ("sensitivity", sensitivity), ("model", model)):
            if v is not None and (not isinstance(v, str)
                                  or len(v) > _MAX_LABEL
                                  or "\n" in v):
                raise ValueError(f"{name} must be a short single-line label")
        if not (isinstance(context_id, str) and len(context_id) == 64
                and all(c in "0123456789abcdef" for c in context_id)):
            raise ValueError("context_id must be a sha256 hex fingerprint — "
                             "use context_passport.fingerprint(text)")
        p = ContextPassport(context_id=context_id, source_type=source_type,
                            tokens_before=int(tokens_before),
                            tokens_after=int(tokens_after), method=method,
                            sensitivity=sensitivity, model=model)
        with _lock:
            _recent.append(p)
            del _recent[:-_RECENT_MAX]
            subs = list(_subs)
        for cb in subs:
            try:
                cb(p)
            except Exception:
                pass
        return p
    except Exception:
        return None
