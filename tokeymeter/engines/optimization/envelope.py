"""
Cache envelope format.

Every cached value is wrapped in an envelope on write and unwrapped on
read. The envelope carries the value plus optional metadata (expiry
timestamp; and, since the self-host capacity work, the token counts of the
ORIGINAL computation so a later hit can report the true avoided volume).

Format (JSON-serializable):
    v1 (legacy):  {"__tokeymeter_v1__": [value, expires_at_or_null]}
    v1 extended:  {"__tokeymeter_v1__": [value, expires_at_or_null, meta_or_null]}

`meta` is an optional dict, currently {"in": <int>, "out": <int>} — the input
and output token counts of the miss that populated this entry. It is appended
as a THIRD element so every pre-existing 2-element envelope (including those
already persisted to disk-backed stores) keeps deserializing unchanged. The
magic key prevents collision with real LLM responses that happen to be dicts.

Legacy / forward compatibility:
    If a stored value isn't an envelope, it's treated as a raw value with no
    TTL and no meta. Two-element envelopes read back with meta=None. This means
    caches written by any prior version keep working — they simply have no
    recovered-token metadata.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

_ENVELOPE_KEY = "__tokeymeter_v1__"


def wrap(value: Any, ttl: Optional[float] = None,
         meta: Optional[Dict[str, Any]] = None) -> dict:
    """Wrap a value with an optional TTL (seconds) and optional metadata.

    ttl=None or ttl<=0 → no expiry (never invalidates).
    meta=None → a 2-element envelope (byte-identical to the legacy format, so
    nothing about existing behavior changes when no meta is supplied). A
    supplied meta is appended as a third element.
    """
    expires_at: Optional[float] = None
    if ttl is not None and ttl > 0:
        expires_at = time.time() + ttl
    if meta is None:
        return {_ENVELOPE_KEY: [value, expires_at]}
    return {_ENVELOPE_KEY: [value, expires_at, meta]}


def _unpack(stored: Any) -> Optional[Tuple[Any, Optional[float], Optional[dict]]]:
    """Return (value, expires_at, meta) for an envelope, or None if it isn't
    one / is malformed. Tolerant of both 2- and 3-element forms."""
    if isinstance(stored, dict) and _ENVELOPE_KEY in stored:
        payload = stored[_ENVELOPE_KEY]
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return None
        value = payload[0]
        expires_at = payload[1]
        meta = payload[2] if len(payload) >= 3 else None
        return value, expires_at, meta
    return None


def unwrap(stored: Any) -> Optional[Any]:
    """Return the value inside an envelope, or None if expired/missing.

    Returns None for:
      - None input (cache miss)
      - Envelope past expiry
      - Malformed envelope: the magic key is present but the payload is the
        wrong shape (defensive — a corrupt entry must not be served as a value)

    For non-envelope values (legacy raw entries), returns as-is.
    """
    if stored is None:
        return None
    # A dict carrying the magic key IS an envelope: if its payload is malformed,
    # return None (never serve a corrupt entry as a value). Only values without
    # the magic key are legacy raw entries returned as-is.
    if isinstance(stored, dict) and _ENVELOPE_KEY in stored:
        unpacked = _unpack(stored)
        if unpacked is None:
            return None  # malformed envelope
        value, expires_at, _meta = unpacked
        if expires_at is not None and time.time() > expires_at:
            return None
        return value
    return stored


def meta(stored: Any) -> Optional[dict]:
    """Return the metadata dict carried by an envelope, or None.

    None when: not an envelope, a legacy 2-element envelope, expired, or
    malformed. Callers use this to recover the original computation's token
    counts on a cache hit; None means "unknown", never a fabricated value.
    """
    if stored is None:
        return None
    unpacked = _unpack(stored)
    if unpacked is None:
        return None
    _value, expires_at, m = unpacked
    if expires_at is not None and time.time() > expires_at:
        return None
    return m if isinstance(m, dict) else None


def is_expired(stored: Any) -> bool:
    """True iff `stored` is an envelope past its expiry."""
    unpacked = _unpack(stored)
    if unpacked is None:
        return False
    _value, expires_at, _meta = unpacked
    return expires_at is not None and time.time() > expires_at
