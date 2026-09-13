"""Budget-Enforced Keys (T3.3, local half) — metering becomes CONTROL.

Three capabilities, all inside the local trust boundary:

  1. KEY IDENTITY, never key VALUE. `register_key(name, value)` stores only
     a SHA-256 fingerprint (+ a 4-char hint). The value is used once, in
     memory, to fingerprint — it is never written, logged, or emitted, and
     `key_name` is deliberately NOT in the TokeNet emitter whitelist: key
     facts stay on the machine that owns the key.
  2. BUDGET ENFORCEMENT in the call path. Bind a key with
     `with tokeymeter.key("prod-openai"):` — every metered MISS accrues to
     it. A soft threshold emits one degraded warning per band; the hard cap
     RAISES KeyBudgetExceeded *before* the model is called. A cap that only
     reports is a dashboard; this one stops spend.
  3. LEAK SCAN. `leak_scan(paths, include_env=True)` reuses the secret
     firewall's detectors over files/env and — the part generic scanners
     can't do — flags when a finding's fingerprint matches a REGISTERED key:
     not just "a secret is here" but "YOUR prod-openai key is in .env.bak".

Enforcement scope (contract, pinned by tests/test_async_parity.py):
caps are enforced at every METERED compute site — sync/async miss,
sync/async high-stakes, and both streaming L3 sites (before the first
chunk). `enabled=False` is the user's explicit stand-down switch and
bypasses everything including budgets; `with_memory` composes around a
metered function and neither records nor enforces itself. Enforcement
without accrual would be incoherent — the cap acts exactly where the
meter sees.

Counters are per-process and in-memory (documented, deliberate: enforcement
must be sub-microsecond and dependency-free). Org-wide rollups belong to the
control plane (N9.3); this module is the local reflex.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from tokeymeter.engines.reliability.degraded import emit_degraded

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,63}$")
_CURRENT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tokeymeter_key", default=None)
_lock = threading.Lock()


class KeyBudgetExceeded(RuntimeError):
    """Raised BEFORE the model call when a key's hard cap is exhausted."""
    def __init__(self, name: str, spent: float, cap: float):
        self.key_name, self.spent_usd, self.cap_usd = name, spent, cap
        super().__init__(
            f"key '{name}' hard budget exhausted: spent ${spent:.4f} of "
            f"${cap:g} cap — call refused before send")


@dataclass
class _KeyInfo:
    name: str
    fingerprint: Optional[str]          # sha256 hex of the value, or None
    hint: str = ""                      # last 4 chars, display only
    monthly_cap_usd: Optional[float] = None
    soft_pct: float = 0.8               # warn threshold as fraction of cap
    spent_usd: float = 0.0
    calls: int = 0
    month: str = field(default_factory=lambda: time.strftime("%Y-%m",
                                                             time.gmtime()))
    _warned_bands: set = field(default_factory=set)


_KEYS: dict[str, _KeyInfo] = {}


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def register_key(name: str, value: Optional[str] = None, *,
                 monthly_cap_usd: Optional[float] = None,
                 soft_pct: float = 0.8) -> dict:
    """Register a key IDENTITY. `value` (optional) is fingerprinted in
    memory and discarded — never stored, logged, or emitted. Returns the
    public view (fingerprint + hint only)."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"key name must match {_NAME_RE.pattern}; got {name!r}")
    if monthly_cap_usd is not None:
        cap = float(monthly_cap_usd)
        if not (cap > 0):
            raise ValueError("monthly_cap_usd must be > 0")
    else:
        cap = None
    if not (0.0 < float(soft_pct) <= 1.0):
        raise ValueError("soft_pct must be in (0, 1]")
    fp = _fingerprint(value) if value else None
    hint = value[-4:] if value and len(value) >= 4 else ""
    with _lock:
        prior = _KEYS.get(name)
        info = _KeyInfo(name=name, fingerprint=fp or (prior.fingerprint
                                                      if prior else None),
                        hint=hint or (prior.hint if prior else ""),
                        monthly_cap_usd=cap, soft_pct=float(soft_pct))
        if prior is not None:            # preserve month-to-date accrual
            info.spent_usd, info.calls = prior.spent_usd, prior.calls
            info.month = prior.month
            info._warned_bands = prior._warned_bands
        _KEYS[name] = info
    return key_status(name)


def unregister_key(name: str) -> bool:
    with _lock:
        return _KEYS.pop(name, None) is not None


def clear_keys() -> None:
    with _lock:
        _KEYS.clear()


def get_current_key() -> Optional[str]:
    return _CURRENT.get()


@contextlib.contextmanager
def key(name: str) -> Iterator[None]:
    """Bind spend in this scope to a registered key. Unregistered names are
    refused loudly — silent accrual to a typo is how budgets rot."""
    with _lock:
        if name not in _KEYS:
            raise ValueError(f"key {name!r} is not registered — call "
                             "tokeymeter.register_key() first")
    token = _CURRENT.set(name)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _roll_month(info: _KeyInfo, now: Optional[float] = None) -> None:
    m = time.strftime("%Y-%m", time.gmtime(now or time.time()))
    if info.month != m:
        info.month, info.spent_usd, info.calls = m, 0.0, 0
        info._warned_bands = set()


def check_current(estimated_cost: float = 0.0) -> None:
    """Called by the decorator BEFORE compute. Raises KeyBudgetExceeded when
    the bound key's hard cap is exhausted; emits one degraded warning per
    soft band. No key bound → no-op (unmetered keys are legal).

    ENFORCEMENT CONTRACT (pinned by tests/test_keys_hardening.py):
    the cap is a spend CIRCUIT-BREAKER, not a reservation ledger. The check
    never blocks or serializes the call path (fail-open doctrine), and call
    cost is unknowable pre-call, so calls already in flight when the cap is
    crossed complete and accrue. Overshoot is therefore bounded by
    (calls concurrently in flight at breach) x (one call's cost); for
    sequential callers that bound is exactly one call. From the first check
    AFTER breach, every call is refused before send — the breach latch.

    Month boundary: UTC calendar month (time.gmtime). On rollover, spend,
    call count, and warned bands reset and the cap re-arms.
    """
    name = _CURRENT.get()
    if name is None:
        return
    warn_msg = None
    with _lock:
        info = _KEYS.get(name)
        if info is None or info.monthly_cap_usd is None:
            return
        _roll_month(info)
        cap = info.monthly_cap_usd
        projected = info.spent_usd + max(0.0, float(estimated_cost or 0.0))
        if info.spent_usd >= cap or projected > cap:
            raise KeyBudgetExceeded(name, info.spent_usd, cap)
        frac = info.spent_usd / cap
        if frac >= info.soft_pct and "soft" not in info._warned_bands:
            info._warned_bands.add("soft")       # decide once, under the lock
            warn_msg = (f"key '{name}' at {frac*100:.0f}% of "
                        f"${cap:g} monthly cap")
    if warn_msg is not None:
        # Emit OUTSIDE the lock: subscribers may call back into keys APIs
        # (e.g. key_status); emitting while holding the non-reentrant module
        # lock would deadlock them. Exactly-once is guaranteed by the
        # band-set update above, which happened atomically under the lock.
        emit_degraded("keys", RuntimeError(warn_msg))


def on_spend(name: Optional[str], cost_usd: float, hit: bool,
             shadow: bool) -> None:
    """Accrual hook (decorator record path). Only real spend counts: misses,
    non-shadow. Fail-safe: never raises."""
    try:
        if name is None or hit or shadow:
            return
        with _lock:
            info = _KEYS.get(name)
            if info is None:
                return
            _roll_month(info)
            info.spent_usd += max(0.0, float(cost_usd or 0.0))
            info.calls += 1
    except Exception:
        pass


def key_status(name: Optional[str] = None) -> dict:
    """Public view — fingerprints and hints only, never values."""
    with _lock:
        items = ([_KEYS[name]] if name else list(_KEYS.values()))
        out = {}
        for i in items:
            _roll_month(i)
            out[i.name] = {
                "fingerprint": i.fingerprint, "hint": i.hint,
                "monthly_cap_usd": i.monthly_cap_usd,
                "spent_usd": round(i.spent_usd, 6), "calls": i.calls,
                "month": i.month,
                "remaining_usd": (round(i.monthly_cap_usd - i.spent_usd, 6)
                                  if i.monthly_cap_usd else None),
            }
    return out[name] if name else out


# ── leak scan ──────────────────────────────────────────────────────────────
def leak_scan(paths: Optional[List[str]] = None, *,
              include_env: bool = True, max_bytes: int = 512 * 1024) -> dict:
    """Scan files/env for secrets via the firewall's detectors, and flag any
    finding whose fingerprint matches a REGISTERED key. Findings carry
    location + kind + severity + (when matched) the key NAME — never the
    secret itself."""
    from tokeymeter.engines.governance.content.secrets import SecretScanner
    scanner = SecretScanner()
    with _lock:
        fps = {i.fingerprint: i.name for i in _KEYS.values() if i.fingerprint}
    findings, registered_leaks, scanned = [], [], 0

    def _scan_blob(source: str, text: str) -> None:
        nonlocal scanned
        scanned += 1
        res = scanner.scan(text)
        for f in getattr(res, "findings", []) or []:
            span = getattr(f, "span", None)
            snippet = (text[span[0]:span[1]]
                       if isinstance(span, (tuple, list)) and len(span) == 2
                       else "")
            entry = {"source": source,
                     "kind": str(getattr(f, "type",
                                         getattr(f, "kind", "?"))),
                     "severity": str(getattr(f, "severity", "?"))}
            if snippet and _fingerprint(snippet) in fps:
                entry["registered_key"] = fps[_fingerprint(snippet)]
                registered_leaks.append(entry)
            findings.append(entry)

    if include_env:
        for k, v in os.environ.items():
            if v and len(v) >= 12:
                _scan_blob(f"env:{k}", v)
    for p in paths or []:
        try:
            if os.path.isfile(p) and os.path.getsize(p) <= max_bytes:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    _scan_blob(p, fh.read())
        except OSError:
            continue
    return {"scanned": scanned, "findings": findings,
            "registered_key_leaks": registered_leaks,
            "ok": not registered_leaks}
