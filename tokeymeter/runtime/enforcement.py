"""The enforcement spine (W4): GOV-1 RBAC/ABAC · GOV-2 rate limits ·
GOV-4 checkpoints · GOV-5 secret/PII pre-execution stage · GOV-6 verdicts.

Laws applied here:
- ENFORCEMENT FAILS CLOSED: every block is a raise on the strict path;
  nothing here goes through safe-failover.
- L4 CONTENT-BLIND: typed errors and verdicts carry detector KINDS, counts,
  policies, and principals — never matched text, never payload.
- FINGERPRINT AFTER REDACTION: the Security stage rewrites the request
  payload (redacted) and REFRESHES ctx["payload_fingerprint"], so no
  fingerprint of secret-bearing text ever enters trust residue (pinned).
- GOV-6: every engine appends to ctx["meta"]["policy_verdicts"] — allow or
  block — and the Trust engine seals the list, INCLUDING on the error path.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from .engine import Engine

try:
    from tokeymeter.engines.governance.identity import get_principal
except Exception:  # pragma: no cover
    def get_principal() -> Optional[str]:
        return None


# ------------------------------------------------------------ verdicts ----
def record_verdict(ctx: Dict[str, Any], policy: str, verdict: str,
                   detail: str = "") -> None:
    """GOV-6: append a content-blind verdict. `detail` must be a kind/label,
    never text from the payload."""
    ctx["meta"].setdefault("policy_verdicts", []).append(
        {"policy": policy, "verdict": verdict, "detail": detail})


# --------------------------------------------------------- typed errors ---
class EnforcementError(PermissionError):
    """Base for every enforcement block. NOT retryable by design."""
    retryable = False
    policy = "enforcement"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(f"{type(self).__name__}({detail})")


class SecretBlocked(EnforcementError):
    policy = "secrets"


class ContentPolicyViolation(EnforcementError):
    policy = "content"


class AccessDenied(EnforcementError):
    policy = "rbac"


class RateLimitExceeded(EnforcementError):
    policy = "rate_limit"

    def __init__(self, detail: str = "", retry_after: float = 0.0) -> None:
        super().__init__(detail)
        self.retry_after = retry_after


class CheckpointDenied(EnforcementError):
    policy = "checkpoint"


class CheckpointPending(EnforcementError):
    """Async checkpoint mode: the request is parked, not denied."""
    policy = "checkpoint"


# ------------------------------------------- GOV-5: Security engine -------
class SecurityEngine(Engine):
    """Secret firewall + PII redaction + content terms, PRE-execution,
    strict. Order matches the shipped call path: secrets first (block),
    then PII (redact), then content terms (block) — nothing reaches the
    adapter, the cache key, or the fingerprint unredacted."""

    name = "security"

    def __init__(self, *, secrets_mode: str = "block",
                 pii: bool = True,
                 blocked_terms: Optional[List[str]] = None) -> None:
        if secrets_mode not in ("block", "off"):
            raise ValueError("secrets_mode must be 'block' or 'off'")
        self._secrets_mode = secrets_mode
        self._pii = pii
        self._terms = [t.lower() for t in (blocked_terms or [])]
        self._scanner: Optional[Any] = None  # lazy: firewall built on first use

    def _scan(self, text: str) -> Any:
        if self._scanner is None:
            from tokeymeter.engines.governance.content.secrets import (
                SecretScanner)
            self._scanner = SecretScanner()
        return self._scanner.scan(text)

    def before_request(self, ctx: Dict[str, Any]) -> None:
        request = ctx["request"]
        payload = request.payload

        if self._secrets_mode == "block":
            result = self._scan(payload)
            if getattr(result, "has_secrets", False):
                kinds = sorted({f.type for f in result.findings})
                record_verdict(ctx, "secrets", "block",
                               detail=",".join(kinds))
                raise SecretBlocked(",".join(kinds))
            record_verdict(ctx, "secrets", "allow")

        if self._pii:
            from tokeymeter.engines.governance.privacy import redact
            redacted = redact(payload)
            if redacted != payload:
                request.payload = redacted
                self._refresh_messages(ctx, redact)
                record_verdict(ctx, "pii", "redact")
            else:
                record_verdict(ctx, "pii", "allow")
            payload = request.payload

        if self._terms:
            lowered = payload.lower()
            hit = next((t for t in self._terms if t in lowered), None)
            if hit is not None:
                record_verdict(ctx, "content", "block", detail=f"term:{hit}")
                raise ContentPolicyViolation(f"term:{hit}")
            record_verdict(ctx, "content", "allow")

        # FINGERPRINT AFTER REDACTION (L4 pin): trust residue must hash the
        # payload as it may lawfully leave the process — never the original.
        ctx["payload_fingerprint"] = hashlib.sha256(
            request.payload.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _refresh_messages(ctx: Dict[str, Any],
                          redactor: Callable[[str], str]) -> None:
        msgs = ctx["meta"].get("messages") or \
            ctx["request"].metadata.get("messages")
        if not msgs:
            return
        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = redactor(m["content"])


# ------------------------------------------- GOV-1: Access engine ---------
class AccessEngine(Engine):
    """RBAC/ABAC: role → {models, actions}; principal → role. DENY BY
    DEFAULT: with the engine active, an unknown principal or an unlisted
    model is a block, never a shrug."""

    name = "access"

    def __init__(self, *, roles: Dict[str, Dict[str, List[str]]],
                 principals: Dict[str, str]) -> None:
        self._roles = {r: {"models": set(v.get("models", [])),
                           "actions": set(v.get("actions", ["execute"]))}
                       for r, v in roles.items()}
        self._principals = dict(principals)

    def _allowed(self, principal: Optional[str], model: str,
                 action: str) -> bool:
        role = self._principals.get(principal or "")
        spec = self._roles.get(role or "")
        if spec is None:
            return False
        models_ok = "*" in spec["models"] or model in spec["models"]
        actions_ok = "*" in spec["actions"] or action in spec["actions"]
        return models_ok and actions_ok

    def before_request(self, ctx: Dict[str, Any]) -> None:
        principal = ctx["meta"].get("principal") or get_principal()
        model = ctx["meta"].get("model", ctx["request"].model)
        if not self._allowed(principal, model, "execute"):
            record_verdict(ctx, "rbac", "block",
                           detail=f"principal:{principal or 'anonymous'}"
                                  f" model:{model}")
            raise AccessDenied(f"principal={principal or 'anonymous'} "
                               f"model={model}")
        record_verdict(ctx, "rbac", "allow", detail=f"model:{model}")


# ------------------------------------------ GOV-2: RateLimit engine -------
class RateLimitEngine(Engine):
    """Sliding-window requests/min and tokens/min per principal. Thread-safe
    deques; token counts recorded post-call (actuals) and pre-checked."""

    name = "rate_limit"

    def __init__(self, *, requests_per_min: Optional[int] = None,
                 tokens_per_min: Optional[int] = None,
                 window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._rpm = requests_per_min
        self._tpm = tokens_per_min
        self._window = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._req: Dict[str, deque] = {}
        self._tok: Dict[str, deque] = {}

    def _prune(self, dq: deque, now: float) -> None:
        while dq and dq[0][0] <= now - self._window:
            dq.popleft()

    def before_request(self, ctx: Dict[str, Any]) -> None:
        principal = ctx["meta"].get("principal") or get_principal() or "_anon"
        now = self._clock()
        with self._lock:
            rq = self._req.setdefault(principal, deque())
            tq = self._tok.setdefault(principal, deque())
            self._prune(rq, now)
            self._prune(tq, now)
            if self._rpm is not None and len(rq) >= self._rpm:
                retry = round(self._window - (now - rq[0][0]), 3)
                record_verdict(ctx, "rate_limit", "block",
                               detail=f"rpm:{self._rpm}")
                raise RateLimitExceeded(f"rpm:{self._rpm}",
                                        retry_after=retry)
            if self._tpm is not None and \
                    sum(n for _, n in tq) >= self._tpm:
                retry = round(self._window - (now - tq[0][0]), 3)
                record_verdict(ctx, "rate_limit", "block",
                               detail=f"tpm:{self._tpm}")
                raise RateLimitExceeded(f"tpm:{self._tpm}",
                                        retry_after=retry)
            rq.append((now, 1))
        record_verdict(ctx, "rate_limit", "allow")

    def after_response(self, ctx: Dict[str, Any]) -> None:
        tokens = (ctx["meta"].get("tokens_in") or 0) + \
                 (ctx["meta"].get("tokens_out") or 0)
        if not tokens or self._tpm is None:
            return
        principal = ctx["meta"].get("principal") or get_principal() or "_anon"
        with self._lock:
            self._tok.setdefault(principal, deque()).append(
                (self._clock(), tokens))


# ------------------------------------------ GOV-4: CheckpointHook ---------
class CheckpointHook:
    """Human-in-loop contract (engine side; plane wiring parked D3).
    decide() returns 'approve' | 'deny' | 'pending'. Blocking approvers wait
    inside decide(); async approvers return 'pending' and the request PARKS
    (CheckpointPending) — never silently proceeds."""

    def decide(self, ctx: Dict[str, Any], reason: str) -> str:
        raise NotImplementedError


class LocalApprover(CheckpointHook):
    """Reference implementation for tests and single-operator setups:
    a callable renders the verdict; content-blind summary only."""

    def __init__(self, fn: Callable[[Dict[str, str]], str]) -> None:
        self._fn = fn

    def decide(self, ctx: Dict[str, Any], reason: str) -> str:
        summary = {  # content-blind by construction
            "reason": reason,
            "principal": str(ctx["meta"].get("principal") or "anonymous"),
            "model": str(ctx["meta"].get("model",
                                         ctx["request"].model)),
            "request_id": ctx["request"].request_id,
        }
        verdict = self._fn(summary)
        if verdict not in ("approve", "deny", "pending"):
            raise ValueError(f"approver returned {verdict!r}")
        return verdict


def run_checkpoint(ctx: Dict[str, Any], hook: CheckpointHook,
                   reason: str) -> None:
    verdict = hook.decide(ctx, reason)
    record_verdict(ctx, "checkpoint", verdict, detail=reason)
    if verdict == "deny":
        raise CheckpointDenied(reason)
    if verdict == "pending":
        raise CheckpointPending(reason)
