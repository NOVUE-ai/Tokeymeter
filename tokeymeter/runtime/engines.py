"""Pipeline engines — thin, delegating, contract-first.

These are COMPATIBILITY-MODE stages: each delegates to shipped Tokeymeter
capability or pins the doc's interface with a minimal default. They do not
duplicate the shipped cache/audit/policy machinery — they give it a place in
the kernel pipeline.

Pipeline placement mirrors the shipped call path: Governance (identity +
policy hooks) runs BEFORE optimization and execution, exactly where
secrets.py/privacy.py sit today.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .engine import Engine
from .providers import ExecutionEngine

try:
    from tokeymeter.identity import get_principal as _get_principal  # type: ignore
except Exception:  # pragma: no cover
    def _get_principal() -> Optional[str]:
        return None


# --------------------------------------------------------------------------
# Governance — WHO + policy hook slot (delegates to shipped identity binding)
# --------------------------------------------------------------------------
class GovernanceEngine(Engine):
    name = "governance"

    def before_request(self, ctx: Dict[str, Any]) -> None:
        principal = _get_principal()
        if principal is not None:
            ctx["meta"]["principal"] = principal
        # Policy attachment surface: subscribers may inspect content-blind
        # metadata and veto by raising, or annotate ctx["meta"].
        # STRICT emit: a veto must stop the request (fail closed) — never
        # the safe-failover path, which would swallow the veto (fail open).
        ctx["kernel"].hooks.emit_strict("governance_check", ctx)


# --------------------------------------------------------------------------
# Cache — the doc's CachePolicy contract + in-memory default (Sprint 5)
# --------------------------------------------------------------------------
class CachePolicy:
    """interface CachePolicy: get(key) -> response?; set(key, response)."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, response: Any) -> None:
        raise NotImplementedError


class InMemoryCache(CachePolicy):
    """Bounded LRU, thread-safe, exact-key. The shipped semantic/Redis caches
    plug in behind the same contract."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._lock = threading.RLock()
        self._max = max(1, int(max_entries))
        self._data: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, response: Any) -> None:
        with self._lock:
            self._data[key] = response
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


def _cache_key(payload: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(payload.encode("utf-8", errors="replace"))
    return h.hexdigest()


class CacheEngine(Engine):
    name = "cache"

    def __init__(self, policy: Optional[CachePolicy] = None) -> None:
        self._policy = policy
        self.hits = 0
        self.misses = 0

    def _resolve_policy(self, ctx: Dict[str, Any]) -> CachePolicy:
        if self._policy is not None:
            return self._policy
        container = ctx["container"]
        if not container.has("cache_policy"):
            container.register_instance(
                "cache_policy",
                InMemoryCache(int(ctx["config"].get("cache.max_entries", 1024))),
            )
        return container.resolve("cache_policy")

    def before_request(self, ctx: Dict[str, Any]) -> None:
        if not ctx["config"].get("cache.enabled", True):
            return
        policy = self._resolve_policy(ctx)
        key = _cache_key(ctx["request"].payload,
                         ctx["meta"].get("model", ctx["request"].model))
        ctx["meta"]["cache_key"] = key
        hit = policy.get(key)
        if hit is not None:
            self.hits += 1
            ctx["response_payload"] = hit
            ctx["meta"]["cache"] = "hit"
            ctx["short_circuit"] = True
        else:
            self.misses += 1
            ctx["meta"]["cache"] = "miss"

    def after_response(self, ctx: Dict[str, Any]) -> None:
        if (
            ctx["config"].get("cache.enabled", True)
            and ctx["meta"].get("cache") == "miss"
            and ctx["response_payload"] is not None
        ):
            self._resolve_policy(ctx).set(
                ctx["meta"]["cache_key"], ctx["response_payload"]
            )


# --------------------------------------------------------------------------
# Trust — tamper-evident, content-blind hash-chain of kernel executions
# --------------------------------------------------------------------------
class TrustEngine(Engine):
    """Chains {request_id, model, principal?, payload fingerprint, cache
    verdict} with prev-hash linking. NEVER records payload text. Full sealed
    audit remains tokeymeter.audit; this is the kernel-path residue."""

    name = "trust"
    _GENESIS = "0" * 64

    def __init__(self, max_entries: int = 5000) -> None:
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = []
        # Bounded in-memory chain: a long-lived runtime must not accumulate
        # sealed entries forever (REL-7 soak caught this). The rolling window
        # keeps recent verifiable history; durable proof lives in the WORM
        # sink / shipped ledger when the proof spine is enabled. verify()
        # checks the retained window; the chain hash still links to genesis
        # via the retained prev-hashes.
        self._max_entries = max(1, int(max_entries))
        self._evicted = 0

    @staticmethod
    def _entry_hash(prev: str, body: str) -> str:
        return hashlib.sha256((prev + "|" + body).encode("utf-8")).hexdigest()

    @staticmethod
    def _verdict_summary(ctx: Dict[str, Any]) -> str:
        """GOV-6: content-blind policy:verdict pairs, order preserved."""
        return ";".join(f"{v['policy']}:{v['verdict']}"
                        for v in ctx["meta"].get("policy_verdicts", []))

    def _seal(self, ctx: Dict[str, Any], outcome: str) -> None:
        body = "|".join([
            ctx["request"].request_id,
            str(ctx["meta"].get("model", ctx["request"].model)),
            str(ctx["meta"].get("principal", "")),
            ctx["payload_fingerprint"],
            str(ctx["meta"].get("cache", "")),
            outcome,
            self._verdict_summary(ctx),
            str(ctx["meta"].get("cost_usd", "")),
        ])
        with self._lock:
            prev = self._entries[-1]["hash"] if self._entries else self._GENESIS
            self._entries.append({
                "prev": prev, "body": body,
                "hash": self._entry_hash(prev, body),
            })
            if len(self._entries) > self._max_entries:
                # evict oldest, keep the window bounded (durable record is
                # the sink's job, not memory's)
                drop = len(self._entries) - self._max_entries
                self._entries = self._entries[drop:]
                self._evicted += drop

    def after_response(self, ctx: Dict[str, Any]) -> None:
        self._seal(ctx, outcome="ok")

    def on_error(self, ctx: Dict[str, Any], exc: BaseException) -> None:
        # A BLOCKED request must still leave proof: the error path seals the
        # verdict ledger too (a governance block with no audit entry is a
        # governance hole — W4 gate).
        self._seal(ctx, outcome=f"error:{type(exc).__name__}")

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._entries]

    def verify(self) -> Tuple[bool, int]:
        """Returns (ok, first_bad_index) — first_bad_index = -1 when ok.
        Verifies the retained window: each entry's hash must equal
        H(prev|body) and chain to the next. When entries have been evicted,
        the window's first prev is taken as its anchor (durable full-chain
        verification is the sink/ledger's responsibility)."""
        with self._lock:
            if not self._entries:
                return True, -1
            prev = self._entries[0]["prev"] if self._evicted else self._GENESIS
            for i, e in enumerate(self._entries):
                if e["prev"] != prev or e["hash"] != self._entry_hash(prev, e["body"]):
                    return False, i
                prev = e["hash"]
        return True, -1


# --------------------------------------------------------------------------
# Reliability — retry + fallback adapter order around execution (Sprint 9 lite)
# --------------------------------------------------------------------------
class ReliabilityEngine(Engine):
    """Wraps the execution engine's adapter call with bounded retries and an
    ordered fallback list. Kernel stays fail-loud; resilience lives here."""

    name = "reliability"

    def __init__(self, execution: ExecutionEngine) -> None:
        self._execution = execution
        self.retries_used = 0
        self.fallbacks_used = 0
        self._orig_execute = execution.execute
        execution.execute = self._execute_with_resilience  # type: ignore[method-assign]

    def _execute_with_resilience(self, ctx: Dict[str, Any]) -> Any:
        cfg = ctx["config"]
        max_retries = int(cfg.get("reliability.max_retries", 1))
        fallback_models: List[str] = list(cfg.get("reliability.fallback_order", []))
        last_exc: Optional[BaseException] = None
        attempts = [ctx["meta"].get("model", ctx["request"].model)] + fallback_models
        for idx, model in enumerate(attempts):
            ctx["meta"]["model"] = model
            for attempt in range(max_retries + 1):
                try:
                    result = self._orig_execute(ctx)
                    if idx > 0:
                        self.fallbacks_used += 1
                        ctx["meta"]["fallback"] = model
                    return result
                except Exception as exc:
                    last_exc = exc
                    # EXEC-4/REL routing: typed non-retryable errors (auth,
                    # malformed) are never retried NOR failed-over — retrying
                    # can't fix them and spams providers. Unknown exceptions
                    # stay retryable (safe default, pinned).
                    if getattr(exc, "retryable", True) is False:
                        raise
                    if attempt < max_retries:
                        self.retries_used += 1
        assert last_exc is not None
        raise last_exc


# --------------------------------------------------------------------------
# Knowledge — STAGE-GATED interface slot (no-op default; content-sovereign
# implementations belong to the locked later stage and are NOT built here)
# --------------------------------------------------------------------------
class KnowledgeEngine(Engine):
    name = "knowledge"

    def __init__(self, retriever: Optional[Any] = None) -> None:
        self._retriever = retriever  # interface reserved; default None = no-op

    def before_request(self, ctx: Dict[str, Any]) -> None:
        if self._retriever is None:
            ctx["meta"]["knowledge"] = "noop"
            return
        # A future, stage-gated retriever contributes context via ctx;
        # deliberately unimplemented at this stage.
        ctx["meta"]["knowledge"] = "active"
