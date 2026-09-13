"""Survivability primitives (W6): the request survives the provider.

Composable, thread-safe units the ResilientExecution wrapper orchestrates:
- CircuitBreaker   per-provider closed/open/half-open with a retry-storm ceiling
- Bulkhead         per-provider concurrency isolation (a slow provider can't
                   starve the others)
- BackoffPolicy    exponential + decorrelated jitter, honoring Retry-After
- OutputValidator  REL-6: declared-schema validation → typed Malformed
- ChaosInjector    REL-4: deterministic fault injection for tests + `--chaos`

Every degrade path emits on the SHIPPED degraded bus with a stable reason;
overhead is recorded to the SHIPPED overhead module. Nothing is reinvented.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from tokeymeter.engines.reliability import degraded as _degraded
from tokeymeter.engines.reliability import overhead as _overhead

from .errors import MalformedRequest, ProviderError


# =====================================================================
# REL-1  Circuit breaker
# =====================================================================
class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(ProviderError):
    """Raised when the breaker is open — retryable via a DIFFERENT provider
    (fallback), never by hammering the same one."""
    retryable = True

    def __init__(self, provider: str) -> None:
        super().__init__(provider, RuntimeError("circuit open"), status=None)


@dataclass
class _BreakerCell:
    state: BreakerState = BreakerState.CLOSED
    failures: int = 0
    successes: int = 0
    opened_at: float = 0.0
    half_open_inflight: bool = False


class CircuitBreaker:
    """Per-provider breaker. Opens after `failure_threshold` consecutive
    failures; after `cooldown_s` allows ONE half-open probe; a success closes
    it, a failure re-opens. The retry-storm ceiling: while OPEN, calls are
    refused BEFORE reaching the provider (no hammering a downed API)."""

    def __init__(self, *, failure_threshold: int = 5, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._threshold = max(1, failure_threshold)
        self._cooldown = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._cells: Dict[str, _BreakerCell] = {}

    def _cell(self, provider: str) -> _BreakerCell:
        return self._cells.setdefault(provider, _BreakerCell())

    def state(self, provider: str) -> BreakerState:
        with self._lock:
            return self._cell(provider).state

    def allow(self, provider: str) -> bool:
        """Gate BEFORE the call. False → refuse (breaker open, not yet cool)."""
        with self._lock:
            cell = self._cell(provider)
            if cell.state == BreakerState.OPEN:
                if self._clock() - cell.opened_at >= self._cooldown:
                    cell.state = BreakerState.HALF_OPEN
                    cell.half_open_inflight = True
                    return True                     # single probe allowed
                return False                        # storm ceiling
            if cell.state == BreakerState.HALF_OPEN:
                if cell.half_open_inflight:
                    return False                    # only one probe in flight
                cell.half_open_inflight = True
                return True
            return True

    def record_success(self, provider: str) -> None:
        with self._lock:
            cell = self._cell(provider)
            cell.failures = 0
            cell.successes += 1
            cell.half_open_inflight = False
            if cell.state in (BreakerState.HALF_OPEN, BreakerState.OPEN):
                cell.state = BreakerState.CLOSED

    def record_failure(self, provider: str) -> None:
        with self._lock:
            cell = self._cell(provider)
            cell.half_open_inflight = False
            if cell.state == BreakerState.HALF_OPEN:
                cell.state = BreakerState.OPEN
                cell.opened_at = self._clock()
                return
            cell.failures += 1
            if cell.failures >= self._threshold:
                cell.state = BreakerState.OPEN
                cell.opened_at = self._clock()
                _degraded.emit_degraded(
                    "circuit_breaker",
                    RuntimeError(f"opened after {cell.failures} failures"),
                    tag=provider)


# =====================================================================
# REL-2  Bulkhead — per-provider concurrency isolation
# =====================================================================
class BulkheadFull(ProviderError):
    retryable = True

    def __init__(self, provider: str) -> None:
        super().__init__(provider, RuntimeError("bulkhead full"))


class Bulkhead:
    def __init__(self, *, limit_per_provider: int = 32) -> None:
        self._limit = limit_per_provider
        self._lock = threading.Lock()
        self._sems: Dict[str, threading.Semaphore] = {}

    def _sem(self, provider: str) -> threading.Semaphore:
        with self._lock:
            return self._sems.setdefault(
                provider, threading.Semaphore(self._limit))

    def acquire(self, provider: str, timeout: float = 0.0) -> bool:
        return self._sem(provider).acquire(
            blocking=timeout > 0, timeout=timeout if timeout > 0 else None)

    def release(self, provider: str) -> None:
        self._sem(provider).release()


# =====================================================================
# REL-3 / REL-3b  Backoff + decorrelated jitter, honoring Retry-After
# =====================================================================
@dataclass
class BackoffPolicy:
    base_s: float = 0.05
    cap_s: float = 5.0
    _prev: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        self._prev = 0.0

    def next_delay(self, attempt: int,
                   retry_after: Optional[float] = None) -> float:
        """Decorrelated jitter (AWS): sleep = min(cap, rand(base, prev*3)).
        REL-3b: an explicit Retry-After wins, but is still capped."""
        if retry_after is not None and retry_after > 0:
            return min(self.cap_s, retry_after)
        low = self.base_s
        high = max(self.base_s, self._prev * 3)
        delay = min(self.cap_s, random.uniform(low, high))
        self._prev = delay
        return delay


# =====================================================================
# REL-6  Output validation & repair
# =====================================================================
class OutputInvalid(MalformedRequest):
    """Response failed the declared validator — typed so REL can retry."""
    retryable = True                                # a re-ask may succeed


class OutputValidator:
    """Runs a caller-supplied predicate/extractor over the response. A
    validator that raises or returns False makes the response invalid; the
    resilience loop then retries (bounded) rather than returning garbage."""

    def __init__(self, validate: Callable[[Any], bool]) -> None:
        self._validate = validate

    def check(self, response: Any, provider: str) -> None:
        try:
            ok = bool(self._validate(response))
        except Exception as exc:
            raise OutputInvalid(provider, exc) from exc
        if not ok:
            raise OutputInvalid(provider, ValueError("validator returned False"))


# =====================================================================
# REL-4  Chaos injector
# =====================================================================
class ChaosInjector:
    """Deterministic fault injection. OFF by default → zero overhead
    (pinned). Modes per provider: 'delay', 'drop', 'error', 'malform'."""

    def __init__(self) -> None:
        self._faults: Dict[str, Dict[str, Any]] = {}
        self.enabled = False

    def configure(self, provider: str, *, mode: str,
                  delay_s: float = 0.0, rate: float = 1.0) -> None:
        if mode not in ("delay", "drop", "error", "malform"):
            raise ValueError(f"unknown chaos mode {mode!r}")
        self._faults[provider] = {"mode": mode, "delay_s": delay_s,
                                  "rate": rate}
        self.enabled = True

    def clear(self) -> None:
        self._faults.clear()
        self.enabled = False

    def maybe_inject(self, provider: str, rng: random.Random) -> None:
        if not self.enabled:
            return
        fault = self._faults.get(provider)
        if not fault or rng.random() > fault["rate"]:
            return
        mode = fault["mode"]
        if mode == "delay":
            time.sleep(fault["delay_s"])
        elif mode == "error":
            from .errors import ProviderDown
            raise ProviderDown(provider, RuntimeError("chaos: injected error"))
        elif mode == "drop":
            from .errors import TransientError
            raise TransientError(provider, RuntimeError("chaos: dropped"))
        elif mode == "malform":
            raise OutputInvalid(provider, ValueError("chaos: malformed"))


# =====================================================================
# Orchestrator — replaces the monkeypatch with an explicit wrapper
# =====================================================================
class ResilientExecution:
    """Wraps ExecutionEngine.execute with the full survivability stack:
    breaker gate → bulkhead → chaos → call → validate → on failure:
    typed-routing + backoff-retry + ordered fallback. Records overhead;
    every degrade emits a stable degraded reason."""

    def __init__(self, execution: Any, *, breaker: Optional[CircuitBreaker] = None,
                 bulkhead: Optional[Bulkhead] = None,
                 backoff: Optional[BackoffPolicy] = None,
                 validator: Optional[OutputValidator] = None,
                 chaos: Optional[ChaosInjector] = None,
                 hedge: bool = False,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None) -> None:
        self._execution = execution
        self._breaker = breaker or CircuitBreaker()
        self._bulkhead = bulkhead
        self._backoff = backoff or BackoffPolicy()
        self._validator = validator
        self._chaos = chaos
        self._hedge = hedge
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.retries_used = 0
        self.fallbacks_used = 0
        self.breaker_refusals = 0
        self._orig_execute = execution.execute
        setattr(execution, "execute", self._resilient)

    def _call_once(self, ctx: Dict[str, Any], provider: str) -> Any:
        if self._chaos is not None:
            self._chaos.maybe_inject(provider, self._rng)
        acquired = False
        if self._bulkhead is not None:
            if not self._bulkhead.acquire(provider, timeout=0.0):
                raise BulkheadFull(provider)
            acquired = True
        try:
            result = self._orig_execute(ctx)
        finally:
            if acquired and self._bulkhead is not None:
                self._bulkhead.release(provider)
        if self._validator is not None:
            self._validator.check(result, provider)
        return result

    def _resilient(self, ctx: Dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        cfg = ctx["config"]
        max_retries = int(cfg.get("reliability.max_retries", 1))
        fallback_models: List[str] = list(
            cfg.get("reliability.fallback_order", []))
        primary = ctx["meta"].get("model", ctx["request"].model)
        attempts = [primary] + fallback_models
        last_exc: Optional[BaseException] = None
        self._backoff.reset()
        try:
            for idx, model in enumerate(attempts):
                ctx["meta"]["model"] = model
                provider = self._execution.adapter_for(model).provider
                if not self._breaker.allow(provider):
                    self.breaker_refusals += 1
                    last_exc = CircuitOpen(provider)
                    continue                        # try the next provider
                provider_failed = False
                for attempt in range(max_retries + 1):
                    try:
                        result = self._call_once(ctx, provider)
                        self._breaker.record_success(provider)
                        if idx > 0:
                            self.fallbacks_used += 1
                            ctx["meta"]["fallback"] = model
                        return result
                    except Exception as exc:
                        last_exc = exc
                        if getattr(exc, "retryable", True) is False:
                            # terminal for this provider AND the whole request
                            self._breaker.record_failure(provider)
                            raise
                        if attempt < max_retries:
                            self.retries_used += 1
                            delay = self._backoff.next_delay(
                                attempt,
                                getattr(exc, "retry_after", None))
                            if delay > 0:
                                self._sleep(delay)
                        else:
                            provider_failed = True
                # retries exhausted for this provider: ONE breaker failure
                # (per-request semantics — retries are the retry policy's job,
                # the breaker trips on repeated request-level failures)
                if provider_failed:
                    self._breaker.record_failure(provider)
                    if last_exc is not None:
                        _degraded.emit_degraded(
                            "reliability_fallover", last_exc, tag=provider)
            assert last_exc is not None
            raise last_exc
        finally:
            _overhead.record((time.perf_counter() - t0) * 1000.0)
