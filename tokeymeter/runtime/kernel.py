"""The Runtime Kernel.

Lifecycle: construct(config) → register engines/hooks → start() →
process(request)* → shutdown() (stop accepting, drain in-flight, engines
shut down in reverse order).

Pipeline per request:
    hooks:before_request
    engines[*].before_request        (registration order)
    execution_engine.execute
    engines[*].after_response        (reverse order)
    hooks:after_response
    -- on failure at any point --
    engines[*].on_error + hooks:on_error, then the exception propagates
    (fail loud; retries/fallbacks are the Reliability engine's job, not the
    kernel's).

CONTENT-BLIND TELEMETRY: the kernel trace records request_id, model,
principal, engine names, durations, and a SHA-256 payload fingerprint.
It never records payload text. `KernelResponse.trace` is safe to emit.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .bus import HookBus
from .config import RuntimeConfig
from .container import Container
from .engine import Engine, EngineRegistry


class KernelStopped(RuntimeError):
    """Raised by process() once shutdown has begun."""


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class KernelRequest:
    payload: str
    model: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class KernelResponse:
    payload: Any
    request_id: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class Kernel:
    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.container = Container()
        self.hooks = HookBus()
        self.engines = EngineRegistry()
        self._started = False
        self._stopping = False
        self._inflight = 0
        self._cv = threading.Condition()

    # -- registration ------------------------------------------------------
    def register_engine(self, engine: Engine) -> Engine:
        return self.engines.register(engine)

    def use(self, hook: str, fn: Any, *, priority: int = 100) -> Any:
        return self.hooks.subscribe(hook, fn, priority=priority)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "Kernel":
        self._started = True
        self.hooks.emit("kernel_started", {"kernel": self})
        return self

    def shutdown(self) -> None:
        with self._cv:
            self._stopping = True
            deadline = time.monotonic() + float(
                self.config.get("kernel.drain_timeout_s", 30.0)
            )
            while self._inflight > 0 and time.monotonic() < deadline:
                self._cv.wait(timeout=0.05)
        for engine in reversed(self.engines.ordered()):
            try:
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                pass
        self.hooks.emit("kernel_stopped", {"kernel": self})

    # -- pipeline -----------------------------------------------------------
    def process(self, request: KernelRequest) -> KernelResponse:
        if not self._started:
            raise RuntimeError("kernel not started; call start()")
        with self._cv:
            if self._stopping:
                raise KernelStopped("kernel is shutting down")
            self._inflight += 1
        try:
            return self._run_pipeline(request)
        finally:
            with self._cv:
                self._inflight -= 1
                self._cv.notify_all()

    async def aprocess(self, request: KernelRequest) -> KernelResponse:
        """KA-1: async twin of process(). Same drain semantics — in-flight
        counter covers awaited execution, shutdown refuses new work and
        waits for completions (pinned by adrain_under_load)."""
        if not self._started:
            raise RuntimeError("kernel not started; call start()")
        with self._cv:
            if self._stopping:
                raise KernelStopped("kernel is shutting down")
            self._inflight += 1
        try:
            return await self._run_pipeline_async(request)
        finally:
            with self._cv:
                self._inflight -= 1
                self._cv.notify_all()

    async def _run_pipeline_async(self, request: KernelRequest) -> KernelResponse:
        ctx = self._build_ctx(request)
        trace = ctx["trace"]
        ordered = self.engines.ordered()
        try:
            self.hooks.emit("before_request", ctx)
            for engine in ordered:
                t0 = time.perf_counter()
                engine.before_request(ctx)
                trace.append({"engine": engine.name, "phase": "before_request",
                              "ms": round((time.perf_counter() - t0) * 1000, 3)})
                if ctx["short_circuit"]:
                    break
            if not ctx["short_circuit"]:
                exec_engine = self.engines.execution_engine()
                t0 = time.perf_counter()
                aexec = getattr(exec_engine, "aexecute", None)
                if callable(aexec):
                    ctx["response_payload"] = await aexec(ctx)
                else:  # engine is sync-only: never block the loop
                    import asyncio as _aio
                    ctx["response_payload"] = await _aio.to_thread(
                        exec_engine.execute, ctx)
                trace.append({"engine": exec_engine.name, "phase": "execute",
                              "ms": round((time.perf_counter() - t0) * 1000, 3)})
            for engine in reversed(ordered):
                t0 = time.perf_counter()
                engine.after_response(ctx)
                trace.append({"engine": engine.name, "phase": "after_response",
                              "ms": round((time.perf_counter() - t0) * 1000, 3)})
            self.hooks.emit("after_response", ctx)
        except Exception as exc:
            for engine in ordered:
                try:
                    engine.on_error(ctx, exc)
                except Exception:
                    pass
            self.hooks.emit("on_error", {**ctx, "error_type": type(exc).__name__})
            raise
        return self._finish_response(ctx, request)

    def _build_ctx(self, request: KernelRequest) -> Dict[str, Any]:
        return {
            "request": request,
            "kernel": self,
            "container": self.container,
            "config": self.config,
            "response_payload": None,
            "short_circuit": False,     # a before_request stage may answer
            "meta": dict(request.metadata),
            "trace": [],
            "payload_fingerprint": _fingerprint(request.payload),
        }

    def _finish_response(self, ctx: Dict[str, Any],
                         request: KernelRequest) -> KernelResponse:
        return KernelResponse(
            payload=ctx["response_payload"],
            request_id=request.request_id,
            model=ctx["meta"].get("model", request.model),
            metadata={k: v for k, v in ctx["meta"].items()
                      if k not in ("payload",)},
            trace=ctx["trace"],
        )

    def _run_pipeline(self, request: KernelRequest) -> KernelResponse:
        ctx = self._build_ctx(request)
        trace = ctx["trace"]
        ordered = self.engines.ordered()
        try:
            self.hooks.emit("before_request", ctx)
            for engine in ordered:
                t0 = time.perf_counter()
                engine.before_request(ctx)
                trace.append({
                    "engine": engine.name, "phase": "before_request",
                    "ms": round((time.perf_counter() - t0) * 1000, 3),
                })
                if ctx["short_circuit"]:
                    break
            if not ctx["short_circuit"]:
                exec_engine = self.engines.execution_engine()
                t0 = time.perf_counter()
                ctx["response_payload"] = exec_engine.execute(ctx)
                trace.append({
                    "engine": exec_engine.name, "phase": "execute",
                    "ms": round((time.perf_counter() - t0) * 1000, 3),
                })
            for engine in reversed(ordered):
                t0 = time.perf_counter()
                engine.after_response(ctx)
                trace.append({
                    "engine": engine.name, "phase": "after_response",
                    "ms": round((time.perf_counter() - t0) * 1000, 3),
                })
            self.hooks.emit("after_response", ctx)
        except Exception as exc:
            for engine in ordered:
                try:
                    engine.on_error(ctx, exc)
                except Exception:
                    pass
            self.hooks.emit("on_error", {**ctx, "error_type": type(exc).__name__})
            raise
        return self._finish_response(ctx, request)
