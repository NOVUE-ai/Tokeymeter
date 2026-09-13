"""Provider adapters and the Execution Engine (doc Sprint 2 contract).

interface ProviderAdapter:
    get_info() -> ProviderInfo
    infer(request) -> response
    stream_infer(request) -> iterator[str]
    health_check() -> HealthStatus

IN-PROCESS DISCIPLINE (binding): adapters wrap the caller's own client code
inside the interpreter — CallableAdapter takes the user's existing function.
There is no proxy and no network hop introduced by this layer; that property
is the moat and is not negotiable in the flagship path.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .engine import Engine


@dataclass
class ProviderInfo:
    provider: str
    models: List[str] = field(default_factory=list)
    version: str = "0"


@dataclass
class HealthStatus:
    healthy: bool
    latency_ms: float = 0.0
    detail: str = ""


class ProviderAdapter:
    # Does this adapter's own path already write a ledger record?
    #
    # The shipped SDK adapters route through tokeymeter.integrations, which
    # records. A raw adapter (a plain callable, a custom provider) does not, so
    # the runtime has to record on its behalf — and must NOT record for the
    # ones that already did, or every call through a wrapped client would be
    # billed twice in the customer's own chargeback.
    emits_ledger_record = False

    """Contract per the architecture's adapter table."""

    provider: str = "adapter"

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(provider=self.provider)

    def infer(self, ctx: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def stream_infer(self, ctx: Dict[str, Any]) -> Iterator[str]:
        # Default: degrade to a single-chunk stream over infer().
        yield str(self.infer(ctx))

    def health_check(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            info = self.get_info()
            ok = bool(info.provider)
        except Exception as exc:
            return HealthStatus(False, 0.0, f"{type(exc).__name__}")
        return HealthStatus(ok, round((time.perf_counter() - t0) * 1000, 3))


class CallableAdapter(ProviderAdapter):
    """Wraps the user's existing in-process model call — the compatibility
    bridge: `CallableAdapter(my_llm_fn, provider="openai")` and the kernel
    executes THEIR code, unchanged."""

    def __init__(
        self,
        fn: Callable[[str], Any],
        *,
        provider: str = "callable",
        models: Optional[List[str]] = None,
    ) -> None:
        self._fn = fn
        self.provider = provider
        self._models = models or ["default"]

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(provider=self.provider, models=list(self._models))

    def infer(self, ctx: Dict[str, Any]) -> Any:
        return self._fn(ctx["request"].payload)


class OpenAIStubAdapter(CallableAdapter):
    """Sprint-2 stub: real SDK wiring stays in tokeymeter.integrations.openai;
    this pins the adapter contract shape for the kernel path."""

    def __init__(self, fn: Callable[[str], Any]) -> None:
        super().__init__(fn, provider="openai", models=["gpt-*"])


class AnthropicStubAdapter(CallableAdapter):
    def __init__(self, fn: Callable[[str], Any]) -> None:
        super().__init__(fn, provider="anthropic", models=["claude-*"])


class UnknownModel(LookupError):
    pass


class ExecutionEngine(Engine):
    """Routes a request to a registered adapter and executes it.

    Routing: exact model → adapter mapping first, then a "route" hook may
    rewrite ctx["meta"]["model"] beforehand (REFLEX alignment), then the
    default adapter if declared.
    """

    name = "execution"
    handles_execution = True

    def __init__(self) -> None:
        self._routes: Dict[str, ProviderAdapter] = {}
        self._default: Optional[ProviderAdapter] = None
        self.calls = 0  # observable for tests/metrics

    def register_adapter(
        self, adapter: ProviderAdapter, *, models: Optional[List[str]] = None,
        default: bool = False,
    ) -> ProviderAdapter:
        for m in (models or adapter.get_info().models):
            self._routes[m] = adapter
        if default or self._default is None:
            self._default = adapter if default else self._default
        return adapter

    def set_default(self, adapter: ProviderAdapter) -> None:
        self._default = adapter

    def adapter_for(self, model: str) -> ProviderAdapter:
        adapter = self._routes.get(model) or self._default
        if adapter is None:
            raise UnknownModel(model)
        return adapter

    def execute(self, ctx: Dict[str, Any]) -> Any:
        model = ctx["meta"].get("model", ctx["request"].model)
        adapter = self.adapter_for(model)
        self.calls += 1
        ctx["meta"]["provider"] = adapter.provider
        ctx["meta"]["model"] = model
        return adapter.infer(ctx)

    async def aexecute(self, ctx: Dict[str, Any]) -> Any:
        """KA-1 async path: awaits native `ainfer` when the adapter has one;
        otherwise runs the sync adapter in a worker thread so the event loop
        is never blocked (pinned by mixed_sync_async battery check)."""
        model = ctx["meta"].get("model", ctx["request"].model)
        adapter = self.adapter_for(model)
        self.calls += 1
        ctx["meta"]["provider"] = adapter.provider
        ctx["meta"]["model"] = model
        ainfer = getattr(adapter, "ainfer", None)
        if callable(ainfer):
            return await ainfer(ctx)
        return await asyncio.to_thread(adapter.infer, ctx)
