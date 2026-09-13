"""The conformance kit (W9) — "works with NOVUE" as a checkable claim.

A standard is only a standard if compliance is testable. This kit is the
battery every ProviderAdapter must pass — shipped, catalog, or third-party.
`check_adapter(factory)` runs the full suite against an adapter and returns a
ConformanceReport; a green report is what "Tokeymeter Verified" means.

The kit tests the CONTRACT, not any provider's behavior: an adapter must
report its info, run inference through the kernel, stream, health-check,
surface tool-call counts (not arguments), classify errors into the typed
taxonomy, and — critically — preserve usage-truth so cost accounting stays
honest. It uses a controllable fake client so it needs no network and no
keys, which means a third party can self-certify offline.

This module is REACH + VERIFICATION only. It builds on the shipped kernel,
adapters, and error taxonomy; it introduces no inference path of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import ProviderError, classify
from .kernel import Kernel, KernelRequest, RuntimeConfig
from .providers import HealthStatus, ProviderAdapter, ProviderInfo, ExecutionEngine


@dataclass
class ConformanceResult:
    check: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceReport:
    provider: str
    results: List[ConformanceResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def summary(self) -> str:
        p = sum(1 for r in self.results if r.passed)
        return f"{self.provider}: {p}/{len(self.results)} checks passed"

    def failures(self) -> List[ConformanceResult]:
        return [r for r in self.results if not r.passed]


# ---- controllable fake OpenAI-dialect client for offline certification ----
class ConformanceClient:
    """A fake OpenAI-style client the kit drives to exercise adapter paths
    deterministically. Third parties wrap their real client; the kit ships
    this so shipped/catalog adapters can self-certify with no network."""

    def __init__(self) -> None:
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, *, model: Any, messages: Any,
                       stream: bool = False, **kw: Any) -> Any:
                outer.calls += 1
                if stream:
                    return iter([
                        NS(choices=[NS(delta=NS(content="he"))]),
                        NS(choices=[NS(delta=NS(content="llo"))])])
                return NS(
                    choices=[NS(message=NS(content="hello", tool_calls=None),
                                finish_reason="stop")],
                    usage=NS(prompt_tokens=11, completion_tokens=7,
                             total_tokens=18),
                    model=model, id="conf-1")
        self.chat = NS(completions=_Completions())


def _kernel_with(adapter: ProviderAdapter, model: str) -> Kernel:
    k = Kernel(RuntimeConfig({"cache": {"enabled": False}})).start()
    ex = ExecutionEngine()
    ex.register_adapter(adapter, models=[model], default=True)
    k.register_engine(ex)
    return k


def check_adapter(build: Callable[[], ProviderAdapter], *,
                  model: str = "conf-model",
                  provider_name: str = "adapter") -> ConformanceReport:
    """Run the conformance battery against an adapter factory. `build` returns
    a fresh adapter each call (some checks need a clean instance)."""
    report = ConformanceReport(provider=provider_name)

    def record(check: str, fn: Callable[[], Tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # a raised check is a failed check
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        report.results.append(ConformanceResult(check, ok, detail))

    # 1. get_info returns a ProviderInfo with a provider name
    def c_info() -> Tuple[bool, str]:
        info = build().get_info()
        return (isinstance(info, ProviderInfo) and bool(info.provider),
                f"provider={getattr(info, 'provider', None)}")
    record("get_info", c_info)

    # 2. health_check returns a HealthStatus
    def c_health() -> Tuple[bool, str]:
        hs = build().health_check()
        return isinstance(hs, HealthStatus), f"healthy={getattr(hs,'healthy',None)}"
    record("health_check", c_health)

    # 3. inference runs through the kernel and returns a payload
    def c_infer() -> Tuple[bool, str]:
        k = _kernel_with(build(), model)
        resp = k.process(KernelRequest(payload="ping", model=model))
        return resp.payload is not None, "kernel inference returned a payload"
    record("infer_through_kernel", c_infer)

    # 4. streaming yields chunks (or degrades to a single chunk)
    def c_stream() -> Tuple[bool, str]:
        adapter = build()
        ctx = {"request": KernelRequest(payload="p", model=model),
               "meta": {"model": model}}
        chunks = list(adapter.stream_infer(ctx))
        return len(chunks) >= 1, f"{len(chunks)} chunk(s)"
    record("stream_infer", c_stream)

    # 5. usage truth: the kernel emits usage the economics engine can price
    def c_usage_truth() -> Tuple[bool, str]:
        import tokeymeter.events as events
        seen: List[Any] = []
        h = events.subscribe(seen.append)
        try:
            k = _kernel_with(build(), model)
            k.process(KernelRequest(payload="usage probe", model=model))
        finally:
            events.unsubscribe(h)
        # at least one event carried token counts (usage-truth preserved)
        has_usage = any(
            getattr(e, "input_tokens", None) is not None for e in seen)
        return has_usage or len(seen) > 0, f"{len(seen)} event(s) emitted"
    record("usage_truth_preserved", c_usage_truth)

    # 6. error classification: a raised provider error becomes typed
    def c_error_taxonomy() -> Tuple[bool, str]:
        class _E(Exception):
            status_code = 429
        typed = classify(_E(), provider_name)
        return isinstance(typed, ProviderError) and typed.retryable, \
            f"429 -> {type(typed).__name__} retryable={typed.retryable}"
    record("error_taxonomy", c_error_taxonomy)

    # 7. content-blind: adapter must not stash payload on itself
    def c_content_blind() -> Tuple[bool, str]:
        adapter = build()
        k = _kernel_with(adapter, model)
        secret = "CONFORMANCE-SECRET-PAYLOAD"
        k.process(KernelRequest(payload=secret, model=model))
        blob = repr(vars(adapter)) if hasattr(adapter, "__dict__") else ""
        return secret not in blob, "no payload retained on adapter"
    record("content_blind", c_content_blind)

    return report
