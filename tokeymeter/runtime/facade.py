"""The Runtime facade (K4, W3) — the essay's one line, made real.

    from tokeymeter import Runtime
    runtime = Runtime(client=my_openai_client)
    response = runtime.execute("Hello!")

Every request runs the full kernel pipeline (governance → reliability →
execution → trust) over the SHIPPED wrapper machinery (W2 adapters), so the
one line buys caching, compression, routing, audit sealing, typed errors,
and usage truth — nothing reimplemented, everything inherited.

THE WOW RECEIPT (first call, TTY only):
    ✓ Routed openai:gpt-4o-mini   ✓ Cache miss   ✓ Cost $0.0017
    ✓ Saved —   ✓ Latency 812ms   ✓ Verified   ✓ Trace 3f9c2a1b04d1

HONESTY LAW (pinned): every figure is read from a real engine counter or
event. A field with no real source renders `—`. The receipt never invents
a number. Suppression: `TOKEYMETER_QUIET=1`, non-TTY stdout, or
`receipt="never"`; force with `receipt="always"`.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import tokeymeter.events as _events

from .config import RuntimeConfig
from .engines import GovernanceEngine, ReliabilityEngine, TrustEngine
from .kernel import Kernel, KernelRequest, KernelResponse
from .providers import CallableAdapter, ExecutionEngine, ProviderAdapter


class RuntimeConfigurationError(RuntimeError):
    pass


def _request_fingerprint(request) -> Optional[str]:
    """A stable, content-blind digest of this request, so a repeated call is
    detectable on the kernel path too. Mirrors the decorator's cache key: a
    digest of the call arguments, never the text itself."""
    try:
        from tokeymeter.utils import make_cache_key
        meta = request.metadata if isinstance(request.metadata, dict) else {}
        msgs = meta.get("messages")
        return make_cache_key((request.payload,), {"messages": msgs},
                              model=request.model or "_default")
    except Exception:
        return None


def _compliance_preflight(model) -> None:
    """Refuse the call if policy does not permit this model.

    The kernel is a SECOND execution route into the same providers, so every
    control the decorator path enforces has to exist here too or the control is
    only as strong as the path a team happens to use. Raises PolicyViolation,
    which is deliberate: this is the one gate that fails closed.
    """
    try:
        from tokeymeter.engines.governance import compliance as _c
    except Exception:
        return
    try:
        _c.check_model(model)
    except Exception as exc:
        from tokeymeter.engines.governance.compliance import PolicyViolation
        if isinstance(exc, PolicyViolation):
            raise
        return


def _task_preflight(request) -> Optional[str]:
    """Evaluate task ceilings before the call. Raises only a deliberate
    TaskLimitExceeded; every other failure is swallowed, because task
    bookkeeping must never be the reason a request fails."""
    try:
        from tokeymeter.engines.execution import task as _task
        key = _request_fingerprint(request)
        _task.before_call(key, False)
        return key
    except Exception as exc:
        from tokeymeter.engines.execution.task import TaskLimitExceeded
        if isinstance(exc, TaskLimitExceeded):
            raise
        return None


def _emit_ledger_record(resp, latency_ms: float, key: Optional[str]) -> None:
    """Write this kernel call into the SAME ledger the decorator writes to.

    The kernel already computed tokens and cost; without this they stayed in
    the response object and never reached chargeback, the close packet, or
    cost-per-task. One ledger is the whole architecture — a second path that
    executes but does not record is a hole in it.

    Never raises: recording is best-effort, execution is not.
    """
    try:
        from tokeymeter.engines.economics.savings import (
            build_call_record, _record)
        from tokeymeter.engines.execution import task as _task
        meta = resp.metadata if isinstance(resp.metadata, dict) else {}
        cost = float(meta.get("cost_usd") or 0.0)
        rfp = _task.fingerprint_response(resp.payload)
        rec = build_call_record(
            model=meta.get("model") or resp.model or "_default",
            hit=False, hit_type=None,
            input_tokens=int(meta.get("tokens_in") or 0),
            output_tokens=int(meta.get("tokens_out") or 0),
            estimated_cost=cost,
            latency_ms=float(latency_ms),
            token_source=meta.get("usage_source"),
            pricing_source=meta.get("cost_source"),
            task_id=_task.current_task_id(),
            agent=_task.current_agent(),
            prompt_fingerprint=_task.fingerprint_of(key),
            response_fingerprint=rfp,
        )
        _record(rec)
        _task.after_call(_task.fingerprint_of(key), cost, executed=True,
                         response_fp=rfp,
                         input_tokens=meta.get("tokens_in"))
    except Exception:
        pass


# Options the RUNTIME itself consumes. Everything else in **wrap_opts is
# forwarded to the provider adapter.
#
# These have to be extracted BEFORE the adapter is built. They were not, so
# `Runtime(client=..., validator=fn)` handed `validator` to the adapter — whose
# own **kwargs swallowed it — and the facade's later pop found nothing. The
# validator silently did not exist, which is the worst way for a safety control
# to fail: configured, deployed, and absent.
_RUNTIME_OPTIONS = ("signer", "relevance_fn", "telemetry_sinks", "checkpoint",
                    "validator")


def _detect_adapter(client: Any, model: str, wrap_opts: Dict[str, Any]
                    ) -> ProviderAdapter:
    """Duck-typed client detection. Adapters imported lazily so a Runtime
    built on a plain callable never touches the integration stack."""
    from .adapters import AnthropicAdapter, AsyncOpenAIAdapter, OpenAIAdapter
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        return AnthropicAdapter(client, models=[model], **wrap_opts)
    if hasattr(client, "chat"):
        from tokeymeter.engines.execution.integrations.openai_async import (
            is_async_client)
        if is_async_client(client):
            return AsyncOpenAIAdapter(client, models=[model], **wrap_opts)
        return OpenAIAdapter(client, models=[model], **wrap_opts)
    raise RuntimeConfigurationError(
        "Unrecognized client: expected an OpenAI-style (.chat) or "
        "Anthropic-style (.messages) client, a callable via Runtime(call=fn), "
        "or an explicit adapter via Runtime(adapter=...).")


class Runtime:
    def __init__(self, config: Optional[Dict[str, Any]] = None, *,
                 client: Any = None, call: Any = None,
                 adapter: Optional[ProviderAdapter] = None,
                 model: str = "default", receipt: str = "auto",
                 **wrap_opts: Any) -> None:
        if receipt not in ("auto", "always", "never"):
            raise RuntimeConfigurationError(
                "receipt must be 'auto', 'always', or 'never'")
        # Take the runtime's own options out of the bag before anything else
        # sees it, so they reach their features rather than an adapter's
        # **kwargs.
        opts = {k: wrap_opts.pop(k) for k in _RUNTIME_OPTIONS if k in wrap_opts}

        overrides = {"cache": {"enabled": False}}  # shipped wrapper caches;
        # the kernel LRU stays off by default to avoid double-caching.
        if config:
            overrides.update(config)
        self._kernel = Kernel(RuntimeConfig(overrides)).start()
        self._model = model
        self._receipt_mode = receipt
        self._receipt_shown = False
        self.last: Optional[KernelResponse] = None
        self.last_events: List[Any] = []

        if adapter is None:
            if call is not None:
                # No adapter will consume the remainder on this path, so an
                # unrecognised keyword is a typo — and a silently ignored
                # `validater=` means a control the operator believes is running
                # simply is not. Fail loudly at construction instead.
                if wrap_opts:
                    raise RuntimeConfigurationError(
                        f"unknown Runtime option(s): "
                        f"{', '.join(sorted(wrap_opts))}. Runtime accepts "
                        f"{', '.join(_RUNTIME_OPTIONS)}; provider options are "
                        f"only forwarded when a client= is supplied.")
                adapter = CallableAdapter(call, provider="callable",
                                          models=[model])
            elif client is not None:
                adapter = _detect_adapter(client, model, wrap_opts)
            else:
                raise RuntimeConfigurationError(
                    "Runtime needs a provider: Runtime(client=...), "
                    "Runtime(call=fn), or Runtime(adapter=...).")
        # The runtime records only for adapters that do not record themselves.
        # Recording for a wrapped SDK client would bill every call twice in the
        # customer's own chargeback.
        self._emit_ledger = not getattr(adapter, "emits_ledger_record", False)

        self._execution = ExecutionEngine()
        self._execution.register_adapter(adapter, models=[model],
                                         default=True)
        cfg = self._kernel.config
        # Proof spine (W5): when trust.proof is enabled, the ProofEngine
        # replaces the bare TrustEngine — same seam-order slot (registered
        # FIRST → unwinds LAST → seals the complete record), plus shipped-
        # ledger mirroring, optional WORM sink, and prove()/Merkle export.
        if cfg.get("trust.proof.enabled", False):
            from .proof import FileAuditSink, ProofEngine
            audit_log = None
            if cfg.get("trust.proof.ledger", False):
                from tokeymeter.engines.trust.audit.log import AuditLog
                audit_log = AuditLog()
            sink = None
            sink_path = cfg.get("trust.proof.sink_path")
            if sink_path:
                sink = FileAuditSink(sink_path)
            signer = opts.get("signer")
            self._trust = ProofEngine(audit_log=audit_log, sink=sink,
                                      signer=signer)
        else:
            self._trust = TrustEngine()  # type: ignore[assignment]
        self._kernel.register_engine(GovernanceEngine())
        # Trust/Proof registered FIRST so it unwinds LAST in after_response
        # and therefore seals the COMPLETE record — every verdict plus the
        # economics cost — that upstream engines wrote (seam-order law).
        self._kernel.register_engine(self._trust)
        # W4 enforcement spine — config-gated, deny-fails-closed, registered
        # in the shipped order: security → access → rate-limit → economics.
        if cfg.get("governance.security.enabled", False):
            from .enforcement import SecurityEngine
            self._kernel.register_engine(SecurityEngine(
                secrets_mode=cfg.get("governance.security.secrets_mode",
                                     "block"),
                pii=bool(cfg.get("governance.security.pii", True)),
                blocked_terms=cfg.get("governance.security.blocked_terms",
                                      []) or []))
        if cfg.get("governance.rbac.enabled", False):
            from .enforcement import AccessEngine
            self._kernel.register_engine(AccessEngine(
                roles=cfg.get("governance.rbac.roles", {}) or {},
                principals=cfg.get("governance.rbac.principals", {}) or {}))
        # W7 optimization: compression + route planning. Registered AFTER
        # security (only sees screened text) and BEFORE economics (so the
        # chosen model and shrunk payload inform cost). Advisory to cost,
        # never to correctness.
        if cfg.get("optimization.enabled", False):
            from .optimization import OptimizationEngine
            self._kernel.register_engine(OptimizationEngine(
                compress=bool(cfg.get("optimization.compress", False)),
                tier=cfg.get("optimization.tier", "structural"),
                route=bool(cfg.get("optimization.route", False)),
                objective=cfg.get("optimization.objective", "cost"),
                cheap_model=cfg.get("optimization.cheap_model", "gpt-4o-mini"),
                capable_model=cfg.get("optimization.capable_model", "gpt-4o"),
                relevance_fn=opts.get("relevance_fn")))
        if cfg.get("governance.rate_limit.enabled", False):
            from .enforcement import RateLimitEngine
            self._kernel.register_engine(RateLimitEngine(
                requests_per_min=cfg.get(
                    "governance.rate_limit.requests_per_min"),
                tokens_per_min=cfg.get(
                    "governance.rate_limit.tokens_per_min")))
        # W8 telemetry: content-blind export to the org's observability stack.
        if cfg.get("telemetry.enabled", False):
            from .telemetry import (InMemorySink, OTelSink, TelemetryEngine,
                                    TelemetrySink)
            sinks: list = []
            if cfg.get("telemetry.otel", False):
                sinks.append(OTelSink())
            provided = opts.get("telemetry_sinks")
            if provided:
                sinks.extend(provided)
            if not sinks:
                sinks.append(InMemorySink())
            self._telemetry = TelemetryEngine(*sinks)
            self._kernel.register_engine(self._telemetry)
        if cfg.get("economics.enabled", True):
            from .economics import EconomicsEngine
            self._kernel.register_engine(EconomicsEngine(
                budget_key=cfg.get("economics.budget_key"),
                budget_mode=cfg.get("economics.budget_mode", "enforce"),
                checkpoint=opts.get("checkpoint")
                if cfg.get("economics.budget_mode") == "approval" else None))
        # W6 survivability: the resilient stack (breaker/bulkhead/backoff/
        # validate/chaos) replaces bare retry+fallback when enabled.
        if cfg.get("reliability.resilient", False):
            from .resilience import (Bulkhead, CircuitBreaker,
                                     OutputValidator, ResilientExecution)
            validator = opts.get("validator")
            self._resilience = ResilientExecution(
                self._execution,
                breaker=CircuitBreaker(
                    failure_threshold=int(cfg.get(
                        "reliability.breaker.failure_threshold", 5)),
                    cooldown_s=float(cfg.get(
                        "reliability.breaker.cooldown_s", 30.0))),
                bulkhead=Bulkhead(limit_per_provider=int(cfg.get(
                    "reliability.bulkhead.limit", 32)))
                if cfg.get("reliability.bulkhead.enabled", False) else None,
                validator=OutputValidator(validator)
                if validator is not None else None)
        else:
            self._kernel.register_engine(ReliabilityEngine(self._execution))
        self._kernel.register_engine(self._execution)

    # ------------------------------------------------------------ sync ----
    def execute(self, prompt: str, *, model: Optional[str] = None,
                messages: Optional[List[Dict[str, str]]] = None,
                stream: bool = False, **provider_kwargs: Any) -> Any:
        request = self._request(prompt, model, messages, provider_kwargs)
        if stream:
            return self._stream(request)
        captured: List[Any] = []
        handle = _events.subscribe(captured.append)
        # Task ceilings are evaluated BEFORE the call, exactly as on the
        # decorator path — a TaskLimitExceeded here means the call never
        # happens. Without this the kernel path was ungoverned: an agent could
        # spin inside a declared envelope because nothing on this route ever
        # consulted it.
        # Compliance first: a refused model must not consume task budget, and
        # "you may not use this model" is a different answer from "you have run
        # out of money".
        _compliance_preflight(request.model or model)
        fp = _task_preflight(request)
        t0 = time.perf_counter()
        try:
            resp = self._kernel.process(request)
        finally:
            _events.unsubscribe(handle)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.last, self.last_events = resp, captured
        if self._emit_ledger:
            _emit_ledger_record(resp, latency_ms, fp)
        self._maybe_receipt(resp, captured, latency_ms)
        return resp.payload

    def _stream(self, request: KernelRequest) -> Iterator[Any]:
        # ARCHITECTURAL DECISION (documented, tested): streaming bypasses the
        # after-response phase, so a STREAMED request produces NO cost record
        # and NO proof-of-execution seal — the totals do not exist until the
        # stream completes, and we refuse to fabricate them. Callers needing
        # per-call cost/proof on streamed responses must use non-streaming
        # execute(). Governance/security STILL gate before the stream opens
        # (secrets/PII/RBAC run in before_request on the adapter route).
        # Receipt is skipped for the same reason: no honest totals mid-stream.
        adapter = self._execution.adapter_for(
            request.metadata.get("model", request.model))
        ctx = {"request": request, "meta": {"model": request.model},
               "container": self._kernel.container,
               "config": self._kernel.config}
        return adapter.stream_infer(ctx)

    # ----------------------------------------------------------- async ----
    async def aexecute(self, prompt: str, *, model: Optional[str] = None,
                       messages: Optional[List[Dict[str, str]]] = None,
                       stream: bool = False, **provider_kwargs: Any) -> Any:
        request = self._request(prompt, model, messages, provider_kwargs)
        if stream:
            return self._astream(request)
        captured: List[Any] = []
        handle = _events.subscribe(captured.append)
        t0 = time.perf_counter()
        try:
            resp = await self._kernel.aprocess(request)
        finally:
            _events.unsubscribe(handle)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.last, self.last_events = resp, captured
        self._maybe_receipt(resp, captured, latency_ms)
        return resp.payload

    def _astream(self, request: KernelRequest) -> AsyncIterator[Any]:
        adapter = self._execution.adapter_for(
            request.metadata.get("model", request.model))
        astream = getattr(adapter, "astream_infer", None)
        if not callable(astream):
            raise RuntimeConfigurationError(
                f"adapter {adapter.provider!r} has no async streaming; "
                "use execute(stream=True) or an async client")
        ctx = {"request": request, "meta": {"model": request.model},
               "container": self._kernel.container,
               "config": self._kernel.config}
        return astream(ctx)

    # ------------------------------------------------------------ misc ----
    def _request(self, prompt: str, model: Optional[str],
                 messages: Optional[List[Dict[str, str]]],
                 provider_kwargs: Dict[str, Any]) -> KernelRequest:
        meta: Dict[str, Any] = {}
        if messages:
            meta["messages"] = messages
        if provider_kwargs:
            meta["provider_kwargs"] = provider_kwargs
        return KernelRequest(payload=prompt, model=model or self._model,
                             metadata=meta)

    def prove(self, request_id: str) -> Any:
        """Mint an offline-verifiable proof packet for a past request.
        Requires the proof spine (config trust.proof.enabled + a signer)."""
        prove = getattr(self._trust, "prove", None)
        if not callable(prove):
            raise RuntimeConfigurationError(
                "prove() requires the proof spine: "
                "Runtime(config={'trust': {'proof': {'enabled': True}}}, "
                "signer=Ed25519Signer.generate())")
        return prove(request_id)

    def shutdown(self) -> None:
        self._kernel.shutdown()

    # --------------------------------------------------------- receipt ----
    def _maybe_receipt(self, resp: KernelResponse, captured: List[Any],
                       latency_ms: float) -> None:
        if self._receipt_mode == "never" or self._receipt_shown:
            return
        if self._receipt_mode == "auto":
            if os.environ.get("TOKEYMETER_QUIET") == "1":
                return
            if not getattr(sys.stdout, "isatty", lambda: False)():
                return
        self._receipt_shown = True
        print(self.render_receipt(resp, captured, latency_ms))

    def render_receipt(self, resp: KernelResponse, captured: List[Any],
                       latency_ms: float) -> str:
        """Every field from a real source, or `—`. Never an invented number."""
        provider = resp.metadata.get("provider")
        routed = f"{provider}:{resp.model}" if provider else "—"
        # ECON-1 (W4): kernel economics meta is the primary cost source;
        # wrapper events remain the fallback for wrapper-only paths.
        # RECEIPT HONESTY: dollars appear only when provenance is a real
        # price ('registered', 'list') — the pricing module's generic
        # 'default' fallback is a guess and renders `—` here (the guess,
        # with its provenance, stays in resp.metadata for `inspect`).
        # cost_usd is None unless provenance is a real price (engine
        # guarantees this), so it is directly receipt-safe.
        meta_cost = resp.metadata.get("cost_usd")
        ev = captured[-1] if captured else None
        lookup = next((e for e in captured
                       if getattr(e, "event_type", "").startswith("lookup")),
                      ev)
        cache = "—"
        cost = saved = None
        if lookup is not None:
            cache = "hit" if getattr(lookup, "hit", False) else "miss"
            c = getattr(lookup, "estimated_cost_usd", None)
            if c is not None:
                if lookup.hit:
                    saved, cost = c, 0.0
                else:
                    cost = c
        if meta_cost is not None:
            cost = meta_cost
        fmt_usd = lambda v: f"${v:.4f}" if v is not None else "—"
        verified = "✓" if self._trust.verify()[0] else "BROKEN"
        fields = [
            f"✓ Routed {routed}",
            f"✓ Cache {cache}",
            f"✓ Cost {fmt_usd(cost)}",
            f"✓ Saved {fmt_usd(saved)}",
            f"✓ Compression —",           # ratio not surfaced by events yet
            f"✓ Latency {latency_ms:.0f}ms",
            f"✓ Verified {verified}",
            f"✓ Trace {resp.request_id[:12]}",
        ]
        return "  ".join(fields)
