"""Telemetry sinks (W8) — AI execution appears in the enterprise's existing
observability stack.

The runtime already records content-blind metadata per request. A TelemetrySink
is the contract for exporting that record to wherever the organization already
watches everything else. Two implementations ship:

- InMemorySink: captures emitted spans/metrics for tests and local inspection.
- OTelSink: maps the runtime's content-blind record onto OpenTelemetry spans
  and metrics. If the OpenTelemetry SDK is not installed, it degrades to a
  no-op that records it degraded (never crashes a request because telemetry
  export is unavailable).

Content-blind law: sinks receive ONLY the content-blind fields — request id,
model, provider, principal, token counts, cost, latency, policy verdicts,
outcome. Never the payload. The sink cannot leak what it never receives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .engine import Engine


# the content-blind record a sink is allowed to see
_ALLOWED_FIELDS = (
    "request_id", "model", "provider", "principal", "tokens_in", "tokens_out",
    "cost_usd", "cost_source", "usage_source", "latency_ms", "outcome",
    "cache", "compression_method", "tokens_saved", "route_tier",
    "route_objective",
)


def content_blind_record(ctx_meta: Dict[str, Any], *, request_id: str,
                         outcome: str, latency_ms: float) -> Dict[str, Any]:
    """Project the request into the ONLY fields a sink may see. Verdicts are
    reduced to policy:verdict pairs (already content-blind)."""
    rec: Dict[str, Any] = {"request_id": request_id, "outcome": outcome,
                           "latency_ms": round(latency_ms, 3)}
    for f in _ALLOWED_FIELDS:
        if f in ctx_meta and f not in rec:
            rec[f] = ctx_meta[f]
    verdicts = ctx_meta.get("policy_verdicts")
    if verdicts:
        rec["verdicts"] = ";".join(
            f"{v['policy']}:{v['verdict']}" for v in verdicts)
    return rec


@runtime_checkable
class TelemetrySink(Protocol):
    def export(self, record: Dict[str, Any]) -> None: ...


@dataclass
class InMemorySink:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def export(self, record: Dict[str, Any]) -> None:
        self.records.append(dict(record))


class OTelSink:
    """Maps the content-blind record onto OpenTelemetry. Degrades to a
    recorded no-op if the SDK is absent — telemetry export must never break a
    request."""

    def __init__(self, *, tracer_name: str = "tokeymeter",
                 meter_name: str = "tokeymeter") -> None:
        self.degraded = False
        self._tracer = None
        self._cost_counter = None
        self._token_counter = None
        self._latency_hist = None
        try:
            from opentelemetry import metrics, trace
            self._tracer = trace.get_tracer(tracer_name)
            meter = metrics.get_meter(meter_name)
            self._cost_counter = meter.create_counter(
                "tokeymeter.cost_usd", unit="USD",
                description="AI call cost")
            self._token_counter = meter.create_counter(
                "tokeymeter.tokens", unit="1",
                description="tokens in+out")
            self._latency_hist = meter.create_histogram(
                "tokeymeter.latency_ms", unit="ms",
                description="request latency")
        except Exception:
            self.degraded = True                     # SDK absent — no-op

    def export(self, record: Dict[str, Any]) -> None:
        if self.degraded:
            return
        try:
            attrs = {k: record[k] for k in (
                "model", "provider", "principal", "outcome", "cost_source",
                "route_tier") if k in record and record[k] is not None}
            if self._tracer is not None:
                with self._tracer.start_as_current_span(
                        "tokeymeter.request", attributes=attrs) as span:
                    if "request_id" in record:
                        span.set_attribute("request_id",
                                           record["request_id"])
                    if record.get("latency_ms") is not None:
                        span.set_attribute("latency_ms", record["latency_ms"])
            cost = record.get("cost_usd")
            if cost is not None and self._cost_counter is not None:
                self._cost_counter.add(float(cost), attrs)
            toks = (record.get("tokens_in") or 0) + \
                   (record.get("tokens_out") or 0)
            if toks and self._token_counter is not None:
                self._token_counter.add(int(toks), attrs)
            lat = record.get("latency_ms")
            if lat is not None and self._latency_hist is not None:
                self._latency_hist.record(float(lat), attrs)
        except Exception:
            # export failure is isolated: never propagate into the request
            pass


class TelemetryEngine(Engine):
    """Kernel engine that projects each request into a content-blind record
    and hands it to the configured sink(s) in the after-response phase."""

    name = "telemetry"

    def __init__(self, *sinks: TelemetrySink) -> None:
        self._sinks = list(sinks)

    def before_request(self, ctx: Dict[str, Any]) -> None:
        ctx["meta"]["_telemetry_t0"] = _now()

    def after_response(self, ctx: Dict[str, Any]) -> None:
        self._emit(ctx, outcome="ok")

    def on_error(self, ctx: Dict[str, Any], exc: BaseException) -> None:
        self._emit(ctx, outcome=f"error:{type(exc).__name__}")

    def _emit(self, ctx: Dict[str, Any], outcome: str) -> None:
        t0 = ctx["meta"].get("_telemetry_t0", _now())
        latency_ms = (_now() - t0) * 1000.0
        record = content_blind_record(
            ctx["meta"], request_id=ctx["request"].request_id,
            outcome=outcome, latency_ms=latency_ms)
        for sink in self._sinks:
            try:
                sink.export(record)
            except Exception:
                pass                                 # sink isolation


def _now() -> float:
    import time
    return time.perf_counter()
