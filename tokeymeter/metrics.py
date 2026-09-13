"""
Prometheus-format metrics exporter.

Subscribes to tokeymeter.events and accumulates counters/histograms. Exposes
a `render()` function returning text in the standard Prometheus exposition
format. No external dependencies — we generate the text by hand.

Drop this into your existing /metrics endpoint:

    from tokeymeter.metrics import PrometheusCollector
    collector = PrometheusCollector()

    # Flask
    @app.get("/metrics")
    def metrics():
        return collector.render(), 200, {"Content-Type": "text/plain"}

    # FastAPI
    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(collector.render())

    # Or just print it
    print(collector.render())

The collector is fail-safe: a buggy emit can't crash the cache pipeline.
Counters are kept in a small lock-protected dict; rendering is O(N) over
the dict size (typically <100 entries).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Tuple

from .events import CacheEvent, subscribe, unsubscribe


# Default histogram buckets (seconds), tuned for LLM cache ops.
# Sub-ms is exact-cache fast path. Single-digit ms is semantic.
# Anything >100ms is a real LLM call.
DEFAULT_BUCKETS_SECONDS: Tuple[float, ...] = (
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.025,
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


class PrometheusCollector:
    """Subscribe to cache events; expose Prometheus-format metrics.

    Tracks (model × hit_type × event_type) breakdowns plus a histogram
    of latencies. All numbers are cumulative counters since process start
    — the standard Prometheus pattern; the scraper computes rates.
    """

    def __init__(
        self,
        *,
        namespace: str = "tokeymeter",
        buckets_seconds: Tuple[float, ...] = DEFAULT_BUCKETS_SECONDS,
        auto_subscribe: bool = True,
    ):
        self._ns = namespace
        self._buckets = buckets_seconds
        self._lock = threading.Lock()

        # Counters keyed by labels.
        # counter name -> {labels frozenset -> count}
        self._counters: Dict[str, Dict[frozenset, float]] = defaultdict(dict)

        # Histogram: buckets keyed by labels.
        # labels frozenset -> [bucket_counts..., sum, count]
        self._histograms: Dict[frozenset, Dict[str, float]] = {}

        self._unsubscribe = None
        if auto_subscribe:
            subscribe(self._on_event)
            # `subscribe` returns the callback; we need a real unsubscribe handle
            self._unsubscribe = lambda: unsubscribe(self._on_event)

    # ---------- Event handler ----------

    def _on_event(self, event: CacheEvent) -> None:
        """Called for every cache event. Must be fast and never raise."""
        try:
            labels = frozenset({
                ("model", event.model),
                ("hit_type", event.hit_type or "none"),
                ("event_type", event.event_type),
                ("function", event.function_name or "unknown"),
            })

            with self._lock:
                # Counter: total events
                self._counters["events_total"][labels] = (
                    self._counters["events_total"].get(labels, 0) + 1
                )

                # Counter: tokens
                if event.input_tokens > 0:
                    self._counters["input_tokens_total"][labels] = (
                        self._counters["input_tokens_total"].get(labels, 0)
                        + event.input_tokens
                    )
                if event.output_tokens > 0:
                    self._counters["output_tokens_total"][labels] = (
                        self._counters["output_tokens_total"].get(labels, 0)
                        + event.output_tokens
                    )

                # Counter: estimated USD cost (split into saved vs spent)
                if event.event_type == "hit":
                    self._counters["estimated_saved_usd_total"][labels] = (
                        self._counters["estimated_saved_usd_total"].get(labels, 0)
                        + event.estimated_cost_usd
                    )
                elif event.event_type == "miss":
                    self._counters["estimated_spent_usd_total"][labels] = (
                        self._counters["estimated_spent_usd_total"].get(labels, 0)
                        + event.estimated_cost_usd
                    )

                # Histogram: latency
                hist_labels = frozenset({
                    ("model", event.model),
                    ("event_type", event.event_type),
                })
                if hist_labels not in self._histograms:
                    self._histograms[hist_labels] = {
                        **{f"le_{b}": 0 for b in self._buckets},
                        "le_+Inf": 0,
                        "sum": 0.0,
                        "count": 0,
                    }
                h = self._histograms[hist_labels]
                latency_s = event.latency_ms / 1000.0
                for b in self._buckets:
                    if latency_s <= b:
                        h[f"le_{b}"] += 1
                h["le_+Inf"] += 1
                h["sum"] += latency_s
                h["count"] += 1
        except Exception:
            pass  # never crash the emit path

    # ---------- Rendering ----------

    def render(self) -> str:
        """Return Prometheus-format text. Safe to call concurrently."""
        lines: List[str] = []

        with self._lock:
            counters = {name: dict(values) for name, values in self._counters.items()}
            histograms = {k: dict(v) for k, v in self._histograms.items()}

        # ---- Counters ----
        counter_metadata = {
            "events_total": ("counter", "Total cache events (hit/miss/error/etc)"),
            "input_tokens_total": ("counter", "Cumulative input token estimate"),
            "output_tokens_total": ("counter", "Cumulative output token estimate"),
            "estimated_saved_usd_total": ("counter", "Estimated USD saved by cache hits"),
            "estimated_spent_usd_total": ("counter", "Estimated USD spent on real API calls"),
        }
        for name, values in counters.items():
            mtype, help_text = counter_metadata.get(name, ("counter", ""))
            full_name = f"{self._ns}_{name}"
            if help_text:
                lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} {mtype}")
            for labels, value in sorted(values.items(), key=lambda kv: _label_sort_key(kv[0])):
                lines.append(f"{full_name}{_render_labels(labels)} {value}")

        # ---- Histograms ----
        if histograms:
            full_name = f"{self._ns}_latency_seconds"
            lines.append(f"# HELP {full_name} Cache operation latency in seconds")
            lines.append(f"# TYPE {full_name} histogram")
            for labels, h in sorted(histograms.items(), key=lambda kv: _label_sort_key(kv[0])):
                label_str_base = _render_labels(labels, trailing_comma=True)
                for b in self._buckets:
                    lines.append(f'{full_name}_bucket{label_str_base}le="{b}"}} {h[f"le_{b}"]}')
                lines.append(f'{full_name}_bucket{label_str_base}le="+Inf"}} {h["le_+Inf"]}')
                lines.append(f'{full_name}_sum{_render_labels(labels)} {h["sum"]}')
                lines.append(f'{full_name}_count{_render_labels(labels)} {h["count"]}')

        # Trailing newline per Prometheus convention
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        """Unsubscribe and stop collecting events."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None


# ---------- Label rendering helpers ----------

def _label_sort_key(labels: frozenset) -> str:
    """Stable sort key for deterministic output."""
    return ",".join(f"{k}={v}" for k, v in sorted(labels))


def _render_labels(labels: frozenset, trailing_comma: bool = False) -> str:
    """Render a label set as Prometheus label string: {k="v",k="v"}.

    If trailing_comma=True, returns just `{k="v",k="v",` for further composition.
    """
    parts = [f'{k}="{_escape(str(v))}"' for k, v in sorted(labels)]
    inner = ",".join(parts)
    if trailing_comma:
        return "{" + (inner + "," if inner else "")
    return "{" + inner + "}" if inner else ""


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
