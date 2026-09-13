"""OptimizationEngine (W7) — compression + route planning in the kernel path.

Runs in the before-execution phase, AFTER security (so it only ever sees
already-screened, already-redacted text — never a secret) and BEFORE
execution (so the provider receives the optimized payload and the chosen
model). It writes:
  - the optimized payload back onto the request (compression)
  - the chosen model onto meta (routing)
  - an auditable route_reason + compression stats onto meta
  - a content-blind optimization event for economics/telemetry

Savings honesty: recorded savings come only from the shipped optimizer/router
math. Compression "tokens saved" is (tokens_before - tokens_after) from the
shipped CompressionResult; routing "est_saved_usd" is the shipped Router's own
figure. Nothing is invented; the savings-parity gate pins this.

Order & safety: optimization is advisory to cost, never to correctness. If a
compressor degrades or a route cannot improve, the request proceeds
unoptimized — optimization must never be the reason a call fails.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .engine import Engine
from .optimize import NoOpOptimizer, Optimizer, RoutePlanner, TieredOptimizer


class OptimizationEngine(Engine):
    name = "optimization"

    def __init__(self, *, optimizer: Optional[Optimizer] = None,
                 compress: bool = False,
                 tier: str = "structural",
                 relevance_fn: Optional[object] = None,
                 route: bool = False,
                 route_planner: Optional[RoutePlanner] = None,
                 objective: str = "cost",
                 cheap_model: str = "gpt-4o-mini",
                 capable_model: str = "gpt-4o",
                 min_tokens_to_compress: int = 24) -> None:
        self._compress = compress
        self._min_tokens = max(0, int(min_tokens_to_compress))
        if optimizer is not None:
            self._optimizer: Optimizer = optimizer
        elif compress:
            self._optimizer = TieredOptimizer(tier=tier,
                                              relevance_fn=relevance_fn)
        else:
            self._optimizer = NoOpOptimizer()
        self._route = route
        self._planner: Optional[RoutePlanner] = None
        if route:
            self._planner = route_planner or RoutePlanner(
                cheap_model=cheap_model, capable_model=capable_model,
                objective=objective)

    def before_request(self, ctx: Dict[str, Any]) -> None:
        request = ctx["request"]

        # ---- route planning (choose the model) ----
        if self._planner is not None:
            plan = self._planner.plan(request.payload)
            meta_updates = plan.as_meta()
            ctx["meta"].update(meta_updates)
            ctx["meta"]["model"] = plan.model          # execution honors this
            if plan.est_saved_usd > 0:
                self._record(ctx, kind="route",
                             saved_usd=plan.est_saved_usd,
                             detail=plan.tier)

        # ---- compression (shrink the payload) ----
        if self._compress and isinstance(request.payload, str):
            query = ctx["meta"].get("query") or \
                request.metadata.get("query")
            result = self._optimizer.optimize(request.payload, query=query)
            # only adopt if it actually helped and stayed safe (shipped
            # safe_compress never expands, but we double-gate on token count
            # and the min-size threshold to avoid churn on tiny prompts)
            if (result.safe and result.tokens_before >= self._min_tokens
                    and result.tokens_after < result.tokens_before):
                request.payload = result.after
                # fingerprint follows the payload the provider will see
                self._refresh_fingerprint(ctx, result.after)
                ctx["meta"]["compression_method"] = result.method
                ctx["meta"]["compression_ratio"] = round(result.ratio, 4)
                ctx["meta"]["tokens_saved"] = (
                    result.tokens_before - result.tokens_after)
                self._record(ctx, kind="compression",
                             tokens_saved=result.tokens_before -
                             result.tokens_after,
                             detail=result.method)
            else:
                ctx["meta"]["compression_method"] = "skipped"

    @staticmethod
    def _refresh_fingerprint(ctx: Dict[str, Any], text: str) -> None:
        import hashlib
        ctx["payload_fingerprint"] = hashlib.sha256(
            text.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _record(ctx: Dict[str, Any], *, kind: str,
                saved_usd: float = 0.0, tokens_saved: int = 0,
                detail: str = "") -> None:
        # content-blind optimization record onto a running list; economics
        # and telemetry consume it. Never any payload text.
        ctx["meta"].setdefault("optimization_events", []).append({
            "kind": kind, "saved_usd": round(saved_usd, 8),
            "tokens_saved": int(tokens_saved), "detail": detail,
        })
