"""Optimization unification (W7): OPT-1 the Optimizer contract over the
shipped compression/salience tiers, OPT-2 the multi-objective RoutePlanner
over the shipped Router.

Design law — WRAP, NEVER REIMPLEMENT. Every gram of savings comes from the
shipped modules (compression.py, salience.py, query_compress.py, router.py,
cascade.py). This layer only *composes and selects* them behind one contract
and writes an auditable reason onto meta. The wave gate (savings-parity)
pins that the numbers the shipped code produces are exactly the numbers this
layer reports — no regression, no invention.

Determinism law — given identical config and input, the optimizer and route
planner produce identical decisions. No hidden randomness on the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from tokeymeter.engines.optimization.compression import (
    CompressionResult, StructuralCompressor, compose, safe_compress)
from tokeymeter.engines.optimization.query_compress import QueryAwareCompressor
from tokeymeter.engines.optimization.router import (
    RouteDecision, Router, WinRateScorer)
from tokeymeter.engines.optimization.salience import SalienceCompressor


# =====================================================================
# OPT-1  The Optimizer contract
# =====================================================================
@runtime_checkable
class Optimizer(Protocol):
    """One question: given text (and optional query), return a
    CompressionResult. Any conformant optimizer plugs into the pipeline."""

    def optimize(self, text: str, *, query: Optional[str] = None
                 ) -> CompressionResult: ...


class NoOpOptimizer:
    """The honest floor: never changes the text, reports a real no-op result.
    Used when optimization is disabled so the contract still holds."""

    method = "noop"

    def optimize(self, text: str, *, query: Optional[str] = None
                 ) -> CompressionResult:
        return safe_compress(None, text)


class TieredOptimizer:
    """The default optimizer: a conservative tier ladder over the shipped
    compressors, selected by config, composed through the shipped compose()
    so ordering and safety semantics are inherited, not re-invented.

    Tiers (each maps to shipped code):
      "structural"   -> StructuralCompressor (lexical, lossless-ish)
      "salience"     -> SalienceCompressor (segment salience)
      "query"        -> QueryAwareCompressor (needs a query; falls back to
                        structural when no query is supplied)
      "compose:a+b"  -> compose(a, b) of any of the above

    A tier that cannot run for a given input (e.g. query tier, no query)
    degrades to the safe floor rather than raising — the pipeline must never
    fail because optimization could not help.
    """

    def __init__(self, *, tier: str = "structural",
                 relevance_fn: Optional[object] = None,
                 salience_target_ratio: float = 0.6,
                 context_target_ratio: float = 0.5) -> None:
        self.tier = tier
        self._relevance_fn = relevance_fn
        self._salience_target = salience_target_ratio
        self._context_target = context_target_ratio
        self.method = f"tiered:{tier}"

    def _build(self, name: str, query: Optional[str]) -> Any:
        name = name.strip()
        if name == "structural":
            return StructuralCompressor()
        if name == "salience":
            return SalienceCompressor(target_ratio=self._salience_target,
                                      query=query)
        if name == "query":
            if not query:
                return StructuralCompressor()      # graceful degrade
            return QueryAwareCompressor(
                query=query, context_target_ratio=self._context_target,
                relevance_fn=self._relevance_fn)
        raise ValueError(f"unknown compression tier {name!r}")

    def optimize(self, text: str, *, query: Optional[str] = None
                 ) -> CompressionResult:
        if not isinstance(text, str) or not text:
            return safe_compress(None, text)
        spec = self.tier
        if spec.startswith("compose:"):
            names = [n for n in spec[len("compose:"):].split("+") if n]
            comps = [self._build(n, query) for n in names]
            compressor = compose(*comps)
        else:
            compressor = self._build(spec, query)
        # safe_compress guarantees: never raises, returns a real result, and
        # NEVER returns an expansion (shipped code keeps the smaller of
        # before/after) — so parity with direct use is exact.
        return safe_compress(compressor, text)


# =====================================================================
# OPT-2  Multi-objective RoutePlanner
# =====================================================================
@dataclass
class RoutePlan:
    """The auditable output of route planning: which model, why, and the
    objective weights that produced it. Written to meta.route_reason."""
    model: str
    tier: str
    reason: str
    est_cost_usd: float
    est_saved_usd: float
    win_rate: float
    confident: bool
    objective: str
    weights: Dict[str, float] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def as_meta(self) -> Dict[str, Any]:
        return {
            "route_model": self.model, "route_tier": self.tier,
            "route_reason": self.reason,
            "route_est_cost_usd": self.est_cost_usd,
            "route_est_saved_usd": self.est_saved_usd,
            "route_win_rate": round(self.win_rate, 4),
            "route_confident": self.confident,
            "route_objective": self.objective,
        }


class RoutePlanner:
    """Multi-objective routing over the SHIPPED Router. The Router already
    produces the win-rate, the cost estimates, and the conservative
    route-up-when-uncertain rule (its savings logic is the source of truth).
    The planner adds an EXPLICIT, AUDITABLE objective on top:

      "cost"     -> take the Router's decision as-is (max savings; default)
      "quality"  -> never take the cheap tier; always the capable model
      "balanced" -> take cheap only when the Router is confident AND the
                    win-rate margin clears a latency/quality safety band

    Determinism: identical config + prompt -> identical plan. The Router's
    cost math is used verbatim; the planner only chooses among the tiers the
    Router already priced, so savings can never exceed — or regress below —
    what the shipped Router computed.
    """

    def __init__(self, *, cheap_model: str = "gpt-4o-mini",
                 capable_model: str = "gpt-4o",
                 objective: str = "cost",
                 balanced_margin: float = 0.15,
                 threshold: float = 0.5,
                 scorer: Optional[WinRateScorer] = None,
                 weights: Optional[Dict[str, float]] = None) -> None:
        if objective not in ("cost", "quality", "balanced"):
            raise ValueError("objective must be cost|quality|balanced")
        self._router = Router(cheap_model=cheap_model,
                              capable_model=capable_model,
                              threshold=threshold, scorer=scorer)
        self._cheap = cheap_model
        self._capable = capable_model
        self.objective = objective
        self._margin = balanced_margin
        self._weights = weights or self._default_weights(objective)

    @staticmethod
    def _default_weights(objective: str) -> Dict[str, float]:
        return {
            "cost": {"cost": 1.0, "quality": 0.0, "latency": 0.0},
            "quality": {"cost": 0.0, "quality": 1.0, "latency": 0.0},
            "balanced": {"cost": 0.5, "quality": 0.4, "latency": 0.1},
        }[objective]

    def plan(self, prompt: str) -> RoutePlan:
        decision: RouteDecision = self._router.route(prompt)
        # The Router already priced both tiers and made a conservative call.
        obj = self.objective
        if obj == "quality":
            # force capable; savings vs capable is zero by definition
            model, tier, saved = self._capable, "capable", 0.0
            reason = "objective=quality: capable model always"
            cost = self._capable_cost(decision)
        elif obj == "balanced":
            take_cheap = (decision.tier == "cheap" and decision.confident and
                          (0.5 - decision.win_rate) >= self._margin)
            if take_cheap:
                model, tier, saved = decision.model, decision.tier, \
                    decision.est_saved_usd
                cost = decision.est_cost_usd
                reason = (f"objective=balanced: cheap accepted "
                          f"(win_rate {decision.win_rate:.2f}, margin ok)")
            else:
                model, tier, saved = self._capable, "capable", 0.0
                cost = self._capable_cost(decision)
                reason = (f"objective=balanced: capable "
                          f"({decision.reason})")
        else:  # cost — Router decision verbatim (max savings)
            model, tier = decision.model, decision.tier
            saved, cost = decision.est_saved_usd, decision.est_cost_usd
            reason = f"objective=cost: {decision.reason}"
        return RoutePlan(
            model=model, tier=tier, reason=reason, est_cost_usd=cost,
            est_saved_usd=saved, win_rate=decision.win_rate,
            confident=decision.confident, objective=obj,
            weights=dict(self._weights),
            candidates=[
                {"tier": "cheap", "model": self._cheap},
                {"tier": "capable", "model": self._capable},
            ])

    def _capable_cost(self, decision: RouteDecision) -> float:
        # When we override to capable, the capable cost is the cheap decision's
        # cost + its own saved delta, or the decision cost if already capable.
        if decision.tier == "capable":
            return decision.est_cost_usd
        return decision.est_cost_usd + decision.est_saved_usd
