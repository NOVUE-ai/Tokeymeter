"""W7 battery — optimization unification.

THE WAVE GATE: test_savings_parity_* — the tokens/cost the unified layer
reports equal, to the unit, what the shipped compressors and Router produce
directly. Optimization adds selection and audit; it never changes the math.
"""
from __future__ import annotations

import hashlib

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import (
    Kernel, KernelRequest, NoOpOptimizer, OptimizationEngine, Optimizer,
    RoutePlan, RoutePlanner, RuntimeConfig, TieredOptimizer,
)
from tokeymeter.runtime.enforcement import SecurityEngine
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine

# shipped modules — the sources of truth for parity
from tokeymeter.engines.optimization.compression import (
    StructuralCompressor, compose, safe_compress)
from tokeymeter.engines.optimization.salience import SalienceCompressor
from tokeymeter.engines.optimization.query_compress import QueryAwareCompressor
from tokeymeter.engines.optimization.router import Router

VERBOSE = ("Please kindly note that in order to proceed you should basically "
           "just go ahead and click the button at this point in time. ") * 4
ANTHROPIC_KEY = "sk-ant-api03-" + "A" * 88


def kernel(*engines, fn=None, model="default", **cfg):
    k = Kernel(RuntimeConfig(cfg or {})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(fn or (lambda p: "ok")),
                        models=[model], default=True)
    for e in engines:
        k.register_engine(e)
    k.register_engine(ex)
    return k, ex


# ============================================= OPT-1 contract ============
def test_noop_optimizer_is_honest_and_conformant():
    r = NoOpOptimizer().optimize(VERBOSE)
    assert r.after == VERBOSE and r.ratio == 1.0 and r.safe
    assert isinstance(NoOpOptimizer(), Optimizer)


def test_tiered_conforms_to_protocol():
    assert isinstance(TieredOptimizer(), Optimizer)


@pytest.mark.parametrize("tier", ["structural", "salience"])
def test_tier_reduces_or_preserves_never_expands(tier):
    r = TieredOptimizer(tier=tier).optimize(VERBOSE)
    assert r.tokens_after <= r.tokens_before          # never an expansion
    assert r.safe


def test_query_tier_without_query_degrades_gracefully():
    # no query supplied -> falls back to structural, never raises
    r = TieredOptimizer(tier="query").optimize(VERBOSE, query=None)
    assert r.safe and r.tokens_after <= r.tokens_before


def test_query_tier_with_query_runs_query_aware():
    r = TieredOptimizer(tier="query").optimize(
        VERBOSE, query="how do I click the button?")
    assert r.safe


def test_compose_tier_chains_shipped_compose():
    r = TieredOptimizer(tier="compose:structural+salience").optimize(VERBOSE)
    assert r.safe and r.tokens_after <= r.tokens_before


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        TieredOptimizer(tier="nonsense").optimize(VERBOSE)


def test_empty_and_tiny_input_safe():
    assert TieredOptimizer().optimize("").after == ""
    assert TieredOptimizer().optimize("hi").safe


# ============================================= THE WAVE GATE =============
def test_savings_parity_structural():
    """Unified structural tier ≡ direct StructuralCompressor via shipped
    safe_compress — after-text, token counts, and ratio identical."""
    direct = safe_compress(StructuralCompressor(), VERBOSE)
    unified = TieredOptimizer(tier="structural").optimize(VERBOSE)
    assert unified.after == direct.after
    assert unified.tokens_before == direct.tokens_before
    assert unified.tokens_after == direct.tokens_after
    assert unified.ratio == direct.ratio


def test_savings_parity_salience():
    direct = safe_compress(
        SalienceCompressor(target_ratio=0.6, query=None), VERBOSE)
    unified = TieredOptimizer(tier="salience").optimize(VERBOSE)
    assert (unified.after, unified.tokens_after, unified.ratio) == \
           (direct.after, direct.tokens_after, direct.ratio)


def test_savings_parity_compose():
    direct = safe_compress(
        compose(StructuralCompressor(), SalienceCompressor(target_ratio=0.6)),
        VERBOSE)
    unified = TieredOptimizer(
        tier="compose:structural+salience").optimize(VERBOSE)
    assert (unified.after, unified.tokens_after) == \
           (direct.after, direct.tokens_after)


def test_route_savings_parity_with_shipped_router():
    """RoutePlanner(objective=cost) reports EXACTLY the shipped Router's
    est_saved_usd and model — no invented savings."""
    prompt = "translate hello to french"
    direct = Router(cheap_model="gpt-4o-mini",
                    capable_model="gpt-4o").route(prompt)
    plan = RoutePlanner(objective="cost").plan(prompt)
    assert plan.model == direct.model
    assert plan.est_saved_usd == direct.est_saved_usd
    assert plan.est_cost_usd == direct.est_cost_usd


# ============================================= OPT-2 route planning =====
def test_route_objective_cost_takes_router_decision():
    plan = RoutePlanner(objective="cost").plan("what is 2 plus 2")
    assert plan.objective == "cost" and plan.tier in ("cheap", "capable")


def test_route_objective_quality_never_cheap():
    plan = RoutePlanner(objective="quality").plan("what is 2 plus 2")
    assert plan.tier == "capable" and plan.est_saved_usd == 0.0
    assert plan.model == "gpt-4o"


def test_route_objective_balanced_requires_margin():
    # a trivial prompt: cost-objective would take cheap; balanced only takes
    # cheap when the confidence margin clears the band
    trivial = "hi"
    cost_plan = RoutePlanner(objective="cost").plan(trivial)
    bal_plan = RoutePlanner(objective="balanced",
                            balanced_margin=0.99).plan(trivial)
    # with an impossibly high margin, balanced must route up to capable
    assert bal_plan.tier == "capable"
    # cost plan is free to choose cheap
    assert cost_plan.tier in ("cheap", "capable")


def test_route_planner_deterministic():
    rp = RoutePlanner(objective="balanced")
    a, b = rp.plan("prove that sqrt 2 is irrational"), \
        rp.plan("prove that sqrt 2 is irrational")
    assert (a.model, a.est_saved_usd, a.reason) == \
           (b.model, b.est_saved_usd, b.reason)


def test_route_plan_meta_is_auditable():
    plan = RoutePlanner(objective="cost").plan("summarize this document")
    meta = plan.as_meta()
    assert meta["route_model"] and "objective=" in meta["route_reason"]
    assert "route_win_rate" in meta and "route_est_saved_usd" in meta


def test_bad_objective_rejected():
    with pytest.raises(ValueError):
        RoutePlanner(objective="fastest")


# ============================================= engine integration =======
def test_engine_compresses_payload_reaching_provider():
    seen = {}
    k, _ = kernel(OptimizationEngine(compress=True, tier="structural"),
                  fn=lambda p: seen.setdefault("p", p) or "ok")
    k.process(KernelRequest(payload=VERBOSE))
    assert len(seen["p"]) < len(VERBOSE)
    assert "kindly" not in seen["p"] or len(seen["p"]) < len(VERBOSE)


def test_engine_sets_model_from_route_plan():
    k, ex = kernel(OptimizationEngine(route=True, objective="quality",
                                      cheap_model="c", capable_model="C"),
                   model="default")
    # register the capable model so execution can find it
    ex.register_adapter(CallableAdapter(lambda p: "cap", provider="p"),
                        models=["C"], default=False)
    resp = k.process(KernelRequest(payload="prove a theorem"))
    assert resp.metadata["route_model"] == "C"
    assert resp.metadata["model"] == "C"


def test_engine_records_content_blind_optimization_events():
    k, _ = kernel(OptimizationEngine(compress=True, tier="structural"))
    resp = k.process(KernelRequest(payload=VERBOSE))
    events = resp.metadata.get("optimization_events", [])
    assert any(e["kind"] == "compression" and e["tokens_saved"] > 0
               for e in events)
    import json
    blob = json.dumps(events)
    assert "kindly" not in blob and "button" not in blob   # content-blind


def test_compression_refreshes_fingerprint_to_sent_payload():
    """The trust fingerprint must hash the payload the PROVIDER sees (post-
    compression), not the original."""
    from tokeymeter.runtime.engines import TrustEngine
    trust = TrustEngine()
    seen = {}
    k, _ = kernel(trust, OptimizationEngine(compress=True, tier="structural"),
                  fn=lambda p: seen.setdefault("p", p) or "ok")
    k.process(KernelRequest(payload=VERBOSE))
    sent_fp = hashlib.sha256(seen["p"].encode()).hexdigest()
    orig_fp = hashlib.sha256(VERBOSE.encode()).hexdigest()
    body = trust.entries()[0]["body"]
    assert sent_fp in body and orig_fp not in body


def test_optimization_only_sees_screened_text():
    """Security runs before optimization: a secret is blocked and never
    reaches the optimizer at all (adapter + optimizer both untouched)."""
    hit = {"opt": 0}

    class SpyOpt(OptimizationEngine):
        def before_request(self, ctx):
            hit["opt"] += 1
            super().before_request(ctx)

    k, _ = kernel(SecurityEngine(secrets_mode="block"),
                  SpyOpt(compress=True))
    with pytest.raises(Exception):
        k.process(KernelRequest(payload=f"deploy {ANTHROPIC_KEY}"))
    assert hit["opt"] == 0                              # never ran


def test_tiny_prompt_not_compressed():
    seen = {}
    k, _ = kernel(OptimizationEngine(compress=True, min_tokens_to_compress=100),
                  fn=lambda p: seen.setdefault("p", p) or "ok")
    k.process(KernelRequest(payload="short prompt here"))
    assert seen["p"] == "short prompt here"            # below threshold
    # (no compression_method churn beyond 'skipped')


def test_disabled_optimization_is_noop_passthrough():
    seen = {}
    k, _ = kernel(OptimizationEngine(compress=False, route=False),
                  fn=lambda p: seen.setdefault("p", p) or "ok")
    k.process(KernelRequest(payload=VERBOSE))
    assert seen["p"] == VERBOSE


# ============================================= facade integration =======
def test_facade_wires_optimization_and_reports_savings():
    seen = {}
    r = Runtime(
        config={"optimization": {"enabled": True, "compress": True,
                                 "tier": "structural"}},
        call=lambda p: seen.setdefault("p", p) or "ok", receipt="never")
    r.execute(VERBOSE)
    assert len(seen["p"]) < len(VERBOSE)
    assert r.last.metadata["tokens_saved"] > 0


def test_facade_route_only_no_compression():
    r = Runtime(
        config={"optimization": {"enabled": True, "route": True,
                                 "objective": "cost"}},
        call=lambda p: "ok", model="gpt-4o-mini", receipt="never")
    r.execute("what is the capital of france")
    assert "route_reason" in r.last.metadata
