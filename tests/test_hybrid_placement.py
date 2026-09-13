"""Hybrid Placement Intelligence (S3) — the crown jewel.

Build-vs-buy economics from one ledger: booked self-host cost vs API-equivalent
cost for the SAME executed volume, with the utilization flip threshold. The
board sees this next to the chargeback statement, so the invariants are hard:

  FLIP MATH RECOMPUTABLE: every ratio/threshold derives from displayed numbers
    with the printed formula; verified by hand.
  CROSS-REPORT CONSISTENCY: S3 executed volume == S2 executed volume from the
    same ledger (two board artifacts must not disagree).
  NO FABRICATED PRICES: registry default/fallback rates are treated as UNPRICED
    and excluded from verdicts — a board decision never rests on a generic guess.
  UTILIZATION HONESTY: utilization > 1.0 (served exceeds declared capacity) is
    reported AS IS with an inconsistency flag, never clamped.
  EVIDENCE ONLY: no placement action; no fabricated quality metric.
  VALIDATION: nan/inf/zero/non-integer cluster inputs raise.
"""
import math
import random

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution.endpoint import endpoint as endpoint_ctx
from tokeymeter.engines.economics.chargeback import chargeback_report
from tokeymeter.engines.economics.hybrid import (
    hybrid_placement_report, hybrid_placement_csv, _api_price_per_1m,
)


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _rec(**kw):
    base = {"timestamp": 100.0, "hit": False, "shadow": False, "model": "m",
            "tag": "t", "endpoint_identity": "e", "principal": "p", "key_name": "k",
            "input_tokens": 1000, "output_tokens": 1000, "estimated_cost": 0.01,
            "token_source": "reported", "pricing_source": "registered",
            "latency_ms": 100.0, "queue_wait_ms": None}
    base.update(kw)
    return base


# ── flip-threshold math (hand-recomputable) ─────────────────────────────

def test_selfhost_full_util_rate_formula():
    # rate=3.6/gpu-hr, tps=1000 -> 3.6/(1000*3600)*1e6 = 1.0
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=3.6, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=[])
    assert math.isclose(
        rep["cluster"]["selfhost_usd_per_1m_at_full_utilization"], 1.0, rel_tol=1e-9)


def test_flip_and_verdict_hand_recompute():
    tokeymeter.register_pricing("api", input_per_1m=1.0, output_per_1m=2.0)
    recs = [_rec(input_tokens=1000, output_tokens=1000, estimated_cost=0.005)]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=3.6, per_gpu_tokens_per_second=1000,
        alternatives=("api",), group_by=("model",), records=recs)
    a = rep["rows"][0]["alternatives"][0]
    # api_equivalent = (1000*1.0 + 1000*2.0)/1e6 = 0.003
    assert math.isclose(a["api_equivalent_usd"], 0.003, rel_tol=1e-9)
    # blended @50/50 of (1,2) = 1.5 ; flip = selfhost_full(1.0)/1.5
    assert math.isclose(a["api_blended_usd_per_1m"], 1.5, rel_tol=1e-9)
    assert math.isclose(a["flip_utilization"], 1.0 / 1.5, rel_tol=1e-6)
    # booked 0.005 > api 0.003 -> api cheaper at current booked cost
    assert a["verdict_at_booked_cost"] == "api_cheaper"


def test_verdict_selfhost_cheaper_when_booked_below_api():
    tokeymeter.register_pricing("api-big", input_per_1m=5.0, output_per_1m=15.0)
    recs = [_rec(input_tokens=1000, output_tokens=1000, estimated_cost=0.001)]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=3.6, per_gpu_tokens_per_second=1000,
        alternatives=("api-big",), group_by=("model",), records=recs)
    a = rep["rows"][0]["alternatives"][0]
    assert a["verdict_at_booked_cost"] == "selfhost_cheaper"


# ── no fabricated prices ────────────────────────────────────────────────

def test_unknown_model_is_unpriced_not_verdicted():
    r = _api_price_per_1m("never-registered-xyz")
    assert r[0] is None and "unpriced" in r[2]


def test_default_and_fallback_sources_treated_as_unpriced():
    # registry returns a generic default for unknowns; must NOT be verdicted
    recs = [_rec()]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=("unknown-model",), group_by=("model",), records=recs)
    alt = rep["rows"][0]["alternatives"][0]
    assert "unpriced" in alt["pricing_source"]
    assert "api_equivalent_usd" not in alt   # no verdict, no fabricated number


def test_registered_alternative_is_priced():
    tokeymeter.register_pricing("real-api", input_per_1m=0.5, output_per_1m=1.5)
    recs = [_rec()]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=("real-api",), group_by=("model",), records=recs)
    alt = rep["rows"][0]["alternatives"][0]
    assert alt["pricing_source"] == "registered"
    assert "api_equivalent_usd" in alt


# ── cross-report consistency with S2 (the crown-jewel invariant) ────────

def test_executed_volume_reconciles_with_chargeback_fuzz():
    rng = random.Random(2024)
    for _ in range(150):
        n = rng.randint(0, 250)
        recs = [_rec(
            timestamp=rng.uniform(0, 1000),
            hit=rng.random() < 0.5, shadow=rng.random() < 0.2,
            model=rng.choice(["m1", "m2"]),
            input_tokens=rng.randint(0, 3000), output_tokens=rng.randint(0, 3000),
            estimated_cost=rng.uniform(0, 1.5),
            token_source=rng.choice(["reported", "estimated"]),
        ) for _ in range(n)]
        cb = chargeback_report(group_by=("model",), records=recs)
        hp = hybrid_placement_report(
            gpu_count=4, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
            alternatives=(), group_by=("model",), records=recs)
        assert cb["totals"]["executed_requests"] == hp["totals"]["executed_requests"]
        assert cb["totals"]["executed_input_tokens"] == hp["totals"]["executed_input_tokens"]
        assert cb["totals"]["executed_output_tokens"] == hp["totals"]["executed_output_tokens"]
        assert abs(cb["totals"]["spend_usd"] - hp["totals"]["selfhost_booked_usd"]) < 1e-9
        assert cb["excluded_shadow_hits"] == hp["excluded_shadow_hits"]


def test_cache_hits_and_shadow_hits_excluded_from_placement():
    recs = [_rec(hit=False, estimated_cost=0.10),           # executed -> counts
            _rec(hit=True, estimated_cost=0.10),            # cache hit -> excluded
            _rec(hit=True, shadow=True, estimated_cost=0.10)]  # shadow hit -> excluded
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=recs)
    assert rep["totals"]["executed_requests"] == 1
    assert rep["excluded_shadow_hits"] == 1


# ── utilization honesty ─────────────────────────────────────────────────

def test_utilization_over_one_reported_not_clamped():
    # capacity = 1 tok/s * 1 gpu * 10s = 10 ; served = 200000
    recs = [_rec(timestamp=5.0, input_tokens=100000, output_tokens=100000)]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=1.0, per_gpu_tokens_per_second=1,
        alternatives=(), group_by=("model",), records=recs,
        period_start=0, period_end=10)
    assert rep["fleet_utilization"] > 1.0          # NOT clamped to 1.0
    assert "utilization > 1.0" in rep["utilization_note"]


def test_normal_utilization_computed():
    recs = [_rec(timestamp=5.0, input_tokens=500, output_tokens=500)]
    rep = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=recs,
        period_start=0, period_end=100)
    # capacity = 1000*8*100 = 800000 ; served=1000 -> 0.00125
    assert math.isclose(rep["fleet_utilization"], 0.00125, rel_tol=1e-6)
    assert rep["utilization_note"] is None


def test_utilization_none_without_bounded_period():
    recs = [_rec()]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=recs)
    assert rep["fleet_utilization"] is None
    assert "bounded period" in rep["utilization_note"]


# ── validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {"gpu_count": 0}, {"gpu_count": -1}, {"gpu_count": 2.5},
    {"gpu_count": float("nan")}, {"gpu_hour_rate_usd": float("inf")},
    {"gpu_hour_rate_usd": -1}, {"gpu_hour_rate_usd": 0},
    {"per_gpu_tokens_per_second": 0}, {"per_gpu_tokens_per_second": float("nan")},
])
def test_bad_cluster_inputs_raise(kw):
    base = dict(gpu_count=1, gpu_hour_rate_usd=1.0, per_gpu_tokens_per_second=1000,
                alternatives=(), group_by=("model",), records=[])
    base.update(kw)
    with pytest.raises(ValueError):
        hybrid_placement_report(**base)


def test_unknown_group_by_raises():
    with pytest.raises(ValueError):
        hybrid_placement_report(
            gpu_count=1, gpu_hour_rate_usd=1.0, per_gpu_tokens_per_second=1000,
            group_by=("bogus",), records=[])


def test_missing_dimension_becomes_unattributed():
    recs = [_rec(model=None, estimated_cost=0.10),
            _rec(model="known", estimated_cost=0.05)]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=recs)
    models = {r["model"] for r in rep["rows"]}
    assert "(unattributed)" in models and "known" in models


# ── CSV ─────────────────────────────────────────────────────────────────

def test_csv_injection_defense_in_dimension():
    recs = [_rec(model='=HYPERLINK("http://x")')]
    csv = hybrid_placement_csv(hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=recs))
    attack = [l for l in csv.splitlines() if "HYPERLINK" in l][0]
    assert not attack.startswith("=")


def test_csv_has_row_per_alternative():
    tokeymeter.register_pricing("a1", input_per_1m=1.0, output_per_1m=2.0)
    tokeymeter.register_pricing("a2", input_per_1m=3.0, output_per_1m=4.0)
    recs = [_rec()]
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=("a1", "a2"), group_by=("model",), records=recs)
    csv = hybrid_placement_csv(rep)
    lines = [l for l in csv.splitlines() if l.strip()]
    # header + one row per (group x alternative) = 1 + 2
    assert len(lines) == 3


# ── end-to-end through the real decorator ───────────────────────────────

def test_end_to_end_build_vs_buy():
    cc = tokeymeter.register_cluster_costs(
        "kimi-sh", measured_tokens_per_second=175,
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36)
    gpu_rate = cc["cluster_costs"]["gpu_hour_rate_usd_full_precision"]
    tokeymeter.register_pricing("gpt-4o-mini", input_per_1m=0.15, output_per_1m=0.60)

    @tokeymeter.cache(model="kimi-sh", tag="support")
    def ask(p):
        set_reported_usage(340, 128)
        return "r"
    with endpoint_ctx("vllm-a100"):
        for i in range(50):
            ask(f"q{i}")            # 50 executed
        for i in range(50):
            ask(f"q{i}")            # 50 hits (excluded)

    rep = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=gpu_rate, per_gpu_tokens_per_second=175,
        alternatives=("gpt-4o-mini",), group_by=("model",))
    row = rep["rows"][0]
    assert row["executed_requests"] == 50        # hits excluded
    assert row["selfhost_booked_usd"] > 0
    alt = row["alternatives"][0]
    assert alt["pricing_source"] == "registered"
    assert "flip_utilization" in alt
    assert alt["verdict_at_booked_cost"] in ("selfhost_cheaper", "api_cheaper")


# ── mixed estate: self-hosted AND API-served in one ledger ──────────────
# The real buyer's ledger holds both. API spend must NEVER be labeled as
# self-host booked cost.

def _mixed_recs():
    return [
        {"timestamp": 5.0, "hit": False, "shadow": False, "model": "kimi-sh",
         "input_tokens": 1000, "output_tokens": 500, "estimated_cost": 0.004,
         "token_source": "reported", "pricing_source": "registered"},
        {"timestamp": 6.0, "hit": False, "shadow": False, "model": "gpt-api",
         "input_tokens": 1000, "output_tokens": 500, "estimated_cost": 0.00045,
         "token_source": "reported", "pricing_source": "list"},
    ]


def test_declared_selfhosted_separates_api_spend():
    rep = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=3.0, per_gpu_tokens_per_second=175,
        alternatives=(), selfhosted_models=("kimi-sh",),
        group_by=("model",), records=_mixed_recs())
    assert rep["estate_mode"] == "declared_selfhosted_models"
    # only the self-hosted model in placement rows
    assert [r["model"] for r in rep["rows"]] == ["kimi-sh"]
    # self-host booked is ONLY the GPU cost, never the API spend
    assert abs(rep["totals"]["selfhost_booked_usd"] - 0.004) < 1e-9
    # API model summarized separately
    api = rep["api_served_workloads"]["rows"]
    assert [r["model"] for r in api] == ["gpt-api"]
    assert abs(rep["api_served_workloads"]["total_api_spend_usd"] - 0.00045) < 1e-9


def test_undeclared_estate_is_backward_compatible():
    rep = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=3.0, per_gpu_tokens_per_second=175,
        alternatives=(), group_by=("model",), records=_mixed_recs())
    # no declaration → all records treated as self-hosted, flagged
    assert rep["estate_mode"] == "all_records_assumed_selfhosted"
    assert len(rep["rows"]) == 2
    assert rep["api_served_workloads"]["rows"] == []


def test_estate_totals_reconcile_to_own_rows():
    rep = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=3.0, per_gpu_tokens_per_second=175,
        alternatives=(), selfhosted_models=("kimi-sh",),
        group_by=("model",), records=_mixed_recs())
    assert abs(rep["totals"]["selfhost_booked_usd"]
               - sum(r["selfhost_booked_usd"] for r in rep["rows"])) < 1e-9
    assert abs(rep["api_served_workloads"]["total_api_spend_usd"]
               - sum(r["api_spend_usd"]
                     for r in rep["api_served_workloads"]["rows"])) < 1e-9
    # whole-estate executed spend = self-host + api
    assert abs(rep["total_executed_spend_usd"] - (0.004 + 0.00045)) < 1e-9


def test_mixed_estate_reconciles_per_model_with_chargeback():
    from tokeymeter.engines.economics.chargeback import chargeback_report
    recs = _mixed_recs()
    cb = chargeback_report(group_by=("model",), records=recs)
    hp = hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=3.0, per_gpu_tokens_per_second=175,
        alternatives=(), selfhosted_models=("kimi-sh",),
        group_by=("model",), records=recs)
    cb_by = {r["model"]: r["spend_usd"] for r in cb["rows"]}
    hp_sh = {r["model"]: r["selfhost_booked_usd"] for r in hp["rows"]}
    hp_api = {r["model"]: r["api_spend_usd"]
              for r in hp["api_served_workloads"]["rows"]}
    for m, s2v in cb_by.items():
        s3v = hp_sh.get(m, hp_api.get(m, 0.0))
        assert abs(s2v - s3v) < 1e-9   # per-model exact across the two reports


# ── hostile data: identical exclusion to chargeback ─────────────────────
# The two board artifacts must drop exactly the same damaged records, or the
# per-model reconciliation a CTO relies on silently breaks.

@pytest.mark.parametrize("bad", [
    {"estimated_cost": float("nan")},
    {"estimated_cost": float("inf")},
    {"estimated_cost": -5.0},
    {"input_tokens": float("nan")},
    {"output_tokens": "abc"},
    {"input_tokens": -100},
])
def test_corrupt_record_excluded_not_poisoning(bad):
    good = _rec(estimated_cost=0.01)
    rep = hybrid_placement_report(
        gpu_count=1, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
        alternatives=(), group_by=("model",), records=[good, _rec(**bad), good])
    assert math.isfinite(rep["totals"]["selfhost_booked_usd"])
    assert rep["totals"]["selfhost_booked_usd"] == 0.02
    assert rep["excluded_malformed_records"] == 1


def test_exclusions_identical_to_chargeback_under_corruption():
    import random
    from tokeymeter.engines.economics.chargeback import chargeback_report
    rng = random.Random(777)
    for _ in range(50):
        recs = []
        for _ in range(rng.randint(0, 120)):
            r = _rec(timestamp=rng.uniform(0, 1000),
                     hit=rng.random() < 0.4, shadow=rng.random() < 0.15,
                     model=rng.choice(["kimi-sh", "gpt-api"]),
                     estimated_cost=rng.uniform(0, 1.5))
            if rng.random() < 0.25:
                r = dict(r, estimated_cost=rng.choice(
                    [float("nan"), float("inf"), -1.0]))
            recs.append(r)
        cb = chargeback_report(group_by=("model",), records=recs,
                               period_start=0, period_end=1000)
        hp = hybrid_placement_report(
            gpu_count=4, gpu_hour_rate_usd=2.0, per_gpu_tokens_per_second=1000,
            alternatives=(), selfhosted_models=("kimi-sh",),
            group_by=("model",), records=recs,
            period_start=0, period_end=1000)
        # SAME exclusion decisions
        assert cb["excluded_malformed_records"] == hp["excluded_malformed_records"]
        assert cb["excluded_shadow_hits"] == hp["excluded_shadow_hits"]
        # per-model reconciliation survives the corruption
        cb_by = {r["model"]: r["spend_usd"] for r in cb["rows"]}
        hp_sh = {r["model"]: r["selfhost_booked_usd"] for r in hp["rows"]}
        hp_api = {r["model"]: r["api_spend_usd"]
                  for r in hp["api_served_workloads"]["rows"]}
        for m, v in cb_by.items():
            v3 = hp_sh.get(m)
            v3 = hp_api.get(m, 0.0) if v3 is None else v3
            assert abs(v - v3) < 1e-9
