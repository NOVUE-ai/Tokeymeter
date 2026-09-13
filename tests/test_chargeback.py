"""Unit-Cost & Chargeback Ledger (S2).

The monthly-close artifact: spend allocated to teams/models/endpoints for a
period, exportable to CSV. Finance runs its close on this, so the invariants
are non-negotiable and pinned hard:

  RECONCILIATION: sum(row spend) == totals.spend, always (the column an
    analyst adds equals the stated total).
  NO RECORD LOST: executed + avoided + excluded_shadow_hits == total records
    in the period. A chargeback that silently drops records is untrustworthy.
  SPEND = EXECUTED: cache hits are credit (avoided_*), never spend; shadow
    real calls ARE spend; shadow hits are excluded entirely.
  CSV SAFETY: formula-injection vectors in user-supplied tags are neutralized.
  PROVENANCE PER ROW: reported vs estimated token mix and pricing-source flags
    travel with each row and the export.
  VALIDATION: unknown dimensions, nan/inf periods, inverted periods all raise.
"""
import math
import random

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution.endpoint import endpoint as endpoint_ctx
from tokeymeter.engines.economics.chargeback import (
    chargeback_report, chargeback_csv, _csv_safe, ALLOWED_DIMENSIONS,
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
    base = {"timestamp": 100.0, "hit": False, "shadow": False, "tag": "team",
            "model": "m", "endpoint_identity": "e", "principal": "p",
            "key_name": "k", "input_tokens": 100, "output_tokens": 50,
            "estimated_cost": 0.01, "token_source": "reported",
            "pricing_source": "registered"}
    base.update(kw)
    return base


# ── allocation basics ───────────────────────────────────────────────────

def test_spend_allocated_per_team():
    recs = [_rec(tag="support", estimated_cost=0.10),
            _rec(tag="support", estimated_cost=0.20),
            _rec(tag="eng", estimated_cost=0.05)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    by = {r["tag"]: r["spend_usd"] for r in rep["rows"]}
    assert by["support"] == 0.30
    assert by["eng"] == 0.05


def test_multi_dimension_grouping():
    recs = [_rec(tag="support", model="kimi", estimated_cost=0.10),
            _rec(tag="support", model="llama", estimated_cost=0.20)]
    rep = chargeback_report(group_by=("tag", "model"), records=recs)
    assert len(rep["rows"]) == 2
    assert {(r["tag"], r["model"]) for r in rep["rows"]} == {
        ("support", "kimi"), ("support", "llama")}


def test_missing_dimension_becomes_unattributed_not_dropped():
    recs = [_rec(tag=None, estimated_cost=0.10),
            _rec(tag="", estimated_cost=0.20),
            _rec(tag="eng", estimated_cost=0.05)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    by = {r["tag"]: r["spend_usd"] for r in rep["rows"]}
    assert by["(unattributed)"] == 0.30    # None and "" merge, nothing lost
    assert by["eng"] == 0.05
    assert rep["totals"]["spend_usd"] == 0.35


# ── spend vs credit semantics ───────────────────────────────────────────

def test_cache_hits_are_credit_not_spend():
    recs = [_rec(hit=False, estimated_cost=0.10),        # executed -> spend
            _rec(hit=True, estimated_cost=0.10)]         # hit -> avoided credit
    rep = chargeback_report(group_by=("tag",), records=recs)
    row = rep["rows"][0]
    assert row["spend_usd"] == 0.10             # only the miss
    assert row["executed_requests"] == 1
    assert row["avoided_requests"] == 1
    assert row["avoided_usd"] == 0.10


def test_shadow_real_call_is_spend():
    # shadow measures but the wrapped call STILL executed and cost money
    recs = [_rec(hit=False, shadow=True, estimated_cost=0.10)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    assert rep["rows"][0]["spend_usd"] == 0.10
    assert rep["excluded_shadow_hits"] == 0


def test_shadow_hits_excluded_entirely():
    recs = [_rec(hit=True, shadow=True, estimated_cost=0.10),
            _rec(hit=False, estimated_cost=0.05)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    assert rep["excluded_shadow_hits"] == 1
    assert rep["totals"]["spend_usd"] == 0.05
    assert rep["totals"]["avoided_requests"] == 0   # shadow hit is NOT avoided


# ── reconciliation + no record loss (the finance guarantees) ────────────

def test_totals_reconcile_to_rows_fuzz():
    rng = random.Random(99)
    for _ in range(200):
        n = rng.randint(0, 300)
        recs = [_rec(
            timestamp=rng.uniform(0, 1000),
            hit=rng.random() < 0.5, shadow=rng.random() < 0.2,
            tag=rng.choice(["support", "eng", "ops", None, ""]),
            model=rng.choice(["kimi", "llama"]),
            endpoint_identity=rng.choice(["a100", "h100", None]),
            estimated_cost=rng.uniform(0, 2.0),
            token_source=rng.choice(["reported", "estimated"]),
            pricing_source=rng.choice(["registered", "list", "fallback"]),
        ) for _ in range(n)]
        rep = chargeback_report(
            group_by=("tag", "model", "endpoint_identity"), records=recs)
        # column reconciles
        assert abs(sum(r["spend_usd"] for r in rep["rows"])
                   - rep["totals"]["spend_usd"]) < 1e-9
        # billable count exact
        assert rep["totals"]["executed_requests"] == sum(
            1 for r in recs if not r["hit"])
        # NOTHING dropped
        accounted = (rep["totals"]["executed_requests"]
                     + rep["totals"]["avoided_requests"]
                     + rep["excluded_shadow_hits"])
        assert accounted == len(recs)


# ── period filtering ────────────────────────────────────────────────────

def test_period_is_half_open():
    recs = [_rec(timestamp=t, estimated_cost=1.0) for t in (10, 20, 30, 40)]
    rep = chargeback_report(group_by=("tag",), records=recs,
                            period_start=20, period_end=40)
    assert rep["totals"]["executed_requests"] == 2   # 20,30 ; not 10 or 40


def test_period_none_includes_all():
    recs = [_rec(timestamp=t, estimated_cost=1.0) for t in (10, 20, 30)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    assert rep["totals"]["executed_requests"] == 3


# ── validation ──────────────────────────────────────────────────────────

def test_unknown_dimension_raises():
    with pytest.raises(ValueError):
        chargeback_report(group_by=("nonsense",), records=[])


def test_empty_group_by_raises():
    with pytest.raises(ValueError):
        chargeback_report(group_by=(), records=[])


def test_nan_inf_period_raises():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            chargeback_report(group_by=("tag",), records=[], period_start=bad)


def test_inverted_period_raises():
    with pytest.raises(ValueError):
        chargeback_report(group_by=("tag",), records=[],
                          period_start=100, period_end=50)


def test_allowed_dimensions_match_record_fields():
    import dataclasses
    from tokeymeter.engines.economics.savings import CallRecord
    fields = {f.name for f in dataclasses.fields(CallRecord)}
    for d in ALLOWED_DIMENSIONS:
        assert d in fields, f"{d} is not a CallRecord field"


# ── provenance honesty ──────────────────────────────────────────────────

def test_row_provenance_tracks_reported_vs_estimated():
    recs = [_rec(token_source="reported", input_tokens=100, output_tokens=0),
            _rec(token_source="estimated", input_tokens=50, output_tokens=0)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    prov = rep["rows"][0]["provenance"]
    assert prov["reported_tokens"] == 100
    assert prov["estimated_tokens"] == 50


def test_fallback_pricing_flagged_not_fully_priced():
    recs = [_rec(pricing_source="fallback")]
    rep = chargeback_report(group_by=("tag",), records=recs)
    assert rep["rows"][0]["provenance"]["fully_priced"] is False
    assert rep["provenance"]["all_rows_fully_priced"] is False


def test_registered_pricing_is_fully_priced():
    recs = [_rec(pricing_source="registered")]
    rep = chargeback_report(group_by=("tag",), records=recs)
    assert rep["rows"][0]["provenance"]["fully_priced"] is True


# ── CSV export + injection defense ──────────────────────────────────────

@pytest.mark.parametrize("attack", [
    "=1+1", "+1+1", "-1+1", "@SUM(A1)",
    '=HYPERLINK("http://evil","x")', '=cmd|"/c calc"!A1', "\t=1", "\r=1",
])
def test_csv_neutralizes_formula_injection(attack):
    out = _csv_safe(attack)
    # neutralized: leading quote (bare) or a quote just inside an RFC-quote
    assert out.startswith("'") or out.startswith('"\'')


def test_csv_full_pipeline_attack_tag_is_safe():
    recs = [_rec(tag='=HYPERLINK("http://x")')]
    csv = chargeback_csv(chargeback_report(group_by=("tag",), records=recs))
    attack_line = [l for l in csv.splitlines() if "HYPERLINK" in l][0]
    assert not attack_line.startswith("=")   # never a bare formula at line start


def test_csv_total_row_reconciles():
    recs = [_rec(tag="a", estimated_cost=0.10),
            _rec(tag="b", estimated_cost=0.20)]
    rep = chargeback_report(group_by=("tag",), records=recs)
    csv = chargeback_csv(rep)
    lines = [l for l in csv.splitlines() if l.strip()]
    total_line = [l for l in lines if l.startswith("TOTAL")][0]
    # TOTAL spend cell equals sum of the two rows
    assert "0.3" in total_line


def test_csv_rfc4180_escaping():
    assert _csv_safe("normal,tag") == '"normal,tag"'
    assert _csv_safe('has"quote') == '"has""quote"'


# ── end-to-end through the real decorator ───────────────────────────────

def test_end_to_end_two_team_chargeback():
    tokeymeter.register_cluster_costs(
        "kimi", measured_tokens_per_second=1400,
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36)

    def team(name):
        @tokeymeter.cache(model="kimi", tag=name, namespace=f"app-{name}")
        def ask(p):
            set_reported_usage(340, 128)
            return "r"
        return ask

    support, eng = team("support"), team("eng")
    with endpoint_ctx("vllm-a100"):
        for i in range(30):
            support(f"s{i % 10}")   # 10 executed, 20 cache hits
        for i in range(10):
            eng(f"e{i}")            # 10 executed, 0 hits

    rep = chargeback_report(group_by=("tag",))
    by = {r["tag"]: r for r in rep["rows"]}
    # both teams executed 10 real calls -> equal spend; support has cache credit
    assert by["support"]["executed_requests"] == 10
    assert by["eng"]["executed_requests"] == 10
    assert by["support"]["avoided_requests"] == 20
    assert by["eng"]["avoided_requests"] == 0
    # reconciliation holds end-to-end
    assert abs(sum(r["spend_usd"] for r in rep["rows"])
               - rep["totals"]["spend_usd"]) < 1e-9
    # spend is real dollars from the cluster-derived rate
    assert by["support"]["spend_usd"] > 0
    assert by["support"]["provenance"]["fully_priced"] is True


# ── hostile data: a deployed ledger will contain damaged lines ──────────
# Bit-flips, partial writes that still parse, buggy external producers. A
# corrupt record must never crash the close, never poison a total, and never
# vanish silently — it is excluded WHOLE and counted.

@pytest.mark.parametrize("bad", [
    {"estimated_cost": float("nan")},
    {"estimated_cost": float("inf")},
    {"estimated_cost": -5.0},          # negative cost would REDUCE the bill
    {"input_tokens": float("nan")},
    {"input_tokens": "abc"},
    {"output_tokens": -100},
    {"input_tokens": None, "output_tokens": "x"},
])
def test_corrupt_record_excluded_not_poisoning(bad):
    good = _rec(estimated_cost=0.01)
    rep = chargeback_report(group_by=("model",),
                            records=[good, _rec(**bad), good])
    assert rep["totals"]["spend_usd"] == 0.02        # good records intact
    assert rep["excluded_malformed_records"] == 1    # counted, not silent
    assert math.isfinite(rep["totals"]["spend_usd"])


def test_corrupt_timestamp_excluded_when_period_bounded():
    good = _rec(timestamp=5.0, estimated_cost=0.01)
    rep = chargeback_report(group_by=("model",),
                            records=[good, _rec(timestamp="nope"), good],
                            period_start=0, period_end=100)
    assert rep["totals"]["spend_usd"] == 0.02
    assert rep["excluded_malformed_records"] == 1


def test_non_string_dimension_retained_not_dropped():
    # a numeric model id is VALID data — dropping it would lose money data;
    # it is keyed as its string form and must sort/render without crashing
    rep = chargeback_report(
        group_by=("model",),
        records=[_rec(estimated_cost=0.01), _rec(model=123, estimated_cost=0.01)])
    assert rep["excluded_malformed_records"] == 0
    assert any(r["model"] == "123" for r in rep["rows"])
    assert "123" in chargeback_csv(rep)


def test_hostile_records_preserve_reconciliation():
    import random
    rng = random.Random(777)
    for _ in range(50):
        recs = []
        for _ in range(rng.randint(0, 120)):
            r = _rec(timestamp=rng.uniform(0, 1000),
                     hit=rng.random() < 0.4, shadow=rng.random() < 0.15,
                     model=rng.choice(["m1", "m2"]),
                     estimated_cost=rng.uniform(0, 1.5))
            if rng.random() < 0.25:
                r = dict(r, estimated_cost=rng.choice(
                    [float("nan"), float("inf"), -1.0]))
            recs.append(r)
        rep = chargeback_report(group_by=("model",), records=recs)
        assert math.isfinite(rep["totals"]["spend_usd"])
        assert rep["totals"]["spend_usd"] >= 0
        # column still reconciles despite exclusions
        assert abs(sum(r["spend_usd"] for r in rep["rows"])
                   - rep["totals"]["spend_usd"]) < 1e-9
