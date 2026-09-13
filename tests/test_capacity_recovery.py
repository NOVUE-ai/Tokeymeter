"""Cluster costs + Capacity Recovery Report (S1).

register_cluster_costs: the enterprise, CFO-auditable pricing path — capital,
power, facility, and staff stated and derived separately. register_selfhost_pricing
stays a thin wrapper over the same token-rate derivation.

capacity_recovery_report: the first sellable artifact. Recovered GPU capacity by
MEASURED mechanism only, with roadmap mechanisms explicitly listed as
not-yet-measured (never fabricated). Every figure derives from the operator's
own measured throughput.

Pinned:
  - component amortization is correct and hand-checkable
  - owned vs leased paths are mutually exclusive and both required-one-of
  - the wrapper equals the manual derivation
  - the report quantifies only shipped mechanisms; not_yet_measured carries NO
    numbers; USD appears only when a rate is supplied; understatement is
    disclosed
"""
import math
import random

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.pricing import (
    register_cluster_costs, register_selfhost_pricing,
    derive_cluster_gpu_hour_rate, derive_selfhost_rate,
)
from tokeymeter.engines.economics.capacity_report import capacity_recovery_report
from tokeymeter.engines.economics.usage import set_reported_usage


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    yield
    # Restore the file-backed default: set_in_memory_savings(True) flips a
    # process-global that other suites (e.g. compression) rely on being off —
    # leaving it on makes their file-backed ledger reads fail. Leave no global
    # state behind.
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.clear_registered_pricing()
    tokeymeter.reset_savings()


# ── cluster cost derivation ─────────────────────────────────────────────

def test_owned_capital_amortization_is_correct():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36,
        hours_per_month=730.0)
    # (240000 / 36) / 8 / 730
    expected = (240000 / 36) / 8 / 730
    assert math.isclose(d["components_per_gpu_hour"]["capital_per_gpu_hour"],
                        round(expected, 6), rel_tol=1e-9)
    assert d["capital_basis"] == "owned_straight_line"


def test_leased_capital_is_correct():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=4, lease_usd_per_month=8000, hours_per_month=730.0)
    expected = 8000 / 4 / 730
    assert math.isclose(d["components_per_gpu_hour"]["capital_per_gpu_hour"],
                        round(expected, 6), rel_tol=1e-9)
    assert d["capital_basis"] == "leased"


def test_power_component_is_kw_times_rate():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=1, lease_usd_per_month=730,  # $1/gpu-hr capital
        power_kw_per_gpu=0.7, power_usd_per_kwh=0.12)
    assert math.isclose(d["components_per_gpu_hour"]["power_per_gpu_hour"],
                        round(0.7 * 0.12, 6), rel_tol=1e-9)


def test_facility_overhead_is_multiplicative_on_base():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=1, lease_usd_per_month=730,   # $1/gpu-hr
        power_kw_per_gpu=1.0, power_usd_per_kwh=1.0,  # +$1/gpu-hr -> base $2
        facility_overhead_factor=1.15)
    # overhead added = base(2.0) * 0.15 = 0.30
    assert math.isclose(d["components_per_gpu_hour"]["facility_overhead_per_gpu_hour"],
                        0.30, rel_tol=1e-9)


def test_staff_spread_across_fleet():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=10, lease_usd_per_month=7300, staff_usd_per_month=14600)
    # 14600 / 10 / 730 = 2.0
    assert math.isclose(d["components_per_gpu_hour"]["staff_per_gpu_hour"],
                        2.0, rel_tol=1e-9)


def test_full_rate_composes_all_components():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=1, lease_usd_per_month=730,          # capital 1.0
        power_kw_per_gpu=1.0, power_usd_per_kwh=1.0,    # power 1.0 -> base 2.0
        facility_overhead_factor=1.5,                   # +1.0 -> 3.0
        staff_usd_per_month=730)                        # +1.0 -> 4.0
    assert math.isclose(d["gpu_hour_rate_usd"], 4.0, rel_tol=1e-9)


def test_displayed_components_sum_exactly_to_stated_rate():
    # A finance analyst who adds the displayed component column MUST get the
    # stated gpu_hour_rate_usd — full-precision rounding once left a last-digit
    # mismatch (3.464181 column vs 3.464180 total). Pinned across a fuzz.
    import random
    rng = random.Random(7)
    cases = [dict(gpu_count=8, gpu_capex_usd=240000, depreciation_months=36,
                  power_kw_per_gpu=0.7, power_usd_per_kwh=0.12,
                  facility_overhead_factor=1.15, staff_usd_per_month=12000)]
    for _ in range(200):
        cases.append(dict(
            gpu_count=rng.randint(1, 5000),
            gpu_capex_usd=rng.uniform(1000, 5e8),
            depreciation_months=rng.randint(1, 72),
            power_kw_per_gpu=rng.uniform(0.1, 1.5),
            power_usd_per_kwh=rng.uniform(0.03, 0.4),
            facility_overhead_factor=rng.uniform(1.0, 2.0),
            staff_usd_per_month=rng.uniform(0, 1e6)))
    for c in cases:
        d = derive_cluster_gpu_hour_rate(**c)
        col = sum(d["components_per_gpu_hour"].values())
        assert abs(col - d["gpu_hour_rate_usd"]) < 1e-9, (c, col, d["gpu_hour_rate_usd"])


def test_full_precision_rate_preserved_and_used_for_registration():
    d = derive_cluster_gpu_hour_rate(
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36,
        power_kw_per_gpu=0.7, power_usd_per_kwh=0.12,
        facility_overhead_factor=1.15, staff_usd_per_month=12000)
    assert "gpu_hour_rate_usd_full_precision" in d
    assert d["gpu_hour_rate_usd_full_precision"] > 0


# ── validation ──────────────────────────────────────────────────────────

def test_owned_and_leased_are_mutually_exclusive():
    with pytest.raises(ValueError):
        derive_cluster_gpu_hour_rate(
            gpu_count=1, gpu_capex_usd=1000, depreciation_months=12,
            lease_usd_per_month=100)


def test_capital_path_is_required():
    with pytest.raises(ValueError):
        derive_cluster_gpu_hour_rate(gpu_count=1)


def test_power_requires_both_inputs():
    with pytest.raises(ValueError):
        derive_cluster_gpu_hour_rate(
            gpu_count=1, lease_usd_per_month=730, power_kw_per_gpu=0.7)


def test_facility_factor_below_one_rejected():
    with pytest.raises(ValueError):
        derive_cluster_gpu_hour_rate(
            gpu_count=1, lease_usd_per_month=730, facility_overhead_factor=0.9)


def test_gpu_count_must_be_positive():
    with pytest.raises(ValueError):
        derive_cluster_gpu_hour_rate(gpu_count=0, lease_usd_per_month=730)


# ── register_cluster_costs end-to-end + wrapper equivalence ─────────────

def test_register_cluster_costs_registers_and_derives():
    r = register_cluster_costs(
        "kimi-k2", measured_tokens_per_second=1400,
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36)
    assert r["model"] == "kimi-k2"
    assert "cluster_costs" in r and "token_rate" in r
    # the registered rate should equal the derived per-1M rate
    assert math.isclose(r["registered"]["input"],
                        r["token_rate"]["usd_per_1m_tokens"], rel_tol=1e-9)


def test_wrapper_equals_manual_derivation():
    w = register_selfhost_pricing(
        "m", gpu_hour_rate_usd=2.10, measured_tokens_per_second=1400)
    manual = 2.10 / (1400 * 3600) * 1e6
    assert math.isclose(w["usd_per_1m_tokens"], manual, rel_tol=1e-9)


def test_cluster_costs_token_rate_matches_derive_selfhost():
    r = register_cluster_costs(
        "m", measured_tokens_per_second=1400,
        gpu_count=1, lease_usd_per_month=1533)   # ~2.10/gpu-hr
    gpu_rate = r["cluster_costs"]["gpu_hour_rate_usd"]
    direct = derive_selfhost_rate(
        gpu_hour_rate_usd=gpu_rate, measured_tokens_per_second=1400)
    assert math.isclose(r["token_rate"]["usd_per_1m_tokens"],
                        direct["usd_per_1m_tokens"], rel_tol=1e-9)


# ── Capacity Recovery Report ────────────────────────────────────────────

def _run_workload(hits=59, model="m"):
    @tokeymeter.cache(model=model, endpoint="vllm-pool")
    def ask(p):
        set_reported_usage(340, 128)
        return "answer"
    ask("repeated")            # miss
    for _ in range(hits):
        ask("repeated")        # hits
    ask("unique")              # another miss


def test_report_quantifies_only_measured_mechanisms():
    _run_workload()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    names = {m["mechanism"] for m in rep["measured_mechanisms"]}
    assert names == {"exact_cache", "semantic_cache", "single_flight"}
    # exact-cache mechanism recorded the hits
    exact = next(m for m in rep["measured_mechanisms"] if m["mechanism"] == "exact_cache")
    assert exact["hits"] == 59


def test_report_not_yet_measured_carries_no_numbers():
    _run_workload()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    for m in rep["not_yet_measured"]:
        assert "not instrumented" in m["status"]
        assert "gpu_seconds_recovered" not in m
        assert "recovered_tokens" not in m
    names = {m["mechanism"] for m in rep["not_yet_measured"]}
    assert names == {"batching", "off_peak_deferral", "consolidation"}


def test_report_usd_only_when_rate_supplied():
    _run_workload()
    no_usd = capacity_recovery_report(measured_tokens_per_second=1400)
    for m in no_usd["measured_mechanisms"]:
        assert "equivalent_usd" not in m
    assert "equivalent_usd_total" not in no_usd["total_recovered"]

    with_usd = capacity_recovery_report(
        measured_tokens_per_second=1400, gpu_hour_rate_usd=2.10)
    for m in with_usd["measured_mechanisms"]:
        assert "equivalent_usd" in m
    assert "equivalent_usd_total" in with_usd["total_recovered"]


def test_report_gpu_seconds_derives_from_throughput():
    _run_workload()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    total = rep["total_recovered"]
    # gpu_seconds = recovered_tokens / tps
    expected = total["recovered_tokens_total"] / 1400
    assert math.isclose(total["gpu_seconds_recovered_total"], round(expected, 4),
                        rel_tol=1e-6)


def test_report_surfaces_pricing_provenance():
    _run_workload()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    assert "pricing" in rep
    assert "all_priced" in rep["pricing"]


def test_report_no_longer_discloses_understatement_limitation():
    # S1.1 resolved the undercount (hits recover the true miss volume), so the
    # known_limitation disclosure must be GONE — its presence would now be a
    # false disclaimer.
    _run_workload()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    assert "known_limitation" not in rep


def test_report_rejects_nonpositive_throughput():
    with pytest.raises(ValueError):
        capacity_recovery_report(measured_tokens_per_second=0)


def test_report_empty_ledger_is_zero_not_error():
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    assert rep["total_recovered"]["recovered_tokens_total"] == 0
    for m in rep["measured_mechanisms"]:
        assert m["hits"] == 0
        assert m["gpu_seconds_recovered"] == 0.0


# ── sum invariant: mechanism rows MUST add up to the total ──────────────
# A report whose columns don't sum fails the first finance review it meets.
# These pin the largest-remainder apportionment, including the exact cases
# that broke the naive per-mechanism round() (banker's rounding on .5 shares).

def _fake(e, s, f, tokens):
    return {"exact_hits": e, "semantic_hits": s, "single_flight_hits": f,
            "saved_input_tokens": tokens // 2,
            "saved_output_tokens": tokens - tokens // 2, "pricing": {}}


def test_mechanism_tokens_sum_exactly_two_way_odd_total():
    rep = capacity_recovery_report(
        measured_tokens_per_second=1400, savings_report=_fake(1, 0, 1, 101))
    s = sum(m["recovered_tokens"] for m in rep["measured_mechanisms"])
    assert s == rep["total_recovered"]["recovered_tokens_total"] == 101


def test_mechanism_tokens_sum_exactly_three_way():
    rep = capacity_recovery_report(
        measured_tokens_per_second=1400, savings_report=_fake(1, 1, 1, 100))
    s = sum(m["recovered_tokens"] for m in rep["measured_mechanisms"])
    assert s == rep["total_recovered"]["recovered_tokens_total"] == 100


def test_mechanism_tokens_sum_invariant_fuzz():
    import random
    rng = random.Random(1234)
    for _ in range(300):
        e, s, f = rng.randint(0, 300), rng.randint(0, 300), rng.randint(0, 300)
        tokens = rng.randint(0, 5_000_000)
        rep = capacity_recovery_report(
            measured_tokens_per_second=1400,
            savings_report=_fake(e, s, f, tokens))
        total = rep["total_recovered"]["recovered_tokens_total"]
        summed = sum(m["recovered_tokens"] for m in rep["measured_mechanisms"])
        if (e + s + f) > 0:
            assert summed == total
        else:
            assert summed == 0
        # and total gpu-seconds derives from total tokens, not the row sum
        assert abs(rep["total_recovered"]["gpu_seconds_recovered_total"]
                   - round(total / 1400, 4)) < 1e-9


def test_apportionment_is_deterministic():
    a = capacity_recovery_report(
        measured_tokens_per_second=1400, savings_report=_fake(3, 3, 3, 1000))
    b = capacity_recovery_report(
        measured_tokens_per_second=1400, savings_report=_fake(3, 3, 3, 1000))
    assert [m["recovered_tokens"] for m in a["measured_mechanisms"]] == \
           [m["recovered_tokens"] for m in b["measured_mechanisms"]]


# ── nan/inf guards (found in stress pass: nan slips past <=0 / <0 checks) ──

def test_report_rejects_nan_and_inf_throughput():
    fake = _fake(1, 0, 0, 200)
    for bad in (float("nan"), float("inf"), 0, -5):
        with pytest.raises(ValueError):
            capacity_recovery_report(
                measured_tokens_per_second=bad, savings_report=fake)


def test_report_rejects_nan_inf_negative_usd_rate():
    fake = _fake(1, 0, 0, 200)
    for bad in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            capacity_recovery_report(
                measured_tokens_per_second=1400,
                gpu_hour_rate_usd=bad, savings_report=fake)


def test_cluster_costs_reject_nan_inf_components():
    for kw in (
        dict(gpu_count=8, gpu_capex_usd=float("nan"), depreciation_months=36),
        dict(gpu_count=8, gpu_capex_usd=float("inf"), depreciation_months=36),
        dict(gpu_count=8, lease_usd_per_month=1000,
             power_kw_per_gpu=float("inf"), power_usd_per_kwh=0.1),
        dict(gpu_count=8, lease_usd_per_month=1000,
             staff_usd_per_month=float("nan")),
    ):
        with pytest.raises((ValueError, TypeError)):
            derive_cluster_gpu_hour_rate(**kw)


# ── S1.5-2: period-bounded capacity ─────────────────────────────────────
# The close packet placed an ALL-TIME recovery figure beside one month's spend
# and could only warn about it. These pin the fix.

def _cap_rec(ts, hit=True, shadow=False, hit_type="exact", tin=10, tout=5, cost=0.001):
    return {"timestamp": ts, "hit": hit, "shadow": shadow, "hit_type": hit_type,
            "tag": "a", "model": "m", "input_tokens": tin, "output_tokens": tout,
            "estimated_cost": cost, "token_source": "reported",
            "pricing_source": "registered"}


def test_period_covering_everything_reproduces_all_time_exactly():
    """The period path must not change a single shipped number."""
    import tokeymeter.engines.economics.savings as _sv
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()

    @tokeymeter.cache(model="m", tag="t")
    def ask(p):
        set_reported_usage(340, 128)
        return "r"
    for i in range(40):
        ask(f"q{i % 12}")

    all_time = capacity_recovery_report(measured_tokens_per_second=1400,
                                        gpu_hour_rate_usd=3.0)
    recs = list(_sv._tracker._iter_records())
    wide = capacity_recovery_report(measured_tokens_per_second=1400,
                                    gpu_hour_rate_usd=3.0, records=recs,
                                    period_start=0, period_end=9e18)
    assert all_time["measured_mechanisms"] == wide["measured_mechanisms"]
    assert all_time["total_recovered"] == wide["total_recovered"]
    assert all_time["scope"] == "all_time_entire_ledger"
    assert wide["scope"] == "period_bounded"


def test_period_windows_partition_cleanly():
    recs = [_cap_rec(float(i)) for i in range(100)]
    whole = capacity_recovery_report(
        measured_tokens_per_second=1400, records=recs,
        period_start=0, period_end=100)["total_recovered"]["recovered_tokens_total"]
    halves = sum(capacity_recovery_report(
        measured_tokens_per_second=1400, records=recs,
        period_start=a, period_end=b)["total_recovered"]["recovered_tokens_total"]
        for a, b in ((0, 50), (50, 100)))
    assert whole == halves == 1500          # no double count, nothing lost


def test_period_boundary_is_half_open():
    recs = [_cap_rec(float(i)) for i in range(100)]
    one = capacity_recovery_report(measured_tokens_per_second=1400, records=recs,
                                   period_start=50, period_end=51)
    assert one["total_recovered"]["recovered_tokens_total"] == 15   # exactly ts=50


def test_shadow_hits_excluded_and_counted():
    recs = [_cap_rec(1.0), _cap_rec(2.0, shadow=True, hit_type="shadow_exact")]
    rep = capacity_recovery_report(measured_tokens_per_second=1400, records=recs,
                                   period_start=0, period_end=10)
    assert rep["total_recovered"]["recovered_tokens_total"] == 15   # only the real hit
    assert rep["excluded_shadow_hits"] == 1


def test_misses_contribute_no_recovered_capacity():
    recs = [_cap_rec(1.0, hit=False, hit_type=None), _cap_rec(2.0)]
    rep = capacity_recovery_report(measured_tokens_per_second=1400, records=recs,
                                   period_start=0, period_end=10)
    assert rep["total_recovered"]["recovered_tokens_total"] == 15


def test_corrupt_records_excluded_whole_and_counted():
    recs = [_cap_rec(1.0),
            _cap_rec(2.0, tin=float("nan")),
            _cap_rec(3.0, cost=float("inf")),
            _cap_rec(4.0, tout=-5)]
    rep = capacity_recovery_report(measured_tokens_per_second=1400, records=recs,
                                   period_start=0, period_end=10)
    assert rep["total_recovered"]["recovered_tokens_total"] == 15   # only the clean one
    assert rep["excluded_malformed_records"] == 3


def test_exclusion_parity_with_chargeback_fuzz():
    """Capacity must drop EXACTLY what chargeback drops, or the close packet
    compares two different populations."""
    rng = random.Random(4242)
    for _ in range(120):
        recs = []
        for _ in range(rng.randint(0, 150)):
            hit = rng.random() < 0.5
            r = _cap_rec(rng.uniform(0, 1000), hit=hit,
                         shadow=rng.random() < 0.18,
                         hit_type=rng.choice(["exact", "semantic",
                                              "single_flight", None]) if hit else None,
                         tin=rng.randint(0, 3000), tout=rng.randint(0, 3000),
                         cost=rng.uniform(0, 1.5))
            if rng.random() < 0.22:
                r[rng.choice(["estimated_cost", "input_tokens",
                              "output_tokens", "timestamp"])] = rng.choice(
                    [float("nan"), float("inf"), -1.0, "xyz", None])
            recs.append(r)
        cb = tokeymeter.chargeback_report(group_by=("tag",), records=recs,
                                          period_start=0, period_end=1000)
        cap = capacity_recovery_report(measured_tokens_per_second=1400,
                                       records=recs, period_start=0, period_end=1000)
        assert cb["excluded_malformed_records"] == cap["excluded_malformed_records"]
        assert cb["excluded_shadow_hits"] == cap["excluded_shadow_hits"]
        # same population counted two ways
        assert cb["totals"]["avoided_tokens"] == \
            cap["total_recovered"]["recovered_tokens_total"]


def test_savings_report_cannot_be_combined_with_period():
    """A pre-aggregate has no timestamps; accepting both would return all-time
    numbers under a period label — the exact mislabelling this fixes."""
    with pytest.raises(ValueError):
        capacity_recovery_report(measured_tokens_per_second=1400,
                                 savings_report={"exact_hits": 1},
                                 period_start=0, period_end=10)
    with pytest.raises(ValueError):
        capacity_recovery_report(measured_tokens_per_second=1400,
                                 savings_report={"exact_hits": 1}, records=[])


@pytest.mark.parametrize("kw", [
    {"period_start": "nope"}, {"period_end": float("nan")},
    {"period_start": float("inf")},
    {"period_start": 100, "period_end": 100},      # not strictly before
    {"period_start": 200, "period_end": 100},      # inverted
])
def test_invalid_period_bounds_rejected(kw):
    with pytest.raises(ValueError):
        capacity_recovery_report(measured_tokens_per_second=1400, records=[], **kw)


def test_all_time_path_flags_its_scope_in_the_note():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    rep = capacity_recovery_report(measured_tokens_per_second=1400)
    assert rep["scope"] == "all_time_entire_ledger"
    assert "ENTIRE ledger" in rep["scope_note"]
    assert "excluded_malformed_records" not in rep     # not a period report
