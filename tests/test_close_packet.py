"""Monthly Close Packet & GL export (Phase S1.5).

The export layer that lets S1–S3 artifacts reach a customer ritual. Pinned:

  THE PACKET IS AN AUDIT, NOT A BUNDLE. It must catch every class of component
    disagreement — tampered totals, drifted estate figures, mismatched periods,
    divergent exclusion decisions — and say DISCREPANCY rather than shipping
    numbers that do not tie. A packet that quietly ships inconsistent artifacts
    launders a defect into a filing.
  LEGITIMATE ROUNDING IS TOLERATED. Two independently-rounded reports can
    differ sub-micro-dollar by rounding order (the documented S3 limit); that
    must NOT trip a discrepancy.
  ACCOUNTS ARE DECLARED, NEVER GUESSED. Unmapped cost centres post to a
    declared suspense account and are flagged; money is never lost by the
    routing.
  GL RECONCILES BY CONSTRUCTION. total_usd is the sum of the displayed rows
    and ties to the chargeback spend total.
  CSV SAFETY. Cost-centre values are operator text landing in Excel.
"""
import copy
import math
import random

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.economics.close_packet import (
    close_packet, general_ledger_rows, general_ledger_csv,
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


def _cb_row(tag, amt, priced=True, reqs=1):
    return {"tag": tag, "spend_usd": amt,
            "executed_requests": reqs,
            "provenance": {"fully_priced": priced, "reported_tokens": 10,
                           "estimated_tokens": 0}}


def _cb(rows, period=(0, 100), shadow=0, malformed=0):
    return {"report": "chargeback", "group_by": ["tag"],
            "period": {"start": period[0], "end": period[1]},
            "rows": rows,
            "totals": {"spend_usd": round(sum(r["spend_usd"] for r in rows), 6),
                       "executed_requests": sum(r["executed_requests"] for r in rows)},
            "provenance": {}, "excluded_shadow_hits": shadow,
            "excluded_malformed_records": malformed}


def _hp(total, period=(0, 100), shadow=0, malformed=0):
    return {"report": "hybrid_placement", "group_by": ["model"],
            "period": {"start": period[0], "end": period[1]},
            "rows": [{"model": "m", "selfhost_booked_usd": total}],
            "totals": {"selfhost_booked_usd": total},
            "total_executed_spend_usd": total,
            "excluded_shadow_hits": shadow, "excluded_malformed_records": malformed}


# ── the packet is an audit ──────────────────────────────────────────────

def test_consistent_components_reconcile():
    cb = _cb([_cb_row("a", 10.0), _cb_row("b", 5.0)])
    pkt = close_packet(chargeback=cb, hybrid=_hp(15.0))
    assert pkt["reconciliation"]["status"] == "reconciled"
    assert all(c["passed"] for c in pkt["reconciliation"]["checks"])


def test_tampered_chargeback_total_caught():
    cb = _cb([_cb_row("a", 10.0)])
    cb["totals"]["spend_usd"] = 99.0
    pkt = close_packet(chargeback=cb, hybrid=_hp(10.0))
    assert pkt["reconciliation"]["status"] == "DISCREPANCY"
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "chargeback_total_equals_its_rows" in failed


def test_drifted_estate_total_caught():
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0)]), hybrid=_hp(10.5))
    assert pkt["reconciliation"]["status"] == "DISCREPANCY"


def test_period_mismatch_caught():
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0)], period=(0, 100)),
                       hybrid=_hp(10.0, period=(0, 200)))
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "components_cover_same_period" in failed


def test_divergent_exclusions_caught():
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0)], shadow=2, malformed=1),
                       hybrid=_hp(10.0, shadow=2, malformed=7))
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "exclusion_decisions_identical" in failed


def test_hybrid_rows_not_summing_caught():
    hp = _hp(10.0)
    hp["totals"]["selfhost_booked_usd"] = 20.0
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0)]), hybrid=hp)
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "hybrid_selfhost_total_equals_its_rows" in failed


def test_sub_tolerance_rounding_is_tolerated():
    # two independently-rounded reports may differ by rounding ORDER; that is
    # the documented limit, not a discrepancy
    hp = _hp(10.0)
    hp["total_executed_spend_usd"] = 10.0 + 2e-6
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0)]), hybrid=hp)
    assert pkt["reconciliation"]["status"] == "reconciled"


def test_exclusions_surfaced_in_packet():
    pkt = close_packet(chargeback=_cb([_cb_row("a", 1.0)], shadow=3, malformed=2))
    assert pkt["exclusions"]["shadow_hits"] == 3
    assert pkt["exclusions"]["malformed_records"] == 2


def test_packet_rejects_non_report_input():
    with pytest.raises(ValueError):
        close_packet(chargeback={"not": "a report"})
    with pytest.raises(ValueError):
        close_packet(chargeback=_cb([_cb_row("a", 1.0)]),
                     hybrid={"report": "chargeback"})


def test_capacity_component_checked():
    cap = {"measured_mechanisms": [{"mechanism": "exact_cache",
                                    "recovered_tokens": 60},
                                   {"mechanism": "single_flight",
                                    "recovered_tokens": 40}],
           "total_recovered": {"recovered_tokens_total": 100}}
    pkt = close_packet(chargeback=_cb([_cb_row("a", 1.0)]), capacity=cap)
    assert pkt["reconciliation"]["status"] == "reconciled"
    cap["total_recovered"]["recovered_tokens_total"] = 101      # now inconsistent
    pkt2 = close_packet(chargeback=_cb([_cb_row("a", 1.0)]), capacity=cap)
    assert pkt2["reconciliation"]["status"] == "DISCREPANCY"


# ── GL export ───────────────────────────────────────────────────────────

def test_declared_accounts_used():
    gl = general_ledger_rows(chargeback=_cb([_cb_row("support", 1.0)]),
                             account_mapping={"support": "6100-AI"})
    assert gl["rows"][0]["account"] == "6100-AI"
    assert gl["unmapped_cost_centers"] == []


def test_unmapped_goes_to_suspense_and_is_flagged():
    gl = general_ledger_rows(
        chargeback=_cb([_cb_row("support", 1.0), _cb_row("newteam", 2.0)]),
        account_mapping={"support": "6100"})
    accounts = {r["cost_center"]: r["account"] for r in gl["rows"]}
    assert accounts["newteam"] == "UNMAPPED-SUSPENSE"
    assert gl["unmapped_cost_centers"] == ["newteam"]
    assert math.isclose(gl["total_usd"], 3.0)      # money not lost by routing


def test_no_mapping_flags_everything_and_preserves_total():
    gl = general_ledger_rows(chargeback=_cb([_cb_row("a", 1.5), _cb_row("b", 2.5)]))
    assert set(gl["unmapped_cost_centers"]) == {"a", "b"}
    assert math.isclose(gl["total_usd"], 4.0)


def test_custom_suspense_account():
    gl = general_ledger_rows(chargeback=_cb([_cb_row("a", 1.0)]),
                             suspense_account="9999-REVIEW")
    assert gl["rows"][0]["account"] == "9999-REVIEW"


def test_gl_total_equals_sum_of_displayed_rows_fuzz():
    rng = random.Random(5)
    for _ in range(200):
        rows = [_cb_row(f"t{i}", round(rng.uniform(0, 999), 6))
                for i in range(rng.randint(1, 20))]
        gl = general_ledger_rows(chargeback=_cb(rows))
        assert abs(sum(r["amount_usd"] for r in gl["rows"])
                   - gl["total_usd"]) < 1e-9


def test_gl_total_ties_to_chargeback_total():
    cb = _cb([_cb_row("a", 1.25), _cb_row("b", 2.5)])
    gl = general_ledger_rows(chargeback=cb)
    assert abs(gl["total_usd"] - cb["totals"]["spend_usd"]) < 1e-9


def test_gl_carries_provenance_flags():
    gl = general_ledger_rows(chargeback=_cb([_cb_row("a", 1.0, priced=False)]))
    assert gl["rows"][0]["fully_priced"] is False


def test_gl_validation():
    with pytest.raises(ValueError):
        general_ledger_rows(chargeback={"report": "nope"})
    with pytest.raises(ValueError):
        general_ledger_rows(chargeback=_cb([_cb_row("a", 1.0)]),
                            cost_center_dimension="nonexistent")
    with pytest.raises(ValueError):
        general_ledger_rows(chargeback=_cb([_cb_row("a", 1.0)]), currency="")


def test_gl_csv_injection_defense():
    gl = general_ledger_rows(chargeback=_cb([_cb_row('=cmd|"/c calc"!A1', 1.0)]))
    csv = general_ledger_csv(gl)
    attack = [l for l in csv.splitlines() if "calc" in l][0]
    assert not attack.startswith("=")


def test_gl_csv_has_total_row():
    gl = general_ledger_rows(chargeback=_cb([_cb_row("a", 1.0), _cb_row("b", 2.0)]))
    lines = [l for l in general_ledger_csv(gl).splitlines() if l.strip()]
    assert len(lines) == 4                      # header + 2 rows + TOTAL
    assert "TOTAL" in lines[-1] and "3.0" in lines[-1]


# ── end-to-end through the real reports ─────────────────────────────────

def test_end_to_end_close_packet_from_live_ledger():
    cc = tokeymeter.register_cluster_costs(
        "kimi-sh", measured_tokens_per_second=175,
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36)
    rate = cc["cluster_costs"]["gpu_hour_rate_usd_full_precision"]
    tokeymeter.register_pricing("gpt-api", input_per_1m=0.15, output_per_1m=0.60)

    def team(t):
        @tokeymeter.cache(model="kimi-sh", tag=t, namespace="app-" + t)
        def ask(p):
            set_reported_usage(340, 128)
            return "r"
        return ask
    sup, eng = team("support"), team("eng")
    for i in range(30):
        sup(f"s{i % 10}")
    for i in range(10):
        eng(f"e{i}")

    cb = tokeymeter.chargeback_report(group_by=("tag",))
    hp = tokeymeter.hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=rate, per_gpu_tokens_per_second=175,
        alternatives=("gpt-api",), selfhosted_models=("kimi-sh",),
        group_by=("model",))
    cap = tokeymeter.capacity_recovery_report(
        measured_tokens_per_second=1400, gpu_hour_rate_usd=rate)

    pkt = close_packet(chargeback=cb, hybrid=hp, capacity=cap,
                       period_label="July 2026")
    assert pkt["reconciliation"]["status"] == "reconciled"
    assert pkt["spend"]["total_usd"] > 0
    assert pkt["components"] == {"chargeback": True, "hybrid_placement": True,
                                 "capacity_recovery": True}

    gl = general_ledger_rows(chargeback=cb,
                             account_mapping={"support": "6100-S", "eng": "6100-E"})
    assert abs(gl["total_usd"] - cb["totals"]["spend_usd"]) < 1e-9
    assert gl["unmapped_cost_centers"] == []


# ── deep-review fixes (seven defects found in the S1.5-1 re-audit) ──────

def test_tolerance_scales_and_never_false_alarms():
    """A FIXED tolerance reported DISCREPANCY on a healthy 50k-row close.
    Display-rounding error is a random walk, so the tolerance scales as
    sqrt(rows) — tight enough to catch real errors, loose enough never to
    cry wolf."""
    import math as _m
    from tokeymeter.engines.economics.close_packet import _tolerance_for
    rng = random.Random(7)
    for n in (10, 1000, 20000, 50000):
        worst = 0.0
        for _ in range(30 if n > 5000 else 100):
            raw = [rng.uniform(0, 50) for _ in range(n)]
            worst = max(worst, abs(round(sum(round(x, 6) for x in raw), 6)
                                   - round(sum(raw), 6)))
        assert _tolerance_for(n) > worst          # no false discrepancy
    # and it must stay far below a cent even at very large scale
    assert _tolerance_for(200000) < 0.01
    # sqrt, not linear — linear would allow ~$0.20 at 200k and hide real errors
    assert _tolerance_for(200000) < 200000 * 1e-6 / 10


def test_capacity_all_time_scope_is_warned_not_hidden():
    cap = {"measured_mechanisms": [{"mechanism": "x", "recovered_tokens": 100}],
           "total_recovered": {"recovered_tokens_total": 100}}
    pkt = close_packet(chargeback=_cb([_cb_row("a", 1.0)]), capacity=cap)
    warns = {w["warning"] for w in pkt["reconciliation"]["warnings"]}
    assert "capacity_scope_is_all_time" in warns
    assert pkt["capacity_recovery"]["scope"].startswith("all_time")
    # a scope caveat is NOT a disagreement
    assert pkt["reconciliation"]["status"] == "reconciled"


def test_packet_is_an_immutable_snapshot():
    cb = _cb([_cb_row("a", 10.0)])
    pkt = close_packet(chargeback=cb)
    cb["rows"][0]["spend_usd"] = 999.0          # mutate source after filing
    assert pkt["spend"]["rows"][0]["spend_usd"] == 10.0


def test_partial_report_dicts_fail_gracefully():
    for bad in ({"report": "chargeback", "group_by": ["t"],
                 "totals": {"spend_usd": 1.0}},                 # no rows
                {"report": "chargeback", "group_by": ["t"], "rows": []},  # no totals
                {"report": "chargeback", "rows": [], "totals": {}}):      # no group_by
        with pytest.raises(ValueError):
            close_packet(chargeback=bad)


def test_non_finite_total_is_caught_not_shipped_silently():
    cb = _cb([_cb_row("a", 10.0)])
    cb["totals"]["spend_usd"] = float("nan")
    pkt = close_packet(chargeback=cb)
    assert pkt["reconciliation"]["status"] == "DISCREPANCY"
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "chargeback_total_is_finite" in failed


def test_api_served_total_must_tie_to_its_own_rows():
    # this passed as "reconciled" before the fix: total 999 vs rows summing to 5
    hp = _hp(10.0)
    hp["totals"]["executed_requests"] = 1
    hp["api_served_workloads"] = {
        "rows": [{"model": "api", "api_spend_usd": 5.0, "requests": 1}],
        "total_api_spend_usd": 999.0}
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0, reqs=2)]), hybrid=hp)
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "hybrid_api_served_total_equals_its_rows" in failed


def test_executed_request_counts_must_match_exactly():
    hp = _hp(10.0)
    hp["totals"]["executed_requests"] = 9999
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0, reqs=5)]), hybrid=hp)
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "executed_request_counts_match" in failed


def test_request_counts_include_api_served_side():
    hp = _hp(10.0)
    hp["totals"]["executed_requests"] = 3
    hp["api_served_workloads"] = {
        "rows": [{"model": "api", "api_spend_usd": 0.0, "requests": 2}],
        "total_api_spend_usd": 0.0}
    # chargeback covers BOTH sides: 3 selfhost + 2 api = 5
    pkt = close_packet(chargeback=_cb([_cb_row("a", 10.0, reqs=5)]), hybrid=hp)
    passed = {c["check"] for c in pkt["reconciliation"]["checks"] if c["passed"]}
    assert "executed_request_counts_match" in passed


def test_fallback_pricing_is_a_warning_not_a_discrepancy():
    cb = _cb([_cb_row("a", 1.0)])
    cb["provenance"] = {"all_rows_fully_priced": False}
    pkt = close_packet(chargeback=cb)
    assert pkt["reconciliation"]["status"] == "reconciled"      # not a disagreement
    warns = {w["warning"] for w in pkt["reconciliation"]["warnings"]}
    assert "not_all_rows_fully_priced" in warns


def test_generated_at_validated():
    with pytest.raises(ValueError):
        close_packet(chargeback=_cb([_cb_row("a", 1.0)]), generated_at="nope")
    with pytest.raises(ValueError):
        close_packet(chargeback=_cb([_cb_row("a", 1.0)]), generated_at=float("nan"))


def test_gl_refuses_non_finite_amount():
    cb = _cb([{"tag": "a", "spend_usd": float("nan"), "executed_requests": 1,
               "provenance": {}}])
    with pytest.raises(ValueError):
        general_ledger_rows(chargeback=cb)


# ── S1.5-2: capacity scope became a CHECK, not a warning ────────────────

def _cap_period(period=(0, 100), tokens=1500, shadow=2, mal=1):
    return {"report": "capacity_recovery", "scope": "period_bounded",
            "period": {"start": period[0], "end": period[1]},
            "measured_mechanisms": [{"mechanism": "exact_cache",
                                     "recovered_tokens": tokens}],
            "total_recovered": {"recovered_tokens_total": tokens},
            "excluded_shadow_hits": shadow, "excluded_malformed_records": mal}


def _cb_with_avoided(avoided=1500, period=(0, 100), shadow=2, mal=1):
    cb = _cb([_cb_row("a", 10.0)], period=period, shadow=shadow, malformed=mal)
    cb["totals"]["avoided_tokens"] = avoided
    return cb


def test_period_bounded_capacity_is_checked_not_warned():
    pkt = close_packet(chargeback=_cb_with_avoided(), capacity=_cap_period())
    names = {c["check"] for c in pkt["reconciliation"]["checks"]}
    assert "capacity_period_matches" in names
    assert "capacity_exclusions_match_chargeback" in names
    assert "capacity_recovered_matches_chargeback_avoided" in names
    assert pkt["reconciliation"]["warnings"] == []      # caveat is gone
    assert pkt["reconciliation"]["status"] == "reconciled"


def test_capacity_period_mismatch_is_a_discrepancy():
    pkt = close_packet(chargeback=_cb_with_avoided(period=(0, 100)),
                       capacity=_cap_period(period=(0, 200)))
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "capacity_period_matches" in failed
    assert pkt["reconciliation"]["status"] == "DISCREPANCY"


def test_capacity_exclusion_divergence_is_a_discrepancy():
    pkt = close_packet(chargeback=_cb_with_avoided(shadow=2, mal=1),
                       capacity=_cap_period(shadow=2, mal=9))
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "capacity_exclusions_match_chargeback" in failed


def test_recovered_must_equal_chargeback_avoided():
    pkt = close_packet(chargeback=_cb_with_avoided(avoided=1500),
                       capacity=_cap_period(tokens=1234))
    failed = {c["check"] for c in pkt["reconciliation"]["checks"] if not c["passed"]}
    assert "capacity_recovered_matches_chargeback_avoided" in failed


def test_legacy_all_time_capacity_still_warns_never_fails():
    """Backward compatibility: an older all-time capacity report inside a period
    packet remains a caveat, not a disagreement."""
    old = {"report": "capacity_recovery", "scope": "all_time_entire_ledger",
           "measured_mechanisms": [{"mechanism": "x", "recovered_tokens": 5}],
           "total_recovered": {"recovered_tokens_total": 5}}
    pkt = close_packet(chargeback=_cb_with_avoided(), capacity=old)
    assert pkt["reconciliation"]["status"] == "reconciled"
    assert any(w["warning"] == "capacity_scope_is_all_time"
               for w in pkt["reconciliation"]["warnings"])


def test_end_to_end_three_artifacts_one_period():
    import tokeymeter.engines.economics.savings as _sv
    cc = tokeymeter.register_cluster_costs(
        "kimi-sh", measured_tokens_per_second=175,
        gpu_count=8, gpu_capex_usd=240000, depreciation_months=36)
    rate = cc["cluster_costs"]["gpu_hour_rate_usd_full_precision"]
    tokeymeter.register_pricing("gpt-api", input_per_1m=0.15, output_per_1m=0.60)

    @tokeymeter.cache(model="kimi-sh", tag="support")
    def ask(p):
        set_reported_usage(340, 128)
        return "r"
    for i in range(60):
        ask(f"q{i % 20}")

    recs = list(_sv._tracker._iter_records())
    lo = min(r["timestamp"] for r in recs)
    hi = max(r["timestamp"] for r in recs) + 1
    cb = tokeymeter.chargeback_report(group_by=("tag",), records=recs,
                                      period_start=lo, period_end=hi)
    hp = tokeymeter.hybrid_placement_report(
        gpu_count=8, gpu_hour_rate_usd=rate, per_gpu_tokens_per_second=175,
        alternatives=("gpt-api",), selfhosted_models=("kimi-sh",),
        group_by=("model",), records=recs, period_start=lo, period_end=hi)
    cap = tokeymeter.capacity_recovery_report(
        measured_tokens_per_second=1400, gpu_hour_rate_usd=rate,
        records=recs, period_start=lo, period_end=hi)

    pkt = close_packet(chargeback=cb, hybrid=hp, capacity=cap,
                       period_label="July 2026")
    assert pkt["reconciliation"]["status"] == "reconciled"
    assert pkt["reconciliation"]["warnings"] == []
    assert pkt["capacity_recovery"]["scope"] == "period_bounded"
    names = {c["check"] for c in pkt["reconciliation"]["checks"]}
    assert {"capacity_period_matches", "capacity_exclusions_match_chargeback",
            "capacity_recovered_matches_chargeback_avoided"} <= names
