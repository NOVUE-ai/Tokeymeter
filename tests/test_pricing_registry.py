"""v0.13 pricing registry — the anti-fabrication layer.

The defect this pins: a self-hosted OSS model (no public list price) silently
fell through to the generic `_default` rate, so the savings report printed USD
figures derived from prices that do not exist for that customer. These tests
pin the four guarantees that close it:

  1. register_pricing installs true rates at runtime, with precedence over the
     static list-price table and the same longest-prefix matching.
  2. pricing_info / estimate_cost_with_source expose PROVENANCE, and the
     savings report labels every figure that rests on the default fallback.
  3. register_selfhost_pricing derives exact rates from two measured inputs
     (GPU-hour cost, throughput) and shows its work.
  4. capacity_report expresses savings in GPU-hours reclaimed — the honest
     unit when there is no invoice — from measured inputs only.
"""
import math
import threading

import pytest

import tokeymeter
from tokeymeter import pricing
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset_savings()
    pricing.clear_registered_pricing()
    yield
    pricing.clear_registered_pricing()
    tokeymeter.reset_savings()


# ── 1. registry semantics ────────────────────────────────────────────────

def test_unregistered_oss_model_resolves_to_default():
    info = tokeymeter.pricing_info("kimi-k2")
    assert info["source"] == "default"


def test_register_exact_and_precedence_over_list():
    # Registering a rate for a model that HAS a list price must win: the
    # operator's number is the truth for their deployment.
    tokeymeter.register_pricing("gpt-4o", input_per_1m=9.99, output_per_1m=19.99)
    info = tokeymeter.pricing_info("gpt-4o")
    assert info["source"] == "registered"
    assert info["input_per_1m"] == 9.99 and info["output_per_1m"] == 19.99


def test_registered_prefix_matching_covers_versioned_names():
    tokeymeter.register_pricing("kimi-k2", input_per_1m=0.40, output_per_1m=0.40)
    info = tokeymeter.pricing_info("kimi-k2-instruct-0905")
    assert info["source"] == "registered_prefix" and info["matched"] == "kimi-k2"
    assert info["input_per_1m"] == 0.40


def test_longest_registered_prefix_wins():
    tokeymeter.register_pricing("deepseek", input_per_1m=1.0, output_per_1m=1.0)
    tokeymeter.register_pricing("deepseek-v3", input_per_1m=0.5, output_per_1m=0.5)
    info = tokeymeter.pricing_info("deepseek-v3-0324")
    assert info["matched"] == "deepseek-v3" and info["input_per_1m"] == 0.5


def test_unregister_restores_prior_resolution():
    tokeymeter.register_pricing("kimi-k2", input_per_1m=0.4, output_per_1m=0.4)
    assert tokeymeter.unregister_pricing("kimi-k2") is True
    assert tokeymeter.unregister_pricing("kimi-k2") is False
    assert tokeymeter.pricing_info("kimi-k2")["source"] == "default"


def test_registration_validates_loudly():
    for bad in (float("nan"), float("inf"), -1, "abc", None):
        with pytest.raises(ValueError):
            tokeymeter.register_pricing("m", input_per_1m=bad, output_per_1m=1.0)
    with pytest.raises(ValueError):
        tokeymeter.register_pricing("", input_per_1m=1.0, output_per_1m=1.0)


def test_estimate_cost_unchanged_for_list_priced_models():
    # Regression guard: adding the registry must not shift existing numbers.
    assert pricing.estimate_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)


def test_estimate_cost_with_source_matches_estimate_cost():
    cost, src = tokeymeter.estimate_cost_with_source("gpt-4o-mini-2024-07-18", 5000, 2000)
    assert src == "list_prefix"
    assert cost == pytest.approx(pricing.estimate_cost("gpt-4o-mini-2024-07-18", 5000, 2000))


def test_registry_thread_safety_smoke():
    errs = []
    def worker(i):
        try:
            for _ in range(200):
                tokeymeter.register_pricing(f"m{i}", input_per_1m=i, output_per_1m=i)
                tokeymeter.pricing_info(f"m{i}-suffix")
                tokeymeter.registered_pricing()
        except Exception as e:  # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs


# ── 2. selfhost derivation shows its work, exactly ───────────────────────

def test_selfhost_derivation_exact_math():
    # $2.10/GPU-hr at 1400 tok/s → 5,040,000 tokens/GPU-hr →
    # 2.10 / 5.04M * 1e6 = $0.416666.../1M. Verifiable by hand.
    d = tokeymeter.derive_selfhost_rate(gpu_hour_rate_usd=2.10,
                                        measured_tokens_per_second=1400)
    assert d["tokens_per_gpu_hour"] == 5_040_000.0
    assert d["usd_per_1m_tokens"] == pytest.approx(2.10 / 5_040_000 * 1_000_000)
    assert "derivation" in d  # the math is embedded, auditable


def test_register_selfhost_pricing_registers_and_prices_calls():
    r = tokeymeter.register_selfhost_pricing("kimi-k2", gpu_hour_rate_usd=2.10,
                                             measured_tokens_per_second=1400)
    rate = r["usd_per_1m_tokens"]
    cost, src = tokeymeter.estimate_cost_with_source("kimi-k2", 1_000_000, 0)
    assert src == "registered"
    assert cost == pytest.approx(rate)


def test_selfhost_derivation_rejects_zero_throughput():
    with pytest.raises(ValueError):
        tokeymeter.derive_selfhost_rate(gpu_hour_rate_usd=2.0,
                                        measured_tokens_per_second=0)


# ── 3. report provenance: fallback USD can never pass silently ───────────

def _run_calls(model, n=3):
    @tokeymeter.cache(model=model)
    def ask(p):
        return "x" * 400  # ~100 output tokens
    for _ in range(n):
        ask("the same prompt " * 20)


def test_report_flags_default_priced_models():
    _run_calls("totally-unknown-oss-model")
    rep = tokeymeter.savings_report()
    p = rep["pricing"]
    assert p["all_priced"] is False
    assert p["default_priced_calls"] == 3
    assert "totally-unknown-oss-model" in p["default_priced_models"]
    assert p["default_priced_usd"] > 0


def test_report_all_priced_after_registration():
    tokeymeter.register_selfhost_pricing("totally-unknown-oss-model",
                                         gpu_hour_rate_usd=2.10,
                                         measured_tokens_per_second=1400)
    _run_calls("totally-unknown-oss-model")
    rep = tokeymeter.savings_report()
    assert rep["pricing"]["all_priced"] is True
    assert rep["pricing"]["default_priced_calls"] == 0


def test_list_priced_models_are_not_flagged():
    _run_calls("gpt-4o-mini")
    rep = tokeymeter.savings_report()
    assert rep["pricing"]["all_priced"] is True


def test_legacy_records_without_source_classified_against_current_registry():
    # Simulate a pre-v0.13 ledger record (no pricing_source field).
    from tokeymeter import savings as sv
    sv._record(sv.CallRecord(timestamp=0.0, model="mystery-oss", hit=False,
                             hit_type=None, input_tokens=10, output_tokens=10,
                             estimated_cost=0.01, latency_ms=1.0,
                             pricing_source=None))
    rep = tokeymeter.savings_report()
    assert "mystery-oss" in rep["pricing"]["default_priced_models"]
    # Registering a price re-classifies legacy records on the next report.
    tokeymeter.register_pricing("mystery-oss", input_per_1m=0.3, output_per_1m=0.3)
    rep2 = tokeymeter.savings_report()
    assert "mystery-oss" not in rep2["pricing"]["default_priced_models"]


# ── 4. capacity view: GPU-hours from measured inputs only ────────────────

def test_capacity_reclaimed_exact_math():
    # 7,200,000 saved tokens at 2000 tok/s = 3600 GPU-seconds = 1.0 GPU-hour.
    c = tokeymeter.capacity_reclaimed(saved_input_tokens=7_000_000,
                                      saved_output_tokens=200_000,
                                      measured_tokens_per_second=2000)
    assert c["saved_tokens_total"] == 7_200_000
    assert c["gpu_seconds_reclaimed"] == pytest.approx(3600.0)
    assert c["gpu_hours_reclaimed"] == pytest.approx(1.0)
    assert "equivalent_usd" not in c  # no rate supplied → no USD invented


def test_capacity_usd_only_when_caller_supplies_rate():
    c = tokeymeter.capacity_reclaimed(saved_input_tokens=7_200_000,
                                      saved_output_tokens=0,
                                      measured_tokens_per_second=2000,
                                      gpu_hour_rate_usd=2.50)
    assert c["equivalent_usd"] == pytest.approx(2.50)


def test_capacity_report_reads_live_ledger():
    tokeymeter.register_selfhost_pricing("kimi-k2", gpu_hour_rate_usd=2.10,
                                         measured_tokens_per_second=1400)
    @tokeymeter.cache(model="kimi-k2")
    def ask(p):
        return "y" * 4000
    ask("same prompt " * 50)   # miss
    ask("same prompt " * 50)   # exact hit → tokens saved
    rep = tokeymeter.savings_report()
    assert rep["saved_input_tokens"] > 0 or rep["saved_output_tokens"] > 0
    cap = tokeymeter.capacity_report(measured_tokens_per_second=1400,
                                     gpu_hour_rate_usd=2.10)
    assert cap["gpu_seconds_reclaimed"] > 0
    assert cap["equivalent_usd"] > 0
    # cross-check: internally consistent (gpu_hours is rounded to 6 dp)
    assert cap["gpu_hours_reclaimed"] == pytest.approx(
        cap["saved_tokens_total"] / 1400 / 3600, abs=5e-7)
