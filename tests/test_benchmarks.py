"""
Tests for the benchmark harness itself.

A benchmark that can silently break or inflate is worse than no benchmark.
These tests protect the integrity of the numbers:
  - Workloads are deterministic per seed, divergent across seeds.
  - "Fresh" requests are genuinely unique (no artificial hit inflation).
  - Cache/compression/memory results satisfy hard invariants
    (hits <= total, cost_with <= cost_without, tokens_after <= tokens_before,
     memory never worse than baseline).
  - Latency overhead stays within the library's design contract.
"""
import pytest

import tokeymeter
# The `benchmarks/` package is part of the repo but NOT shipped in the sdist
# (it's a dev/CI-only harness). When installed from PyPI the import will fail;
# skip the whole module cleanly instead of breaking pytest collection.
benchmarks = pytest.importorskip("benchmarks", reason="benchmarks/ not installed (sdist)")
from benchmarks.workloads import (
    generate_agent_workload,
    generate_rag_workload,
    generate_support_bot_workload,
)
from benchmarks.harness import (
    run_cache_benchmark,
    run_compression_benchmark,
    run_memory_benchmark,
)


# =================================================================
#                Workload determinism
# =================================================================

def test_support_workload_deterministic():
    a = generate_support_bot_workload(n=300, seed=42)
    b = generate_support_bot_workload(n=300, seed=42)
    assert a.prompts() == b.prompts()


def test_support_workload_diverges_by_seed():
    a = generate_support_bot_workload(n=300, seed=1)
    b = generate_support_bot_workload(n=300, seed=2)
    assert a.prompts() != b.prompts()


def test_fresh_requests_are_unique():
    """Fresh requests must carry a unique detail so they genuinely miss.
    This guards against the artificial-collision inflation we caught and fixed."""
    wl = generate_support_bot_workload(n=500, seed=42)
    fresh = [r["prompt"] for r in wl.requests if r["kind"] == "fresh"]
    # Every fresh prompt is distinct
    assert len(fresh) == len(set(fresh)), "fresh requests are not all unique"


def test_rag_fresh_requests_unique():
    wl = generate_rag_workload(n=300, seed=42)
    fresh = [r["prompt"] for r in wl.requests if r["kind"] == "fresh"]
    assert len(fresh) == len(set(fresh))


def test_agent_workload_shape():
    wl = generate_agent_workload(n_sessions=10, turns_per_session=8, seed=42)
    assert len(wl.requests) == 80
    # Each session has exactly turns_per_session turns
    from collections import Counter
    counts = Counter(r["session_id"] for r in wl.requests)
    assert all(c == 8 for c in counts.values())
    assert len(counts) == 10


# =================================================================
#                Cache benchmark invariants
# =================================================================

def test_cache_benchmark_invariants():
    wl = generate_support_bot_workload(n=400, seed=42)
    r = run_cache_benchmark(wl, model="gpt-4o-mini", semantic=False)

    # Hard invariants that must ALWAYS hold
    assert r.cache_hits <= r.requests_total
    assert r.exact_hits + r.semantic_hits <= r.cache_hits + 1  # rounding tolerance
    assert 0.0 <= r.hit_rate_pct <= 100.0
    assert r.cost_with_tokeymeter_usd <= r.cost_without_tokeymeter_usd
    assert 0.0 <= r.cost_reduction_pct <= 100.0
    assert r.tokens_sent_with_tokeymeter <= r.tokens_sent_without_tokeymeter


def test_cache_hit_rate_tracks_repeat_rate():
    """With semantic off, hit rate should be at least the exact-repeat rate
    (verbatim repeats are always cacheable) and well below 100%."""
    wl = generate_support_bot_workload(
        n=1000, exact_repeat_rate=0.30, paraphrase_rate=0.0, seed=42
    )
    r = run_cache_benchmark(wl, model="gpt-4o-mini", semantic=False)
    # At least ~25% (exact repeats minus warmup), and not a suspicious ~100%
    assert 15.0 <= r.hit_rate_pct <= 70.0, f"hit rate {r.hit_rate_pct} implausible"


def test_zero_repeat_workload_low_hit_rate():
    """If almost nothing repeats, hit rate must be low — guards against
    a benchmark that fakes hits."""
    wl = generate_support_bot_workload(
        n=500, exact_repeat_rate=0.0, paraphrase_rate=0.0, seed=42
    )
    r = run_cache_benchmark(wl, model="gpt-4o-mini", semantic=False)
    # All requests are fresh+unique → essentially no hits
    assert r.hit_rate_pct < 5.0, f"unexpected hits on no-repeat workload: {r.hit_rate_pct}%"


# =================================================================
#                Latency overhead within design contract
# =================================================================

def test_overhead_within_design_contract():
    """The library's design contract claims <5ms miss overhead. Verify the
    measured overhead respects a generous ceiling (CI machines are slow)."""
    wl = generate_support_bot_workload(n=200, seed=42)
    r = run_cache_benchmark(wl, model="gpt-4o-mini", semantic=False,
                            latency_samples=100)
    # Generous ceiling: 5ms p50, well above what we measure (~0.05ms) but
    # catches a real regression (e.g. accidental network call).
    assert r.overhead_hit_p50_ms < 5.0, f"hit overhead {r.overhead_hit_p50_ms}ms too high"
    assert r.overhead_miss_p50_ms < 5.0, f"miss overhead {r.overhead_miss_p50_ms}ms too high"


# =================================================================
#                Compression benchmark invariants
# =================================================================

def test_compression_benchmark_invariants():
    wl = generate_support_bot_workload(n=300, seed=42)
    r = run_compression_benchmark(wl, model="gpt-4o-mini")
    assert r.tokens_after <= r.tokens_before          # never expands
    assert 0.0 <= r.reduction_pct <= 100.0
    assert r.cost_after_usd <= r.cost_before_usd
    assert 0.0 < r.mean_ratio <= 1.0


def test_compression_honest_about_rag():
    """RAG context has little filler — compression should be modest, not faked."""
    wl = generate_rag_workload(n=200, seed=42)
    r = run_compression_benchmark(wl, model="gpt-4o")
    # We claim ~0% in the README; just assert it's not implausibly high
    assert r.reduction_pct < 15.0


# =================================================================
#                Memory benchmark invariants
# =================================================================

def test_memory_benchmark_never_worse_than_baseline():
    """The core memory invariant: summarized context is never larger than
    the full history baseline."""
    wl = generate_agent_workload(n_sessions=20, turns_per_session=12, seed=42)
    r = run_memory_benchmark(wl, model="gpt-4o")
    assert r.tokens_with_memory <= r.tokens_without_memory
    assert 0.0 <= r.reduction_pct <= 100.0
    assert r.cost_with_memory_usd <= r.cost_without_memory_usd


def test_memory_reduction_grows_with_turns():
    """Longer conversations should yield MORE memory savings (the quadratic
    -> linear effect). Verify monotonic direction."""
    short = run_memory_benchmark(
        generate_agent_workload(n_sessions=10, turns_per_session=6, seed=42),
        model="gpt-4o",
    )
    long = run_memory_benchmark(
        generate_agent_workload(n_sessions=10, turns_per_session=20, seed=42),
        model="gpt-4o",
    )
    assert long.reduction_pct >= short.reduction_pct


# =================================================================
#                Cost figures are internally consistent
# =================================================================

def test_cost_uses_library_pricing():
    """Cost figures must come from tokeymeter.pricing, not invented constants."""
    from tokeymeter.pricing import estimate_cost
    # gpt-4o-mini input price is 0.15/1M; 1M input tokens => $0.15
    assert abs(estimate_cost("gpt-4o-mini", 1_000_000, 0) - 0.15) < 1e-9
    # The benchmark must produce costs consistent with this table
    wl = generate_support_bot_workload(n=200, seed=42)
    r = run_cache_benchmark(wl, model="gpt-4o-mini", semantic=False)
    # cost_without should equal estimate_cost on the measured tokens
    expected = estimate_cost(
        "gpt-4o-mini",
        r.measured["total_input_tokens"],
        r.measured["total_output_tokens"],
    )
    assert abs(r.cost_without_tokeymeter_usd - round(expected, 6)) < 1e-6
