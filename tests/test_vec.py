"""
Tests for sqlite-vec-accelerated semantic cache.

These tests run only when sqlite-vec is importable. We verify:
  - SemanticCache auto-selects sqlite-vec when available.
  - vec-backed lookups return the same results as linear-scan for small datasets.
  - The vec backend scales reasonably (1k entries → <100ms per lookup).
  - Eviction cleans both the regular table and the vec table.
"""
import time

import numpy as np
import pytest

try:
    import sqlite_vec  # noqa: F401
    _has_vec = True
except ImportError:
    _has_vec = False

from tokeymeter.semantic import SemanticCache, is_vec_index_available


def _bag(t: str):
    t = t.lower()
    v = np.zeros(26, dtype=np.float32)
    for ch in t:
        if "a" <= ch <= "z":
            v[ord(ch) - ord("a")] += 1
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def _rand_unit(dim=26):
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


pytestmark = pytest.mark.skipif(not _has_vec, reason="sqlite-vec not installed")


def test_vec_index_available_when_installed():
    assert is_vec_index_available() is True


def test_auto_selects_vec_backend(tmp_path):
    sem = SemanticCache(
        path=str(tmp_path / "auto.db"),
        threshold=0.9,
        encoder=_bag,
        dim=26,
    )
    assert sem.backend == "sqlite-vec"


def test_can_force_linear_scan(tmp_path):
    sem = SemanticCache(
        path=str(tmp_path / "linear.db"),
        threshold=0.9,
        encoder=_bag,
        use_vec_index=False,
        dim=26,
    )
    assert sem.backend == "linear-scan"


def test_vec_lookup_matches_linear_for_small_dataset(tmp_path):
    """For a small dataset, vec0 and linear scan should return the same answer."""
    encoder = _bag

    sem_lin = SemanticCache(
        path=str(tmp_path / "lin.db"),
        threshold=0.95,
        encoder=encoder,
        use_vec_index=False,
        dim=26,
    )
    sem_vec = SemanticCache(
        path=str(tmp_path / "vec.db"),
        threshold=0.95,
        encoder=encoder,
        use_vec_index=True,
        dim=26,
    )

    prompts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
        "nu xi omicron",
    ]
    for i, p in enumerate(prompts):
        sem_lin.store(p, f"r-{i}")
        sem_vec.store(p, f"r-{i}")

    # Query with a similar-but-not-identical prompt
    query = "gamma alpha beta"  # anagram of first → high similarity
    r_lin = sem_lin.lookup(query)
    r_vec = sem_vec.lookup(query)
    assert r_lin == r_vec == "r-0"


def test_vec_lookup_respects_threshold(tmp_path):
    """If the best match is below threshold, vec backend returns None."""
    sem = SemanticCache(
        path=str(tmp_path / "thresh.db"),
        threshold=0.999,  # very strict
        encoder=_bag,
        use_vec_index=True,
        dim=26,
    )
    sem.store("alpha beta gamma", "r0")
    # Different distribution of letters → low similarity → miss
    assert sem.lookup("zzzzz") is None


def test_vec_lookup_under_100ms_at_1k_entries(tmp_path):
    """Sanity check that the indexed backend is actually fast."""
    np.random.seed(42)
    sem = SemanticCache(
        path=str(tmp_path / "perf.db"),
        threshold=0.0,
        encoder=lambda t: _rand_unit(26),  # random vectors regardless of input
        use_vec_index=True,
        dim=26,
        max_entries=10_000,
    )

    # Populate 1000 random entries
    for i in range(1000):
        sem.store(f"prompt-{i}", f"response-{i}")

    # Time a lookup
    t0 = time.perf_counter()
    sem.lookup("query")
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n  vec lookup at 1k entries: {ms:.2f}ms")
    assert ms < 100, f"vec lookup too slow at 1k entries: {ms:.1f}ms"


def test_vec_eviction_cleans_both_tables(tmp_path):
    sem = SemanticCache(
        path=str(tmp_path / "evict.db"),
        threshold=0.0,
        encoder=_bag,
        use_vec_index=True,
        max_entries=3,
        dim=26,
    )
    for i in range(5):
        sem.store(f"prompt-{i}", f"r-{i}")
    # Only 3 entries should remain after eviction
    assert len(sem) == 3
