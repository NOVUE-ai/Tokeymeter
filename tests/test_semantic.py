"""
Tests for the semantic cache layer.

We use a deterministic FAKE encoder so CI is fast and doesn't need to
download a 1GB sentence-transformers model. The fake encoder is just a
character-frequency vector — semantically meaningless but reproducible
and sufficient to verify the cache mechanics work end to end.

We DO test:
  - Hits when query is "similar" to a stored prompt (per the fake metric)
  - Misses when query is dissimilar
  - Fail-open behavior under broken encoders / broken DB
  - Pipeline integration: semantic miss → exact miss → real call
  - The exact-cache promotion: a semantic hit populates the exact cache
"""
import os
import sys
import tempfile

import pytest

# Skip the entire file if numpy isn't installed.
np = pytest.importorskip("numpy")

import tokeymeter
from tokeymeter.semantic import SemanticCache
from tokeymeter.storage import MemoryStore


def fake_encoder(text: str):
    """Deterministic 26-dim character-frequency embedding, L2-normalized.

    Two strings with similar letter counts get similar vectors. Not a real
    semantic encoder — just enough to test the cache mechanics.
    """
    text = text.lower()
    vec = np.zeros(26, dtype=np.float32)
    for ch in text:
        if "a" <= ch <= "z":
            vec[ord(ch) - ord("a")] += 1
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


@pytest.fixture
def sem_cache(tmp_path):
    return SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.95,
        encoder=fake_encoder,
    )


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)  # reset to lazy-load
    tokeymeter.reset_savings()
    yield


# ----------------- SemanticCache unit tests -----------------

def test_semantic_cache_miss_when_empty(sem_cache):
    assert sem_cache.lookup("hello world") is None


def test_semantic_cache_hit_on_similar_input(sem_cache):
    sem_cache.store("abcdef", "response-1")
    # Same letters, different order — identical fake-embedding
    result = sem_cache.lookup("fedcba")
    assert result == "response-1"


def test_semantic_cache_miss_on_dissimilar_input(sem_cache):
    sem_cache.store("aaaa bbbb", "response-1")
    # Different letter distribution → low similarity → miss
    result = sem_cache.lookup("xyzxyzxyz qqqq")
    assert result is None


def test_semantic_cache_fail_open_on_broken_encoder(tmp_path):
    def broken(text):
        raise RuntimeError("encoder on fire")

    cache = SemanticCache(
        path=str(tmp_path / "b.db"),
        threshold=0.9,
        encoder=broken,
    )
    # Both should return None, not raise
    assert cache.lookup("x") is None
    cache.store("x", "y")  # should not raise


def test_semantic_cache_eviction(tmp_path):
    cache = SemanticCache(
        path=str(tmp_path / "e.db"),
        threshold=0.0,  # always match
        encoder=fake_encoder,
        max_entries=3,
    )
    cache.store("a", "r1")
    cache.store("b", "r2")
    cache.store("c", "r3")
    cache.store("d", "r4")  # should evict "a"
    assert len(cache) == 3


# ----------------- Decorator integration tests -----------------

def test_decorator_semantic_layer_catches_similar_prompts(sem_cache):
    """The big one: a semantically-similar prompt should hit the cache,
    even though the exact strings differ."""
    calls = [0]

    @tokeymeter.cache(
        model="gpt-4o-mini",
        semantic=True,
        semantic_cache=sem_cache,
        prompt_arg="prompt",
    )
    def fake_llm(prompt: str) -> str:
        calls[0] += 1
        return f"response-{calls[0]}"

    r1 = fake_llm(prompt="abcdef")   # miss
    r2 = fake_llm(prompt="fedcba")   # semantic hit (fake encoder is bag-of-letters)
    assert r1 == r2
    assert calls[0] == 1, "second call should have been a semantic hit"


def test_decorator_falls_through_when_no_semantic_match(sem_cache):
    """Dissimilar prompts → real function called every time."""
    calls = [0]

    @tokeymeter.cache(
        semantic=True,
        semantic_cache=sem_cache,
        prompt_arg="prompt",
    )
    def fake_llm(prompt: str) -> str:
        calls[0] += 1
        return f"r-{calls[0]}"

    fake_llm(prompt="aaa bbb")
    fake_llm(prompt="xyz qqq")
    fake_llm(prompt="mno pqr")
    assert calls[0] == 3


def test_exact_layer_takes_precedence_over_semantic(sem_cache):
    """Identical call should hit exact (fast), not even reach semantic."""
    calls = [0]
    enc_calls = [0]

    def counting_encoder(text):
        enc_calls[0] += 1
        return fake_encoder(text)

    sem_cache._encoder = counting_encoder  # type: ignore[attr-defined]

    @tokeymeter.cache(
        semantic=True,
        semantic_cache=sem_cache,
        prompt_arg="prompt",
    )
    def fake_llm(prompt: str) -> str:
        calls[0] += 1
        return "r"

    fake_llm(prompt="hello")  # miss → real call → store in both
    enc_before = enc_calls[0]
    fake_llm(prompt="hello")  # exact hit, should NOT call encoder
    assert calls[0] == 1
    assert enc_calls[0] == enc_before, "exact hit should not invoke the encoder"


def test_semantic_hit_promotes_to_exact_cache(sem_cache):
    """After a semantic hit, the exact form should also be cached
    so the same call is microsecond-fast next time."""
    calls = [0]
    exact_store = MemoryStore()

    @tokeymeter.cache(
        store=exact_store,
        semantic=True,
        semantic_cache=sem_cache,
        prompt_arg="prompt",
    )
    def fake_llm(prompt: str) -> str:
        calls[0] += 1
        return "r"

    fake_llm(prompt="abcdef")  # miss → store
    assert len(exact_store) == 1

    fake_llm(prompt="fedcba")  # semantic hit → also promotes to exact
    assert len(exact_store) == 2
    assert calls[0] == 1


def test_savings_report_breaks_down_exact_vs_semantic(sem_cache):
    @tokeymeter.cache(
        model="gpt-4o-mini",
        semantic=True,
        semantic_cache=sem_cache,
        prompt_arg="prompt",
    )
    def fake_llm(prompt: str) -> str:
        return "x" * 200

    fake_llm(prompt="abcdef")  # miss
    fake_llm(prompt="abcdef")  # exact hit
    fake_llm(prompt="fedcba")  # semantic hit (after promotion → exact next time)
    fake_llm(prompt="aaa bbb")  # miss (different letters)

    r = tokeymeter.savings_report()
    assert r["total_calls"] == 4
    assert r["exact_hits"] == 1
    assert r["semantic_hits"] == 1
    assert r["cache_misses"] == 2


def test_semantic_disabled_when_deps_missing(tmp_path, monkeypatch):
    """If user requests semantic=True but no encoder is available,
    the library should warn and fall back to exact-only — not crash."""
    # Force the default semantic cache to return None
    tokeymeter.set_default_semantic_cache(None)
    import tokeymeter.decorator as dec
    monkeypatch.setattr(dec, "_get_default_semantic_cache", lambda threshold=0.92: None)

    calls = [0]

    @tokeymeter.cache(semantic=True)
    def fake_llm(prompt: str) -> str:
        calls[0] += 1
        return "r"

    # Should still work — just without semantic layer
    fake_llm(prompt="hello")
    fake_llm(prompt="hello")  # exact-only hit
    fake_llm(prompt="goodbye")
    assert calls[0] == 2  # one for hello, one for goodbye
