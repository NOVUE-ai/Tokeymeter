"""Tests for the core @tokeymeter.cache decorator behavior."""
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_store():
    """Each test gets its own in-memory store. No shared state."""
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset_savings()
    yield


def test_basic_cache_hit():
    """Same input twice = function called once."""
    calls = [0]

    @tokeymeter.cache(model="gpt-4o-mini")
    def fake_llm(prompt):
        calls[0] += 1
        return f"response: {prompt}"

    r1 = fake_llm("hello")
    r2 = fake_llm("hello")
    assert r1 == r2 == "response: hello"
    assert calls[0] == 1


def test_different_inputs_dont_collide():
    """Different inputs = different cache keys = function called each time."""
    calls = [0]

    @tokeymeter.cache(model="gpt-4o-mini")
    def fake_llm(prompt):
        calls[0] += 1
        return f"r:{prompt}"

    fake_llm("A")
    fake_llm("B")
    fake_llm("C")
    assert calls[0] == 3


def test_kwargs_order_doesnt_matter():
    """Kwargs in different order should hit the same cache key."""
    calls = [0]

    @tokeymeter.cache()
    def fake_llm(prompt, temperature=0.7, max_tokens=100):
        calls[0] += 1
        return "ok"

    fake_llm("x", temperature=0.5, max_tokens=50)
    fake_llm("x", max_tokens=50, temperature=0.5)
    assert calls[0] == 1


def test_fail_open_on_broken_get():
    """If cache.get() raises, the wrapped function still runs and returns."""
    class BrokenGetStore:
        def get(self, k): raise RuntimeError("disk on fire")
        def set(self, k, v): pass

    @tokeymeter.cache(store=BrokenGetStore())
    def fake_llm(prompt):
        return "ok"

    assert fake_llm("x") == "ok"
    assert fake_llm("x") == "ok"  # both calls work despite broken store


def test_fail_open_on_broken_set():
    """If cache.set() raises, the wrapped function still returns normally."""
    class BrokenSetStore:
        def get(self, k): return None
        def set(self, k, v): raise RuntimeError("disk on fire")

    @tokeymeter.cache(store=BrokenSetStore())
    def fake_llm(prompt):
        return "ok"

    assert fake_llm("x") == "ok"


def test_fail_open_on_broken_key_fn():
    """If the user's key_fn raises, we bypass cache and call the function."""
    calls = [0]

    def bad_key(args, kwargs):
        raise ValueError("nope")

    @tokeymeter.cache(key_fn=bad_key)
    def fake_llm(prompt):
        calls[0] += 1
        return "ok"

    assert fake_llm("x") == "ok"
    assert fake_llm("x") == "ok"
    assert calls[0] == 2  # cache was bypassed both times


def test_user_exceptions_propagate():
    """If the wrapped function raises, the exception must NOT be swallowed."""
    @tokeymeter.cache()
    def bad_llm(prompt):
        raise ValueError("api down")

    with pytest.raises(ValueError, match="api down"):
        bad_llm("x")


def test_decorator_works_without_parens():
    """@tokeymeter.cache should work the same as @tokeymeter.cache()."""
    calls = [0]

    @tokeymeter.cache
    def fake_llm(prompt):
        calls[0] += 1
        return "ok"

    fake_llm("x")
    fake_llm("x")
    assert calls[0] == 1


def test_enabled_false_disables_cache():
    """enabled=False should bypass caching entirely."""
    calls = [0]

    @tokeymeter.cache(enabled=False)
    def fake_llm(prompt):
        calls[0] += 1
        return "ok"

    fake_llm("x")
    fake_llm("x")
    fake_llm("x")
    assert calls[0] == 3


def test_custom_key_fn():
    """key_fn lets you ignore irrelevant args like request IDs."""
    calls = [0]

    @tokeymeter.cache(key_fn=lambda args, kwargs: kwargs.get("prompt", args[0] if args else ""))
    def fake_llm(prompt, request_id=None):
        calls[0] += 1
        return f"r:{prompt}"

    fake_llm(prompt="hello", request_id="req-1")
    fake_llm(prompt="hello", request_id="req-2")  # different request_id, same prompt
    assert calls[0] == 1  # only one real call


def test_savings_report_basic():
    """After some calls, savings report should reflect hits and savings."""
    @tokeymeter.cache(model="gpt-4o-mini")
    def fake_llm(prompt):
        return "x" * 400  # ~100 tokens of output

    fake_llm("a")  # miss
    fake_llm("a")  # hit
    fake_llm("a")  # hit
    fake_llm("b")  # miss

    r = tokeymeter.savings_report()
    assert r["total_calls"] == 4
    assert r["cache_hits"] == 2
    assert r["cache_misses"] == 2
    assert r["hit_rate_pct"] == 50.0
    assert r["estimated_saved_usd"] > 0
    assert r["estimated_spent_usd"] > 0


def test_clear_cache():
    """clear_cache() should wipe entries so the next call is a miss."""
    calls = [0]

    @tokeymeter.cache()
    def fake_llm(prompt):
        calls[0] += 1
        return "ok"

    fake_llm("x")  # miss
    fake_llm("x")  # hit
    assert calls[0] == 1

    tokeymeter.clear_cache()
    fake_llm("x")  # miss again after clear
    assert calls[0] == 2
