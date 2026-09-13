"""
Tests for the async @tokeymeter.cache and the streaming @tokeymeter.cache_stream.

Critical behaviors verified:
  - Async functions get the async pipeline; sync functions get the sync pipeline.
  - Cache hits on async functions return without awaiting the wrapped function.
  - Fail-open works in the async path.
  - Streaming caches the full sequence of chunks and replays them in order.
  - If a consumer breaks out of a stream early, NOTHING is cached
    (partial responses must never be served as full responses).
  - If the wrapped generator raises mid-stream, NOTHING is cached.
  - Semantic caching works for streams.
"""
import asyncio

import numpy as np
import pytest

import tokeymeter
from tokeymeter.semantic import SemanticCache
from tokeymeter.storage import MemoryStore


# ----- shared fixtures -----

@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.reset_savings()
    yield


def _bag_of_letters(text: str):
    text = text.lower()
    v = np.zeros(26, dtype=np.float32)
    for ch in text:
        if "a" <= ch <= "z":
            v[ord(ch) - ord("a")] += 1
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


# =================================================================
#                      ASYNC @tokeymeter.cache TESTS
# =================================================================

@pytest.mark.asyncio
async def test_async_basic_cache_hit():
    calls = [0]

    @tokeymeter.cache(model="gpt-4o-mini")
    async def ask(prompt):
        calls[0] += 1
        await asyncio.sleep(0)  # actually be async
        return f"response: {prompt}"

    r1 = await ask("hello")
    r2 = await ask("hello")
    assert r1 == r2
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_async_does_not_call_func_on_hit():
    """A cache hit must not invoke the underlying coroutine at all."""
    from tokeymeter.utils import make_cache_key

    invoked = [False]

    # Seed a memory store so we know the hit comes from the cache, not the func
    store = MemoryStore()
    tokeymeter.set_default_store(store)

    @tokeymeter.cache(shared_namespace=True)
    async def expensive(prompt):
        invoked[0] = True
        await asyncio.sleep(10)  # would hang the test if actually awaited
        return "ok"

    key = make_cache_key(("hello",), {}, model="_default")
    store.set(key, "ok")

    result = await asyncio.wait_for(expensive("hello"), timeout=0.5)
    assert result == "ok"
    assert invoked[0] is False


@pytest.mark.asyncio
async def test_async_user_exceptions_propagate():
    @tokeymeter.cache()
    async def bad(prompt):
        raise ValueError("upstream down")

    with pytest.raises(ValueError, match="upstream down"):
        await bad("x")


@pytest.mark.asyncio
async def test_async_fail_open_on_broken_store():
    class BrokenStore:
        def get(self, k): raise RuntimeError("boom")
        def set(self, k, v): raise RuntimeError("boom")

    calls = [0]

    @tokeymeter.cache(store=BrokenStore())
    async def ask(prompt):
        calls[0] += 1
        return "ok"

    assert await ask("x") == "ok"
    assert await ask("x") == "ok"
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_async_semantic_cache(tmp_path):
    sem = SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.95,
        encoder=_bag_of_letters,
    )

    calls = [0]

    @tokeymeter.cache(
        model="gpt-4o-mini",
        semantic=True,
        semantic_cache=sem,
        prompt_arg="prompt",
    )
    async def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = await ask(prompt="abcdef")  # miss
    r2 = await ask(prompt="fedcba")  # semantic hit
    assert r1 == r2
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_async_single_encode_per_miss(tmp_path):
    """A semantic miss must encode the prompt exactly ONCE — not twice.
    A semantic hit must also encode once (just for lookup).
    An exact-match hit must not encode at all (short-circuits before semantic).

    This guards the lookup_by_embedding / store_by_embedding optimization.
    """
    encode_count = [0]

    def counting_encoder(text):
        encode_count[0] += 1
        return _bag_of_letters(text)

    sem = SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.95,
        encoder=counting_encoder,
    )

    @tokeymeter.cache(
        model="gpt-4o-mini",
        semantic=True,
        semantic_cache=sem,
        prompt_arg="prompt",
    )
    async def ask(prompt):
        return f"r-{prompt}"

    # Cold miss → 1 encode (used for both lookup and store)
    encode_count[0] = 0
    await ask(prompt="hello world")
    assert encode_count[0] == 1, f"expected 1 encode/miss, got {encode_count[0]}"

    # Semantic hit on anagram → 1 encode (lookup only)
    encode_count[0] = 0
    await ask(prompt="dlrow olleh")
    assert encode_count[0] == 1, f"expected 1 encode/semantic-hit, got {encode_count[0]}"

    # Exact hit on same prompt → 0 encodes (short-circuits before semantic layer)
    encode_count[0] = 0
    await ask(prompt="hello world")
    assert encode_count[0] == 0, f"exact hit should not encode, got {encode_count[0]}"


@pytest.mark.asyncio
async def test_async_semantic_does_not_block_event_loop(tmp_path):
    """The async semantic encode runs in a worker thread; the event loop
    must remain responsive for other coroutines."""
    import time

    def slow_encoder(text: str):
        time.sleep(0.05)  # 50ms sync blocking work
        return _bag_of_letters(text)

    sem = SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.5,
        encoder=slow_encoder,
    )

    @tokeymeter.cache(semantic=True, semantic_cache=sem, prompt_arg="prompt")
    async def ask(prompt):
        return "r"

    bg_ticks = [0]

    async def background():
        for _ in range(10):
            bg_ticks[0] += 1
            await asyncio.sleep(0.01)

    await asyncio.gather(ask(prompt="hello"), background())
    # If the encode blocked the loop, bg_ticks would be 0 or 1.
    # With to_thread, it should be 3+.
    assert bg_ticks[0] >= 3, f"event loop blocked, only {bg_ticks[0]} ticks"


# =================================================================
#                  STREAMING @tokeymeter.cache_stream TESTS
# =================================================================

@pytest.mark.asyncio
async def test_stream_basic_cache():
    """First call materializes the stream; second call replays cached chunks."""
    calls = [0]

    @tokeymeter.cache_stream(model="gpt-4o-mini")
    async def stream_chat(prompt):
        calls[0] += 1
        for word in prompt.split():
            yield word + " "

    chunks1 = [c async for c in stream_chat("the quick brown fox")]
    chunks2 = [c async for c in stream_chat("the quick brown fox")]

    assert chunks1 == ["the ", "quick ", "brown ", "fox "]
    assert chunks2 == ["the ", "quick ", "brown ", "fox "]
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_stream_does_not_cache_on_consumer_early_break():
    """If the consumer breaks out early, the partial response MUST NOT be cached
    — otherwise the next call would get a truncated response."""
    calls = [0]

    @tokeymeter.cache_stream()
    async def stream_chat(prompt):
        calls[0] += 1
        for word in ["the ", "quick ", "brown ", "fox "]:
            yield word

    partial = []
    async for chunk in stream_chat("the quick brown fox"):
        partial.append(chunk)
        if len(partial) == 2:
            break
    assert partial == ["the ", "quick "]

    full = [c async for c in stream_chat("the quick brown fox")]
    assert full == ["the ", "quick ", "brown ", "fox "]
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_stream_does_not_cache_on_generator_exception():
    """If the underlying generator raises mid-stream, don't cache."""
    calls = [0]

    @tokeymeter.cache_stream()
    async def flaky_stream(prompt):
        calls[0] += 1
        yield "first "
        yield "second "
        if calls[0] == 1:
            raise RuntimeError("upstream blew up")
        yield "third "

    with pytest.raises(RuntimeError, match="upstream"):
        async for _ in flaky_stream("x"):
            pass

    chunks = [c async for c in flaky_stream("x")]
    assert chunks == ["first ", "second ", "third "]
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_stream_preserves_chunk_granularity_on_replay():
    """Cached replay should yield individual chunks, not one big chunk."""
    @tokeymeter.cache_stream()
    async def stream(prompt):
        for word in ["alpha", "beta", "gamma"]:
            yield word

    _ = [c async for c in stream("x")]
    chunks = [c async for c in stream("x")]
    assert chunks == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_stream_semantic_cache(tmp_path):
    sem = SemanticCache(
        path=str(tmp_path / "sem_stream.db"),
        threshold=0.95,
        encoder=_bag_of_letters,
    )

    calls = [0]

    @tokeymeter.cache_stream(
        model="gpt-4o-mini",
        semantic=True,
        semantic_cache=sem,
        prompt_arg="prompt",
    )
    async def stream(prompt):
        calls[0] += 1
        for word in ["alpha ", "beta ", "gamma "]:
            yield word

    chunks1 = [c async for c in stream(prompt="abcdef")]
    chunks2 = [c async for c in stream(prompt="fedcba")]  # semantic hit

    assert chunks1 == chunks2 == ["alpha ", "beta ", "gamma "]
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_stream_rejects_non_async_generator():
    """@cache_stream must reject regular async functions and sync generators."""
    with pytest.raises(TypeError, match="async generator"):
        @tokeymeter.cache_stream()
        async def not_a_generator(prompt):
            return "ok"

    with pytest.raises(TypeError, match="async generator"):
        @tokeymeter.cache_stream()
        def sync_gen(prompt):
            yield "ok"


@pytest.mark.asyncio
async def test_stream_real_time_passthrough_on_miss():
    """On miss, consumer should see each chunk in real time as produced
    — not all-at-once after the generator completes."""
    import time

    @tokeymeter.cache_stream()
    async def stream(prompt):
        for i in range(3):
            await asyncio.sleep(0.02)
            yield f"chunk-{i}"

    seen_at = []
    start = time.perf_counter()
    async for chunk in stream("x"):
        seen_at.append((chunk, time.perf_counter() - start))

    times = [t for _, t in seen_at]
    assert len(times) == 3
    assert times[0] < 0.04, f"first chunk too late: {times[0]:.3f}s"
    assert times[2] - times[0] >= 0.03, "chunks not arriving incrementally"
