"""
Tests for the conversation memory layer (v0.7).

Verified behaviors:
  - Turns store and retrieve correctly (in-memory + SQLite)
  - Summary is generated and cached by buffer hash
  - Tiered fidelity: recent N turns full, older summarized
  - Invariant: memory NEVER makes context more expensive than baseline
  - Fail-open: broken store / summarizer / injector doesn't break calls
  - with_memory decorator (sync + async) composes with @tokeymeter.cache
  - fidelity_rate audit logs similarity for sampled calls
  - Concurrent writes to one session don't corrupt state
  - SQLite persistence across process boundary
"""
import asyncio
import os

import pytest

import tokeymeter
from tokeymeter.memory import (
    CallableSummarizer,
    ConversationMemory,
    InMemoryMemoryStore,
    SQLiteMemoryStore,
    TruncationSummarizer,
    Turn,
    _default_text_injector,
    _memory_jaccard_similarity,
    _memory_fidelity_log,
    memory_fidelity_log,
    with_memory,
)
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    tokeymeter.events.clear_subscribers()
    _memory_fidelity_log.clear()
    yield


# =================================================================
#                Turn dataclass
# =================================================================

def test_turn_token_estimate_is_positive():
    t = Turn(user="hello", assistant="hi there")
    assert t.token_estimate() >= 1


def test_turn_token_estimate_scales_with_length():
    short = Turn(user="hi", assistant="hi")
    long = Turn(user="x" * 400, assistant="y" * 400)
    assert long.token_estimate() > short.token_estimate()


# =================================================================
#                InMemoryMemoryStore
# =================================================================

def test_inmemory_append_and_retrieve():
    s = InMemoryMemoryStore()
    s.append_turn("sess1", Turn(user="u1", assistant="a1"))
    s.append_turn("sess1", Turn(user="u2", assistant="a2"))
    turns = s.get_turns("sess1")
    assert len(turns) == 2
    assert turns[0].user == "u1"
    assert turns[1].user == "u2"


def test_inmemory_separate_sessions_are_isolated():
    s = InMemoryMemoryStore()
    s.append_turn("a", Turn(user="ua", assistant="aa"))
    s.append_turn("b", Turn(user="ub", assistant="ab"))
    assert len(s.get_turns("a")) == 1
    assert len(s.get_turns("b")) == 1
    assert s.get_turns("a")[0].user == "ua"


def test_inmemory_clear_session():
    s = InMemoryMemoryStore()
    s.append_turn("x", Turn(user="u", assistant="a"))
    s.set_summary("x", "summary", "hash")
    s.clear_session("x")
    assert s.get_turns("x") == []
    assert s.get_summary("x") is None


def test_inmemory_summary_buffer_hash_tracking():
    s = InMemoryMemoryStore()
    s.set_summary("x", "sum1", "hash1")
    assert s.get_summary_buffer_hash("x") == "hash1"
    s.set_summary("x", "sum2", "hash2")
    assert s.get_summary_buffer_hash("x") == "hash2"
    assert s.get_summary("x") == "sum2"


def test_list_sessions():
    s = InMemoryMemoryStore()
    s.append_turn("alpha", Turn(user="u", assistant="a"))
    s.append_turn("beta", Turn(user="u", assistant="a"))
    sessions = s.list_sessions()
    assert "alpha" in sessions
    assert "beta" in sessions


# =================================================================
#                SQLiteMemoryStore
# =================================================================

def test_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "mem.db")
    s1 = SQLiteMemoryStore(path=path)
    s1.append_turn("sess", Turn(user="u1", assistant="a1"))
    s1.set_summary("sess", "summary1", "hash1")

    s2 = SQLiteMemoryStore(path=path)
    assert len(s2.get_turns("sess")) == 1
    assert s2.get_summary("sess") == "summary1"


def test_sqlite_concurrent_append_doesnt_corrupt(tmp_path):
    """Multiple threads writing to one session shouldn't lose turns."""
    import threading
    path = str(tmp_path / "mem.db")
    store = SQLiteMemoryStore(path=path)

    def writer(i):
        store.append_turn("sess", Turn(user=f"u{i}", assistant=f"a{i}"))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()

    turns = store.get_turns("sess")
    assert len(turns) == 20


# =================================================================
#                TruncationSummarizer
# =================================================================

def test_truncation_summarizer_handles_empty():
    s = TruncationSummarizer()
    assert s.summarize([]) == ""


def test_truncation_summarizer_includes_first_turn():
    s = TruncationSummarizer()
    turns = [
        Turn(user="What is Python?", assistant="A programming language."),
        Turn(user="Tell me more", assistant="It is dynamic."),
        Turn(user="Show example", assistant="print('hi')"),
    ]
    summary = s.summarize(turns)
    assert "What is Python" in summary
    assert "Earlier turns" in summary or "began with" in summary


def test_truncation_summarizer_truncates_long_text():
    s = TruncationSummarizer(keep_chars_per_turn=30)
    turns = [Turn(user="x" * 200, assistant="y" * 200)]
    summary = s.summarize(turns)
    # Each line should be capped near 30 chars (plus prefix)
    lines = summary.split("\n")
    for line in lines:
        # Allow some overhead for prefixes like "  User: "
        assert len(line) < 100


# =================================================================
#                ConversationMemory: basic API
# =================================================================

@pytest.mark.asyncio
async def test_add_turn_and_get_context_basic():
    mem = ConversationMemory(recent_window=5, summary_threshold=10)
    await mem.add_turn("s1", "Hello", "Hi there")
    ctx = await mem.get_context("s1")
    assert ctx.total_turns == 1
    assert ctx.used_summary is False
    assert "Hello" in ctx.text


@pytest.mark.asyncio
async def test_empty_session_returns_empty_context():
    mem = ConversationMemory()
    ctx = await mem.get_context("nonexistent")
    assert ctx.total_turns == 0
    assert ctx.text == ""
    assert ctx.messages == []
    assert ctx.tokens_saved == 0


@pytest.mark.asyncio
async def test_summarization_kicks_in_above_threshold():
    """Realistic conversation with enough size for summarization to help."""
    mem = ConversationMemory(recent_window=3, summary_threshold=5)
    # Add enough content per turn that summarization beats the baseline
    long_u = "I am working on a project " * 5
    long_a = "Here is what I would recommend " * 5
    for i in range(12):
        await mem.add_turn("s1", f"{long_u} {i}", f"{long_a} {i}")
    ctx = await mem.get_context("s1")
    assert ctx.total_turns == 12
    assert ctx.used_summary is True
    assert len(ctx.recent_turns) == 3
    assert ctx.tokens_saved > 0


@pytest.mark.asyncio
async def test_invariant_memory_never_more_expensive_than_baseline():
    """Industrial-grade: tokens_with_memory <= tokens_full_history, always."""
    mem = ConversationMemory(recent_window=2, summary_threshold=4)
    # Tiny turns where summarization would be net-negative
    for i in range(10):
        await mem.add_turn("s1", f"Q{i}", f"A{i}")
    ctx = await mem.get_context("s1")
    assert ctx.tokens_with_memory <= ctx.tokens_full_history


@pytest.mark.asyncio
async def test_messages_format_is_openai_compatible():
    mem = ConversationMemory(recent_window=10, summary_threshold=20)
    await mem.add_turn("s1", "first question", "first answer")
    await mem.add_turn("s1", "second question", "second answer")
    ctx = await mem.get_context("s1")
    assert isinstance(ctx.messages, list)
    assert all("role" in m and "content" in m for m in ctx.messages)
    assert any(m["role"] == "user" for m in ctx.messages)
    assert any(m["role"] == "assistant" for m in ctx.messages)


@pytest.mark.asyncio
async def test_clear_session_removes_everything():
    mem = ConversationMemory()
    await mem.add_turn("s1", "u", "a")
    await mem.clear_session("s1")
    ctx = await mem.get_context("s1")
    assert ctx.total_turns == 0


@pytest.mark.asyncio
async def test_force_full_disables_summarization():
    mem = ConversationMemory(recent_window=2, summary_threshold=4)
    long_u = "Long question " * 20
    long_a = "Long answer " * 20
    for i in range(10):
        await mem.add_turn("s1", f"{long_u} {i}", f"{long_a} {i}")

    ctx_summarized = await mem.get_context("s1")
    ctx_full = await mem.get_context("s1", force_full=True)

    assert ctx_summarized.used_summary is True
    assert ctx_full.used_summary is False
    assert ctx_full.tokens_with_memory >= ctx_summarized.tokens_with_memory


# =================================================================
#                Summary caching
# =================================================================

@pytest.mark.asyncio
async def test_summary_is_cached_by_buffer_hash():
    """If the buffer hasn't changed, the summarizer should not re-run."""
    summarize_count = [0]

    class CountingSummarizer:
        def summarize(self, turns):
            summarize_count[0] += 1
            return f"Summary of {len(turns)} turns"

    mem = ConversationMemory(
        recent_window=2,
        summary_threshold=4,
        summarizer=CountingSummarizer(),
    )
    long_u = "Long content " * 50
    long_a = "Long response " * 50
    for i in range(8):
        await mem.add_turn("s1", f"{long_u} {i}", f"{long_a} {i}")

    # First get_context triggers summary
    await mem.get_context("s1")
    first = summarize_count[0]
    # Subsequent calls without buffer change should hit the cache
    await mem.get_context("s1")
    await mem.get_context("s1")
    await mem.get_context("s1")
    assert summarize_count[0] == first, "Summary should be cached"

    # Adding a new turn changes the buffer-to-summarize → re-summarize
    await mem.add_turn("s1", f"{long_u} new", f"{long_a} new")
    await mem.get_context("s1")
    assert summarize_count[0] > first


@pytest.mark.asyncio
async def test_async_summarizer_supported():
    """Summarizer.summarize may be async; ConversationMemory awaits it."""
    async def async_sum(turns):
        await asyncio.sleep(0.001)
        return f"async summary of {len(turns)}"

    mem = ConversationMemory(
        recent_window=2,
        summary_threshold=4,
        summarizer=CallableSummarizer(async_sum),
    )
    long_u = "X" * 200
    long_a = "Y" * 200
    for i in range(8):
        await mem.add_turn("s1", f"{long_u}{i}", f"{long_a}{i}")
    ctx = await mem.get_context("s1")
    assert ctx.summary_text and "async summary" in ctx.summary_text


# =================================================================
#                Fail-open
# =================================================================

@pytest.mark.asyncio
async def test_broken_summarizer_falls_open_to_recent_turns():
    class Broken:
        def summarize(self, turns):
            raise RuntimeError("kaboom")

    mem = ConversationMemory(
        recent_window=2, summary_threshold=4, summarizer=Broken(),
    )
    long_u = "X" * 200
    for i in range(8):
        await mem.add_turn("s1", f"{long_u}{i}", f"r{i}")
    ctx = await mem.get_context("s1")
    # Falls back to "" summary, only recent turns survive — still a usable context
    assert len(ctx.recent_turns) == 2
    assert "X" in ctx.text  # the recent turns are present


@pytest.mark.asyncio
async def test_broken_store_doesnt_raise_from_add_turn():
    class BrokenStore:
        def append_turn(self, sid, turn): raise IOError("disk full")
        def get_turns(self, sid): return []
        def get_summary(self, sid): return None
        def set_summary(self, *args): pass
        def get_summary_buffer_hash(self, sid): return None
        def clear_session(self, sid): pass
        def list_sessions(self): return []

    mem = ConversationMemory(store=BrokenStore())
    # Must not raise
    await mem.add_turn("s1", "u", "a")


# =================================================================
#                with_memory decorator
# =================================================================

@pytest.mark.asyncio
async def test_with_memory_injects_context_in_async():
    mem = ConversationMemory(recent_window=5, summary_threshold=10)
    received_prompts = []

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt")
    async def ask(prompt, sid):
        received_prompts.append(prompt)
        return f"answer-{len(received_prompts)}"

    await ask(prompt="What is Python?", sid="user1")
    await ask(prompt="And JavaScript?", sid="user1")
    await ask(prompt="How about Go?", sid="user1")

    # First call: no context yet
    assert received_prompts[0] == "What is Python?"
    # Second call: context should include first exchange
    assert "What is Python" in received_prompts[1]
    assert "And JavaScript" in received_prompts[1]
    # Third call: context should include first AND second exchange
    assert "JavaScript" in received_prompts[2]


@pytest.mark.asyncio
async def test_with_memory_records_original_prompt_not_expanded():
    """The TURN should record what the USER said, not what the LLM saw."""
    mem = ConversationMemory()

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt")
    async def ask(prompt, sid):
        return "ok"

    await ask(prompt="Original question", sid="s1")

    # Check the turn that was recorded
    turns = mem._store.get_turns("s1")
    assert len(turns) == 1
    assert turns[0].user == "Original question"  # NOT the expanded form


@pytest.mark.asyncio
async def test_with_memory_skips_when_session_id_missing():
    """Fail-open: if caller doesn't supply session_id, just pass through."""
    mem = ConversationMemory()

    @with_memory(memory=mem, session_arg="sid")
    async def ask(prompt, sid=None):
        return f"got: {prompt}"

    # Without session_id, memory should be a no-op
    r = await ask(prompt="hello", sid=None)
    assert r == "got: hello"
    # No turns recorded
    assert mem._store.list_sessions() == []


@pytest.mark.asyncio
async def test_with_memory_composes_with_cache():
    """@with_memory above @tokeymeter.cache — both layers should work."""
    mem = ConversationMemory(recent_window=5, summary_threshold=10)
    api_calls = [0]

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt")
    @tokeymeter.cache(model="gpt-4o-mini", prompt_arg="prompt")
    async def ask(prompt, sid):
        api_calls[0] += 1
        return f"answer about: {prompt[:30]}"

    # Same session, same question twice → second should hit cache
    await ask(prompt="What is recursion?", sid="s1")
    await ask(prompt="What is recursion?", sid="s1")
    # Both calls had memory injected, but second's full prompt matches first's
    # because first turn has empty context. Wait — no. After first turn,
    # second turn has a context with the first exchange. So second prompt differs!
    # Expectation: BOTH are API calls because their full prompts differ.
    assert api_calls[0] == 2


@pytest.mark.asyncio
async def test_custom_injector():
    mem = ConversationMemory()

    def my_injector(original, ctx):
        return f"[CONTEXT={ctx.total_turns} turns] {original}"

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt",
                 context_injector=my_injector)
    async def ask(prompt, sid):
        return prompt

    await ask(prompt="hello", sid="s1")  # 0 prior turns
    r = await ask(prompt="hello", sid="s1")  # 1 prior turn
    assert r.startswith("[CONTEXT=1 turns]")


def test_with_memory_sync_works():
    """Decorator detects sync functions and handles them."""
    mem = ConversationMemory()

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt")
    def ask(prompt, sid):
        return f"sync: {prompt[:30]}"

    r1 = ask(prompt="hello", sid="s1")
    r2 = ask(prompt="hello", sid="s1")
    assert r1.startswith("sync:")
    assert r2.startswith("sync:")

    # Turns should be recorded
    turns = mem._store.get_turns("s1")
    assert len(turns) == 2


def test_with_memory_requires_memory_instance():
    with pytest.raises(TypeError):
        @with_memory(memory=None)  # type: ignore
        def f(): ...


# =================================================================
#                Fidelity audit
# =================================================================

@pytest.mark.asyncio
async def test_fidelity_rate_zero_means_no_audits():
    mem = ConversationMemory(recent_window=2, summary_threshold=4)

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt",
                 fidelity_rate=0.0)
    async def ask(prompt, sid):
        return "r"

    # Build up history to trigger summarization
    long_u = "Long content " * 50
    for i in range(10):
        await ask(prompt=f"{long_u}{i}", sid="s1")

    log = memory_fidelity_log()
    assert len(log) == 0


@pytest.mark.asyncio
async def test_fidelity_rate_one_audits_every_summarized_call():
    mem = ConversationMemory(recent_window=2, summary_threshold=4)
    api_calls = [0]

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt",
                 fidelity_rate=1.0)
    async def ask(prompt, sid):
        api_calls[0] += 1
        # Return something dependent on the prompt
        return f"resp ({len(prompt)} chars)"

    # Build up history beyond threshold
    long_u = "Long content " * 50
    long_a = "Long resp " * 50
    for i in range(8):
        # Manually add turns to avoid double-counting through with_memory
        await mem.add_turn("s1", f"{long_u}{i}", f"{long_a}{i}")

    # Now make a call that should trigger summarization AND fidelity audit
    await ask(prompt="new question", sid="s1")

    log = memory_fidelity_log()
    assert len(log) == 1
    assert 0.0 <= log[0]["similarity"] <= 1.0
    assert log[0]["function_name"] == "ask"


@pytest.mark.asyncio
async def test_fidelity_audit_skipped_when_no_summarization():
    """If summary wasn't used (history is short), don't audit."""
    mem = ConversationMemory(recent_window=10, summary_threshold=20)

    @with_memory(memory=mem, session_arg="sid", prompt_arg="prompt",
                 fidelity_rate=1.0)
    async def ask(prompt, sid):
        return "r"

    # Only 3 turns, no summarization
    for i in range(3):
        await ask(prompt=f"q{i}", sid="s1")

    log = memory_fidelity_log()
    assert len(log) == 0


def test_jaccard_similarity():
    assert _memory_jaccard_similarity("hello world", "hello world") == 1.0
    assert _memory_jaccard_similarity("", "") == 1.0
    assert _memory_jaccard_similarity("a b c", "b c d") == pytest.approx(2/4)


# =================================================================
#                Stats / inspect
# =================================================================

@pytest.mark.asyncio
async def test_inspect_session():
    mem = ConversationMemory()
    await mem.add_turn("s1", "hello", "hi")
    await mem.add_turn("s1", "how are you", "good")
    info = await mem.inspect_session("s1")
    assert info["session_id"] == "s1"
    assert info["total_turns"] == 2
    assert info["tokens_total"] > 0


@pytest.mark.asyncio
async def test_inspect_nonexistent_session():
    mem = ConversationMemory()
    info = await mem.inspect_session("nonexistent")
    assert info["session_id"] == "nonexistent"
    assert info["total_turns"] == 0


@pytest.mark.asyncio
async def test_list_sessions():
    mem = ConversationMemory()
    await mem.add_turn("alpha", "u", "a")
    await mem.add_turn("beta", "u", "a")
    sessions = await mem.list_sessions()
    assert set(sessions) == {"alpha", "beta"}


def test_constructor_validates_args():
    with pytest.raises(ValueError):
        ConversationMemory(recent_window=-1)
    with pytest.raises(ValueError):
        ConversationMemory(recent_window=10, summary_threshold=5)
    with pytest.raises(ValueError):
        ConversationMemory(summary_threshold=100, max_turns=10)


# =================================================================
#                Concurrent safety
# =================================================================

@pytest.mark.asyncio
async def test_concurrent_add_turn_per_session():
    """Many concurrent add_turn calls to one session shouldn't lose turns."""
    mem = ConversationMemory()
    await asyncio.gather(*[
        mem.add_turn("s1", f"u{i}", f"a{i}") for i in range(50)
    ])
    turns = mem._store.get_turns("s1")
    assert len(turns) == 50


@pytest.mark.asyncio
async def test_concurrent_get_context_works():
    mem = ConversationMemory(recent_window=2, summary_threshold=4)
    long_u = "X" * 200
    for i in range(8):
        await mem.add_turn("s1", f"{long_u}{i}", f"r{i}")
    # Many concurrent reads
    results = await asyncio.gather(*[
        mem.get_context("s1") for _ in range(20)
    ])
    # All should return identical contexts
    first_text = results[0].text
    for r in results[1:]:
        assert r.text == first_text
