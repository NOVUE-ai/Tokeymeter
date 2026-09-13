"""Regression tests for the exception-policy pass (#3).

Policy (docs/EXCEPTION_POLICY.md): pure-logic paths handle EXPECTED exceptions
quietly and fail safe, but UNEXPECTED exceptions (Tokeymeter bugs) are surfaced as
`internal_error:<where>` degraded events instead of vanishing — while the call
still succeeds. I/O boundaries keep failing open with source-specific events.
"""
import tokeymeter
import tokeymeter.decorator as dec
from tokeymeter import degraded
from tokeymeter.storage import MemoryStore


def test_unexpected_internal_bug_is_surfaced_not_hidden():
    """An unexpected error in pure-logic key composition must surface as an
    internal_error event, yet the call must still succeed (fail-safe miss)."""
    tokeymeter.set_default_store(MemoryStore())
    degraded.clear_subscribers()
    orig = dec.make_cache_key

    def buggy(*a, **k):
        raise RuntimeError("simulated internal bug")

    dec.make_cache_key = buggy
    try:
        @tokeymeter.cache(model="gpt-4o")
        def ask(p):
            return f"real:{p}"

        assert ask("hello") == "real:hello"   # fail-safe: call still works
        counts = degraded.degraded_counts()
        assert counts.get("internal_error:key_generation", 0) >= 1
    finally:
        dec.make_cache_key = orig
        degraded.clear_subscribers()


def test_key_generation_bug_fails_safe_as_miss_not_wrong_hit():
    """Even when key logic breaks, two different prompts must never collide onto
    one cached value — broken keying degrades to recompute, never a wrong hit."""
    tokeymeter.set_default_store(MemoryStore())
    degraded.clear_subscribers()
    orig = dec.make_cache_key
    dec.make_cache_key = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        calls = {"n": 0}

        @tokeymeter.cache(model="gpt-4o")
        def ask(p):
            calls["n"] += 1
            return f"answer-for:{p}"

        a = ask("prompt A")
        b = ask("prompt B")
        assert a == "answer-for:prompt A"
        assert b == "answer-for:prompt B"     # B never gets A's value
        assert calls["n"] == 2                 # broken keying -> always recompute
    finally:
        dec.make_cache_key = orig
        degraded.clear_subscribers()


def test_expected_arg_type_errors_stay_quiet():
    """Token counting on an exotic/uncountable arg is EXPECTED — it must NOT raise
    an internal_error (that would be noise), and must not break the call."""
    tokeymeter.set_default_store(MemoryStore())
    degraded.clear_subscribers()

    class Uncountable:
        def __iter__(self):
            raise TypeError("not countable")
        def __str__(self):
            raise TypeError("nope")

    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        return "ok"

    assert ask(Uncountable()) == "ok"          # call succeeds
    # no internal_error from the expected uncountable-arg path
    assert not any(s.startswith("internal_error") for s in degraded.degraded_counts())
    degraded.clear_subscribers()


def test_io_boundary_still_fails_open_with_source_event():
    """A backend get/set failure remains fail-open and emits a source-specific
    (not internal_error) event — boundaries are handled differently from logic."""
    degraded.clear_subscribers()

    class BrokenStore(MemoryStore):
        def get(self, k):
            raise ConnectionError("down")
        def set(self, k, v):
            raise ConnectionError("down")

    @tokeymeter.cache(model="gpt-4o", store=BrokenStore())
    def ask(p):
        return f"real:{p}"

    assert ask("x") == "real:x"                # fail-open
    counts = degraded.degraded_counts()
    assert counts.get("store.get", 0) >= 1     # source-specific, not internal_error
    assert not any(s.startswith("internal_error") for s in counts)
    degraded.clear_subscribers()


# ---- module-wide boundary visibility (memory.py / semantic.py) ----

def test_memory_wrapper_failures_are_visible_and_fail_open():
    """The memory async wrappers are fail-open BOUNDARIES (broad catch is correct
    per policy), but must not be SILENT: a store failure surfaces a memory.* event
    while every call still returns."""
    import asyncio
    from tokeymeter.memory import ConversationMemory

    class BoomStore:
        def append_turn(self, *a, **k): raise RuntimeError("down")
        def get_turns(self, *a, **k): raise RuntimeError("down")
        def clear_session(self, *a, **k): raise RuntimeError("down")
        def list_sessions(self, *a, **k): raise RuntimeError("down")
        def set_summary(self, *a, **k): raise RuntimeError("down")

    degraded.clear_subscribers()
    mem = ConversationMemory(max_turns=10, store=BoomStore())

    async def run():
        await mem.add_turn("s", "u", "a")
        await mem.get_context("s")
        await mem.clear_session("s")
        await mem.inspect_session("s")
        await mem.list_sessions()
        await mem.get_full_context("s")

    asyncio.run(run())   # must not raise (fail-open)
    sources = {s for s in degraded.degraded_counts() if s.startswith("memory.")}
    for op in ("add_turn", "get_context", "clear_session",
               "inspect_session", "list_sessions", "get_full_context"):
        assert f"memory.{op}" in sources, f"memory.{op} should be visible"
    degraded.clear_subscribers()


def test_semantic_clear_expected_vec_absence_stays_quiet():
    """clear() on a cache whose vec table was never created hits the EXPECTED
    OperationalError path — that is narrowed and must NOT emit a degraded event."""
    import tempfile, os
    from tokeymeter.semantic import SemanticCache
    degraded.clear_subscribers()
    c = SemanticCache(path=os.path.join(tempfile.mkdtemp(), "s.db"),
                      encoder=lambda t: [0.0, 0.0, 0.0], dim=3)
    c.store("hello", "world")
    c.clear()   # vec table may be absent -> expected -> quiet
    assert not any(s.startswith("semantic.clear") for s in degraded.degraded_counts())
    degraded.clear_subscribers()


def test_semantic_clear_outer_failure_is_visible():
    """If clear() can't even open the store, surface semantic.clear (not silent)."""
    import tempfile, os, sqlite3
    from tokeymeter.semantic import SemanticCache
    degraded.clear_subscribers()
    c = SemanticCache(path=os.path.join(tempfile.mkdtemp(), "s.db"),
                      encoder=lambda t: [0.0, 0.0, 0.0], dim=3)

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk gone")

    c._connect = boom  # type: ignore[assignment]
    c.clear()          # must not raise
    assert degraded.degraded_counts().get("semantic.clear", 0) >= 1
    degraded.clear_subscribers()
