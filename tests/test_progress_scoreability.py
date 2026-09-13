"""Progress scoreability (S4-5) — refuse to score what cannot be measured.

THE BUG THIS CLOSES, and it was in our own flagship signal:

An object with no meaningful `__str__` falls back to CPython's default repr,
which embeds its memory address — `<generator object g at 0x7f9968...>`. That
address changes every call, so hashing it gave two IDENTICAL responses two
DIFFERENT fingerprints. Progress read a perfect 1.00 for an agent that was
completely stuck, and nothing errored or warned. Verified before the fix: two
identical generator responses produced different fingerprints.

THE PRINCIPLE: a metric that lies is worse than a missing metric. The same
discipline that made `plan` predict enforcement to $0.00000000 error applies
here — the progress column has to be trustworthy or nobody acts on it.

WHAT IS AND IS NOT SCOREABLE:
  scoreable      str, bytes, a list of chunks (what cache_stream assembles),
                 any object whose text form is content
  not scoreable  a live generator or iterator — reading it would consume the
                 caller's stream — and any object whose text form is an
                 identity repr
"""
import asyncio

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution import task as T
from tokeymeter.engines.governance.agents import agent_report, render_agents
import tokeymeter.engines.economics.savings as sv


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _records():
    return list(sv._tracker._iter_records())


# ── what is scoreable ───────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    "a text response",
    b"raw bytes",
    ["tool ", "error"],                 # the shape cache_stream assembles
    ("chunk one", "chunk two"),
])
def test_stable_content_is_scoreable_and_deterministic(value):
    a, b = T.fingerprint_response(value), T.fingerprint_response(value)
    assert a is not None and a == b


def test_different_content_gets_different_fingerprints():
    assert T.fingerprint_response("one") != T.fingerprint_response("two")


def test_a_provider_object_is_refused_without_an_extractor():
    """The attack that forced this design: a real SDK response carries a unique
    request id, so its text form differs on every call even when the content is
    identical. Scoring it would give a stuck agent a perfect 1.00 forever."""
    import uuid

    class SDKResponse:
        def __init__(self, content):
            self.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            self.content = content

        def __str__(self):
            return f"ChatCompletion(id={self.id!r}, content={self.content!r})"

    assert T.fingerprint_response(SDKResponse("tool error")) is None


def test_a_provider_object_is_scoreable_through_extract_text():
    import uuid

    class SDKResponse:
        def __init__(self, content):
            self.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            self.content = content

        def __str__(self):
            return f"ChatCompletion(id={self.id!r}, content={self.content!r})"

    ex = lambda r: r.content
    assert (T.fingerprint_response(SDKResponse("err"), ex)
            == T.fingerprint_response(SDKResponse("err"), ex))
    assert (T.fingerprint_response(SDKResponse("err"), ex)
            != T.fingerprint_response(SDKResponse("other"), ex))


def test_a_broken_extractor_never_breaks_a_call():
    def boom(_):
        raise RuntimeError("bad extractor")
    assert T.fingerprint_response(object(), boom) is None


def test_a_chunk_list_containing_non_content_is_refused():
    """A container inherits whatever instability its elements carry."""
    class NoStr:
        pass
    assert T.fingerprint_response(["ok", NoStr()]) is None


# ── what is NOT scoreable, and must be refused ──────────────────────────

def test_a_live_generator_is_refused_not_guessed():
    """Reading it would consume the caller's stream. Refuse, do not guess."""
    def g():
        yield "tool "
        yield "error"
    g1, g2 = g(), g()                   # both held alive: no address reuse
    assert T.fingerprint_response(g1) is None
    assert T.fingerprint_response(g2) is None


def test_a_generator_is_not_consumed_by_fingerprinting():
    """The caller must still get every chunk."""
    def g():
        yield "a"
        yield "b"
    gen = g()
    T.fingerprint_response(gen)
    assert list(gen) == ["a", "b"]


def test_an_identity_repr_object_is_refused():
    """A default repr embeds a memory address that changes every call."""
    class NoStr:
        pass
    o1, o2 = NoStr(), NoStr()
    assert T.fingerprint_response(o1) is None
    assert T.fingerprint_response(o2) is None


def test_a_dict_is_refused_because_it_may_carry_a_request_id():
    """Conservative on purpose: we cannot tell a stable dict from one carrying
    a per-call id, and refusing costs a column entry while guessing costs the
    signal's credibility."""
    assert T.fingerprint_response({"content": "error"}) is None


def test_an_async_iterator_is_refused():
    class AGen:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration
    assert T.fingerprint_response(AGen()) is None


@pytest.mark.parametrize("value", [None, "", b""])
def test_empty_is_not_scoreable(value):
    assert T.fingerprint_response(value) is None


def test_an_exploding_object_never_breaks_a_call():
    class Exploding:
        def __str__(self):
            raise RuntimeError("boom")
    assert T.fingerprint_response(Exploding()) is None


# ── the streaming path still measures correctly ─────────────────────────

def test_cache_stream_assembles_chunks_and_is_scoreable():
    """A streamed response IS measurable once assembled, which is what
    cache_stream passes in. This is the common streaming path and it must keep
    working."""
    @tokeymeter.cache_stream(model="m")
    async def stream(p):
        set_reported_usage(1000, 200)
        for c in ("tool ", "error"):
            yield c

    async def run():
        with tokeymeter.task("t", agent="streamer"):
            async for _ in stream("alpha"):
                pass
            async for _ in stream("beta"):
                pass
    asyncio.run(run())
    fps = [r["response_fingerprint"] for r in _records()]
    assert all(f is not None for f in fps)
    assert len(set(fps)) == 1          # identical responses, identical prints


# ── the number must say "not scored", never a fabricated one ────────────

def test_novelty_is_none_when_nothing_could_be_scored():
    """Dividing by every sample would report 0.0 — indistinguishable from a
    total stall — for a task whose responses simply could not be measured."""
    @tokeymeter.cache(model="m")
    def gen_agent(p):
        set_reported_usage(1200, 200)

        def g():
            yield "tool error"
        return g()

    history = []
    with tokeymeter.task("t", agent="gen", stall_window=8) as st:
        for i in range(20):
            history.append(f"turn {i}")
            gen_agent(tuple(history))
    snap = st.snapshot()
    assert snap["progress_novelty"] is None
    assert snap["responses_scored"] == 0
    assert snap["stalled"] is False     # never guess a stall from no evidence


def test_an_unmeasurable_agent_is_not_reported_as_healthy():
    """Before the fix this agent scored a perfect 1.00. It must now read as
    not scored, with the reason stated."""
    @tokeymeter.cache(model="m")
    def gen_agent(p):
        set_reported_usage(1200, 200)

        def g():
            yield "tool error"
        return g()

    history = []
    for i in range(4):
        with tokeymeter.task(f"gen-{i}", agent="streaming-agent"):
            for j in range(8):
                history.append(f"t{j}")
                gen_agent(tuple(history) + (i,))
    rep = agent_report(_records())
    row = next(r for r in rep["agents"] if r["agent"] == "streaming-agent")
    assert row["median_progress"] is None
    assert row["responses_scored"] == 0
    text = render_agents(rep)
    assert "not scored" in text


def test_report_states_partial_scoring_coverage():
    """A task with 20 calls of which 2 could be measured must not present a
    progress figure as fact."""
    toggle = {"n": 0}

    @tokeymeter.cache(model="m")
    def mixed(p):
        set_reported_usage(1200, 200)
        toggle["n"] += 1
        if toggle["n"] % 2:
            def g():
                yield "unmeasurable"
            return g()
        return f"measurable {toggle['n']}"

    for i in range(4):
        with tokeymeter.task(f"mix-{i}", agent="mixed-agent"):
            for j in range(10):
                mixed(f"{i}-{j}")
    rep = agent_report(_records())
    row = next(r for r in rep["agents"] if r["agent"] == "mixed-agent")
    assert 0 < row["scoring_coverage_pct"] < 100
    assert row["responses_scored"] < row["responses_executed"]
    assert "scored on" in render_agents(rep)


def test_a_task_with_too_few_scoreable_responses_is_not_scored():
    """Two responses that happen to agree are noise, not evidence."""
    calls = {"n": 0}

    @tokeymeter.cache(model="m")
    def mostly_unmeasurable(p):
        set_reported_usage(1200, 200)
        calls["n"] += 1
        if calls["n"] <= 2:
            return "measurable"

        def g():
            yield "unmeasurable"
        return g()

    with tokeymeter.task("t", agent="sparse"):
        for i in range(12):
            mostly_unmeasurable(f"p{i}")
    rep = agent_report(_records())
    row = next(r for r in rep["agents"] if r["agent"] == "sparse")
    assert row["median_progress"] is None      # 2 samples is not a score


# ── the signal still works where it can measure ─────────────────────────

def test_a_measurable_stuck_agent_is_still_halted():
    """The fix must not have blunted the detector it was protecting."""
    @tokeymeter.cache(model="m")
    def step(messages, tokens):
        set_reported_usage(tokens, 200)
        return "tool error: cannot parse"

    history = []
    with pytest.raises(tokeymeter.TaskStalled):
        with tokeymeter.task("stuck", stall_window=8, enforce=True):
            for i in range(40):
                history.append(f"retry {i}")
                step(tuple(history), 1000 + i * 400)


def test_a_measurable_working_agent_is_still_clean():
    @tokeymeter.cache(model="m")
    def step(messages, tokens, n):
        set_reported_usage(tokens, 200)
        return f"extracted {n}"

    history = []
    with tokeymeter.task("working", stall_window=8, enforce=True) as st:
        for i in range(30):
            history.append(f"s{i}")
            step(tuple(history), 1000 + i * 400, i)
    assert st.snapshot()["stalled"] is False
    assert st.snapshot()["progress_novelty"] == 1.0


def test_scoreability_never_breaks_a_request():
    """Whatever the response shape, the caller gets their answer."""
    class Exploding:
        def __str__(self):
            raise RuntimeError("boom")

    @tokeymeter.cache(model="m")
    def weird(p):
        set_reported_usage(100, 50)
        return Exploding()

    with tokeymeter.task("t", agent="weird", stall_window=4, enforce=True):
        for i in range(10):
            assert isinstance(weird(f"p{i}"), Exploding)


# ── the most severe defect the audit found ──────────────────────────────

def test_prompt_extractor_is_never_used_as_a_response_extractor():
    """`extract_text` means "pull the PROMPT out for token estimation" in this
    codebase — the shipped OpenAI and Anthropic integrations set it to the
    request messages. Wiring it into response fingerprinting hashed a GROWING
    prompt and called the result a response print: thirty identical failures
    produced thirty distinct fingerprints, progress read 1.00, and the stall
    was never detected. Confident and wrong is worse than "not scored"."""
    class SDKResp:
        def __init__(self, content):
            self.content = content

    history = []

    @tokeymeter.cache(model="m",
                      extract_text=lambda *_a, **_k: " ".join(history))
    def agent(prompt, tokens):
        set_reported_usage(tokens, 200)
        return SDKResp("tool error: cannot parse")      # SAME answer every time

    with tokeymeter.task("t", agent="a", stall_window=8, enforce=True) as st:
        for i in range(30):
            history.append(f"turn {i}")                  # prompt GROWS
            agent(tuple(history), 1000 + i * 400)

    prints = {r["response_fingerprint"] for r in _records()}
    assert prints == {None}                              # not scored, honestly
    assert st.snapshot()["progress_novelty"] is None     # never a fake 1.00


def test_a_declared_response_extractor_catches_the_stall():
    class SDKResp:
        def __init__(self, content):
            self.content = content

    history = []

    @tokeymeter.cache(model="m", extract_response_text=lambda r: r.content)
    def agent(prompt, tokens):
        set_reported_usage(tokens, 200)
        return SDKResp("tool error: cannot parse")

    with pytest.raises(tokeymeter.TaskStalled):
        with tokeymeter.task("t", agent="a", stall_window=8, enforce=True):
            for i in range(30):
                history.append(f"turn {i}")
                agent(tuple(history), 1000 + i * 400)


def test_a_response_extractor_does_not_flag_a_working_agent():
    class SDKResp:
        def __init__(self, content):
            self.content = content

    history = []

    @tokeymeter.cache(model="m", extract_response_text=lambda r: r.content)
    def agent(prompt, tokens, n):
        set_reported_usage(tokens, 200)
        return SDKResp(f"new finding {n}")

    with tokeymeter.task("t", agent="a", stall_window=8, enforce=True) as st:
        for i in range(30):
            history.append(f"t{i}")
            agent(tuple(history), 1000 + i * 400, i)
    assert st.snapshot()["stalled"] is False
    assert st.snapshot()["progress_novelty"] == 1.0
