"""A refusal must reach the caller, and must stop the upstream call.

THE BUG THIS CLOSES, found only by building a real-API test:

The wrapped client's fail-open handler caught every exception and called the
UNWRAPPED client. So a halted task looked bounded — 4 metered calls — while 20
calls went upstream. Sixteen were billed by the provider and invisible to the
ledger.

That breaks two claims at once: enforcement stops nothing on the primary
integration path, and "the ledger reconciles with your provider invoice" is
false by exactly the number of refused calls.

THE DISTINCTION: fail-open exists so a bug in METERING never breaks a caller's
request. A refusal is not a bug — it is the feature. Catching it and calling the
provider anyway is the one thing fail-open must never do.
"""
import itertools

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.execution.integrations import openai as oai
import tokeymeter.engines.economics.savings as sv

_seq = itertools.count()


class _FakeOpenAI:
    """Counts REAL upstream calls, which is the number that matters here."""

    def __init__(self):
        self.upstream = 0
        outer = self

        class _Msg:
            def __init__(self, c):
                self.content = c

        class _Choice:
            def __init__(self, c):
                self.message = _Msg(c)

        class _Usage:
            def __init__(self, i, o):
                self.prompt_tokens, self.completion_tokens = i, o
                self.total_tokens = i + o

        class _Resp:
            def __init__(self, c, i, o):
                self.id = f"chatcmpl-{next(_seq)}"
                self.choices = [_Choice(c)]
                self.usage = _Usage(i, o)
                self.model = "gpt-4o-mini"

        class _Completions:
            def create(self, *, model, messages, **kw):
                outer.upstream += 1
                text = " ".join(m["content"] for m in messages)
                return _Resp("same answer", max(10, len(text) // 4), 5)

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_halt_handlers()
    yield
    tokeymeter.clear_halt_handlers()
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()


def _drive(client, asks=20, identical=False, **task_kwargs):
    """`identical=True` sends a byte-identical request every turn, which is what
    max_repeats exists to catch — a growing conversation has a unique
    fingerprint per call and no repeat to count."""
    history = [{"role": "system", "content": "sys"}]
    raised = None
    try:
        with tokeymeter.task("t", agent="a", enforce=True, **task_kwargs):
            for i in range(asks):
                if identical:
                    msgs = [{"role": "user", "content": "the same request"}]
                else:
                    history.append({"role": "user",
                                    "content": f"attempt {i} " + "x" * 200})
                    msgs = history
                r = client.chat.completions.create(model="gpt-4o-mini",
                                                   messages=msgs)
                if not identical:
                    history.append({"role": "assistant",
                                    "content": r.choices[0].message.content})
    except tokeymeter.TaskLimitExceeded as exc:
        raised = type(exc).__name__
    metered = len(list(sv._tracker._iter_records()))
    return raised, metered


@pytest.mark.parametrize("kwargs,expected", [
    ({"stall_window": 4}, "TaskStalled"),
    ({"max_calls": 5}, "TaskCallLimitExceeded"),
    ({"envelope": 0.000001, "reserve": 0.0000005}, "TaskEnvelopeExceeded"),
])
def test_a_refusal_reaches_the_caller_through_the_wrapped_client(kwargs,
                                                                 expected):
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    raised, _ = _drive(client, **kwargs)
    assert raised == expected, (
        "the caller never learned it was stopped, so their agent loop keeps "
        "running")


@pytest.mark.parametrize("kwargs", [
    {"stall_window": 4}, {"max_calls": 5},
    {"envelope": 0.000001, "reserve": 0.0000005},
])
def test_no_upstream_call_escapes_a_refusal(kwargs):
    """The number that decides whether enforcement is real: what the PROVIDER
    saw. Measured before the fix: 4 metered, 20 upstream."""
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    _, metered = _drive(client, asks=20, **kwargs)
    assert fake.upstream == metered, (
        f"{fake.upstream - metered} call(s) were billed by the provider and "
        f"never recorded — the ledger cannot reconcile with an invoice")
    assert fake.upstream < 20


def test_the_ledger_matches_what_the_provider_saw():
    """The reconciliation claim, stated as a test."""
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    _drive(client, asks=20, stall_window=4)
    assert len(list(sv._tracker._iter_records())) == fake.upstream


def test_a_compliance_refusal_also_propagates():
    from tokeymeter.engines.governance import rules as R
    tokeymeter.register_pricing("gpt-4o-mini", input_per_1m=0.15,
                                output_per_1m=0.60)
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"only": ["approved-model"]}}]))
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    try:
        with pytest.raises(tokeymeter.ModelNotPermitted):
            with tokeymeter.data_class("PHI"):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "patient record"}])
        assert fake.upstream == 0          # the call never went out
    finally:
        R.clear_rules()


def test_fail_open_still_works_when_the_METER_breaks():
    """The distinction that makes the fix safe: a bug in metering must still
    never break a caller's request."""
    import tokeymeter.engines.execution.task as T
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    original = T.before_call

    def explode(*a, **k):
        raise RuntimeError("meter exploded")

    T.before_call = explode
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        assert r.choices[0].message.content == "same answer"
        assert fake.upstream == 1
    finally:
        T.before_call = original


def test_a_halt_still_notifies_on_the_wrapped_path():
    seen = []
    tokeymeter.on_halt(seen.append)
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    _drive(client, stall_window=4)
    assert len(seen) == 1


def test_a_repeated_identical_call_is_refused_and_propagates():
    """max_repeats catches byte-identical retries. A growing conversation has a
    unique fingerprint per call, which is exactly why stall detection exists
    alongside it — the two are orthogonal."""
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    raised, _ = _drive(client, identical=True, max_repeats=3)
    assert raised == "TaskLoopDetected"
    assert fake.upstream <= 3
