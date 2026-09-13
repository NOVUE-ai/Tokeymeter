"""The model on the record must be the model the caller actually used.

WHY THIS FILE EXISTS
--------------------
A wrapped SDK client is decorated once, but the caller picks a model per
request, so the decorator is bound with a placeholder. For a period, that
placeholder was what every record carried — and the damage was not cosmetic:

  * cost was computed from a generic fallback price. Measured against a real
    OpenAI invoice: $0.0250 reported for ~$0.0039 of actual usage, 6.5x high,
    while the request and token counts reconciled EXACTLY (98 / 22,842). A
    ledger that agrees on tokens and disagrees on money is worse than one that
    is obviously broken.
  * chargeback by model collapsed every model into one meaningless bucket.
  * a model allowlist compared the placeholder against the permitted names and
    refused EVERY call, INCLUDING THE APPROVED ONE. Turning on `only:` would
    have taken production down.

This exact defect was found and fixed once before, and returned in a later
refactor because no test guarded it. That is what this file is for.
"""
import collections
import concurrent.futures as cf
import itertools

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.execution.integrations import openai as oai
from tokeymeter.engines.governance import rules as R
from tokeymeter.engines.economics.pricing import estimate_cost
from tokeymeter.engines.economics.usage import set_reported_usage
import tokeymeter.engines.economics.savings as sv

_seq = itertools.count()


class _FakeOpenAI:
    def __init__(self):
        outer = self
        self.upstream = 0

        class _Msg:
            def __init__(self, c): self.content = c

        class _Choice:
            def __init__(self, c): self.message = _Msg(c)

        class _Usage:
            def __init__(self, i, o):
                self.prompt_tokens, self.completion_tokens = i, o
                self.total_tokens = i + o

        class _Resp:
            def __init__(self, c, i, o):
                self.id = f"chatcmpl-{next(_seq)}"
                self.choices = [_Choice(c)]
                self.usage = _Usage(i, o)
                self.model = "reported-by-provider"

        class _Completions:
            def create(self, *, model, messages, **kw):
                outer.upstream += 1
                return _Resp("ok", 100, 20)

        class _Chat:
            def __init__(self): self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    yield
    R.clear_rules()
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()


def _records():
    return list(sv._tracker._iter_records())


def test_the_record_carries_the_model_the_caller_asked_for():
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    client.chat.completions.create(model="gpt-4o-mini",
                                   messages=[{"role": "user", "content": "hi"}])
    recs = _records()
    assert recs, "the call produced no record at all"
    assert recs[0]["model"] == "gpt-4o-mini"
    assert recs[0]["model"] != "_default"


def test_the_cost_matches_that_models_price():
    """The number a customer reconciles against their invoice."""
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    client.chat.completions.create(model="gpt-4o-mini",
                                   messages=[{"role": "user", "content": "hi"}])
    r = _records()[0]
    expected = estimate_cost("gpt-4o-mini", r["input_tokens"], r["output_tokens"])
    assert abs(r["estimated_cost"] - expected) < 1e-9


def test_a_model_allowlist_permits_the_approved_model():
    """The failure that would take production down: `only:` refusing the very
    model it names."""
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"only": ["gpt-4o-secure"]}}]))
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    with tokeymeter.data_class("PHI"):
        client.chat.completions.create(
            model="gpt-4o-secure", messages=[{"role": "user", "content": "p"}])
    assert _records()[0]["model"] == "gpt-4o-secure"


def test_a_model_allowlist_still_refuses_an_unapproved_model():
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"only": ["gpt-4o-secure"]}}]))
    fake = _FakeOpenAI()
    client = oai.wrap(fake, semantic=False)
    with pytest.raises(tokeymeter.ModelNotPermitted):
        with tokeymeter.data_class("PHI"):
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "p"}])
    assert fake.upstream == 0          # and the call never went out


def test_chargeback_separates_models():
    """Note the distinct content per call: two identical requests would make
    the second a cache hit, which correctly does not count as executed. That
    is the same trap that has produced three wrong test expectations in this
    project."""
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    for i, m in enumerate(("gpt-4o-mini", "gpt-4o", "gpt-4o-mini")):
        client.chat.completions.create(
            model=m, messages=[{"role": "user", "content": f"q-{i}-{m}"}])
    rows = {r["model"]: r["executed_requests"]
            for r in tokeymeter.chargeback_report(group_by=("model",))["rows"]}
    assert rows == {"gpt-4o-mini": 2, "gpt-4o": 1}


def test_concurrent_calls_do_not_see_each_others_model():
    """A ContextVar rather than an attribute, so eight threads billing four
    models never cross."""
    client = oai.wrap(_FakeOpenAI(), semantic=False)

    def work(i):
        m = f"model-{i % 4}"
        client.chat.completions.create(
            model=m, messages=[{"role": "user", "content": f"u{i}"}])
        return m

    with cf.ThreadPoolExecutor(8) as ex:
        sent = list(ex.map(work, range(80)))
    recorded = collections.Counter(r["model"] for r in _records())
    assert dict(recorded) == dict(collections.Counter(sent))


def test_the_plain_decorator_keeps_its_own_model():
    """The side channel must not leak into ordinary decorated functions."""
    @tokeymeter.cache(model="claude-sonnet-4-5")
    def direct(prompt):
        set_reported_usage(100, 20)
        return "x"

    direct("a")
    assert _records()[0]["model"] == "claude-sonnet-4-5"


def test_the_channel_is_cleared_after_a_call():
    """A leaked value would mislabel the NEXT unrelated call."""
    from tokeymeter.decorator import _PER_CALL_MODEL
    client = oai.wrap(_FakeOpenAI(), semantic=False)
    client.chat.completions.create(model="gpt-4o-mini",
                                   messages=[{"role": "user", "content": "hi"}])
    assert _PER_CALL_MODEL.get() is None
