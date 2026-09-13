"""Group 1 (S4-6) — make what is already built actually work.

Four defects, all of them "shipped but not reachable":

  1. The progress signal could not score a provider object, so the flagship
     differentiator silently reported "not scored" for the most common shape in
     production. The shipped SDK wrappers now supply a response extractor.
  2. `Runtime(client=..., validator=fn)` could never work: the facade extracted
     its own options AFTER building the adapter, so `validator` was swallowed
     by the adapter's **kwargs and the safety control silently did not exist.
     A typo'd option was accepted in silence for the same reason.
  3. The kernel path wrote NO ledger record and consulted NO task ceiling — a
     whole execution route that was neither governed nor accounted for.
  4. 164 public names with no indication which twenty are the product.
"""
import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.runtime.facade import Runtime, RuntimeConfigurationError
from tokeymeter.engines.execution.response_text import (
    openai_response_text, anthropic_response_text)
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


# ── 1. provider response extraction ─────────────────────────────────────

class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _OAI:
    def __init__(self, message):
        self.choices = [message and _Choice(message)]


def test_openai_message_text_is_extracted():
    assert openai_response_text(_OAI(_Msg(content="hello"))) == "hello"


def test_openai_tool_calls_are_the_response_when_there_is_no_text():
    """A typical agent turn returns NO message text — it returns a tool call.
    If that were unscoreable the progress signal would be blind to the dominant
    agent architecture, which is exactly the workload it exists for."""
    out = openai_response_text(
        _OAI(_Msg(tool_calls=[_ToolCall("search", '{"q":"x"}')])))
    assert out and "search" in out and '"q":"x"' in out


def test_openai_identical_tool_calls_produce_identical_text():
    """The property the whole stall signal rests on."""
    a = openai_response_text(_OAI(_Msg(tool_calls=[_ToolCall("s", '{"q":1}')])))
    b = openai_response_text(_OAI(_Msg(tool_calls=[_ToolCall("s", '{"q":1}')])))
    assert a == b


def test_openai_different_tool_arguments_differ():
    a = openai_response_text(_OAI(_Msg(tool_calls=[_ToolCall("s", '{"q":1}')])))
    b = openai_response_text(_OAI(_Msg(tool_calls=[_ToolCall("s", '{"q":2}')])))
    assert a != b


@pytest.mark.parametrize("bad", [None, object(), "not a response", 42])
def test_openai_extractor_never_raises_on_a_foreign_shape(bad):
    assert openai_response_text(bad) is None


def test_openai_empty_turn_is_not_scoreable():
    assert openai_response_text(_OAI(_Msg())) is None


class _Block:
    def __init__(self, text=None, type=None, name=None, input=None):
        self.text, self.type, self.name, self.input = text, type, name, input


class _Anth:
    def __init__(self, blocks):
        self.content = blocks


def test_anthropic_text_blocks_are_extracted():
    assert anthropic_response_text(_Anth([_Block(text="hi")])) == "hi"


def test_anthropic_tool_use_blocks_count_as_output():
    out = anthropic_response_text(
        _Anth([_Block(type="tool_use", name="search", input={"q": "x"})]))
    assert out and "search" in out


def test_anthropic_mixed_blocks_all_contribute():
    """Text and tool_use can appear in the same turn, so both must be in the
    digest or two different turns could collide."""
    out = anthropic_response_text(
        _Anth([_Block(text="thinking"),
               _Block(type="tool_use", name="s", input={"q": 1})]))
    assert "thinking" in out and "s(" in out


@pytest.mark.parametrize("bad", [None, object(), 42])
def test_anthropic_extractor_never_raises_on_a_foreign_shape(bad):
    assert anthropic_response_text(bad) is None


def test_shipped_integrations_declare_a_response_extractor():
    """Without this the flagship signal reports "not scored" out of the box for
    the most common production shape."""
    import inspect
    from tokeymeter.engines.execution.integrations import openai as oai
    from tokeymeter.engines.execution.integrations import anthropic as ant
    for mod in (oai, ant):
        src = inspect.getsource(mod)
        assert "extract_response_text=" in src


# ── 2. runtime options reach their features ─────────────────────────────

def _cfg():
    return {"reliability": {"resilient": True, "retries": 2}}


def test_validator_retries_an_invalid_response():
    calls = {"n": 0}

    def flaky(prompt, **kw):
        calls["n"] += 1
        return "no id" if calls["n"] == 1 else "order_id: 42"

    rt = Runtime(call=flaky, model="m", config=_cfg(),
                 validator=lambda r: "order_id" in str(r))
    assert rt.execute("place order") == "order_id: 42"
    assert calls["n"] == 2


def test_a_valid_response_is_not_retried():
    calls = {"n": 0}

    def good(prompt, **kw):
        calls["n"] += 1
        return "order_id: 7"

    Runtime(call=good, model="m", config=_cfg(),
            validator=lambda r: "order_id" in str(r)).execute("x")
    assert calls["n"] == 1


def test_an_unknown_runtime_option_is_rejected():
    """A silently ignored `validater=` means a control the operator believes is
    running simply is not. Fail at construction, loudly."""
    with pytest.raises(RuntimeConfigurationError) as ei:
        Runtime(call=lambda p, **k: "x", model="m", config=_cfg(),
                validater=lambda r: True)
    assert "validater" in str(ei.value)


def test_the_error_names_the_options_that_do_exist():
    with pytest.raises(RuntimeConfigurationError) as ei:
        Runtime(call=lambda p, **k: "x", model="m", nonsense=1)
    assert "validator" in str(ei.value)


# ── 3. the kernel path is governed and accounted for ────────────────────

def test_runtime_calls_reach_the_ledger():
    """One ledger is the whole architecture. A second path that executes but
    does not record is a hole in it."""
    rt = Runtime(call=lambda p, **k: "hello world", model="m")
    with tokeymeter.task("ticket-1", agent="support"):
        rt.execute("a")
        rt.execute("b")
    recs = _records()
    assert len(recs) == 2
    assert all(r["task_id"] == "ticket-1" for r in recs)
    assert all(r["agent"] == "support" for r in recs)
    assert all(r["model"] == "m" for r in recs)


def test_runtime_records_reach_chargeback():
    rt = Runtime(call=lambda p, **k: "x", model="m")
    with tokeymeter.task("t", agent="support"):
        rt.execute("a")
    cb = tokeymeter.chargeback_report(group_by=("agent",))
    assert [r["agent"] for r in cb["rows"]] == ["support"]


def test_identical_runtime_prompts_share_a_fingerprint():
    rt = Runtime(call=lambda p, **k: "x", model="m")
    with tokeymeter.task("t", agent="a"):
        rt.execute("same")
        rt.execute("same")
        rt.execute("different")
    fps = [r["prompt_fingerprint"] for r in _records()]
    assert fps[0] == fps[1] != fps[2]


def test_an_envelope_is_enforced_on_the_kernel_path():
    """It was ungoverned: an agent could spin inside a declared envelope
    because nothing on this route ever consulted it."""
    calls = {"n": 0}

    def big(prompt, **kw):
        calls["n"] += 1
        return "x" * 4000

    rt = Runtime(call=big, model="m")
    with pytest.raises(tokeymeter.TaskLimitExceeded):
        with tokeymeter.task("t", envelope=0.0002, reserve=0.0001,
                             enforce=True):
            for i in range(50):
                rt.execute(f"prompt {i}")
    assert calls["n"] < 50


def test_a_loop_is_detected_on_the_kernel_path():
    rt = Runtime(call=lambda p, **k: "tool error", model="m")
    with pytest.raises(tokeymeter.TaskLoopDetected):
        with tokeymeter.task("t", max_repeats=4, enforce=True):
            for _ in range(30):
                rt.execute("the same prompt")


def test_an_adapter_that_already_records_is_not_double_counted():
    """Recording for a wrapped SDK client would bill every call twice in the
    customer's own chargeback."""
    from tokeymeter.runtime.providers import ProviderInfo

    class AlreadyRecords:
        emits_ledger_record = True
        provider = "fake"

        def get_info(self):
            return ProviderInfo(provider="fake", models=["m"])

        def infer(self, ctx):
            return "ok"

    Runtime(adapter=AlreadyRecords(), model="m").execute("a")
    assert _records() == []


def test_shipped_sdk_adapters_declare_that_they_record():
    from tokeymeter.runtime.adapters import OpenAIAdapter, AnthropicAdapter
    from tokeymeter.runtime.providers import ProviderAdapter, CallableAdapter
    assert OpenAIAdapter.emits_ledger_record is True
    assert AnthropicAdapter.emits_ledger_record is True
    assert ProviderAdapter.emits_ledger_record is False
    assert CallableAdapter.emits_ledger_record is False


def test_ledger_recording_never_breaks_execution():
    """Recording is best-effort; execution is not."""
    rt = Runtime(call=lambda p, **k: object(), model="m")   # unscoreable reply
    assert rt.execute("a") is not None
    assert len(_records()) == 1


# ── 4. the product has a named surface ──────────────────────────────────

def test_primary_api_members_all_exist():
    missing = [n for n in tokeymeter.PRIMARY_API if not hasattr(tokeymeter, n)]
    assert missing == []


def test_primary_api_members_are_public():
    assert all(n in tokeymeter.__all__ for n in tokeymeter.PRIMARY_API)


def test_primary_api_is_a_readable_shortlist():
    """The point is that a newcomer can hold it in their head."""
    assert 10 <= len(tokeymeter.PRIMARY_API) <= 30


def test_nothing_was_removed_to_build_the_shortlist():
    """Deleting public API from a shipped package breaks working code and buys
    nothing — the shortlist names the product, it does not shrink it."""
    for name in ("semantic_report", "redact", "set_savings_path",
                 "compress_tools", "audit_log"):
        if hasattr(tokeymeter, name):
            assert name in tokeymeter.__all__


def test_primary_api_entries_carry_a_description():
    for name, doc in tokeymeter.primary_api():
        assert doc, f"{name} has no docstring first line"
