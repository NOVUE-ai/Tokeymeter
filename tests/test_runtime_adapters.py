"""W2 battery — real adapters (K3) + error taxonomy (EXEC-4) + tool calls
(EXEC-6). The wave gate lives here: test_usage_truth_parity_* pins that the
kernel path emits an envelope IDENTICAL to the direct shipped-wrapper path.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import tokeymeter.events as events
from tokeymeter.runtime import (
    AnthropicAdapter, AsyncOpenAIAdapter, AuthError, Kernel, KernelRequest,
    MalformedRequest, OpenAIAdapter, ProviderDown, RateLimited,
    ReliabilityEngine, RuntimeConfig, TransientError, classify,
)
from tokeymeter.runtime.providers import ExecutionEngine


# ---------------------------------------------------------------- fakes ----
def _oai_response(text="hello", pt=11, ct=7, tool_calls=None, model="gpt-x"):
    return NS(choices=[NS(message=NS(content=text, tool_calls=tool_calls),
                          finish_reason="stop")],
              usage=NS(prompt_tokens=pt, completion_tokens=ct,
                       total_tokens=pt + ct),
              model=model, id="fake-1")


class FakeOpenAI:
    def __init__(self, responder=None):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, *, model, messages, stream=False, **kw):
                outer.calls += 1
                if responder:
                    return responder(model=model, messages=messages,
                                     stream=stream, **kw)
                if stream:
                    return iter([NS(choices=[NS(delta=NS(content="he"))]),
                                 NS(choices=[NS(delta=NS(content="llo"))])])
                return _oai_response(model=model)

        self.chat = NS(completions=_Completions())


class FakeAnthropic:
    def __init__(self):
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, *, model, messages, max_tokens, stream=False, **kw):
                outer.calls += 1
                if stream:
                    return iter([NS(type="content_block_delta",
                                    delta=NS(text="hi"))])
                return NS(content=[NS(type="text", text="hi from claude")],
                          usage=NS(input_tokens=9, output_tokens=4),
                          model=model, id="fake-a1", stop_reason="end_turn")

        self.messages = _Messages()


class FakeAsyncOpenAI:
    def __init__(self):
        self.calls = 0
        outer = self

        class _Completions:
            async def create(self, *, model, messages, stream=False, **kw):
                outer.calls += 1
                if stream:
                    async def gen():
                        yield NS(choices=[NS(delta=NS(content="a"))])
                        yield NS(choices=[NS(delta=NS(content="sync"))])
                    return gen()
                return _oai_response(model=model)

        self.chat = NS(completions=_Completions())


def make_kernel(adapter, model="default", **cfg):
    k = Kernel(RuntimeConfig(cfg or {"cache": {"enabled": False}})).start()
    ex = ExecutionEngine()
    ex.register_adapter(adapter, models=[model], default=True)
    k.register_engine(ex)
    return k, ex


def collect_events(fn):
    seen = []
    h = events.subscribe(seen.append)
    try:
        fn()
    finally:
        events.unsubscribe(h)
    return seen


def envelope(evts):
    """The usage/cost envelope: what Economics sees. Parity is exact."""
    return [(e.event_type, e.hit, e.model, e.input_tokens, e.output_tokens,
             round(e.estimated_cost_usd, 10)) for e in evts]


# --------------------------------------------------- adapter round-trips ---
def test_openai_adapter_infer_roundtrip_and_health():
    k, _ = make_kernel(OpenAIAdapter(FakeOpenAI(), semantic=False))
    resp = k.process(KernelRequest(payload="ping"))
    assert resp.payload.choices[0].message.content == "hello"
    hs = OpenAIAdapter(FakeOpenAI(), semantic=False).health_check()
    assert hs.healthy


def test_anthropic_adapter_infer_roundtrip_and_health():
    k, _ = make_kernel(AnthropicAdapter(FakeAnthropic(), semantic=False))
    resp = k.process(KernelRequest(payload="ping"))
    assert resp.payload.content[0].text == "hi from claude"
    assert AnthropicAdapter(FakeAnthropic(), semantic=False).health_check().healthy


def test_openai_stream_chunk_order():
    adapter = OpenAIAdapter(FakeOpenAI(), semantic=False)
    k, _ = make_kernel(adapter)
    ctx = {"request": KernelRequest(payload="p"), "meta": {"model": "default"}}
    chunks = [c.choices[0].delta.content for c in adapter.stream_infer(ctx)]
    assert chunks == ["he", "llo"]


# ------------------------------------------------ THE WAVE GATE: parity ----
def test_usage_truth_parity_openai():
    """Kernel-path envelope ≡ direct-wrapper envelope, field for field.
    Same fake client config, same prompt, same wrap opts."""
    from tokeymeter.engines.execution.integrations import openai as oai
    prompt = "parity probe: identical envelopes required"

    direct = collect_events(lambda: oai.wrap(FakeOpenAI(), semantic=False)
                            .chat.completions.create(
                                model="gpt-x",
                                messages=[{"role": "user", "content": prompt}]))

    k, _ = make_kernel(OpenAIAdapter(FakeOpenAI(), semantic=False),
                       model="gpt-x")
    kernel = collect_events(
        lambda: k.process(KernelRequest(payload=prompt, model="gpt-x")))

    assert envelope(direct) == envelope(kernel), (
        "USAGE-TRUTH PARITY BROKEN: kernel path and direct wrapper path "
        "emitted different economics envelopes")
    assert envelope(kernel), "no events captured — parity test is vacuous"


def test_usage_truth_parity_openai_cache_hit_second_call():
    """Parity must hold through the shipped cache too: two identical calls on
    each path → same (miss, store, hit) envelope sequence."""
    from tokeymeter.engines.execution.integrations import openai as oai
    prompt = "parity with cache"

    def twice_direct():
        w = oai.wrap(FakeOpenAI(), semantic=False)
        for _ in range(2):
            w.chat.completions.create(
                model="gpt-x", messages=[{"role": "user", "content": prompt}])

    def twice_kernel():
        k, _ = make_kernel(OpenAIAdapter(FakeOpenAI(), semantic=False),
                           model="gpt-x")
        for _ in range(2):
            k.process(KernelRequest(payload=prompt, model="gpt-x"))

    assert envelope(collect_events(twice_direct)) == \
        envelope(collect_events(twice_kernel))


# ----------------------------------------------------- async parity --------
def test_async_sync_result_parity():
    sync_a = OpenAIAdapter(FakeOpenAI(), semantic=False)
    async_a = AsyncOpenAIAdapter(FakeAsyncOpenAI(), semantic=False)
    ctx = {"request": KernelRequest(payload="p", model="gpt-x"),
           "meta": {"model": "gpt-x"}}
    s = sync_a.infer(dict(ctx, meta=dict(ctx["meta"])))
    a = asyncio.run(async_a.ainfer(dict(ctx, meta=dict(ctx["meta"]))))
    assert (s.choices[0].message.content, s.usage.prompt_tokens,
            s.usage.completion_tokens) == \
           (a.choices[0].message.content, a.usage.prompt_tokens,
            a.usage.completion_tokens)


def test_async_stream_order():
    async_a = AsyncOpenAIAdapter(FakeAsyncOpenAI(), semantic=False)
    ctx = {"request": KernelRequest(payload="p"), "meta": {"model": "gpt-x"}}

    async def run():
        return [c.choices[0].delta.content
                async for c in async_a.astream_infer(ctx)]
    assert asyncio.run(run()) == ["a", "sync"]


def test_async_adapter_sync_infer_fails_loud():
    with pytest.raises(RuntimeError):
        AsyncOpenAIAdapter(FakeAsyncOpenAI(), semantic=False).infer(
            {"request": KernelRequest(payload="p"), "meta": {}})


# ------------------------------------------------- EXEC-4 taxonomy ---------
class _E(Exception):
    def __init__(self, status=None, headers=None):
        self.status_code = status
        self.response = NS(headers=headers or {}, status_code=status)


@pytest.mark.parametrize("exc,expected,retryable", [
    (type("RateLimitError", (_E,), {})(429), RateLimited, True),
    (type("AuthenticationError", (_E,), {})(401), AuthError, False),
    (type("APITimeoutError", (_E,), {})(), TransientError, True),
    (type("BadRequestError", (_E,), {})(400), MalformedRequest, False),
    (type("InternalServerError", (_E,), {})(500), ProviderDown, True),
    (_E(503), ProviderDown, True),                      # status fallback
    (ValueError("mystery"), TransientError, True),      # unknown → safe default
], ids=["429", "401", "timeout", "400", "500", "503-fallback", "unknown"])
def test_taxonomy_table(exc, expected, retryable):
    typed = classify(exc, "openai")
    assert isinstance(typed, expected) and typed.retryable is retryable
    assert typed.provider == "openai" and typed.original_type


def test_retry_after_attached():
    exc = type("RateLimitError", (_E,), {})(429, headers={"Retry-After": "2.5"})
    assert classify(exc, "openai").retry_after == 2.5


def test_no_payload_in_typed_errors():
    secret = "SUPER-SECRET-PROMPT-TEXT"
    exc = ValueError(f"failed while sending {secret}")
    typed = classify(exc, "openai")
    assert secret not in str(typed)                     # L4: message is clean
    assert typed.original_type == "ValueError"


def test_rel_routes_on_type_auth_never_retried():
    """The seam: Reliability retries Transient, never retries Auth."""
    calls = {"n": 0}

    def dying(model, messages, **kw):
        calls["n"] += 1
        raise type("AuthenticationError", (_E,), {})(401)

    # Baseline first: the shipped wrapper is fail-open and makes its own
    # second inner call on exception (measured, not assumed).
    from tokeymeter.engines.execution.integrations import openai as oai
    baseline = FakeOpenAI(responder=dying)
    with pytest.raises(Exception):
        oai.wrap(baseline, semantic=False).chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "p"}])
    wrapper_baseline = calls["n"]

    calls["n"] = 0
    adapter = OpenAIAdapter(FakeOpenAI(responder=dying), semantic=False)
    k, ex = make_kernel(adapter, model="gpt-x",
                        cache={"enabled": False},
                        reliability={"max_retries": 3})
    k.engines.register(ReliabilityEngine(ex))
    rel = k.engines.get("reliability")
    with pytest.raises(AuthError):
        k.process(KernelRequest(payload="p", model="gpt-x"))
    # Reliability contributed ZERO extra attempts for a non-retryable error:
    assert calls["n"] == wrapper_baseline
    assert rel.retries_used == 0


def test_rel_still_retries_transient():
    calls = {"n": 0}

    def flaky(model, messages, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise type("APITimeoutError", (_E,), {})()
        return _oai_response(text="recovered", model=model)

    adapter = OpenAIAdapter(FakeOpenAI(responder=flaky), semantic=False)
    k, ex = make_kernel(adapter, model="gpt-x",
                        cache={"enabled": False},
                        reliability={"max_retries": 3})
    k.engines.register(ReliabilityEngine(ex))
    resp = k.process(KernelRequest(payload="p", model="gpt-x"))
    assert resp.payload.choices[0].message.content == "recovered"
    assert calls["n"] == 3


# ------------------------------------------------- EXEC-6 tool calls -------
def test_tool_call_roundtrip_and_count_only_in_meta():
    secret_args = '{"city": "CONFIDENTIAL-LOCATION"}'
    tcs = [NS(id="c1", function=NS(name="get_weather", arguments=secret_args)),
           NS(id="c2", function=NS(name="get_time", arguments="{}"))]

    def with_tools(model, messages, **kw):
        return _oai_response(text=None, tool_calls=tcs, model=model)

    k, _ = make_kernel(OpenAIAdapter(FakeOpenAI(responder=with_tools),
                                     semantic=False), model="gpt-x")
    resp = k.process(KernelRequest(payload="what's the weather", model="gpt-x"))
    # round-trip: caller receives the tool calls untouched
    assert resp.payload.choices[0].message.tool_calls[0].function.name == \
        "get_weather"
    # meta carries COUNT only
    assert resp.metadata["tool_calls"] == 2


def test_tool_args_never_in_trace():
    secret_args = '{"account": "SECRET-ACCT-9"}'
    tcs = [NS(id="c1", function=NS(name="lookup", arguments=secret_args))]
    k, _ = make_kernel(OpenAIAdapter(
        FakeOpenAI(responder=lambda model, messages, **kw:
                   _oai_response(text=None, tool_calls=tcs, model=model)),
        semantic=False), model="gpt-x")
    resp = k.process(KernelRequest(payload="q", model="gpt-x"))
    import json
    blob = json.dumps(resp.trace) + json.dumps(resp.metadata)
    assert "SECRET-ACCT-9" not in blob and "lookup" not in blob   # L4 pin
