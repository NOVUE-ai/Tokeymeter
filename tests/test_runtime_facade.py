"""W3 battery — Runtime facade (K4) + async kernel path (KA-1).

Wave gate lives at the bottom: the README quickstart block executes
verbatim. Receipt honesty law pinned: figures equal real counters, absent
sources render `—`, payload text never appears.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tokeymeter import Runtime, RuntimeConfigurationError
from tokeymeter.runtime import Kernel, KernelRequest, KernelStopped, RuntimeConfig
from tokeymeter.runtime.providers import CallableAdapter, ExecutionEngine

SECRET = "ULTRA-SECRET-PROMPT-CONTENT"


def rt(fn=None, **kw):
    return Runtime(call=fn or (lambda p: "ok:" + p), receipt=kw.pop(
        "receipt", "never"), **kw)


# ------------------------------------------------------------- facade -----
def test_facade_e2e_callable_full_pipeline():
    r = rt()
    assert r.execute("hello") == "ok:hello"
    engines = {t["engine"] for t in r.last.trace}
    assert engines == {"governance", "economics", "reliability", "trust", "execution"}
    ok, bad = r._trust.verify()
    assert ok and bad == -1


def test_facade_requires_a_provider():
    with pytest.raises(RuntimeConfigurationError):
        Runtime()


def test_facade_client_autodetection():
    class FakeOpenAIClient:
        chat = NS(completions=NS(create=lambda **kw: NS(
            choices=[NS(message=NS(content="o", tool_calls=None))],
            usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kw.get("model"), id="x")))

    class FakeAnthropicClient:
        messages = NS(create=lambda **kw: NS(
            content=[NS(type="text", text="a")],
            usage=NS(input_tokens=1, output_tokens=1),
            model=kw.get("model"), id="y", stop_reason="end_turn"))

    ro = Runtime(client=FakeOpenAIClient(), receipt="never", semantic=False)
    ra = Runtime(client=FakeAnthropicClient(), receipt="never", semantic=False)
    assert ro.execute("q").choices[0].message.content == "o"
    assert ra.execute("q").content[0].text == "a"
    with pytest.raises(RuntimeConfigurationError):
        Runtime(client=object(), receipt="never")


def test_exceptions_surface_unchanged():
    def boom(p):
        raise ValueError("provider exploded")
    with pytest.raises(ValueError):
        rt(boom, config={"reliability": {"max_retries": 0}}).execute("x")


def test_streaming_chunks_sync():
    class StreamAdapter(CallableAdapter):
        def stream_infer(self, ctx):
            yield "chunk1"
            yield "chunk2"
    r = Runtime(adapter=StreamAdapter(lambda p: "full"), receipt="never")
    assert list(r.execute("p", stream=True)) == ["chunk1", "chunk2"]


# ------------------------------------------------------------ receipt -----
def _receipt_of(r, prompt="hi"):
    resp_payload = r.execute(prompt)
    return resp_payload, r.render_receipt(r.last, r.last_events, 12.3)


def test_receipt_fields_equal_counters_and_dash_for_absent():
    r = rt()
    _, receipt = _receipt_of(r)
    assert "✓ Routed callable:default" in receipt        # real meta
    assert "✓ Compression —" in receipt                  # no source → dash
    assert "✓ Verified ✓" in receipt                     # trust chain real
    assert f"✓ Trace {r.last.request_id[:12]}" in receipt
    # callable adapter emits no economics events → cost/saved must be dash,
    # never an invented number (HONESTY LAW):
    assert "✓ Cost —" in receipt and "✓ Saved —" in receipt
    assert not re.search(r"Cost \$\d", receipt)


def test_receipt_content_blind():
    r = rt()
    r.execute(SECRET)
    receipt = r.render_receipt(r.last, r.last_events, 1.0)
    assert SECRET not in receipt and "ULTRA-SECRET" not in receipt


def test_receipt_quiet_env_suppresses(monkeypatch, capsys):
    monkeypatch.setenv("TOKEYMETER_QUIET", "1")
    r = Runtime(call=lambda p: "x", receipt="auto")
    r.execute("p")
    assert capsys.readouterr().out == ""


def test_receipt_non_tty_suppresses(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    Runtime(call=lambda p: "x", receipt="auto").execute("p")
    assert capsys.readouterr().out == ""


def test_receipt_always_prints_once(capsys):
    r = Runtime(call=lambda p: "x", receipt="always")
    r.execute("p")
    r.execute("p")
    out = capsys.readouterr().out
    assert out.count("✓ Routed") == 1                    # first call only


# --------------------------------------------------------- KA-1 async -----
def test_aparity_results():
    r = rt()
    sync_out = r.execute("same prompt")
    async_out = asyncio.run(rt().aexecute("same prompt"))
    assert sync_out == async_out == "ok:same prompt"


def test_async_streaming():
    class AStream(CallableAdapter):
        async def astream_infer(self, ctx):
            yield "a1"
            yield "a2"

    r = Runtime(adapter=AStream(lambda p: "full"), receipt="never")

    async def run():
        stream = await r.aexecute("p", stream=True)
        return [c async for c in stream]
    assert asyncio.run(run()) == ["a1", "a2"]


def test_adrain_under_load():
    release = threading.Event()
    started = threading.Event()

    def slow(p):
        started.set()
        release.wait(timeout=5)
        return "done"

    k = Kernel(RuntimeConfig({"kernel": {"drain_timeout_s": 5.0}})).start()
    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(slow), models=["default"],
                        default=True)
    k.register_engine(ex)

    async def scenario():
        task = asyncio.create_task(k.aprocess(KernelRequest(payload="p")))
        await asyncio.to_thread(started.wait, 5)
        stopper = asyncio.create_task(asyncio.to_thread(k.shutdown))
        await asyncio.sleep(0.1)
        with pytest.raises(KernelStopped):
            await k.aprocess(KernelRequest(payload="new"))   # refused
        release.set()
        resp = await task
        await stopper
        return resp

    assert asyncio.run(scenario()).payload == "done"     # in-flight completed


def test_mixed_sync_async_safety():
    """Sync process() on a worker thread while aprocess() runs on the loop —
    same kernel, no interference, loop never blocked by the sync adapter."""
    k = Kernel(RuntimeConfig({})).start()
    ex = ExecutionEngine()
    ex.register_adapter(
        CallableAdapter(lambda p: (time.sleep(0.05), f"r:{p}")[1]),
        models=["default"], default=True)
    k.register_engine(ex)

    async def scenario():
        a = asyncio.create_task(k.aprocess(KernelRequest(payload="async")))
        s = asyncio.to_thread(k.process, KernelRequest(payload="sync"))
        ra, rs = await asyncio.gather(a, s)
        return ra.payload, rs.payload

    assert asyncio.run(scenario()) == ("r:async", "r:sync")


# ------------------------------------------- THE WAVE GATE: README --------
def test_readme_quickstart_executes_verbatim():
    # Explicit UTF-8: read_text() inherits the platform default, which is
    # cp1252 on Windows — and README.md contains bytes that cp1252 cannot
    # decode, so this test crashed on Windows runners.
    readme = (Path(__file__).parent.parent / "README.md").read_text(
        encoding="utf-8")
    m = re.search(r"```python quickstart-runtime\n(.*?)```", readme, re.S)
    assert m, "README quickstart-runtime block missing"
    code = m.group(1)
    namespace: dict = {}
    exec(compile(code, "README.quickstart", "exec"), namespace)  # noqa: S102
