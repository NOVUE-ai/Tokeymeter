"""Over-the-wire integration: real OpenAI SDK -> local HTTP server speaking the
vLLM/OpenAI response schema -> Tokeymeter's real wrapper -> the ledger.

This is the exact wiring a self-hoster runs (client = OpenAI(base_url=...);
wrapped = tokeymeter.integrations.openai.wrap(client)) — over a real socket,
through the SDK's real response parsing, not function stubs. Pinned:

  1. miss records carry the SERVER's reported token counts (token_source
     "reported"), through the SDK's pydantic parsing
  2. a repeated request is served from cache WITHOUT touching the server
     (server request-count is the ground truth)
  3. endpoint identity from the contextvar lands on wire-path records
  4. queue-wait: if the SDK surfaces vLLM's extra usage field, it is captured;
     if the SDK strips extra fields, the record honestly carries None — this
     test asserts whichever the SDK actually does, and NEVER a fabricated 0
  5. server-side 500s fail open into the caller as exceptions without
     corrupting the ledger
  6. two named endpoints against one physical server stay partitioned

Skipped automatically when the openai package is not installed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

openai = pytest.importorskip("openai")

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.execution.endpoint import endpoint as endpoint_ctx
from tokeymeter.integrations.openai import wrap as wrap_openai


class _VLLMishHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /v1/chat/completions with vLLM-style usage
    extras (queue_time_ms). Counts requests; can be told to fail once."""
    server_version = "vllmish/0.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        if self.server.fail_always or self.server.fail_next:
            self.server.fail_next = False
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "injected failure"}')
            return
        payload = {
            "id": "chatcmpl-local-1",
            "object": "chat.completion",
            "created": 1784650000,
            "model": body.get("model", "kimi-k2-instruct"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": f"echo:{len(self.server.requests)}"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 337,
                "completion_tokens": 129,
                "total_tokens": 466,
                # vLLM-style extension the schema-tolerant extractor probes:
                "queue_time_ms": 47.5,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence request logging in test output
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _VLLMishHandler)
    srv.requests = []
    srv.fail_next = False
    srv.fail_always = False
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


def _client(srv):
    return openai.OpenAI(
        base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
        api_key="not-needed-locally",
    )


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_wire_miss_carries_server_reported_tokens(server):
    wrapped = wrap_openai(_client(server), semantic=False)
    with endpoint_ctx("vllm-wire-pool"):
        resp = wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("hello over the wire"))
    assert resp.choices[0].message.content.startswith("echo:")
    assert len(server.requests) == 1
    rec = _records()[-1]
    assert rec["hit"] is False
    assert (rec["input_tokens"], rec["output_tokens"]) == (337, 129)
    assert rec["token_source"] == "reported"
    assert rec["endpoint_identity"] == "vllm-wire-pool"


def test_wire_repeat_served_from_cache_server_untouched(server):
    wrapped = wrap_openai(_client(server), semantic=False)
    with endpoint_ctx("vllm-wire-pool"):
        r1 = wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("cache me"))
        r2 = wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("cache me"))
    assert len(server.requests) == 1          # ground truth: ONE wire request
    assert r2.choices[0].message.content == r1.choices[0].message.content
    recs = _records()
    assert [r["hit"] for r in recs] == [False, True]
    assert recs[1]["endpoint_identity"] == "vllm-wire-pool"


def test_wire_queue_wait_capture_matches_sdk_behavior(server):
    """The extractor reads resp.usage.queue_time_ms IF the SDK surfaces extra
    fields on its parsed models. Assert consistency with what the SDK actually
    does — captured value when surfaced, honest None when stripped — and that
    a fabricated 0.0 never appears either way."""
    wrapped = wrap_openai(_client(server), semantic=False)
    resp = wrapped.chat.completions.create(
        model="kimi-k2-instruct", messages=_msgs("queue probe"))
    sdk_surfaces = getattr(resp.usage, "queue_time_ms", None) is not None
    rec = _records()[-1]
    if sdk_surfaces:
        assert rec["queue_wait_ms"] == 47.5
    else:
        assert rec["queue_wait_ms"] is None
    assert rec["queue_wait_ms"] != 0.0        # never a fabricated zero


def test_wire_persistent_server_error_fails_open_ledger_uncorrupted(server):
    # The OpenAI SDK retries transient 5xx by default, so a SINGLE injected
    # failure self-heals (see the transient test) — correct resilient behavior.
    # To prove fail-open on a PERSISTENT error we disable retries and fail every
    # request, asserting the error reaches the caller and the ledger stays sane.
    client = openai.OpenAI(
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="not-needed-locally", max_retries=0)
    wrapped = wrap_openai(client, semantic=False)
    wrapped.chat.completions.create(
        model="kimi-k2-instruct", messages=_msgs("good one"))   # 1 clean miss

    server.fail_always = True
    with pytest.raises(Exception):
        wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("this one 500s"))
    server.fail_always = False

    # a later request (server recovered) records cleanly
    wrapped.chat.completions.create(
        model="kimi-k2-instruct", messages=_msgs("after the failure"))
    recs = _records()
    assert all(r["token_source"] in ("reported", "estimated") for r in recs)
    # two successful misses recorded; the failed call produced NO record
    assert sum(1 for r in recs if not r["hit"]) == 2


def test_wire_transient_error_self_heals_via_sdk_retry(server):
    # A single 5xx with default retries enabled must recover transparently —
    # transient serving blips should not surface to the caller.
    wrapped = wrap_openai(_client(server), semantic=False)
    server.fail_next = True
    resp = wrapped.chat.completions.create(
        model="kimi-k2-instruct", messages=_msgs("transient blip"))
    assert resp.choices[0].message.content.startswith("echo:")
    assert len(server.requests) == 2          # first 500, retry succeeded
    rec = _records()[-1]
    assert rec["hit"] is False and rec["token_source"] == "reported"


def test_wire_two_endpoints_two_ledger_partitions(server):
    """Two named endpoints against the same physical server — the ledger must
    keep them separate (the join key consolidation/unit-cost relies on)."""
    wrapped = wrap_openai(_client(server), semantic=False)
    with endpoint_ctx("vllm-a100-primary"):
        wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("to primary"))
    with endpoint_ctx("tgi-spillover"):
        wrapped.chat.completions.create(
            model="kimi-k2-instruct", messages=_msgs("to spillover"))
    eps = [r["endpoint_identity"] for r in _records()]
    assert eps == ["vllm-a100-primary", "tgi-spillover"]
