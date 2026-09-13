"""Tests for Tier 1: prefix-cacheability optimization + tool compression.

Locks the guarantees: prefix measured correctly, breakpoint placed only when
it qualifies, conversation order preserved (semantics safe), tool schemas never
broken, and everything falls open on bad input.
"""
import pytest

from tokeymeter.cache_optimize import CacheOptimizer, compress_tools


BIG = "You are an expert assistant. " * 200  # ~1200+ tokens


def test_identifies_cacheable_prefix():
    msgs = [{"role": "system", "content": BIG},
            {"role": "user", "content": "What is the refund policy?"}]
    out, rep = CacheOptimizer(min_prefix_tokens=1024).optimize(msgs)
    assert rep.meets_min_prefix
    assert rep.cacheable_prefix_tokens >= 1024
    assert rep.breakpoint_index == 0


def test_places_cache_control_breakpoint():
    msgs = [{"role": "system", "content": BIG},
            {"role": "user", "content": "hi"}]
    out, rep = CacheOptimizer(place_breakpoint=True).optimize(msgs)
    assert out[0].get("cache_control") == {"type": "ephemeral"}
    # original input not mutated
    assert "cache_control" not in msgs[0]


def test_below_min_does_not_place_breakpoint():
    msgs = [{"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"}]
    out, rep = CacheOptimizer(min_prefix_tokens=1024).optimize(msgs)
    assert not rep.meets_min_prefix
    assert rep.breakpoint_index is None
    assert all("cache_control" not in m for m in out)


def test_warns_on_bad_structure():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "system", "content": BIG}]
    out, rep = CacheOptimizer().optimize(msgs)
    # system after user -> no leading static prefix
    assert rep.cacheable_prefix_tokens == 0
    assert any("front" in n for n in rep.notes)


def test_conversation_order_preserved():
    msgs = [{"role": "system", "content": BIG},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}]
    out, rep = CacheOptimizer().optimize(msgs)
    # the user/assistant/user order must be identical (no reordering of turns)
    roles = [m["role"] for m in out if m["role"] in ("user", "assistant")]
    assert roles == ["user", "assistant", "user"]
    assert not rep.reordered


@pytest.mark.parametrize("bad", [None, [], "not a list", [1, 2, 3], [{"no": "role"}]])
def test_optimizer_falls_open(bad):
    out, rep = CacheOptimizer().optimize(bad)
    assert out is bad or isinstance(out, list)  # never raises


# ---- tool compression ----

def test_compress_tools_trims_filler():
    tools = [{"type": "function", "function": {
        "name": "analyze_cpu",
        "description": "This tool is used to analyze CPU usage and should be used when the user wants to understand CPU spikes.",
        "parameters": {"type": "object", "properties": {"log": {"type": "string"}}}}}]
    out, stats = compress_tools(tools)
    assert stats["tokens_after"] < stats["tokens_before"]
    # name + parameters preserved exactly
    assert out[0]["function"]["name"] == "analyze_cpu"
    assert out[0]["function"]["parameters"]["properties"]["log"]["type"] == "string"


def test_compress_tools_preserves_schema_when_no_filler():
    tools = [{"type": "function", "function": {
        "name": "f", "description": "Add two numbers.",
        "parameters": {"type": "object"}}}]
    out, stats = compress_tools(tools)
    assert out[0]["function"]["name"] == "f"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_compress_tools_never_empties_description():
    tools = [{"type": "function", "function": {
        "name": "f", "description": "Use this tool to", "parameters": {}}}]
    out, stats = compress_tools(tools)
    # trimming would near-empty it -> keep original
    assert out[0]["function"]["description"]


@pytest.mark.parametrize("bad", [None, "x", [1, 2], [{"no": "function"}]])
def test_compress_tools_falls_open(bad):
    out, stats = compress_tools(bad)
    assert out is bad or isinstance(out, list)


def test_does_not_mutate_input_tools():
    desc = "This tool is used to do a thing and should be used when needed for the task."
    tools = [{"type": "function", "function": {"name": "f", "description": desc, "parameters": {}}}]
    compress_tools(tools)
    assert tools[0]["function"]["description"] == desc  # original unchanged


# ---- Tier A: volatile-content detection (detect + recommend, never rewrite) ----

BIG_SYS = "You are an expert assistant for ACME Corp. " * 40


def test_detects_timestamp_in_prefix():
    msgs = [{"role": "system", "content": BIG_SYS + "Current time: 2026-06-04T14:23:01Z."},
            {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    kinds = {f.kind for f in rep.volatile_findings}
    assert "timestamp" in kinds or "current_datetime_label" in kinds
    assert rep.estimated_cache_health == "broken"


def test_detects_uuid_and_request_id():
    msgs = [{"role": "system", "content": BIG_SYS +
             "Session id: 7f3a9c21-1b2e-4f8a-9c0d-2e1f4a6b8c0d. request_id: req-8842aa9931."},
            {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    kinds = {f.kind for f in rep.volatile_findings}
    assert "uuid" in kinds
    assert "request_id" in kinds


def test_detects_token_as_security_smell():
    msgs = [{"role": "system", "content": BIG_SYS + "API key: sk-abcdef0123456789abcdef0123."},
            {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    assert any(f.kind == "token_or_key" for f in rep.volatile_findings)


def test_findings_are_content_blind():
    secrets = {
        "uuid": "7f3a9c21-1b2e-4f8a-9c0d-2e1f4a6b8c0d",
        "email": "tuhin@example.com",
        "reqid": "req-8842aa9931",
        "key": "sk-abcdef0123456789abcdef0123",
    }
    content = (BIG_SYS + f"Current time: 2026-06-04T14:23:01Z. Session id: {secrets['uuid']}. "
               f"Logged in as: {secrets['email']}. request_id: {secrets['reqid']}. "
               f"API key: {secrets['key']}.")
    msgs = [{"role": "system", "content": content}, {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    blob = repr(rep.volatile_findings) + repr(rep.notes)
    for label, secret in secrets.items():
        assert secret not in blob, f"LEAK: {label} value appeared in findings"
    # also no raw timestamp/date
    assert "14:23:01" not in blob and "2026-06-04" not in blob


def test_clean_prompt_has_no_findings():
    msgs = [{"role": "system", "content": BIG_SYS}, {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    assert rep.volatile_findings == []
    assert rep.estimated_cache_health == "good"
    assert not rep.has_cache_breakers


def test_volatile_in_user_turn_is_ignored():
    # volatile content in the DYNAMIC turn is fine — it's supposed to change
    msgs = [{"role": "system", "content": BIG_SYS},
            {"role": "user", "content": "It is now 2026-06-04T14:23:01Z, what time is it?"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    # detector only scans the prefix (head), not the user turn
    assert rep.volatile_findings == []


def test_findings_carry_recommendation_and_location():
    msgs = [{"role": "system", "content": BIG_SYS + "request_id: req-8842aa9931."},
            {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100).optimize(msgs)
    f = next(f for f in rep.volatile_findings if f.kind == "request_id")
    assert f.recommendation
    assert f.message_index == 0
    assert f.role == "system"


def test_detect_volatile_can_be_disabled():
    msgs = [{"role": "system", "content": BIG_SYS + "Current time: 2026-06-04T14:23:01Z."},
            {"role": "user", "content": "hi"}]
    _, rep = CacheOptimizer(min_prefix_tokens=100, detect_volatile=False).optimize(msgs)
    assert rep.volatile_findings == []


def test_check_cache_does_not_modify_messages():
    import tokeymeter as tk
    content = BIG_SYS + "request_id: req-8842aa9931."
    msgs = [{"role": "system", "content": content}, {"role": "user", "content": "hi"}]
    rep = tk.check_cache(msgs, min_prefix_tokens=100)
    assert msgs[0]["content"] == content          # unchanged
    assert "cache_control" not in msgs[0]          # no breakpoint placed
    assert rep.has_cache_breakers
