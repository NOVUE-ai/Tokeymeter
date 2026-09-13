"""Regression tests for P0: nested message structures bypass redaction.

Audit finding (Codex enterprise probe): with a configured redactor, top-level
string args were redacted but OpenAI-style nested inputs were not —
messages=[{"role":"user","content":"...PII..."}] sent raw PII to the wrapped
function and to the event/preview path, silently breaking the redaction
guarantee for the most common real input shape.

Fix: argument redaction recurses into dict/list/tuple (shared _redact_deep
walker), redacting string leaves before keying, embedding, the model call,
events, and storage — while preserving keys, container types, and non-string
scalars, and bounding recursion depth.
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.privacy import DefaultRedactor


def _store():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset()


def test_nested_messages_are_redacted_before_function_sees_them():
    """The headline probe: messages=[{content: PII}] must not reach the function raw."""
    _store()
    red = DefaultRedactor()
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=red, prompt_arg="messages")
    def chat(messages):
        seen["v"] = messages
        return "ok"

    chat(messages=[{"role": "user", "content": "email alice@example.com ssn 123-45-6789"}])
    blob = str(seen["v"])
    assert "alice@example.com" not in blob
    assert "123-45-6789" not in blob
    assert "[EMAIL]" in blob and "[SSN]" in blob


def test_deep_mixed_structure_redacted_and_shape_preserved():
    """Redaction descends through dict/list/tuple at depth, redacts only string
    leaves, and leaves keys + non-string scalars untouched."""
    _store()
    red = DefaultRedactor()
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=red)
    def fn(payload):
        seen["v"] = payload
        return "ok"

    fn(payload={"a": [{"b": ("call bob@x.com", 42, {"c": "ssn 999-88-7777"})}], "n": 7})
    out = seen["v"]
    blob = str(out)
    # PII gone
    assert "bob@x.com" not in blob and "999-88-7777" not in blob
    # structure + types + non-string scalars preserved
    assert isinstance(out, dict) and out["n"] == 7
    assert isinstance(out["a"][0]["b"], tuple)
    assert out["a"][0]["b"][1] == 42


def test_dict_keys_are_never_redacted():
    """Only values are redacted; keys keep the structure intact."""
    _store()

    # A redactor that would mangle any string it touches, to prove keys are skipped.
    def aggressive(text):
        return "REDACTED"

    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=aggressive)
    def fn(payload):
        seen["v"] = payload
        return "ok"

    fn(payload={"role": "user", "content": "secret"})
    assert set(seen["v"].keys()) == {"role", "content"}  # keys preserved
    assert seen["v"]["content"] == "REDACTED"             # value redacted


def test_redaction_count_includes_nested_pii():
    """Audit evidence: the redaction count must reflect nested PII, not just
    top-level strings (otherwise we redact silently and report zero)."""
    from tokeymeter.decorator import _count_redactions
    red = DefaultRedactor()

    # Two emails nested inside a list-of-dict (the realistic shape).
    n = _count_redactions(
        red, (), {"messages": [{"role": "user", "content": "alice@example.com and bob@example.com"}]}
    )
    assert n >= 2, f"nested PII undercounted: {n}"

    # Sanity: a flat string with one email still counts as before.
    assert _count_redactions(red, ("contact carol@example.com",), {}) >= 1


def test_pathological_depth_does_not_crash():
    """Deeply nested input must fail safe (no stack blow-up, no raise)."""
    _store()
    red = DefaultRedactor()

    # build nesting deeper than the recursion guard
    payload = "alice@example.com"
    for _ in range(60):
        payload = [payload]

    @tokeymeter.cache(model="gpt-4o", redactor=red)
    def fn(payload):
        return "ok"

    # must not raise
    assert fn(payload=payload) == "ok"
