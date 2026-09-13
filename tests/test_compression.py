"""
Tests for the prompt-compression module (v0.6).

Covers:
  - StructuralCompressor rules (whitespace, fillers, substitutions, examples, quotes)
  - compose() pipeline including partial failure
  - Fail-open contract (broken compressor → original prompt returned)
  - Decorator integration (compression before keying)
  - Compression metrics tracked in events and savings report
  - verify_rate sampling and similarity logging
  - Cache key compounding (two verbose prompts compressing to same → 1 entry)
"""
import asyncio

import pytest

import tokeymeter
from tokeymeter import compression, events
from tokeymeter.compression import (
    CompressionResult,
    StructuralCompressor,
    compose,
    safe_compress,
)
from tokeymeter.decorator import (
    _apply_compressor,
    _jaccard_similarity,
    compression_verification_log,
)
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    # Clear verification log
    from tokeymeter.decorator import _compression_verifications
    _compression_verifications.clear()
    yield


# =================================================================
#                  StructuralCompressor unit tests
# =================================================================

def test_normalize_whitespace_collapses_runs():
    c = StructuralCompressor()
    r = c.compress("hello     world\n\n\n\nfoo  \n\nbar")
    assert r.safe
    # Multiple spaces → one space, 3+ newlines → 2
    assert "     " not in r.after
    assert "\n\n\n" not in r.after


def test_remove_fillers_strips_polite_phrases():
    c = StructuralCompressor()
    r = c.compress("Could you please tell me about photosynthesis.")
    assert r.safe
    assert "could you please" not in r.after.lower()
    assert "photosynthesis" in r.after.lower()


def test_substitutions_replace_verbose_phrases():
    c = StructuralCompressor()
    r = c.compress("In order to succeed, due to the fact that practice helps.")
    assert r.safe
    assert "in order to" not in r.after.lower()
    assert "due to the fact that" not in r.after.lower()
    assert "to succeed" in r.after.lower()
    assert "because" in r.after.lower()


def test_substitutions_can_be_disabled():
    c = StructuralCompressor(substitutions=False)
    r = c.compress("In order to succeed.")
    assert r.safe
    assert "in order to" in r.after.lower()


def test_skip_inside_quotes_preserves_quoted_text():
    """Filler removal should NOT touch text inside quotes."""
    c = StructuralCompressor()
    r = c.compress('Summarize this: "Could you please be more specific?"')
    assert r.safe
    # The OUTSIDE "Could you please" should be gone... but wait, it's not outside
    # In this test the only "could you please" IS inside quotes, so it should survive
    assert "Could you please be more specific" in r.after


def test_skip_inside_code_blocks():
    c = StructuralCompressor()
    text = "Could you please rewrite this:\n```python\ndef please_help():\n    pass\n```"
    r = c.compress(text)
    assert r.safe
    # The "please_help" function name is inside a code block — must survive
    assert "please_help" in r.after
    # The OUTSIDE "Could you please" should be gone
    assert "Could you please rewrite" not in r.after


def test_max_examples_caps_few_shot_prompts():
    c = StructuralCompressor(max_examples=2)
    text = """Instruction text.

Example 1: foo
Example 2: bar
Example 3: baz
Example 4: qux

End instruction."""
    r = c.compress(text)
    assert r.safe
    assert "Example 1:" in r.after
    assert "Example 2:" in r.after
    assert "Example 3:" not in r.after
    assert "Example 4:" not in r.after


def test_compression_reduces_token_count():
    c = StructuralCompressor()
    verbose = (
        "Could you please kindly tell me about photosynthesis.  "
        "I would like you to be thorough.  "
        "In order to understand this well, "
        "due to the fact that I am a beginner."
    )
    r = c.compress(verbose)
    assert r.safe
    assert r.tokens_after < r.tokens_before
    assert r.ratio < 1.0


def test_empty_input_yields_safe_noop():
    c = StructuralCompressor()
    r = c.compress("")
    assert r.safe
    assert r.before == "" and r.after == ""
    assert r.ratio == 1.0


def test_non_string_input_yields_safe_noop():
    c = StructuralCompressor()
    # Type-hint says str but we should be defensive
    r = c.compress(None)  # type: ignore[arg-type]
    assert r.safe


def test_custom_fillers_apply():
    c = StructuralCompressor(custom_fillers=[r"\bplease note that\b"])
    r = c.compress("Please note that the sky is blue.")
    assert r.safe
    assert "please note that" not in r.after.lower()


def test_custom_substitutions_apply():
    c = StructuralCompressor(custom_substitutions=[(r"\bcolour\b", "color")])
    r = c.compress("The colour is bright.")
    assert r.safe
    assert "color" in r.after


# =================================================================
#                      compose() tests
# =================================================================

def test_compose_chains_compressors():
    c1 = StructuralCompressor(substitutions=True, remove_fillers=False)
    c2 = StructuralCompressor(substitutions=False, remove_fillers=True)
    p = compose(c1, c2)

    r = p.compress("Could you please in order to succeed.")
    assert r.safe
    # Both transformations should have happened
    assert "could you please" not in r.after.lower()
    assert "in order to" not in r.after.lower()
    assert r.method.startswith("compose:")


def test_compose_skips_failing_stage():
    class BrokenCompressor:
        def compress(self, text):
            raise RuntimeError("kaboom")

    good = StructuralCompressor()
    p = compose(BrokenCompressor(), good, BrokenCompressor())

    r = p.compress("Could you please help me?")
    # The good stage should still have run
    assert r.safe
    assert "could you please" not in r.after.lower()
    assert r.extra.get("any_stage_unsafe") is True


def test_compose_requires_at_least_one_compressor():
    with pytest.raises(ValueError):
        compose()


# =================================================================
#                  Fail-open contract
# =================================================================

def test_safe_compress_returns_noop_for_none_compressor():
    r = safe_compress(None, "hello world")
    assert r.safe
    assert r.before == r.after == "hello world"


def test_safe_compress_returns_failed_on_exception():
    class Boom:
        def compress(self, text):
            raise ValueError("nope")

    r = safe_compress(Boom(), "hello")
    assert not r.safe
    assert r.before == r.after == "hello"  # original returned


# =================================================================
#                  Decorator integration tests
# =================================================================

def test_decorator_applies_compression_before_keying():
    """Two verbose prompts that compress to the same form should share a cache entry."""
    calls = [0]

    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        prompt_arg="prompt",
    )
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    # Both phrasings strip down to "tell me about Python." after filler removal
    r1 = ask(prompt="Could you please tell me about Python.")
    r2 = ask(prompt="I would like you to tell me about Python.")

    assert r1 == r2
    assert calls[0] == 1


def test_decorator_compression_metric_appears_in_savings_report():
    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        model="gpt-4o-mini",
        prompt_arg="prompt",
    )
    def ask(prompt):
        return "response"

    ask(prompt="Could you please in order to help me understand photosynthesis.")
    ask(prompt="A simple prompt.")

    # The savings JSONL should have the compression fields. Resolve the ledger's
    # actual path (honors TOKEYMETER_HOME / set_savings_path) rather than assuming
    # the default location.
    import json
    path = tokeymeter.savings_ledger_health()["path"]
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # At least one record should show compression happened
    compressed = [r for r in records if r.get("compression_method") == "structural"]
    assert len(compressed) >= 1
    assert all(r["tokens_saved_via_compression"] >= 0 for r in compressed)


def test_decorator_with_broken_compressor_falls_open():
    """If the compressor raises, the call still works (original prompt sent)."""
    class Bad:
        def compress(self, text):
            raise RuntimeError("boom")

    calls = [0]

    @tokeymeter.cache(compressor=Bad(), prompt_arg="prompt")
    def ask(prompt):
        calls[0] += 1
        return "ok"

    assert ask(prompt="hello") == "ok"
    assert ask(prompt="hello") == "ok"  # cache still works
    assert calls[0] == 1


def test_event_carries_compression_metadata():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        prompt_arg="prompt",
    )
    def ask(prompt):
        return "ok"

    ask(prompt="Could you please tell me about Python.")
    ev = seen[0]
    assert ev.extra.get("compression_method") == "structural"
    assert ev.extra.get("compression_ratio") is not None
    assert ev.extra["compression_ratio"] < 1.0


def test_decorator_without_compressor_unchanged():
    """No compressor configured → behavior identical to v0.5."""
    calls = [0]

    @tokeymeter.cache(prompt_arg="prompt")
    def ask(prompt):
        calls[0] += 1
        return "ok"

    ask(prompt="hello")
    ask(prompt="hello")
    assert calls[0] == 1


# =================================================================
#                  verify_rate fidelity audit
# =================================================================

def test_verify_rate_zero_means_no_verification():
    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        verify_rate=0.0,
        prompt_arg="prompt",
    )
    def ask(prompt):
        return "r"

    ask(prompt="Could you please describe Python")
    log = compression_verification_log()
    assert len(log) == 0


def test_verify_rate_one_always_verifies():
    """verify_rate=1.0 should verify every call (when compression applies)."""
    calls = [0]

    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        verify_rate=1.0,
        prompt_arg="prompt",
    )
    def ask(prompt):
        calls[0] += 1
        return f"answer about {prompt}"

    ask(prompt="Could you please describe Python")  # miss → 1 real call + 1 verify call

    log = compression_verification_log()
    assert len(log) == 1
    # The verification compares two calls; calls[0] should be 2 (one for compressed, one for original)
    assert calls[0] == 2
    # similarity should be a float in [0, 1]
    assert 0.0 <= log[0]["similarity"] <= 1.0


def test_verify_skipped_when_compression_was_noop():
    """If compression didn't actually change the prompt, no need to verify."""
    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        verify_rate=1.0,
        prompt_arg="prompt",
    )
    def ask(prompt):
        return "r"

    # This prompt has nothing to compress (no fillers, no extra whitespace)
    ask(prompt="Python")
    log = compression_verification_log()
    # No verification because compression was effectively a no-op
    assert len(log) == 0


def test_jaccard_similarity_basic():
    assert _jaccard_similarity("hello world", "hello world") == 1.0
    assert _jaccard_similarity("hello world", "goodbye world") == pytest.approx(1/3)
    assert _jaccard_similarity("", "") == 1.0
    assert _jaccard_similarity("hello", "") == 0.0


def test_custom_similarity_fn_is_used():
    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        verify_rate=1.0,
        verify_similarity_fn=lambda a, b: 0.42,
        prompt_arg="prompt",
    )
    def ask(prompt):
        return "answer"

    ask(prompt="Could you please describe Python")
    log = compression_verification_log()
    assert log[0]["similarity"] == 0.42


# =================================================================
#                  Async integration
# =================================================================

@pytest.mark.asyncio
async def test_async_decorator_applies_compression():
    calls = [0]

    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        prompt_arg="prompt",
    )
    async def ask(prompt):
        calls[0] += 1
        return "ok"

    await ask(prompt="Could you please tell me about JavaScript.")
    await ask(prompt="I would like you to tell me about JavaScript.")
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_async_verify_rate_works():
    calls = [0]

    @tokeymeter.cache(
        compressor=StructuralCompressor(),
        verify_rate=1.0,
        prompt_arg="prompt",
    )
    async def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    await ask(prompt="Could you please tell me about Python.")
    log = compression_verification_log()
    assert len(log) == 1


# =================================================================
#                  _apply_compressor unit test
# =================================================================

def test_apply_compressor_writes_back_to_kwarg():
    c = StructuralCompressor()
    new_args, new_kwargs, result = _apply_compressor(
        (), {"prompt": "Could you please help"}, "prompt", None, c
    )
    assert result is not None and result.safe
    assert "could you please" not in new_kwargs["prompt"].lower()


def test_apply_compressor_writes_back_to_positional():
    c = StructuralCompressor()
    new_args, new_kwargs, result = _apply_compressor(
        ("Could you please help",), {}, None, None, c
    )
    assert result is not None and result.safe
    assert "could you please" not in new_args[0].lower()


def test_apply_compressor_none_compressor_is_noop():
    new_args, new_kwargs, result = _apply_compressor(
        ("hello",), {}, None, None, None
    )
    assert result is None
    assert new_args == ("hello",)


# --- Regression test for CP-1: verbatim preservation of code/quotes ---

def test_cp1_fenced_blocks_and_quotes_preserved_verbatim():
    from tokeymeter.compression import StructuralCompressor
    c = StructuralCompressor()
    fenced = "Code:\n```python\ndef f():\n    if True:\n        return     [1,  2]\n```\nExplain."
    out = c.compress(fenced).after
    block = fenced.split("```")[1]
    assert block in out, "fenced code block must be preserved verbatim"
    quoted = 'Reformat exactly: "a    b\tc" please.'
    assert '"a    b\tc"' in c.compress(quoted).after, "quoted run must be verbatim"
    # ...but filler prose outside protected regions still compresses:
    r = c.compress("Please kindly note that as per our previous discussion " * 3)
    assert r.tokens_after < r.tokens_before
