"""THE FIVE INVARIANTS — frozen guarantees of Tokeymeter.

This file is the single, named home for the five promises that must never erode.
Each test reads as the specification of one guarantee and fails loudly if it is
violated. Do not weaken these without a deliberate, reviewed decision.

  1. no silent quality loss
  2. no cross-lineage bleed
  3. no sensitive preview leaks
  4. no over-aggressive compression
  5. no unbounded optimization
"""
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.compression import StructuralCompressor, CompressionResult
from tokeymeter.pricing import estimate_tokens
from tokeymeter import events as ev
from tokeymeter import decorator as _dec


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.set_event_preview_policy("full")
    _dec._COMPRESSION_MAX_REDUCTION = 0.8
    _dec._RUNTIME_GUARDS = True
    yield


# 1 ----------------------------------------------------------------------------
def test_invariant_no_silent_quality_loss():
    """Compression never alters code/quoted content; verbatim is verbatim."""
    c = StructuralCompressor()
    src = "Explain:\n```python\ndef f():\n    return    [1,  2]\n```\nand \"a    b\"."
    out = c.compress(src).after
    assert src.split("```")[1] in out          # fenced block byte-for-byte
    assert '"a    b"' in out                    # quoted run byte-for-byte


# 2 ----------------------------------------------------------------------------
def test_invariant_no_cross_lineage_bleed():
    calls = [0]
    @tokeymeter.cache(model="m")
    def ask(p):
        calls[0] += 1
        return f"r{calls[0]}"
    with tokeymeter.lineage("A"):
        a = ask("same")
    with tokeymeter.lineage("B"):
        b = ask("same")
    assert calls[0] == 2 and a != b             # no answer crossed lineages
    # and a lineage cannot read the global pool
    calls[0] = 0
    ask("g")
    with tokeymeter.lineage("C"):
        ask("g")
    assert calls[0] == 2


# 3 ----------------------------------------------------------------------------
def test_invariant_no_sensitive_preview_leak():
    captured = []
    ev.subscribe(lambda e: captured.append(e))
    secret = "SSN 123-45-6789 confidential diagnosis"
    @tokeymeter.cache(model="m", high_stakes=True)
    def critical(p):
        return "ok"
    critical(secret)
    prev = captured[-1].prompt_preview or ""
    assert secret[:12] not in prev and prev.startswith("sha256:")
    # the runtime guard holds even if policy is forced permissive
    tokeymeter.set_event_preview_policy("full")
    captured.clear()
    critical(secret + "!")
    assert (captured[-1].prompt_preview or "").startswith("sha256:")


# 4 ----------------------------------------------------------------------------
def test_invariant_no_over_aggressive_compression():
    class Over:
        def compress(self, text):
            kept = text[: max(1, len(text)//10)]
            return CompressionResult(before=text, after=kept,
                tokens_before=estimate_tokens(text), tokens_after=estimate_tokens(kept),
                ratio=estimate_tokens(kept)/max(1, estimate_tokens(text)),
                duration_ms=0.0, method="over", safe=True)
    seen = {}
    @tokeymeter.cache(model="m", compressor=Over())
    def ask(prompt):
        seen["p"] = prompt
        return "ok"
    long = "Important clause. " * 40
    ask(long)
    assert seen["p"] == long                    # over-compression withheld -> original used


# 5 ----------------------------------------------------------------------------
def test_invariant_no_unbounded_optimization():
    calls = [0]
    @tokeymeter.cache(model="m", high_stakes=True)
    def hs(p):
        calls[0] += 1
        return "v"
    hs("k"); hs("k"); hs("k")
    assert calls[0] == 3                        # never served from cache
    # high_stakes result must not populate the cache for normal readers
    @tokeymeter.cache(model="m")
    def normal(p):
        return "NORMAL"
    @tokeymeter.cache(model="m", high_stakes=True)
    def hs2(p):
        return "HS"
    hs2("shared")
    assert normal("shared") == "NORMAL"


# Guards are real (fail-safe + loud), and toggleable -------------------------
def test_runtime_guards_toggle_exists_and_default_on():
    assert _dec._RUNTIME_GUARDS is True
    tokeymeter.set_runtime_guards(False)
    assert _dec._RUNTIME_GUARDS is False
    tokeymeter.set_runtime_guards(True)


def test_runtime_guard_actually_fires_and_degrades_safely():
    """A guard must demonstrably catch a forced violation and take the safe
    action — proving guards are enforcement, not decoration."""
    from tokeymeter import events as ev, decorator as _dec
    captured = []
    ev.subscribe(lambda e: captured.append(e))

    _dec._RUNTIME_GUARDS = True
    captured.clear()
    _dec._record_and_emit(("p",), {}, "r", False, "high_stakes", "m", None, 1.0,
                          "LEAKED_KEY", "sensitive", False, None, high_stakes=True)
    assert captured[-1].cache_key is None, "guard ON must scrub the cache_key"

    _dec._RUNTIME_GUARDS = False
    captured.clear()
    _dec._record_and_emit(("p",), {}, "r", False, "high_stakes", "m", None, 1.0,
                          "LEAKED_KEY", "sensitive", False, None, high_stakes=True)
    assert captured[-1].cache_key == "LEAKED_KEY", "guard OFF proves the guard is the cause"
    _dec._RUNTIME_GUARDS = True
