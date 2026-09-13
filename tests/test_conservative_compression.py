"""Conservative-compression-by-default: Tokeymeter refuses over-aggressive compression
and falls back to the original prompt, so it never silently degrades quality."""
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.compression import CompressionResult, StructuralCompressor
from tokeymeter.pricing import estimate_tokens
from tokeymeter import decorator as _dec


class OverCompressor:
    """Nukes ~90% of the prompt — simulates pathological over-compression."""
    def compress(self, text):
        kept = text[: max(1, len(text) // 10)]
        return CompressionResult(before=text, after=kept,
                                 tokens_before=estimate_tokens(text),
                                 tokens_after=estimate_tokens(kept),
                                 ratio=estimate_tokens(kept) / max(1, estimate_tokens(text)),
                                 duration_ms=0.0, method="over", safe=True)


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    yield
    _dec._COMPRESSION_MAX_REDUCTION = 0.8   # restore default


LONG = "Important clause A. Important clause B. " * 30


def test_default_withholds_over_compression():
    seen = {}
    @tokeymeter.cache(model="ov1", compressor=OverCompressor())
    def ask(prompt):
        seen["p"] = prompt
        return "ok"
    ask(LONG)
    assert seen["p"] == LONG, "extreme compression must fall back to original prompt"


def test_modest_compression_still_applies():
    seen = {}
    @tokeymeter.cache(model="mod1", compressor=StructuralCompressor())
    def ask(prompt):
        seen["p"] = prompt
        return "ok"
    filler = "Please kindly note that as per our previous discussion, " * 4 + "answer is 42."
    ask(filler)
    assert len(seen["p"]) < len(filler), "modest compression should still apply"


def test_cap_is_the_mechanism():
    # A very low cap must withhold even modest compression -> proves the cap drives it.
    tokeymeter.set_compression_max_reduction(0.01)
    seen = {}
    @tokeymeter.cache(model="cap1", compressor=StructuralCompressor())
    def ask(prompt):
        seen["p"] = prompt
        return "ok"
    filler = "Please kindly note that as per our previous discussion, " * 4 + "answer is 42."
    ask(filler)
    assert seen["p"] == filler, "low cap must withhold compression (original used)"


def test_setter_validates():
    with pytest.raises(ValueError):
        tokeymeter.set_compression_max_reduction(0.0)
    with pytest.raises(ValueError):
        tokeymeter.set_compression_max_reduction(1.5)
