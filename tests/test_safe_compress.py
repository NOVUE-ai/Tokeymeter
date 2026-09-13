"""Tests for SafeCompressor — the confidence-gated safety layer.

Locks the correctness guarantee: good compressions ship, suspect ones fall
back to the original (never silently shipped), and the fallback rate is counted.
"""
import pytest

from tokeymeter.compression import CompressionResult
from tokeymeter.safe_compress import (
    SafeCompressor,
    compression_stats,
    reset_compression_stats,
)
from tokeymeter.salience import SalienceCompressor


RAG = """You are a support assistant. Answer using only the context.
Context:
The refund policy allows returns within 30 days of purchase. Items must be unused.
Our refund policy permits returns within thirty days, provided the item is unused.
Customers can return products within 30 days if the goods remain unused.
Shipping is free on orders over $50. Standard shipping takes 5-7 business days.
The company is headquartered in Austin, Texas.
Question: What is the refund window for an unused item?"""


@pytest.fixture(autouse=True)
def _fresh_stats():
    reset_compression_stats()
    yield


class _Broken:
    """Inner compressor that destroys content — must trigger fallback."""
    def compress(self, text):
        return CompressionResult(
            before=text, after="totally unrelated text",
            tokens_before=100, tokens_after=3, ratio=0.03,
            duration_ms=0.1, method="broken", safe=True, extra={},
        )


class _NoGain:
    """Inner compressor that achieves nothing — must fall back (no_gain/negligible)."""
    def compress(self, text):
        return CompressionResult(
            before=text, after=text,
            tokens_before=100, tokens_after=100, ratio=1.0,
            duration_ms=0.1, method="nogain", safe=True, extra={},
        )


def test_good_compression_ships():
    safe = SafeCompressor(inner=SalienceCompressor(target_ratio=0.55)).for_query(
        "What is the refund window for an unused item?"
    )
    r = safe.compress(RAG)
    assert r.extra.get("gated") is True
    assert not r.extra.get("fell_back")
    assert "30 days" in r.after  # answer fact preserved
    assert r.tokens_after < r.tokens_before


def test_extreme_compression_falls_back():
    safe = SafeCompressor(inner=_Broken())
    r = safe.compress(RAG)
    assert r.extra.get("fell_back") is True
    assert r.after == RAG  # returned the original, unharmed
    assert r.reason if hasattr(r, "reason") else r.extra.get("reason") == "too_extreme"


def test_query_terms_lost_falls_back():
    # compressor that shrinks but drops the query terms entirely
    class _DropsAnswer:
        def compress(self, text):
            return CompressionResult(
                before=text, after="Shipping is free on orders over fifty dollars.",
                tokens_before=100, tokens_after=40, ratio=0.4,
                duration_ms=0.1, method="drops", safe=True, extra={},
            )
    safe = SafeCompressor(inner=_DropsAnswer()).for_query("What is the refund window?")
    r = safe.compress(RAG)
    assert r.extra.get("fell_back") is True
    assert r.extra.get("reason") == "query_terms_lost"
    assert r.after == RAG


def test_no_gain_falls_back():
    safe = SafeCompressor(inner=_NoGain())
    r = safe.compress(RAG)
    assert r.extra.get("fell_back") is True
    assert r.after == RAG


def test_fallback_rate_is_counted():
    reset_compression_stats()
    SafeCompressor(inner=_Broken()).compress(RAG)        # fallback
    SafeCompressor(inner=SalienceCompressor()).for_query(
        "What is the refund window for an unused item?"
    ).compress(RAG)                                       # ship
    st = compression_stats()
    assert st["total_compressions"] == 2
    assert st["fell_back"] == 1
    assert st["shipped"] == 1
    assert st["fallback_rate_pct"] == 50.0
    assert "too_extreme" in st["fallback_reasons"]


def test_inner_exception_falls_back_safely():
    class _Raises:
        def compress(self, text):
            raise RuntimeError("boom")
    r = SafeCompressor(inner=_Raises()).compress(RAG)
    assert r.extra.get("fell_back") is True
    assert r.after == RAG
    assert r.safe is True  # falling back IS the safe outcome


def test_for_query_returns_copy():
    base = SafeCompressor(inner=SalienceCompressor())
    q = base.for_query("question")
    assert q.query == "question"
    assert base.query is None
