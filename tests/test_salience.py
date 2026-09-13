"""Tests for SalienceCompressor — the model-free differentiator.

Locks two things: (1) it achieves real compression on redundant/boilerplate-
heavy prompts, and (2) it never violates safety invariants (never raises,
never empties, never grows, preserves protected content).
"""
import pytest

from tokeymeter.salience import SalienceCompressor


RAG_PROMPT = """You are a helpful support assistant. Answer using only the context below.

Context:
The refund policy allows returns within 30 days of purchase. Items must be unused.
Our refund policy permits returns within thirty days, provided the item is unused and in original packaging.
Customers can return products within 30 days if the goods remain unused.
Shipping is free on orders over $50. Standard shipping takes 5-7 business days.
As mentioned, please note that shipping is complimentary for orders above fifty dollars.
The company was founded in 2011 and is headquartered in Austin, Texas.
Our headquarters, by the way, has been located in Austin since the founding year of 2011.

Question: What is the refund window for an unused item?"""


def test_real_compression_on_redundant_prompt():
    r = SalienceCompressor(target_ratio=0.55).compress(RAG_PROMPT)
    assert r.safe
    assert r.ratio < 0.75, f"expected meaningful compression, got ratio {r.ratio}"
    assert r.tokens_after < r.tokens_before


def test_preserves_question_and_instruction():
    r = SalienceCompressor(target_ratio=0.55).compress(RAG_PROMPT)
    assert "What is the refund window" in r.after  # the question
    assert "Answer using only the context" in r.after  # the instruction


def test_removes_redundancy():
    r = SalienceCompressor(target_ratio=0.55).compress(RAG_PROMPT)
    # three refund restatements should collapse; count remaining "30 days"/"thirty"
    mentions = r.after.lower().count("30 days") + r.after.lower().count("thirty days")
    assert mentions <= 2, f"redundant refund lines not collapsed: {mentions}"


@pytest.mark.parametrize("text", [
    "",
    "hello",
    "What is 2+2? Answer concisely.",
    "You must return JSON. Do not include prose. Always validate. Never guess.",
    "The quick brown fox jumps over the lazy dog near the riverbank this morning.",
])
def test_never_raises_never_empties_never_grows(text):
    r = SalienceCompressor().compress(text)
    assert r.safe in (True, False)
    if text.strip():
        assert r.after.strip(), "emptied non-empty input"
    assert r.tokens_after <= r.tokens_before, "grew tokens"


def test_code_block_preserved():
    txt = "Explain this:\n```python\ndef f(x):\n    return x * 2\n```\nWhat does it return?"
    r = SalienceCompressor().compress(txt)
    assert "```python" in r.after
    assert "return x * 2" in r.after


def test_short_prompt_is_noop():
    r = SalienceCompressor().compress("Just one short line here.")
    assert r.ratio == 1.0
    assert r.extra.get("noop") is True


def test_method_label():
    r = SalienceCompressor().compress(RAG_PROMPT)
    assert r.method == "salience"


# ---- query-aware compression (the safety mechanism for context/RAG) ----

QA_PROMPT = """Background notes for the assistant.
The product launched in March and sold well in Q2.
Our refund policy is 30 days. Shipping is free over $50.
The company is headquartered in Austin, Texas.
Customer satisfaction was 92 percent last quarter.
The mobile app supports iOS and Android.
Question: Where is the company headquartered?"""


def test_query_aware_protects_answer_fact():
    blind = SalienceCompressor(target_ratio=0.5)
    aware = blind.for_query("Where is the company headquartered?")
    ra = aware.compress(QA_PROMPT)
    assert "Austin" in ra.after, "query-aware compression dropped the answer fact"


def test_query_aware_preserves_negation():
    prompt = """Policy notes for the agent.
Customers on the free plan are not eligible for phone support.
Free plan users receive email support only during business hours.
Premium customers get 24/7 phone and email support.
The billing cycle resets on the first of each month.
Question: Are free plan customers eligible for phone support?"""
    aware = SalienceCompressor(target_ratio=0.5).for_query(
        "Are free plan customers eligible for phone support?"
    )
    r = aware.compress(prompt)
    assert "not eligible" in r.after.lower(), "dropped the negation — would flip the answer"


def test_for_query_returns_query_aware_copy():
    base = SalienceCompressor()
    q = base.for_query("some question")
    assert q.query == "some question"
    assert base.query is None  # original unchanged


def test_query_aware_still_compresses():
    aware = SalienceCompressor(target_ratio=0.5).for_query("Where is the company headquartered?")
    r = aware.compress(QA_PROMPT)
    assert r.tokens_after < r.tokens_before  # still saves tokens, just safely
