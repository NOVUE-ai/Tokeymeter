"""Tests for Cascade — cheap-first, escalate-on-low-confidence (Camp A lever)."""
import pytest

from tokeymeter.router import Cascade, CascadeResult


def _cheap_factory(counter):
    def cheap(p, *a, **k):
        counter["n"] += 1
        if "capital of france" in p.lower():
            return "The capital of France is Paris."
        if "merge sort" in p.lower():
            return "I'm not sure, I cannot determine the bound."
        return "A reasonable answer with enough detail to pass."
    return cheap


def _capable_factory(counter):
    def capable(p, *a, **k):
        counter["n"] += 1
        return "Capable model: detailed correct answer."
    return capable


def test_confident_cheap_answer_accepted():
    cc, pc = {"n": 0}, {"n": 0}
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o")
    r = c.run("What is the capital of France?")
    assert r.tier == "cheap"
    assert not r.escalated
    assert pc["n"] == 0  # capable never called
    assert r.est_saved_usd > 0


def test_low_confidence_escalates():
    cc, pc = {"n": 0}, {"n": 0}
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o")
    r = c.run("Derive the time complexity of merge sort and prove the bound.")
    assert r.tier == "capable"
    assert r.escalated
    assert pc["n"] == 1
    assert "Capable model" in r.answer


def test_escalated_cost_reflects_both_calls():
    cc, pc = {"n": 0}, {"n": 0}
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o")
    r = c.run("Derive the time complexity of merge sort and prove the bound.")
    # escalated path pays both -> net saving negative vs always-capable
    assert r.est_saved_usd < 0


def test_empty_response_escalates():
    def empty_cheap(p, *a, **k): return ""
    pc = {"n": 0}
    c = Cascade(empty_cheap, _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o")
    r = c.run("Some question that matters?")
    assert r.escalated
    assert r.confidence == 0.0


def test_custom_confidence_fn():
    cc, pc = {"n": 0}, {"n": 0}
    # force always-low confidence -> always escalate
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o",
                confidence_fn=lambda p, r: 0.0)
    res = c.run("What is the capital of France?")
    assert res.escalated
    assert res.reason.startswith("custom_scorer")


def test_custom_confidence_fn_failure_falls_back():
    cc, pc = {"n": 0}, {"n": 0}
    def boom(p, r): raise ValueError("nope")
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o",
                confidence_fn=boom)
    res = c.run("What is the capital of France?")
    # heuristic still works despite broken scorer
    assert isinstance(res, CascadeResult)


def test_result_shape():
    cc, pc = {"n": 0}, {"n": 0}
    c = Cascade(_cheap_factory(cc), _capable_factory(pc),
                cheap_model="gpt-4o-mini", capable_model="gpt-4o")
    r = c.run("What is the capital of France?")
    assert isinstance(r, CascadeResult)
    assert 0.0 <= r.confidence <= 1.0
    assert r.answer
    assert r.reason
