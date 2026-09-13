"""Tests for the model Router (Camp A cost lever: easy->cheap, hard->capable)."""
import pytest

from tokeymeter.router import Router, RouteDecision


@pytest.fixture
def rt():
    return Router(cheap_model="gpt-4o-mini", capable_model="gpt-4o", threshold=0.5)


def test_easy_factual_routes_cheap(rt):
    d = rt.route("What is the capital of France?")
    assert d.tier == "cheap"
    assert d.model == "gpt-4o-mini"
    assert d.est_saved_usd > 0


def test_hard_reasoning_routes_capable(rt):
    d = rt.route("Derive the time complexity of merge sort step by step and prove the bound.")
    assert d.tier == "capable"
    assert d.model == "gpt-4o"
    assert d.est_saved_usd == 0.0  # no saving when we use the capable model


def test_code_routes_capable(rt):
    d = rt.route("Debug this:\n```python\ndef f(x): return x/0\n```\nWhy does it fail?")
    assert d.tier == "capable"


def test_translate_routes_cheap(rt):
    d = rt.route("Translate 'good morning' to Spanish.")
    assert d.tier == "cheap"


def test_uncertain_routes_up(rt):
    # a mid-complexity prompt with a hard cue but no clear easy signal -> route up
    d = rt.route("Why does water boil at a lower temperature at altitude?")
    assert d.tier == "capable"  # conservative: uncertain -> capable


def test_empty_routes_up(rt):
    d = rt.route("")
    assert d.tier == "capable"
    assert not d.confident


def test_decision_has_reason_and_cost(rt):
    d = rt.route("What is 2+2?")
    assert isinstance(d, RouteDecision)
    assert d.reason
    assert d.est_cost_usd >= 0
    assert 0.0 <= d.complexity <= 1.0


def test_custom_complexity_fn_overrides():
    # force everything to "hard" via custom scorer
    rt = Router(cheap_model="gpt-4o-mini", capable_model="gpt-4o",
                complexity_fn=lambda p: 0.99)
    d = rt.route("What is the capital of France?")
    assert d.tier == "capable"
    assert d.reason == "custom_scorer"


def test_custom_fn_failure_falls_back_to_heuristic():
    def boom(p): raise ValueError("nope")
    rt = Router(cheap_model="gpt-4o-mini", capable_model="gpt-4o", complexity_fn=boom)
    d = rt.route("What is the capital of France?")
    # heuristic still works despite the broken custom scorer
    assert d.tier in ("cheap", "capable")
    assert d.reason != "custom_scorer"


def test_lower_threshold_is_more_conservative():
    cautious = Router("gpt-4o-mini", "gpt-4o", threshold=0.2)
    relaxed = Router("gpt-4o-mini", "gpt-4o", threshold=0.8)
    p = "Summarize the following text in one sentence."
    # cautious router routes up more often than relaxed
    dc = cautious.route(p)
    dr = relaxed.route(p)
    # at minimum, relaxed should never route up where cautious routes cheap
    assert not (dc.tier == "cheap" and dr.tier == "capable")
