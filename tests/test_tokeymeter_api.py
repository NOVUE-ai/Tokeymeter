"""Tests for the Tokeymeter public API layer (the marketable surface).

These lock the developer-facing contract: meter() lowers cost by default,
works on any callable, and report() returns measured savings. The underlying
engine has its own exhaustive suite; here we protect the branded surface.
"""
import pytest

import tokeymeter
import tokeymeter as tk


@pytest.fixture(autouse=True)
def _fresh_store():
    tokeymeter.set_default_store(tokeymeter.storage.MemoryStore())
    tk.reset()
    yield


def test_public_api_surface():
    for name in ("meter", "report", "reset", "compressor", "no_optimize", "__version__"):
        assert hasattr(tk, name), f"missing public symbol {name}"


def test_meter_exact_cache_collapses_identical_calls():
    calls = {"n": 0}

    @tk.meter(model="gpt-4o-mini", tag="t")
    def ask(prompt: str) -> str:
        calls["n"] += 1
        return "ans:" + prompt

    ask("same question")
    ask("same question")
    assert calls["n"] == 1  # second served from cache


def test_meter_distinct_prompts_all_execute():
    calls = {"n": 0}

    @tk.meter(model="gpt-4o-mini")
    def ask(prompt: str) -> str:
        calls["n"] += 1
        return prompt

    for p in ("a", "b", "c"):
        ask(p)
    assert calls["n"] == 3


def test_meter_works_on_any_callable_signature():
    # provider-agnostic: any text-in/text-out callable
    @tk.meter(model="_default")
    def custom_provider(prompt: str) -> str:
        return f"[custom] {prompt}"

    assert custom_provider("hi") == "[custom] hi"
    # cached second time, same output
    assert custom_provider("hi") == "[custom] hi"


def test_meter_inline_wrap_form():
    def raw(p: str) -> str:
        return p.upper()

    wrapped = tk.meter(raw, model="_default")
    assert wrapped("hello") == "HELLO"


def test_compress_true_uses_salience():
    calls = {"n": 0}

    @tk.meter(model="gpt-4o-mini", compress=True)
    def ask(prompt: str) -> str:
        calls["n"] += 1
        return "ok"

    ask("a fairly wordy prompt that the compressor can shrink a bit")
    assert calls["n"] == 1  # still works end-to-end with compression on


def test_compressor_builder_salience():
    c = tk.compressor("salience")
    assert type(c).__name__ == "SalienceCompressor"
    c2 = tk.compressor()  # default
    assert type(c2).__name__ == "SalienceCompressor"


def test_compressor_builder_light_is_structural():
    c = tk.compressor("light")
    assert type(c).__name__ == "StructuralCompressor"


def test_compressor_builder_unknown_raises():
    with pytest.raises(ValueError):
        tk.compressor("nonsense-kind")


def test_report_returns_measured_fields():
    @tk.meter(model="gpt-4o-mini", tag="rep")
    def ask(p: str) -> str:
        return p

    ask("x")
    ask("x")  # one hit
    rep = tk.report()
    assert isinstance(rep, dict)
    assert rep["total_calls"] >= 2
    assert "estimated_saved_usd" in rep
    assert "hit_rate_pct" in rep
    assert "by_tag" in rep


def test_shadow_mode_does_not_change_output():
    calls = {"n": 0}

    @tk.meter(model="gpt-4o-mini", shadow=True)
    def ask(p: str) -> str:
        calls["n"] += 1
        return p

    ask("same")
    ask("same")
    # shadow = measurement only, real function runs both times
    assert calls["n"] == 2


def test_high_stakes_never_cached():
    calls = {"n": 0}

    @tk.meter(model="gpt-4o-mini", high_stakes=True)
    def ask(p: str) -> str:
        calls["n"] += 1
        return p

    ask("critical")
    ask("critical")
    assert calls["n"] == 2  # correctness over savings


# ---- unified routing + pipeline (metered_route) ----

def test_metered_route_routes_and_caches():
    cheap = {"n": 0}; capable = {"n": 0}

    def cheap_fn(p): cheap["n"] += 1; return f"[mini]{p}"
    def capable_fn(p): capable["n"] += 1; return f"[4o]{p}"

    ask = tk.metered_route(
        cheap_fn=cheap_fn, capable_fn=capable_fn,
        cheap_model="gpt-4o-mini", capable_model="gpt-4o", tag="t",
    )
    a_easy = ask("What is the capital of France?")
    a_hard = ask("Derive the complexity of quicksort step by step and prove the bound.")
    ask("What is the capital of France?")  # identical -> cached

    assert a_easy.startswith("[mini]")   # easy routed to cheap
    assert a_hard.startswith("[4o]")     # hard routed to capable
    assert cheap["n"] == 1               # identical easy prompt cached


def test_metered_route_exposes_routing():
    ask = tk.metered_route(
        cheap_fn=lambda p: p, capable_fn=lambda p: p,
        cheap_model="gpt-4o-mini", capable_model="gpt-4o",
    )
    d = ask.route("What is 2+2?")
    assert d.tier in ("cheap", "capable")
    assert hasattr(ask, "router")


def test_metered_route_in_public_api():
    assert hasattr(tk, "metered_route")
    assert "metered_route" in tk.__all__
    assert "router" in tk.__all__
