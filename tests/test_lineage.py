"""Context-lineage protection: a lineage can never receive a cached/semantic
answer produced in a different lineage (or the global pool)."""
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def _mem():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)


def _counter_fn():
    calls = [0]
    @tokeymeter.cache(model="m")
    def ask(p):
        calls[0] += 1
        return f"r{calls[0]}:{p}"
    return ask, calls


def test_default_no_lineage_shares_cache():
    ask, calls = _counter_fn()
    ask("p"); ask("p")
    assert calls[0] == 1


def test_cross_lineage_never_shares():
    ask, calls = _counter_fn()
    with tokeymeter.lineage("A"):
        ra = ask("same")
    with tokeymeter.lineage("B"):
        rb = ask("same")
    assert calls[0] == 2, "different lineages must not share a cached answer"
    assert ra != rb


def test_within_lineage_still_caches():
    ask, calls = _counter_fn()
    with tokeymeter.lineage("C"):
        ask("p"); ask("p"); ask("p")
    assert calls[0] == 1, "caching must still work within a single lineage"


def test_lineage_cannot_read_global_cache():
    ask, calls = _counter_fn()
    ask("p")                       # global
    with tokeymeter.lineage("D"):
        ask("p")                   # must miss
    assert calls[0] == 2


def test_static_lineage_param_matches_context():
    calls = [0]
    @tokeymeter.cache(model="m", lineage="X")
    def ask(p):
        calls[0] += 1
        return "v"
    ask("p"); ask("p")
    assert calls[0] == 1           # cached within static lineage X
    # A different static lineage is isolated:
    @tokeymeter.cache(model="m", lineage="Y")
    def ask_y(p):
        calls[0] += 1
        return "v"
    ask_y("p")
    assert calls[0] == 2


def test_lineage_none_raises():
    with pytest.raises(ValueError):
        with tokeymeter.lineage(None):
            pass
