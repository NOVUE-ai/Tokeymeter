"""v0.14 identity binding — principal stamped on every record and event.

Pins the keystone guarantees: the contextvar composes correctly across
scopes, exceptions, threads, and async tasks; the decorator stamps it into
BOTH the savings CallRecord and the public CacheEvent; validation rejects
identifiers that could smuggle content; and None (unattributed) is always a
legal state.
"""
import asyncio
import threading

import pytest

import tokeymeter
from tokeymeter import events as ev
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.set_principal(None)
    yield
    tokeymeter.set_principal(None)
    tokeymeter.reset_savings()


def _last_record():
    from tokeymeter import savings as sv
    recs = list(sv._tracker._iter_records())
    assert recs, "no records written"
    return recs[-1]


def test_record_and_event_carry_principal():
    seen = []
    cb = lambda e: seen.append(e.principal)
    ev.subscribe(cb)
    try:
        @tokeymeter.cache(model="m")
        def ask(p):
            return "x" * 50
        with tokeymeter.principal("person:priya"):
            ask("hello world " * 5)
    finally:
        ev.unsubscribe(cb)
    assert _last_record().get("principal") == "person:priya"
    assert seen and seen[-1] == "person:priya"


def test_unbound_is_none_not_error():
    @tokeymeter.cache(model="m")
    def ask(p):
        return "y"
    ask("q1 " * 5)
    assert _last_record().get("principal") is None


def test_scope_restores_prior_including_on_exception():
    tokeymeter.set_principal("agent:outer")
    try:
        with tokeymeter.principal("agent:inner"):
            assert tokeymeter.get_principal() == "agent:inner"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tokeymeter.get_principal() == "agent:outer"


def test_set_principal_returns_previous():
    assert tokeymeter.set_principal("a:1") is None
    assert tokeymeter.set_principal("a:2") == "a:1"


def test_validation_rejects_content_shaped_ids():
    for bad in ("has spaces", "x" * 200, "line\nbreak", "", 42):
        with pytest.raises((ValueError, TypeError)):
            tokeymeter.set_principal(bad)
    # a rejected set must not clobber the current binding
    tokeymeter.set_principal("agent:ok")
    with pytest.raises(ValueError):
        tokeymeter.set_principal("nope nope")
    assert tokeymeter.get_principal() == "agent:ok"


def test_thread_isolation():
    tokeymeter.set_principal("person:main")
    seen = {}
    def worker():
        seen["initial"] = tokeymeter.get_principal()
        tokeymeter.set_principal("person:worker")
        seen["after"] = tokeymeter.get_principal()
    t = threading.Thread(target=worker)
    t.start(); t.join()
    # worker's binding never leaks back into the main thread
    assert seen["after"] == "person:worker"
    assert tokeymeter.get_principal() == "person:main"


def test_async_task_isolation():
    async def task(pid, out):
        with tokeymeter.principal(pid):
            await asyncio.sleep(0.01)
            out.append((pid, tokeymeter.get_principal()))
    async def main():
        out = []
        await asyncio.gather(task("agent:a", out), task("agent:b", out))
        return out
    out = asyncio.run(main())
    assert all(want == got for want, got in out)


def test_emitter_whitelist_includes_principal():
    # The control-plane client is not part of the open-source distribution —
    # it is the paid surface — so this integration property can only be checked
    # in a tree that has it.
    pytest.importorskip("integrations.tokenet",
                        reason="TokeNet client not present in this tree")
    from integrations.tokenet import emitter as em
    assert "principal" in em._FIELDS
