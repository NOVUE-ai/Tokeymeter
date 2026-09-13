"""Regression tests for Anomaly-2: tenant isolation by default.

Two leak paths were reproduced and fixed:
  - cross-TENANT: same function + same prompt + shared store served tenant B
    tenant A's cached answer (no tenant dimension in the key). Fixed with a
    first-class tenant boundary: tokeymeter.tenant_scope(id) contextvar + a
    static tenant= param, composed as the OUTERMOST cache-key prefix, and (like
    lineage) disabling fuzzy semantic serving while a tenant is active.
  - cross-SCRIPT: two different scripts whose top-level functions both live in
    module "__main__" collided on the shared default cache.db. Fixed by salting
    "__main__" namespaces with a stable entry-point id.
"""
import asyncio
import concurrent.futures as cf
import threading

import pytest

import tokeymeter
import tokeymeter.decorator as dec
from tokeymeter.storage import MemoryStore


def _fresh():
    tokeymeter.set_default_store(MemoryStore())


# ---------- tenant isolation: context manager ----------
def test_tenant_scope_isolates_identical_prompts():
    _fresh()
    calls = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        calls["n"] += 1
        return f"computed#{calls['n']}"

    with tokeymeter.tenant_scope("ACME"):
        a1 = ask("same question")
    with tokeymeter.tenant_scope("GLOBEX"):
        b1 = ask("same question")     # identical prompt, different tenant
    with tokeymeter.tenant_scope("ACME"):
        a2 = ask("same question")     # ACME again -> must reuse ACME's entry

    assert a1 != b1, "cross-tenant leak: tenants must not share cached answers"
    assert a1 == a2, "same-tenant caching must still hit"
    assert calls["n"] == 2            # one compute per distinct tenant, not 3


def test_static_tenant_param_isolates():
    _fresh()

    def build(t):
        @tokeymeter.cache(model="gpt-4o", tenant=t)
        def ans(prompt):
            return f"[{t}] {prompt}"
        return ans

    assert build("ACME")("q") == "[ACME] q"
    assert build("GLOBEX")("q") == "[GLOBEX] q"   # not ACME's value


def test_static_tenant_overrides_contextvar():
    _fresh()

    @tokeymeter.cache(model="gpt-4o", tenant="STATIC")
    def ask(prompt):
        return "x"

    with tokeymeter.tenant_scope("CTX"):
        ask("p")
    keys = list(dec._default_store._data.keys())
    assert any("t=STATIC" in k for k in keys)
    assert not any("t=CTX" in k for k in keys)


def test_no_tenant_means_no_prefix_backward_compat():
    _fresh()

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return "x"

    ask("p")
    keys = list(dec._default_store._data.keys())
    assert keys and not any(k.startswith("t=") for k in keys)


def test_tenant_scope_none_raises():
    with pytest.raises(ValueError):
        with tokeymeter.tenant_scope(None):
            pass


# ---------- tenant disables cross-tenant semantic bleed ----------
def test_active_tenant_disables_semantic_serving():
    _fresh()

    class TattleSemantic:
        def __init__(self):
            self.lookups = 0
        def lookup(self, *a, **k):
            self.lookups += 1
            return None
        def store(self, *a, **k):
            pass
        def add(self, *a, **k):
            pass

    sem = TattleSemantic()

    @tokeymeter.cache(model="gpt-4o", semantic=True, semantic_cache=sem)
    def ask(prompt):
        return "x"

    with tokeymeter.tenant_scope("ACME"):
        ask("hello there")
    assert sem.lookups == 0, "semantic serving must be disabled under an active tenant"


# ---------- async + concurrency ----------
def test_tenant_isolation_async():
    _fresh()
    calls = {"n": 0}

    @tokeymeter.cache(model="gpt-4o")
    async def ask(prompt):
        calls["n"] += 1
        return f"r{calls['n']}"

    async def run():
        async def one(t):
            with tokeymeter.tenant_scope(t):
                return await ask("same")
        a = await one("ACME")
        b = await one("GLOBEX")
        return a, b

    a, b = asyncio.run(run())
    assert a != b


def test_tenant_isolation_under_thread_concurrency():
    _fresh()

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return f"[{tokeymeter.decorator._tenant_var.get()}]"

    errors = {"n": 0}

    def worker(tenant):
        for _ in range(50):
            with tokeymeter.tenant_scope(tenant):
                out = ask("identical prompt")
            if out != f"[{tenant}]":
                errors["n"] += 1

    tenants = [f"tenant{i}" for i in range(20)]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, tenants))
    assert errors["n"] == 0, "contextvar tenant scope must be thread-isolated"


# ---------- cross-script __main__ salting ----------
def test_main_namespace_salted_per_entry_point():
    import hashlib

    def resolve_for(entry_path):
        dec._entry_point_id_cache = hashlib.sha256(entry_path.encode()).hexdigest()[:12]
        dec._auto_ns_owner.clear()

        def ask(prompt):
            return "x"
        ask.__module__ = "__main__"
        return dec._resolve_namespace(ask, None, False)

    try:
        ns_a = resolve_for("/apps/script_a.py")
        ns_b = resolve_for("/apps/script_b.py")
        ns_a_again = resolve_for("/apps/script_a.py")
        assert ns_a != ns_b              # different scripts -> isolated
        assert ns_a == ns_a_again        # same script -> stable (persistence)
        assert ns_a.startswith("__main__[")
    finally:
        dec._entry_point_id_cache = None
        dec._auto_ns_owner.clear()
