"""K2 engine-layout battery (W1, doc #7 §2.2-K2).

Pins the alias-shim identity technique: every old import path IS the
canonical module object, so imports, from-imports, monkeypatching, and
pickling behave identically on both paths — and the split subpackages
(integrations→execution+economics, backends→trust+optimization) resolve
without duplicate module objects.
"""
from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
import time

import pytest

PAIRS = [  # (old path, canonical path) — one per engine + hubs
    ("tokeymeter.policy", "tokeymeter.engines.governance.policy"),
    ("tokeymeter.pricing", "tokeymeter.engines.economics.pricing"),
    ("tokeymeter.degraded", "tokeymeter.engines.reliability.degraded"),
    ("tokeymeter.memory", "tokeymeter.engines.knowledge.memory"),
    ("tokeymeter.integrity", "tokeymeter.engines.trust.integrity"),
    ("tokeymeter.compression", "tokeymeter.engines.optimization.compression"),
    ("tokeymeter.salience", "tokeymeter.engines.optimization.salience"),
    ("tokeymeter.audit.log", "tokeymeter.engines.trust.audit.log"),
    ("tokeymeter.content.secrets",
     "tokeymeter.engines.governance.content.secrets"),
    ("tokeymeter.integrations.openai",
     "tokeymeter.engines.execution.integrations.openai"),
    ("tokeymeter.integrations.reconcile",
     "tokeymeter.engines.economics.reconcile"),
    ("tokeymeter.backends.cipher", "tokeymeter.engines.trust.cipher"),
    ("tokeymeter.backends.redis_store",
     "tokeymeter.engines.optimization.redis_store"),
]


@pytest.mark.parametrize("old,new", PAIRS, ids=[p[0] for p in PAIRS])
def test_alias_identity(old, new):
    a, b = importlib.import_module(old), importlib.import_module(new)
    assert a is b, f"{old} is not the canonical module object {new}"


def test_no_duplicate_module_objects_in_sys_modules():
    for old, new in PAIRS:
        importlib.import_module(old), importlib.import_module(new)
        assert sys.modules[old] is sys.modules[new]


def test_from_import_both_paths_same_object():
    from tokeymeter.policy import get_security_policy as f_old
    from tokeymeter.engines.governance.policy import (
        get_security_policy as f_new)
    assert f_old is f_new
    from tokeymeter.audit.log import AuditLog as A_old
    from tokeymeter.engines.trust.audit.log import AuditLog as A_new
    assert A_old is A_new


def test_monkeypatch_propagates_both_directions(monkeypatch):
    import tokeymeter.pricing as old
    import tokeymeter.engines.economics.pricing as new
    monkeypatch.setattr(old, "_k2_probe", "via_old", raising=False)
    assert getattr(new, "_k2_probe") == "via_old"
    monkeypatch.setattr(new, "_k2_probe2", "via_new", raising=False)
    assert getattr(old, "_k2_probe2") == "via_new"


def test_backends_public_names_survive():
    from tokeymeter.backends import (RedisStore, Cipher, FernetCipher,
                                     NoOpCipher)
    from tokeymeter.engines.trust.cipher import FernetCipher as FC
    assert FernetCipher is FC and all((RedisStore, Cipher, NoOpCipher))


def test_split_package_attribute_access():
    import tokeymeter.integrations as pkg
    from tokeymeter.engines.economics import reconcile
    assert pkg.reconcile is reconcile          # split child rehomed
    from tokeymeter.engines.execution.integrations import universal
    assert pkg.universal is universal


def test_pickle_qualname_survives_move():
    from tokeymeter.compression import CompressionResult
    cls = pickle.loads(pickle.dumps(CompressionResult))
    assert cls is CompressionResult


def test_engines_package_is_lazy():
    """engines/__init__ itself imports nothing: loading it adds zero modules
    beyond what `import tokeymeter` (eager by long-standing design) already
    loaded."""
    code = ("import sys, tokeymeter; before=set(sys.modules); "
            "import tokeymeter.engines; "
            "new=set(sys.modules)-before-{'tokeymeter.engines'}; "
            "assert not new, f'engines init eagerly loaded: {new}'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_shim_reimport_idempotent():
    m1 = importlib.import_module("tokeymeter.policy")
    m2 = importlib.import_module("tokeymeter.policy")
    assert m1 is m2


def test_public_api_intact_via_top_level():
    import tokeymeter
    for name in ("cache", "principal", "set_default_store"):
        assert hasattr(tokeymeter, name), f"top-level API lost: {name}"


def test_cold_import_budget():
    """Cold `import tokeymeter` stays under a sane absolute bound; value
    recorded for future ±10% comparisons (pre-move baseline not captured —
    honest note in CHANGELOG)."""
    code = "import time;t=time.perf_counter();import tokeymeter;" \
           "print(time.perf_counter()-t)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    secs = float(r.stdout.strip())
    assert secs < 2.0, f"cold import {secs:.2f}s exceeds 2.0s bound"


def test_stdlib_only_core_still_holds():
    code = ("import sys;b=set(sys.modules);import tokeymeter.runtime;"
            "n={m.split('.')[0] for m in set(sys.modules)-b};"
            "tp=n-set(sys.stdlib_module_names)-{'tokeymeter'};"
            "assert not tp, tp")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_engine_map_doc_exists_and_matches():
    import pathlib
    doc = pathlib.Path(__file__).parent.parent / "docs" / "ENGINE_MAP.md"
    assert doc.exists(), "docs/ENGINE_MAP.md missing"
    text = doc.read_text()
    for _, canonical in PAIRS:
        mod = canonical.rsplit(".", 1)[-1]
        assert mod in text, f"ENGINE_MAP.md missing module {mod}"
