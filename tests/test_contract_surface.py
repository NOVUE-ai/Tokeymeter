"""Contract-diff guard (W0.3, feeds Exit-Gate P1).

Snapshots the PUBLIC SURFACE of the runtime contract — exported names and
callable signatures of `tokeymeter.runtime` — into a frozen JSON. Any change
to the surface fails this test until the snapshot is deliberately
regenerated, making every contract change a REVIEWED change.

Regenerate (reviewed changes only):
    python -m tests.test_contract_surface --update

Policy (doc #8 P1 / DEC-3): additions are minor, removals/signature changes
are major and require a deprecation entry first.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import tokeymeter.runtime as rt

SNAPSHOT = Path(__file__).parent / "contract_snapshot.json"


def _describe(obj) -> str:
    if inspect.isclass(obj):
        methods = {}
        for name, member in sorted(vars(obj).items()):
            if name.startswith("_") and name not in ("__init__",):
                continue
            if callable(member):
                try:
                    methods[name] = str(inspect.signature(member))
                except (TypeError, ValueError):
                    methods[name] = "<uninspectable>"
        return json.dumps({"kind": "class", "methods": methods}, sort_keys=True)
    if callable(obj):
        try:
            return json.dumps(
                {"kind": "function", "sig": str(inspect.signature(obj))})
        except (TypeError, ValueError):
            return json.dumps({"kind": "function", "sig": "<uninspectable>"})
    return json.dumps({"kind": type(obj).__name__})


def current_surface() -> dict:
    return {name: _describe(getattr(rt, name))
            for name in sorted(getattr(rt, "__all__", []))}


def test_contract_surface_frozen():
    assert SNAPSHOT.exists(), (
        "contract_snapshot.json missing — run "
        "`python -m tests.test_contract_surface --update` once, review, commit."
    )
    frozen = json.loads(SNAPSHOT.read_text())
    live = current_surface()
    removed = sorted(set(frozen) - set(live))
    added = sorted(set(live) - set(frozen))
    changed = sorted(
        n for n in set(frozen) & set(live) if frozen[n] != live[n])
    assert not removed and not changed and not added, (
        "PUBLIC CONTRACT SURFACE CHANGED — this must be a reviewed change.\n"
        f"  removed: {removed}\n  added: {added}\n  changed: {changed}\n"
        "If intentional: bump per semver policy (add=minor, remove/change=major"
        " after deprecation), then regenerate the snapshot."
    )


if __name__ == "__main__":
    if "--update" in sys.argv:
        SNAPSHOT.write_text(json.dumps(current_surface(), indent=1,
                                       sort_keys=True))
        print(f"snapshot updated: {SNAPSHOT} "
              f"({len(current_surface())} public names)")
    else:
        print("run with --update to regenerate the snapshot (reviewed only)")
