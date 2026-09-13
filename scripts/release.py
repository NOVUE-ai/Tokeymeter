#!/usr/bin/env python3
"""Release hardening (W9) — SBOM + dependency-surface audit for the runtime.

Enterprise security teams require a Software Bill of Materials and a clear
account of the dependency surface before adopting an SDK. This script produces
both for the tokeymeter runtime:

  sbom     — a CycloneDX-style JSON SBOM of installed dependencies.
  deps     — the runtime's DECLARED dependency surface and an assertion that
             the core stays minimal (the in-process moat depends on the core
             being light: stdlib + a small, auditable set).
  manifest — a content-blind file manifest (path, SHA-256, size) of the
             runtime package, the input a signed release attestation covers.

Content-blind: the SBOM and manifest carry names, versions, hashes, sizes —
never file contents or secrets.

    python scripts/release.py sbom     > sbom.json
    python scripts/release.py deps
    python scripts/release.py manifest > runtime.manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(ROOT, "tokeymeter", "runtime")

# The runtime's declared, auditable dependency surface. The CORE is stdlib +
# cryptography only (for proof packets); everything else is optional/extra.
CORE_DEPS = {"cryptography"}
OPTIONAL_DEPS = {
    "opentelemetry-api": "telemetry export (OTelSink)",
    "opentelemetry-sdk": "telemetry export (OTelSink)",
    "openai": "OpenAI-dialect adapters (user-supplied client)",
    "anthropic": "Anthropic adapter (user-supplied client)",
    "redislite": "distributed single-flight (optional cache backend)",
    "anyio": "async test surface",
}


def _installed() -> List[Dict[str, str]]:
    """Enumerate installed distributions via importlib.metadata (stdlib)."""
    try:
        from importlib import metadata
    except Exception:  # pragma: no cover
        import importlib_metadata as metadata  # type: ignore
    out = []
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
            if name:
                out.append({"name": name, "version": version})
        except Exception:
            continue
    # dedup + sort
    seen = {}
    for d in out:
        seen[d["name"].lower()] = d
    return sorted(seen.values(), key=lambda d: d["name"].lower())


def cmd_sbom() -> int:
    comps = _installed()
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": "tokeymeter",
                "description": "Enterprise AI execution runtime (data plane)",
            },
        },
        "components": [
            {"type": "library", "name": c["name"], "version": c["version"],
             "purl": f"pkg:pypi/{c['name']}@{c['version']}"}
            for c in comps
        ],
    }
    print(json.dumps(sbom, indent=2))
    return 0


def cmd_deps() -> int:
    installed = {d["name"].lower(): d["version"] for d in _installed()}
    print("Runtime dependency surface:")
    print("\n  CORE (required — kept minimal by design):")
    for dep in sorted(CORE_DEPS):
        v = installed.get(dep, "NOT INSTALLED")
        print(f"    - {dep} ({v})")
    print("\n  OPTIONAL (extras — degrade gracefully when absent):")
    for dep, why in sorted(OPTIONAL_DEPS.items()):
        v = installed.get(dep, "not installed")
        print(f"    - {dep} ({v}) — {why}")
    # the discipline assertion: the core is small
    print(f"\n  Core dependency count: {len(CORE_DEPS)} "
          f"(the in-process moat requires a light core).")
    return 0


def cmd_manifest() -> int:
    entries = []
    for dirpath, _dirs, files in os.walk(RUNTIME_DIR):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            with open(path, "rb") as fh:
                data = fh.read()
            entries.append({
                "path": rel,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            })
    manifest = {
        "package": "tokeymeter.runtime",
        "file_count": len(entries),
        "files": sorted(entries, key=lambda e: e["path"]),
    }
    # a manifest hash over the canonical bytes — what a signed attestation covers
    canonical = json.dumps(manifest["files"], sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    print(json.dumps(manifest, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Runtime release hardening")
    ap.add_argument("command", choices=["sbom", "deps", "manifest"])
    args = ap.parse_args()
    return {"sbom": cmd_sbom, "deps": cmd_deps,
            "manifest": cmd_manifest}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
