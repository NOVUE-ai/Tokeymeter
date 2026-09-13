#!/usr/bin/env python
"""
Build-time manifest generator.

Run by the release pipeline BEFORE building the wheel, so the signed
file manifest is embedded in the distributed package. Users then verify
their install with `tokeymeter.verify_self()`.

Signing key:
    Read from the TOKEYMETER_MANIFEST_SIGNING_KEY environment variable (hex or
    raw bytes >= 16 chars). If absent, the manifest is generated UNSIGNED
    (still useful for detecting accidental corruption, but not tamper-proof).
    In CI, provide it as a repository secret.

Usage:
    python scripts/build_manifest.py
    TOKEYMETER_MANIFEST_SIGNING_KEY=<hex> python scripts/build_manifest.py
"""
import os
import subprocess
import sys

# Make the in-tree package importable without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokeymeter import integrity            # noqa: E402
from tokeymeter.audit import HMACSigner     # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_signer():
    raw = os.environ.get("TOKEYMETER_MANIFEST_SIGNING_KEY")
    if not raw:
        print("WARNING: TOKEYMETER_MANIFEST_SIGNING_KEY not set — manifest will be UNSIGNED.")
        return None
    # Accept hex or raw
    key: bytes
    try:
        key = bytes.fromhex(raw)
        if len(key) < 16:
            raise ValueError
    except ValueError:
        key = raw.encode("utf-8")
    if len(key) < 16:
        print("ERROR: signing key must be >= 16 bytes; refusing to sign.")
        return None
    return HMACSigner(key)


def main() -> int:
    pkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokeymeter"
    )
    signer = _load_signer()

    manifest = integrity.generate_manifest(
        pkg_dir,
        signer=signer,
        build_metadata={
            "git_commit": _git_commit(),
            "builder": os.environ.get("GITHUB_WORKFLOW", "local"),
            "ci_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        },
    )
    path = integrity.write_manifest(manifest, pkg_dir)
    signed = "SIGNED" if manifest.signature else "UNSIGNED"
    print(f"Wrote {signed} manifest: {path}")
    print(f"  files: {len(manifest.files)}")
    print(f"  version: {manifest.package_version}")
    print(f"  manifest_hash: {manifest.manifest_hash()}")
    print(f"  git_commit: {manifest.build_metadata.get('git_commit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
