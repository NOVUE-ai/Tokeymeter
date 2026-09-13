"""
Unified, cross-platform check runner for NOVUE (Tokeymeter + TokeNet).

One command runs every verification — the engine quality suite, all TokeNet
control-plane suites, the platform e2e, the large-workload enterprise simulation,
and (optionally) the production stress battery — on Windows, macOS, and Linux.

No shell, no Unix-only commands. Pure subprocess + sys.executable + an explicit
UTF-8 / PYTHONPATH child environment. All transient artifacts (databases, audit
chains, the savings ledger) are redirected under a controlled temp workspace via
TOKENET_WORKSPACE and TOKEYMETER_HOME, then cleaned up unless --keep is given.

    python scripts/run_all_checks.py             # full suite (no stress)
    python scripts/run_all_checks.py --stress    # also run the stress battery
    python scripts/run_all_checks.py --quick     # control-plane suites only (fast)
    python scripts/run_all_checks.py --only auth,policy
    python scripts/run_all_checks.py --keep      # keep the workspace for inspection

Exit code is 0 iff every selected check passed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple


def repo_root() -> str:
    """The novue/ repo root (this file lives at <root>/scripts/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


ROOT = repo_root()

# The control-plane checks exercise integrations/tokenet, which is NOT part of
# the open-source distribution — that package is the paid surface. In a tree
# without it these checks are skipped rather than failed: a gate that cannot
# pass on the code it ships is a gate nobody trusts.
_HAS_CONTROL_PLANE = os.path.isdir(os.path.join(ROOT, "integrations", "tokenet"))

# (name, kind, target)
#   kind "module" -> python -m target   (self-validating; exit 0 == pass)
#   kind "pytest" -> pytest <target> -q
CONTROL_PLANE: List[Tuple[str, str, str]] = [] if not _HAS_CONTROL_PLANE else [
    ("auth",      "module", "integrations.tokenet.tests.test_auth"),
    ("policy",    "module", "integrations.tokenet.tests.test_policy"),
    ("budget",    "module", "integrations.tokenet.tests.test_budget"),
    ("reconcile", "module", "integrations.tokenet.tests.test_reconcile"),
    ("approval",  "module", "integrations.tokenet.tests.test_approval"),
    ("gating",    "module", "integrations.tokenet.tests.test_gating_audit"),
    ("alerts",    "module", "integrations.tokenet.tests.test_alerts"),
    ("digest",    "module", "integrations.tokenet.tests.test_digest"),
    ("graph",     "module", "integrations.tokenet.tests.test_graph"),
    ("release",   "module", "integrations.tokenet.tests.test_release"),
    ("selfhost",  "module", "integrations.tokenet.tests.test_selfhost"),
    ("incidents", "module", "integrations.tokenet.tests.test_incidents"),
    ("registry",  "module", "integrations.tokenet.tests.test_registry"),
    ("reflex",    "module", "integrations.tokenet.tests.test_reflex"),
    ("e2e",       "module", "integrations.tokenet.tests.e2e_platform"),
]
QUALITY: List[Tuple[str, str, str]] = [
    # A small, always-present suite so the runner itself can be exercised
    # end-to-end without running everything. Deliberately NOT the runner's own
    # test file: that file invokes the runner, and pointing this at it would
    # recurse.
    ("selftest", "pytest", "tests/test_per_call_model.py"),
    ("engine", "pytest", "tests/"),
]
LARGE_WORKLOAD: List[Tuple[str, str, str]] = [] if not _HAS_CONTROL_PLANE else [
    ("enterprise_sim", "module", "integrations.tokenet.harnesses.enterprise_sim"),
]
STRESS: List[Tuple[str, str, str]] = [] if not _HAS_CONTROL_PLANE else [
    ("stress_battery", "module", "integrations.tokenet.harnesses.stress_all"),
]

# W6: chaos gate — the resilience battery's fault-injection + survivability
# checks, run as a first-class runner check under --chaos.
CHAOS: List[Tuple[str, str, str]] = [
    ("chaos_resilience", "pytest",
     "tests/test_runtime_resilience.py"),
]


def child_env(workspace: str) -> dict:
    """A child environment that is correct on every OS: PYTHONPATH joined with the
    platform separator, UTF-8 forced (so box-drawing/✓ output never crashes a
    Windows cp1252 console), and all local state redirected into the workspace."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ROOT + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENET_WORKSPACE"] = workspace
    env["TOKEYMETER_HOME"] = os.path.join(workspace, "tokeymeter-home")
    # Redirect the platform temp dir into the workspace (TMPDIR on POSIX, TEMP/TMP
    # on Windows) so every check's mkdtemp lands here too — fully controlled and
    # removed by a single cleanup; nothing touches system temp, home, or cwd.
    tmp = os.path.join(workspace, "tmp")
    env["TMPDIR"] = tmp
    env["TEMP"] = tmp
    env["TMP"] = tmp
    return env


def run_one(name: str, kind: str, target: str, env: dict, timeout: int):
    if kind == "pytest":
        cmd = [sys.executable, "-m", "pytest", target, "-q"]
    else:
        cmd = [sys.executable, "-m", target]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout,
                              capture_output=True, encoding="utf-8", errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        return proc.returncode == 0, time.time() - t0, lines
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, [f"TIMEOUT after {timeout}s"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run all NOVUE checks (cross-platform).")
    ap.add_argument("--quick", action="store_true", help="control-plane suites only")
    ap.add_argument("--stress", action="store_true", help="include the stress battery")
    ap.add_argument("--chaos", action="store_true", help="include the chaos/resilience gate")
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    ap.add_argument("--only", default="", help="comma-separated check names")
    ap.add_argument("--timeout", type=int, default=600, help="per-check timeout (s)")
    args = ap.parse_args(argv)

    checks = list(CONTROL_PLANE)
    if not args.quick:
        checks += QUALITY + LARGE_WORKLOAD
    if args.stress:
        checks += STRESS
    if args.chaos:
        checks += CHAOS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        checks = [c for c in checks if c[0] in want]
    if not checks:
        print("no checks selected", file=sys.stderr)
        return 2

    workspace = tempfile.mkdtemp(prefix="tokenet-checks-")
    env = child_env(workspace)
    os.makedirs(env["TOKEYMETER_HOME"], exist_ok=True)
    os.makedirs(env["TMPDIR"], exist_ok=True)

    print(f"NOVUE check runner · {len(checks)} checks · "
          f"python {sys.version.split()[0]} · {sys.platform}")
    print(f"workspace: {workspace}")
    print("-" * 66)

    results = []
    t_start = time.time()
    for name, kind, target in checks:
        ok, dur, lines = run_one(name, kind, target, env, args.timeout)
        results.append((name, ok, dur, lines))
        summary = lines[-1][:78] if lines else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {dur:6.1f}s  {summary}")
        if not ok:
            print("        ── last output ──")
            for ln in lines[-15:]:
                print("        " + ln[:96])

    if not args.keep:
        shutil.rmtree(workspace, ignore_errors=True)

    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    print("-" * 66)
    print(f"  {passed}/{total} checks passed in {time.time() - t_start:.1f}s   "
          f"(workspace {'kept at ' + workspace if args.keep else 'cleaned'})")
    if passed != total:
        print("  FAILED: " + ", ".join(n for n, ok, _, _ in results if not ok))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
