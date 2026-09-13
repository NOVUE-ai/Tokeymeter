#!/usr/bin/env python3
"""run_perfection_gate.py — the Exit Gate of TOKEYMETER_PERFECTION_PLAN (doc #8).

Machine-checks P1–P6. v0 (W0): P-criteria that have machinery run for real;
criteria whose machinery arrives in later waves report PENDING(wave). The
gate is honest: it never reports green for a check it cannot execute.

Usage:
    python scripts/run_perfection_gate.py            # full (slow: suites+runner)
    python scripts/run_perfection_gate.py --quick    # skip full runner
    python scripts/run_perfection_gate.py --coverage # also measure coverage
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "perfection_baseline.json"


def run(cmd, timeout=2400):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr, round(time.time() - t0, 1)


def check_import_guard():
    code = (
        "import sys\n"
        "before=set(sys.modules)\n"
        "import tokeymeter.runtime\n"
        "new={m.split('.')[0] for m in set(sys.modules)-before}\n"
        "tp=new-set(sys.stdlib_module_names)-{'tokeymeter'}\n"
        "sys.exit(1 if tp else 0)\n"
    )
    rc, _, _ = run([sys.executable, "-c", code])
    return rc == 0, "runtime imports stdlib-only" if rc == 0 else "THIRD-PARTY LEAK in core import"


def parse_pytest_tail(out):
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?", out)
    m2 = re.search(r"(\d+) failed", out)
    passed = int(m.group(1)) if m else 0
    failed = int(m2.group(1)) if m2 else 0
    return passed, failed


def main():
    quick = "--quick" in sys.argv
    do_cov = "--coverage" in sys.argv
    results = {}

    print("═" * 66)
    print(" TOKEYMETER PERFECTION GATE  (doc #8 §1)")
    print("═" * 66)

    # --- P4 components that exist today -----------------------------------
    ok, msg = check_import_guard()
    results["P4.import_guard"] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] stdlib-only core: {msg}")

    cov_args = ["--cov=tokeymeter", "--cov-report=term"] if do_cov else []
    rc, out, secs = run([sys.executable, "-m", "pytest", "tests/", "-q",
                         "-p", "no:cacheprovider", *cov_args])
    passed, failed = parse_pytest_tail(out)
    results["P4.engine_suite"] = rc == 0
    print(f"  [{'PASS' if rc == 0 else 'FAIL'}] engine suite: "
          f"{passed} passed, {failed} failed ({secs}s)")
    coverage_pct = None
    if do_cov:
        cm = re.search(r"TOTAL.*?(\d+)%", out)
        if cm:
            coverage_pct = int(cm.group(1))
            print(f"         coverage (branchless line %): {coverage_pct}% "
                  f"(P4 target ≥90 — enforced from W6)")

    rc, out, secs = run([sys.executable, "-m", "pytest",
                         "integrations/tokenet/tests/", "-q",
                         "-p", "no:cacheprovider"])
    results["P4.plane_suite"] = rc == 0
    p2, f2 = parse_pytest_tail(out)
    print(f"  [{'PASS' if rc == 0 else 'FAIL'}] plane suite: "
          f"{p2} passed, {f2} failed ({secs}s)")

    if not quick:
        rc, out, secs = run([sys.executable, "scripts/run_all_checks.py"])
        results["P4.full_runner"] = rc == 0
        tail = [l for l in out.splitlines() if "checks passed" in l]
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] full runner: "
              f"{tail[-1].strip() if tail else 'see output'} ({secs}s)")
    else:
        print("  [SKIP] full runner (--quick)")

    # --- P1 contract ---------------------------------------------------------
    rc, out, _ = run([sys.executable, "-m", "pytest",
                      "tests/test_contract_surface.py", "-q",
                      "-p", "no:cacheprovider"])
    results["P1.contract_frozen"] = rc == 0
    print(f"  [{'PASS' if rc == 0 else 'FAIL'}] P1 contract surface frozen "
          f"(spec doc publishes in W9)")

    # --- Pending machinery (honest) ---------------------------------------
    pending = [
        ("P2 engines complete", "W2–W9"),
        ("P3 conformance kits self-green", "W9"),
        ("P4 coverage>=90 / chaos / fuzz-14d / 1000-concurrent / soak", "W6+"),
        ("P5 developer wedge complete", "W3+W9"),
        ("P6 release-ready (signed wheels, SBOM, docs-execute)", "W9"),
    ]
    for name, wave in pending:
        print(f"  [PEND] {name}  → arrives {wave}")

    hard_fail = [k for k, v in results.items() if v is False]
    print("─" * 66)
    if hard_fail:
        print(f"  GATE: RED — failing: {hard_fail}")
        return 1
    print("  GATE: GREEN on all machinery that exists today "
          f"({sum(results.values())}/{len(results)} live checks)")
    if BASELINE.exists() or coverage_pct is not None:
        base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        base.update({"engine_passed": passed, "plane_passed": p2,
                     "live_checks": results,
                     **({"coverage_pct": coverage_pct}
                        if coverage_pct is not None else {})})
        BASELINE.write_text(json.dumps(base, indent=1, sort_keys=True))
        print(f"  baseline recorded → {BASELINE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
