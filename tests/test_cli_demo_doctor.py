"""B0 polish items stay pinned: `tokeymeter demo` and doctor rendering."""
import os
import subprocess
import sys


def _child_env(tmp_path):
    env = dict(os.environ)                 # inherit real PATH (Windows-safe)
    env["PYTHONPATH"] = os.getcwd()
    env["TOKEYMETER_HOME"] = str(tmp_path)
    return env


def test_demo_runs_clean_and_asserts_its_own_invariants(tmp_path):
    # Fresh interpreter, isolated home: the demo must pass offline, keyless,
    # and exit 0 only when its own honesty invariants hold.
    r = subprocess.run(
        [sys.executable, "-m", "tokeymeter", "demo"],
        capture_output=True, text=True, timeout=120,
        env=_child_env(tmp_path),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL ACTS PASSED" in r.stdout
    assert '"all_priced": false' in r.stdout          # Act 1 flagged
    assert "usd_per_1m_tokens" in r.stdout            # Act 2 derivation shown
    assert "gpu_seconds_reclaimed" in r.stdout        # Act 3 capacity view
    # the demo must never leave ledger files behind (in-memory mode)
    assert not (tmp_path / "savings.jsonl").exists()


def test_doctor_renders_na_and_verdict(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "tokeymeter", "doctor"],
        capture_output=True, text=True, timeout=60,
        env=_child_env(tmp_path),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    first = r.stdout.splitlines()[0]
    assert "HEALTHY" in first or "DEGRADED" in first  # verdict leads
    assert "Nonems" not in r.stdout                   # the old bug, pinned dead
    assert "p50=n/a" in r.stdout                      # zero samples → n/a
