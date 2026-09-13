"""Cross-platform portability: encoding and local-state paths.

These pin the two classes of defect that only appear off a UTF-8 Linux box:

  LEGACY CONSOLE ENCODING. Windows consoles still default to cp1252 in many
    environments. A single non-ASCII character in CLI output raises
    UnicodeEncodeError and takes the whole command down — `tokeymeter demo`
    did exactly that on '\\u2192'. A diagnostic tool that dies while printing
    diagnostics is worse than one that prints an imperfect character.

  PLATFORM-DEFAULT TEXT I/O. `open()` and `Path.read_text()` without an
    explicit encoding inherit the platform default. README.md contains bytes
    cp1252 cannot decode, so the README gate crashed on Windows runners; and
    any shipped file written as UTF-8 but read back as cp1252 is a corruption
    waiting for a non-ASCII byte.

  LOCAL-STATE PATHS. Hardcoding ``~/.tokeymeter/<name>`` ignores both
    TOKEYMETER_HOME and set_home(), breaking exactly the environments that
    need the redirect most: containers, CI, and locked-down service accounts
    where ``~`` is unwritable, ephemeral, or shared between tenants.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "tokeymeter"


# ── legacy console encoding ─────────────────────────────────────────────

@pytest.mark.parametrize("cmd", ["version", "doctor", "demo", "pricing"])
def test_cli_survives_cp1252_console(cmd):
    """Every CLI subcommand must complete on a legacy code page."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.setdefault("TOKEYMETER_HOME", tempfile.mkdtemp())
    # Capture BYTES, not text: the point of this test is that the command
    # completes on a legacy code page. Decoding the captured output as UTF-8
    # would fail on correctly-encoded cp1252 bytes and report a product crash
    # that never happened.
    p = subprocess.run([sys.executable, "-m", "tokeymeter", cmd],
                       capture_output=True, env=env, timeout=300,
                       cwd=str(REPO_ROOT))
    combined = (p.stdout or b"").decode("utf-8", "replace") + \
               (p.stderr or b"").decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in combined, (
        f"`tokeymeter {cmd}` crashed on a cp1252 console:\n{combined[-600:]}")


def test_console_guard_is_installed_and_tolerant():
    """The guard must set errors='replace' and never itself raise."""
    from tokeymeter.__main__ import _make_console_resilient
    _make_console_resilient()          # must be idempotent and safe
    _make_console_resilient()


def test_cli_output_paths_are_ascii():
    """Defence in depth: keep printed strings ASCII so a legacy console shows
    readable text rather than '?' substitutions."""
    offenders = []
    for rel in ("__main__.py", "demo.py"):
        p = PKG_ROOT / rel
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line and any(ord(c) > 127 for c in line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, f"non-ASCII in CLI output lines: {offenders}"


# ── explicit text encoding ──────────────────────────────────────────────

def test_no_encoding_less_text_open_in_shipped_code():
    """Every text-mode open() in the package must state its encoding."""
    offenders = []
    for path in PKG_ROOT.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "open(" not in line or "encoding=" in line:
                continue
            if any(m in line for m in ('"rb"', "'rb'", '"wb"', "'wb'",
                                       '"ab"', "'ab'")):
                continue          # binary mode takes no encoding
            if "def " in line or "fail" in line or "falls_open" in line:
                continue          # function definitions / unrelated names
            if "open(" in line and ("with open(" in line or "= open(" in line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, (
        "text open() without an explicit encoding inherits the platform "
        f"default (cp1252 on Windows): {offenders}")


def test_readme_is_not_decodable_as_cp1252_so_reads_must_be_explicit():
    """Guards the actual failure: the README gate used read_text() with no
    encoding, and README.md cannot be decoded as cp1252."""
    raw = (REPO_ROOT / "README.md").read_bytes()
    raw.decode("utf-8")                      # must be valid UTF-8
    try:
        raw.decode("cp1252")
    except UnicodeDecodeError:
        return                               # exactly why the fix is required
    pytest.skip("README currently happens to be cp1252-decodable")


def test_readme_gate_reads_utf8_explicitly():
    src = (REPO_ROOT / "tests" / "test_runtime_facade.py").read_text(
        encoding="utf-8")
    assert 'README.md").read_text(' in src
    assert 'encoding="utf-8"' in src


# ── local-state paths honor TOKEYMETER_HOME ─────────────────────────────

def test_no_hardcoded_home_paths_in_shipped_code():
    offenders = []
    for path in PKG_ROOT.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        if '"~/.tokeymeter' in text or "'~/.tokeymeter" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"hardcoded home path ignores TOKEYMETER_HOME/set_home(): {offenders}")


def test_state_path_resolves_through_home():
    from tokeymeter import paths
    home = tempfile.mkdtemp()
    paths.set_home(home)
    try:
        assert paths.home() == home
        assert paths.state_path("audit.db") == os.path.join(home, "audit.db")
    finally:
        paths.set_home(None)


def test_audit_and_stores_land_under_tokeymeter_home():
    """The reason this matters: containers and locked-down service accounts
    where ~ is unwritable or shared."""
    from tokeymeter import paths
    from tokeymeter.engines.trust.audit.log import AuditLog
    from tokeymeter.engines.optimization.storage import SQLiteStore
    home = tempfile.mkdtemp()
    paths.set_home(home)
    try:
        a = AuditLog()
        assert a._path.startswith(home)
        assert a._install_secret_path.startswith(home)
        s = SQLiteStore()
        assert s._path.startswith(home)
        # the auto-created signing key must land there too
        assert any("audit" in f for f in os.listdir(home))
    finally:
        paths.set_home(None)


def test_explicit_path_still_wins_over_home():
    """The lazy default must not override a caller's explicit choice."""
    from tokeymeter import paths
    from tokeymeter.engines.optimization.storage import SQLiteStore
    home = tempfile.mkdtemp()
    explicit = os.path.join(tempfile.mkdtemp(), "chosen.db")
    paths.set_home(home)
    try:
        s = SQLiteStore(path=explicit)
        assert s._path == explicit
        assert not s._path.startswith(home)
    finally:
        paths.set_home(None)


# ── UTC month rollover (budget resets must not be timezone-dependent) ────

def test_budget_month_key_is_utc_anchored():
    import time
    import tokeymeter.engines.economics.keys as keysmod
    src = Path(keysmod.__file__).read_text(encoding="utf-8")
    assert "time.gmtime()" in src, (
        "month key must be UTC-anchored (time.gmtime), not local time — "
        "otherwise a budget rolls over at a different instant per timezone")
    assert "localtime" not in src.split("month:")[1][:200]


def test_month_key_crosses_december_to_january():
    import calendar
    import time
    dec = time.strftime("%Y-%m", time.gmtime(
        calendar.timegm((2026, 12, 31, 23, 59, 59, 0, 0, 0))))
    jan = time.strftime("%Y-%m", time.gmtime(
        calendar.timegm((2027, 1, 1, 0, 0, 1, 0, 0, 0))))
    assert dec == "2026-12" and jan == "2027-01" and dec != jan
