"""
Self-integrity verification (v0.8.1).

Tokeymeter audits its own code the same way it audits your LLM calls.

The threat: the LiteLLM supply-chain compromise (March 2026) slipped a
credential stealer into trusted PyPI packages. A `.pth` file executed on
every Python startup. Tens of thousands of installs were hit before anyone
noticed.

This module provides defense-in-depth against the same class of attack:

  - verify_self():     recompute hashes of every installed Tokeymeter file and
                       compare to a signed manifest shipped with the package.
                       Catches modified, missing, or injected files.

  - scan_environment(): detect the LiteLLM attack vector specifically —
                       `.pth` files in site-packages that contain executable
                       code (import/exec/subprocess/base64). Warns about ANY
                       package doing this, not just Tokeymeter.

  - generate_manifest(): build-time helper that produces tokeymeter/_manifest.json,
                       the signed inventory verify_self() checks against.

Honest threat model (no overclaiming):
  Self-verification catches ACCIDENTAL corruption and UNSOPHISTICATED
  tampering. A sophisticated attacker who can modify files can also
  regenerate an UNSIGNED manifest. Real protection requires:
    (a) a SIGNED manifest, AND
    (b) the verification key obtained OUT-OF-BAND (not from the package).
  The strongest protection is the release pipeline itself: PyPI Trusted
  Publishing (OIDC, no stealable tokens) + Sigstore. See SECURITY.md.
  This module is one layer of several, not a silver bullet.

Everything here is fail-safe: a missing manifest, unreadable file, or
verification error returns a structured report — never raises into the
caller's code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("tokeymeter.integrity")

MANIFEST_SCHEMA_VERSION = "v1"
MANIFEST_FILENAME = "_manifest.json"


def _default_package_dir() -> str:
    """Resolve the `tokeymeter/` package root.

    This module lives at `tokeymeter/engines/trust/integrity.py`, so
    `dirname(__file__)` is `tokeymeter/engines/trust/` — NOT the package root
    where `_manifest.json` is shipped (see `[tool.setuptools.package-data]`).
    Defaulting to `dirname(__file__)` made `verify_self()` — the documented,
    zero-argument call — look in the wrong directory and return
    `no_manifest` every time, silently disabling tamper detection on every
    installed wheel. Resolve from the top-level package instead, and fall back
    to walking up from this file if the import is unavailable."""
    try:
        import tokeymeter as _pkg
        return os.path.dirname(os.path.abspath(_pkg.__file__))
    except Exception:
        # engines/trust/integrity.py -> engines/trust -> engines -> tokeymeter
        return os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

# Files that are intentionally excluded from the manifest (they change
# per-environment or are the manifest itself).
_MANIFEST_EXCLUDES = {
    MANIFEST_FILENAME,
}
_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git"}
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so")


# ============================================================
#                       Data model
# ============================================================

@dataclass
class FileManifest:
    """Signed inventory of every file in the Tokeymeter package."""
    schema_version: str
    package_version: str
    generated_at: float
    files: Dict[str, str]            # relative_posix_path -> sha256 hex
    python_requires: str
    build_metadata: dict
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None

    def content_bytes(self) -> bytes:
        """Deterministic serialization of everything EXCEPT the signature.

        This is what gets signed and what manifest_hash() hashes.
        """
        payload = {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "files": dict(sorted(self.files.items())),
            "python_requires": self.python_requires,
            "build_metadata": self.build_metadata,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.content_bytes()).hexdigest()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> "FileManifest":
        return cls(**json.loads(s))


@dataclass
class IntegrityReport:
    """Result of verify_self()."""
    status: str                      # "ok" | "tampered" | "no_manifest" | "error"
    package_version: Optional[str] = None
    files_checked: int = 0
    files_ok: int = 0
    modified: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)       # present but not in manifest
    signature_status: str = "unsigned"  # "unsigned"|"unverified"|"valid"|"invalid"
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        if self.status == "ok":
            return (f"Tokeymeter integrity OK — {self.files_ok}/{self.files_checked} files "
                    f"verified, signature {self.signature_status}.")
        if self.status == "no_manifest":
            return "No manifest found — cannot verify (development install?)."
        if self.status == "unverified":
            return ("UNVERIFIED — file hashes match but the manifest is unsigned; "
                    "not tamper-evidence. " + ("; ".join(self.notes) if self.notes else ""))
        if self.status == "tampered":
            return (f"INTEGRITY FAILURE — {len(self.modified)} modified, "
                    f"{len(self.missing)} missing, {len(self.extra)} unexpected files.")
        return f"Integrity check error: {'; '.join(self.notes) or 'unknown'}"


@dataclass
class EnvironmentScanReport:
    """Result of scan_environment()."""
    status: str                      # "clean" | "suspicious" | "error"
    pth_files_scanned: int = 0
    suspicious_pth: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    def summary(self) -> str:
        if self.status == "clean":
            return (f"Environment clean — scanned {self.pth_files_scanned} .pth "
                    f"file(s), no executable code found.")
        if self.status == "suspicious":
            names = ", ".join(s["path"].split("/")[-1] for s in self.suspicious_pth[:5])
            return (f"WARNING — {len(self.suspicious_pth)} suspicious .pth file(s) "
                    f"with executable code: {names}. This is the vector used in "
                    f"the March 2026 LiteLLM supply-chain attack. Investigate before "
                    f"trusting this environment.")
        return f"Environment scan error: {'; '.join(self.notes) or 'unknown'}"


# ============================================================
#                  Hashing helpers
# ============================================================

def _hash_file(path: str) -> Optional[str]:
    """SHA-256 of a file's bytes. None on error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        log.debug("integrity: cannot hash %s: %s", path, e)
        return None


def _iter_package_files(package_dir: str):
    """Yield (relative_posix_path, absolute_path) for every source file
    in the package, applying excludes. Deterministic order."""
    base = Path(package_dir)
    paths: List[Path] = []
    for root, dirs, files in os.walk(package_dir):
        # Prune excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
        for fn in files:
            if fn in _MANIFEST_EXCLUDES:
                continue
            if fn.endswith(_EXCLUDE_SUFFIXES):
                continue
            paths.append(Path(root) / fn)
    for p in sorted(paths):
        rel = p.relative_to(base).as_posix()
        yield rel, str(p)


# ============================================================
#                  Manifest generation (build-time)
# ============================================================

def generate_manifest(
    package_dir: Optional[str] = None,
    *,
    package_version: Optional[str] = None,
    signer=None,
    build_metadata: Optional[dict] = None,
) -> FileManifest:
    """Build the file manifest. Run at release/build time.

    Args:
        package_dir: path to the `tokeymeter/` package directory. Defaults to the
            directory containing this module.
        package_version: version string. Defaults to tokeymeter.__version__.
        signer: optional Signer (from tokeymeter.audit.signers) to sign the
            manifest. The same Signer abstraction as the audit layer.
        build_metadata: extra metadata (git commit, builder, CI run id).

    Writes nothing; returns a FileManifest. Caller persists it via
    write_manifest().
    """
    if package_dir is None:
        package_dir = _default_package_dir()

    if package_version is None:
        try:
            from tokeymeter import __version__ as v
            package_version = v
        except Exception:
            package_version = "unknown"

    files: Dict[str, str] = {}
    for rel, absolute in _iter_package_files(package_dir):
        digest = _hash_file(absolute)
        if digest is not None:
            files[rel] = digest

    meta = dict(build_metadata or {})
    meta.setdefault("python_version", sys.version.split()[0])
    meta.setdefault("file_count", len(files))

    manifest = FileManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        package_version=package_version,
        generated_at=time.time(),
        files=files,
        python_requires=">=3.9",
        build_metadata=meta,
    )

    if signer is not None:
        try:
            sig = signer.sign(manifest.content_bytes())
            manifest.signature = sig.hex()
            manifest.signature_algorithm = getattr(signer, "algorithm", "unknown")
        except Exception as e:
            log.warning("integrity: manifest signing failed: %s", e)

    return manifest


def write_manifest(manifest: FileManifest, package_dir: Optional[str] = None) -> str:
    """Persist a manifest to <package_dir>/_manifest.json. Returns the path."""
    if package_dir is None:
        package_dir = _default_package_dir()
    path = os.path.join(package_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    return path


# ============================================================
#                  Self-verification (runtime)
# ============================================================

def _load_manifest(package_dir: str) -> Optional[FileManifest]:
    path = os.path.join(package_dir, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return FileManifest.from_json(f.read())
    except Exception as e:
        log.debug("integrity: manifest load failed: %s", e)
        return None


def verify_self(
    package_dir: Optional[str] = None,
    *,
    signer=None,
    require_signature: bool = True,
) -> IntegrityReport:
    """Verify the installed Tokeymeter package against its shipped manifest.

    Recomputes the SHA-256 of every installed source file and compares to
    the manifest. Reports modified, missing, and unexpected files.

    Args:
        package_dir: the tokeymeter/ directory. Defaults to this module's dir.
        signer: a Signer to verify the manifest signature. For real
            tamper-protection, the verification key MUST be obtained
            out-of-band (not from the package). See module docstring.
        require_signature: if True, an unsigned/unverifiable manifest
            yields a non-ok status.

    Never raises. Returns an IntegrityReport.
    """
    if package_dir is None:
        package_dir = _default_package_dir()

    report = IntegrityReport(status="ok")

    try:
        manifest = _load_manifest(package_dir)
        if manifest is None:
            report.status = "no_manifest"
            report.notes.append(
                "No _manifest.json shipped. This is normal for editable/dev "
                "installs (pip install -e). Released wheels include a manifest."
            )
            return report

        report.package_version = manifest.package_version

        # ---- Signature check ----
        if manifest.signature:
            if signer is None:
                report.signature_status = "unverified"
                report.notes.append(
                    "Manifest is signed but no verification key provided; "
                    "signature not checked."
                )
                if require_signature:
                    report.status = "error"
                    report.notes.append("require_signature=True but no signer given.")
                    return report
            else:
                try:
                    sig_bytes = bytes.fromhex(manifest.signature)
                    ok = signer.verify(manifest.content_bytes(), sig_bytes)
                    report.signature_status = "valid" if ok else "invalid"
                    if not ok:
                        report.status = "tampered"
                        report.notes.append("Manifest signature did not verify.")
                        return report
                except Exception as e:
                    report.signature_status = "invalid"
                    report.status = "error"
                    report.notes.append(f"Signature verification raised: {e}")
                    return report
        else:
            report.signature_status = "unsigned"

        # C2 fix: an UNSIGNED manifest provides only corruption-detection, not
        # tamper-detection — an attacker who modifies files can regenerate an
        # unsigned manifest over them. We still run the file check below (so
        # genuine corruption is caught), but when a signature is required (the
        # default) a would-be "ok" result is downgraded to "unverified": an
        # unsigned manifest can never be reported as trustworthy.
        unsigned_untrusted = (not manifest.signature) and require_signature

        # ---- File-by-file check ----
        manifest_files = dict(manifest.files)
        on_disk = dict(_iter_package_files(package_dir))

        report.files_checked = len(manifest_files)
        for rel, expected_hash in manifest_files.items():
            abs_path = os.path.join(package_dir, rel)
            if not os.path.exists(abs_path):
                report.missing.append(rel)
                continue
            actual = _hash_file(abs_path)
            if actual == expected_hash:
                report.files_ok += 1
            else:
                report.modified.append(rel)

        # Files present on disk but NOT in the manifest (possible injection)
        for rel in on_disk:
            if rel not in manifest_files:
                report.extra.append(rel)

        if report.modified or report.missing or report.extra:
            report.status = "tampered"
        elif unsigned_untrusted:
            report.status = "unverified"
            report.notes.append(
                "File hashes match, but the manifest is UNSIGNED and is not "
                "tamper-evidence (an attacker who altered files could regenerate "
                "it). Provide a verification key via signer=, or pass "
                "require_signature=False for corruption-only checking."
            )

        return report
    except Exception as e:
        report.status = "error"
        report.notes.append(f"verify_self raised: {e}")
        return report


# ============================================================
#          Environment scan (the LiteLLM .pth vector)
# ============================================================

# CRITICAL tokens: essentially never legitimate in a .pth file. These are
# the primitives a credential stealer / reverse shell needs. Flagged ALWAYS,
# even if the filename looks like a known-benign hook (an attacker can rename).
_CRITICAL_TOKENS = (
    "subprocess",
    "popen",
    "socket",
    "/dev/tcp",
    "b64decode",
    "b64encode",
    "fromhex",
    "marshal",
    "ctypes",
    "os.system",
    "urllib",
    "urlopen",
    "requests.",
    "http.client",
    "httplib",
    "smtplib",
    "ftplib",
    "telnetlib",
    "pickle.loads",
    "codecs.decode",
    "__import__('os'",
    '__import__("os"',
)

# SOFT tokens: can be legitimate (coverage.py uses exec() in its startup hook).
# Flagged only when the .pth filename is NOT a recognized benign startup hook.
_SOFT_TOKENS = (
    "exec(",
    "eval(",
    "compile(",
)

# Known-benign .pth filename fragments that legitimately run a startup hook.
# Soft-token matches on these are suppressed; CRITICAL tokens are NOT.
_KNOWN_BENIGN_PTH = (
    "distutils-precedence",
    "__editable__",
    "coverage",
    "protobuf",
    "setuptools",
    "_virtualenv",
    "matplotlib",
    "_distutils_hack",
)


def _scan_startup_module(path: str) -> Optional[dict]:
    """Scan a sitecustomize.py/usercustomize.py for attack-grade primitives.

    These run on every interpreter startup. Unlike .pth files, .py modules
    legitimately contain code, so we flag only CRITICAL tokens (exfil /
    deserialization primitives), not soft exec/eval."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    low = text.lower()
    hits = [t for t in _CRITICAL_TOKENS if t.lower() in low]
    if hits:
        return {
            "path": path,
            "severity": "critical",
            "reason": f"startup module contains attack-grade primitives: {hits}",
            "tokens": hits,
        }
    return None


def _scan_pth_file(path: str) -> Optional[dict]:
    """Return a finding dict if a .pth file contains attack-grade code.

    Two-tier logic:
      - CRITICAL tokens (subprocess, socket, payload-decode, ...) flag
        regardless of filename — these are the LiteLLM-attack signature.
      - SOFT tokens (exec/eval/compile) flag ONLY when the filename is not
        a recognized startup hook (coverage.py legitimately uses exec()).

    A bare `import coverage` never triggers. A `subprocess.Popen` always does.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    fname = os.path.basename(path).lower()
    is_known_name = any(b in fname for b in _KNOWN_BENIGN_PTH)

    critical_hits: List[str] = []
    soft_hits: List[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        for tok in _CRITICAL_TOKENS:
            if tok in low:
                critical_hits.append(f"CRITICAL '{tok}': {stripped[:70]}")
                break
        else:
            for tok in _SOFT_TOKENS:
                if tok in low:
                    soft_hits.append(f"soft '{tok}': {stripped[:70]}")
                    break

    findings: List[str] = list(critical_hits)
    # Soft hits only count if the file isn't a recognized startup hook
    if not is_known_name:
        findings.extend(soft_hits)

    if not findings:
        return None

    return {
        "path": path,
        "findings": findings[:10],
        "size_bytes": len(content),
        "severity": "critical" if critical_hits else "review",
        "name_looks_benign": is_known_name,
    }


def scan_environment(site_packages_dirs: Optional[List[str]] = None) -> EnvironmentScanReport:
    """Tripwire for the LiteLLM-style startup-execution attack vector.

    A `.pth` file is supposed only to add paths to sys.path; the March 2026
    LiteLLM attack shipped one that ran a credential stealer on every Python
    startup. This also scans `sitecustomize.py`/`usercustomize.py`, which
    auto-execute on startup too.

    IMPORTANT — this is a heuristic tripwire, NOT a sandbox. It flags known
    attack-grade primitives (the LiteLLM signature and common exfil calls).
    A determined attacker can evade a token scan via obfuscation or by using
    primitives not on the list. Treat a clean result as "no KNOWN-pattern
    payload found," not "proven safe." Real protection is defense-in-depth:
    PyPI Trusted Publishing, signed artifacts, pinned hashes, and an SBOM
    (see SECURITY.md). Never raises.
    """
    report = EnvironmentScanReport(status="clean")

    try:
        if site_packages_dirs is None:
            site_packages_dirs = [
                p for p in sys.path
                if p and ("site-packages" in p or "dist-packages" in p)
            ]
            # De-dup while preserving order
            seen = set()
            site_packages_dirs = [
                d for d in site_packages_dirs
                if not (d in seen or seen.add(d))
            ]

        for sp in site_packages_dirs:
            if not os.path.isdir(sp):
                continue
            try:
                for fn in os.listdir(sp):
                    if fn.endswith(".pth"):
                        full = os.path.join(sp, fn)
                        report.pth_files_scanned += 1
                        finding = _scan_pth_file(full)
                        if finding is not None:
                            report.suspicious_pth.append(finding)
                    elif fn in ("sitecustomize.py", "usercustomize.py"):
                        # These modules ALSO auto-execute on interpreter startup
                        # and are an equivalent supply-chain vector to .pth.
                        full = os.path.join(sp, fn)
                        report.pth_files_scanned += 1
                        finding = _scan_startup_module(full)
                        if finding is not None:
                            report.suspicious_pth.append(finding)
            except OSError as e:
                report.notes.append(f"could not list {sp}: {e}")

        if report.suspicious_pth:
            report.status = "suspicious"
        return report
    except Exception as e:
        report.status = "error"
        report.notes.append(f"scan_environment raised: {e}")
        return report


# ============================================================
#                  Convenience: full self-check
# ============================================================

def self_check(*, signer=None, verbose: bool = True) -> dict:
    """Run both verify_self() and scan_environment(); return a combined dict.

    Convenience for the CLI / a startup health check. Never raises.
    """
    integrity = verify_self(signer=signer)
    environment = scan_environment()
    result = {
        "integrity": asdict(integrity),
        "environment": asdict(environment),
        "overall_ok": integrity.ok and environment.clean,
    }
    if verbose:
        log.info("tokeymeter.self_check: %s", integrity.summary())
        log.info("tokeymeter.self_check: %s", environment.summary())
    return result
