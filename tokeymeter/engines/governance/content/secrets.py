"""
NOVUE Content Governance — Increment 1: the structured-secret firewall.

This is the first tier of the content layer: deterministic, high-precision
detection of *structured secrets* — credentials, keys, and checksum-validatable
identifiers — inspected in-process, before any prompt leaves the trust boundary,
with the verdict (never the content) sealed into the audit chain.

Why structured secrets first
----------------------------
A leaked credential is *access*, not merely data — the single most catastrophic
class of AI-prompt leak (the Samsung/engineer-pasting-keys scenario). And it is
the most precisely detectable: an AWS key has a known prefix and high entropy; a
credit card passes Luhn or it does not. So this tier delivers the highest-stakes
protection at near-zero false-positive risk, with no external dependency and no
model — it is pure, auditable, deterministic code.

What this is NOT
----------------
This is not PII detection (names, addresses — that is Tier 2, GLiNER/Presidio),
not injection detection (Tier 4), and not "is this proprietary source code"
(that is a policy/routing decision, not a content detector). Those are later
increments. This module does one thing extremely well: catch structured secrets.

Design guarantees
-----------------
- **In-process / content-blind**: detection runs on content in memory; only the
  verdict (type, confidence, action, offset — never the secret) is recorded.
- **Fail-closed**: if scanning raises in BLOCK posture, the call is stopped, not
  leaked. (Inverse of the engine's fail-open optimization path — deliberate.)
- **Graduated by confidence**: a finding maps to an action by confidence band;
  an uncertain match never hard-blocks legitimate work.
- **Reversible, feature-preserving redaction**: secrets are replaced with typed
  placeholders ([AWS_KEY_1]); a per-call, in-memory-only map restores them in the
  response. The map is never persisted, never logged, never transmitted.
- **Deterministic & reproducible**: the same input yields the same findings —
  what an auditor requires.
- **Evasion-resistant**: a second normalized pass (zero-width stripped,
  boundary-tolerant credential patterns) defeats glued-text and zero-width-char
  evasions. Validated by an adversarial red-team battery (fuzz/ReDoS/exhaustion/
  concurrency) in tests-real/08_secret_firewall_redteam.py.

Known Tier-1 boundaries (documented — none are silent leaks):
  - A credential fragmented with internal whitespace ("AKIA IOSF ODNN ...") is
    not reassembled; mitigated by a high-secret-density block policy + later tiers.
  - Homoglyph / full-width-digit substitution of checksum identifiers is not
    normalized here; deferred to a later normalization tier.
The realistic, high-value leak paths — plain, embedded, code-fenced, JSON, glued,
zero-width, large-document — are all caught.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Pattern, Tuple

# Reuse the engine's vetted Luhn implementation rather than duplicate it.
from tokeymeter.engines.governance.privacy import _luhn_ok

# ── performance guard (P0.4) ────────────────────────────────────────────────
# A model-bound prompt is normally small; a 5 MB paste is either a dump or an
# attempt to time-out the scanner. These bounds keep scan latency bounded and
# memory-safe without losing precision on real prompts:
#   - inputs up to _SINGLE_PASS_BYTES scan in one pass (unchanged fast path)
#   - larger inputs scan in overlapping chunks under a wall-clock budget
#   - a cheap literal prefilter skips a detector whose required marker is absent
#   - an input-size budget caps total bytes scanned (deterministic across HW)
#   - if a budget is hit, the scan stops and reports complete=False; the policy
#     posture (fail_closed) then decides block vs proceed-partial
#
# LATENCY CLASSES (the documented performance contract):
#   Class A — small (<= 256 KB / _SINGLE_PASS_BYTES): single pass, sub-millisecond
#             on a normal prompt; this is the overwhelming common case.
#   Class B — large (256 KB .. 16 MB / _MAX_SCAN_BYTES): chunked scan, bounded by
#             the wall-clock budget (~0.5 s default) and the findings cap; a real
#             secret anywhere in the input is still caught. complete=True if the
#             whole input was scanned within budget, else complete=False.
#   Class C — oversized (> 16 MB) or adversarial backtracking that exhausts the
#             time budget: the scan stops early, complete=False, bytes_scanned
#             reports how far it got. Posture decides: fail_closed → BLOCK (an
#             unscanned region may hide a secret), fail_open → proceed-partial.
# All classes are content-blind: results carry type/span/confidence only, never
# the secret value, and an incomplete scan carries no content at all.
_SINGLE_PASS_BYTES = 262_144        # <=256 KB -> single pass
_CHUNK_BYTES = 65_536               # 64 KB scan chunks for larger inputs
_CHUNK_OVERLAP = 1_024              # overlap >= any plausible secret length
_TIME_BUDGET_S = 5.0                # wall-clock SAFETY NET for pathological
                                   # backtracking only — NOT the scan guarantee.
                                   # The deterministic guarantee is the 16 MB
                                   # byte cap below (identical on every machine).
                                   # A tighter net (was 0.5s) aborted honest
                                   # multi-MB linear scans on slower/Windows
                                   # hardware, silently missing a real key past
                                   # the abort point. 5s clears any legitimate
                                   # in-cap scan while still catching a runaway.
_MAX_FINDINGS = 2_000               # stop once this many findings accrue
_MAX_SCAN_BYTES = 16_777_216        # 16 MB: deterministic input-size budget. Unlike
                                    # the wall-clock budget (varies by hardware), a byte
                                    # cap bounds work identically on every machine, so a
                                    # pathological 1 GB paste can never exhaust memory/CPU.


# ─────────────────────────────────────────────────────────────────────────────
# Verdict model
# ─────────────────────────────────────────────────────────────────────────────
class Action(str, Enum):
    """What the policy decided to do about a finding, graduated by confidence."""
    ALLOW = "allow"          # recorded, permitted (low confidence / informational)
    MUTATE = "mutate"        # redact + reversible placeholder, call proceeds
    ESCALATE = "escalate"    # route to human approval
    BLOCK = "block"          # stop the call entirely


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"    # live credentials = access


@dataclass(frozen=True)
class Finding:
    """A single detected secret. Carries NO raw secret value off this object
    except `span` (start,end) into the original text, used only in-process for
    redaction. The recorded verdict uses type/confidence/offset, never the value.
    """
    type: str                     # e.g. "AWS_ACCESS_KEY"
    severity: Severity
    confidence: float             # 0.0–1.0
    span: Tuple[int, int]         # (start, end) offsets in the scanned text
    placeholder_label: str        # e.g. "AWS_KEY" → becomes [AWS_KEY_1]

    def redacted_descriptor(self) -> dict:
        """The content-blind descriptor safe to record (no secret value)."""
        return {
            "type": self.type,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "offset": self.span[0],
            "length": self.span[1] - self.span[0],
        }


@dataclass
class ScanResult:
    """The outcome of scanning one piece of text."""
    findings: List[Finding] = field(default_factory=list)
    scanned: bool = True          # False if scanning errored (drives fail-closed)
    error: Optional[str] = None
    complete: bool = True         # False if the scan hit the size/time budget
    bytes_scanned: int = 0

    @property
    def has_secrets(self) -> bool:
        return bool(self.findings)

    def max_severity(self) -> Optional[Severity]:
        if not self.findings:
            return None
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return max((f.severity for f in self.findings), key=order.index)


# ─────────────────────────────────────────────────────────────────────────────
# Detectors — deterministic, high-precision, typed
# ─────────────────────────────────────────────────────────────────────────────
def _shannon_entropy(s: str) -> float:
    """Bits-per-character Shannon entropy. High entropy ⇒ likely a random secret,
    which lets us distinguish real keys/tokens from ordinary words."""
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


@dataclass(frozen=True)
class _Detector:
    """A typed detector: a regex plus optional validator and entropy gate.

    The validator (e.g. Luhn) and the entropy floor are what keep precision high
    and false positives near zero — we do not flag a 16-digit number unless it
    passes Luhn, nor a long token unless it is actually high-entropy.
    """
    type: str
    severity: Severity
    pattern: Pattern[str]
    placeholder_label: str
    base_confidence: float
    validator: Optional[Callable[[str], bool]] = None
    entropy_floor: Optional[float] = None     # min bits/char on the matched span
    entropy_group: int = 0                     # which regex group to entropy-check
    required_any: Optional[Tuple[str, ...]] = None   # cheap literal prefilter

    def scan(self, text: str) -> List[Finding]:
        # Cheap prefilter: if the pattern requires a literal marker (e.g. "sk-ant-",
        # "AKIA", "eyJ") and none is present, the regex cannot match — skip it. This
        # is a C-level substring scan, far cheaper than running the regex, and sound
        # (only ever skips when the regex provably can't match).
        if self.required_any and not any(lit in text for lit in self.required_any):
            return []
        out: List[Finding] = []
        for m in self.pattern.finditer(text):
            value = m.group(0)
            conf = self.base_confidence

            # checksum / structural validator (e.g. Luhn) — strongest signal
            if self.validator is not None:
                cleaned = re.sub(r"[^\dA-Za-z]", "", value)
                if not self.validator(cleaned):
                    continue                    # not a real instance → skip
                conf = min(1.0, conf + 0.05)

            # entropy gate — distinguishes real secrets from dictionary words
            if self.entropy_floor is not None:
                span_text = m.group(self.entropy_group) if self.entropy_group else value
                ent = _shannon_entropy(span_text)
                if ent < self.entropy_floor:
                    continue                    # too low-entropy → not a secret
                # nudge confidence up with entropy headroom
                conf = min(1.0, conf + min(0.05, (ent - self.entropy_floor) * 0.02))

            out.append(Finding(
                type=self.type,
                severity=self.severity,
                confidence=conf,
                span=m.span(),
                placeholder_label=self.placeholder_label,
            ))
        return out


# Detector registry. Ordered most-specific first so overlapping matches resolve
# to the most precise type (handled by span de-duplication below).
def _build_detectors() -> List[_Detector]:
    return [
        # ---- Critical: live credentials (access, not just data) ----
        _Detector(
            type="ANTHROPIC_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
            placeholder_label="ANTHROPIC_KEY", base_confidence=0.99,
        ),
        _Detector(
            type="OPENAI_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}\b"),
            placeholder_label="OPENAI_KEY", base_confidence=0.97,
            entropy_floor=3.0,
        ),
        _Detector(
            type="AWS_ACCESS_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[A-Z0-9]{16}\b"),
            placeholder_label="AWS_KEY", base_confidence=0.98,
        ),
        _Detector(
            type="AWS_SECRET_KEY", severity=Severity.CRITICAL,
            # 40-char base64-ish secret, gated hard on entropy to avoid FPs
            pattern=re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
            placeholder_label="AWS_SECRET", base_confidence=0.80,
            entropy_floor=4.3,
        ),
        _Detector(
            type="GITHUB_TOKEN", severity=Severity.CRITICAL,
            pattern=re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
            placeholder_label="GITHUB_TOKEN", base_confidence=0.98,
        ),
        _Detector(
            type="GCP_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
            placeholder_label="GCP_KEY", base_confidence=0.97,
        ),
        _Detector(
            type="SLACK_TOKEN", severity=Severity.CRITICAL,
            pattern=re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
            placeholder_label="SLACK_TOKEN", base_confidence=0.97,
        ),
        _Detector(
            type="STRIPE_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b"),
            placeholder_label="STRIPE_KEY", base_confidence=0.98,
        ),
        _Detector(
            type="PRIVATE_KEY_BLOCK", severity=Severity.CRITICAL,
            pattern=re.compile(
                r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
                r"[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
            ),
            placeholder_label="PRIVATE_KEY", base_confidence=0.99,
        ),
        _Detector(
            type="JWT", severity=Severity.HIGH,
            pattern=re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
            placeholder_label="JWT", base_confidence=0.92,
        ),
        _Detector(
            type="GENERIC_BEARER_TOKEN", severity=Severity.HIGH,
            # "Authorization: Bearer <high-entropy>" — context + entropy gated
            pattern=re.compile(r"(?i)bearer\s+([A-Za-z0-9_\-\.=]{20,})"),
            placeholder_label="BEARER_TOKEN", base_confidence=0.85,
            entropy_floor=3.5, entropy_group=1,
        ),
        _Detector(
            type="DB_CONNECTION_STRING", severity=Severity.CRITICAL,
            # postgres://user:pass@host/db , mysql:// , mongodb+srv:// , redis://
            pattern=re.compile(
                r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://"
                r"[^\s:@/]+:[^\s:@/]+@[^\s/]+",
            ),
            placeholder_label="DB_URI", base_confidence=0.95,
        ),
        # ---- High: checksum-validatable financial identifiers ----
        _Detector(
            type="CREDIT_CARD", severity=Severity.HIGH,
            pattern=re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
            placeholder_label="CC", base_confidence=0.90,
            validator=lambda s: _luhn_ok(s),
        ),
        _Detector(
            type="US_SSN", severity=Severity.HIGH,
            # dashed/spaced only (bare 9-digit is too FP-prone for Tier 1)
            pattern=re.compile(r"\b(?!000|666|9\d\d)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b"),
            placeholder_label="SSN", base_confidence=0.88,
        ),
        _Detector(
            type="IBAN", severity=Severity.HIGH,
            pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
            placeholder_label="IBAN", base_confidence=0.80,
            validator=lambda s: _iban_ok(s),
        ),
    ]


def _iban_ok(iban: str) -> bool:
    """ISO 13616 mod-97 checksum for IBANs (high precision when it passes)."""
    s = iban.strip().upper()
    if len(s) < 15 or len(s) > 34:
        return False
    rearranged = s[4:] + s[:4]
    digits = ""
    for ch in rearranged:
        if ch.isdigit():
            digits += ch
        elif ch.isalpha():
            digits += str(ord(ch) - 55)
        else:
            return False
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# The scanner
# ─────────────────────────────────────────────────────────────────────────────
def _build_loose_detectors() -> List[_Detector]:
    """Boundary-tolerant variants of the high-value credential detectors.

    These drop the \\b word-boundary anchors so a secret glued directly to
    adjacent text (e.g. 'xEjAKIA...OX') is still caught. They are applied ONLY
    to CRITICAL/HIGH credential types whose patterns are specific enough
    (fixed prefixes, fixed lengths, checksums, entropy gates) that removing the
    anchor does not introduce false positives. Generic patterns (SSN, phone,
    bare card candidates) are NOT loosened, since for those the boundary anchor
    is what keeps precision high.
    """
    return [
        _Detector(
            type="ANTHROPIC_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
            placeholder_label="ANTHROPIC_KEY", base_confidence=0.99,
        ),
        _Detector(
            type="OPENAI_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}"),
            placeholder_label="OPENAI_KEY", base_confidence=0.97, entropy_floor=3.0,
        ),
        _Detector(
            type="AWS_ACCESS_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[A-Z0-9]{16}"),
            placeholder_label="AWS_KEY", base_confidence=0.98,
        ),
        _Detector(
            type="GITHUB_TOKEN", severity=Severity.CRITICAL,
            pattern=re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}"),
            placeholder_label="GITHUB_TOKEN", base_confidence=0.98,
        ),
        _Detector(
            type="GCP_API_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
            placeholder_label="GCP_KEY", base_confidence=0.97,
        ),
        _Detector(
            type="STRIPE_KEY", severity=Severity.CRITICAL,
            pattern=re.compile(r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}"),
            placeholder_label="STRIPE_KEY", base_confidence=0.98,
        ),
        _Detector(
            type="JWT", severity=Severity.HIGH,
            pattern=re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
            placeholder_label="JWT", base_confidence=0.92,
        ),
    ]


_ZERO_WIDTH_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def _strip_zero_width(text: str) -> Tuple[str, List[int]]:
    """Remove zero-width / soft-hyphen obfuscation characters and return the
    stripped text plus an index map (stripped position → original position), so
    findings on the stripped text can be reported at their true original offsets.
    """
    out_chars: List[str] = []
    index_map: List[int] = []
    for i, ch in enumerate(text):
        if ord(ch) in _ZERO_WIDTH_CHARS:
            continue
        out_chars.append(ch)
        index_map.append(i)
    return "".join(out_chars), index_map


class SecretScanner:
    """Scans text for structured secrets. Deterministic and side-effect-free.

    Performance: linear in input length; runs in well under a millisecond on
    ordinary prompts. Safe to run on every call (Tier 1 of the cascade).

    Evasion resistance: secrets can be hidden by gluing them to adjacent text
    (defeating word-boundary anchors), by inserting zero-width characters, or by
    fragmenting tokens with whitespace. The scanner defends against all three by
    additionally scanning a *normalized* view of the text (zero-width stripped,
    a boundary-tolerant pass) and mapping any findings back to the original
    offsets. Findings from the raw and normalized passes are merged and
    de-duplicated, so detection is robust without inflating false positives
    (the validators and entropy gates still apply to every candidate).
    """

    # Characters used to break/obfuscate tokens; stripped in the normalized pass.
    _ZERO_WIDTH = _ZERO_WIDTH_CHARS

    # Cheap literal prefilter per detector type: a substring the pattern REQUIRES.
    # If absent from the text, the regex provably can't match, so it is skipped.
    # Only set for patterns with a mandatory literal; the literal-less detectors
    # (40-char secret, credit card, SSN, IBAN) are bounded by the chunk/time budget.
    _PREFILTER: Dict[str, Tuple[str, ...]] = {
        "ANTHROPIC_API_KEY": ("sk-ant-",),
        "OPENAI_API_KEY": ("sk-",),
        "AWS_ACCESS_KEY": ("AKIA", "ASIA", "AGPA", "AIDA", "AROA", "ANPA"),
        "GITHUB_TOKEN": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        "GCP_API_KEY": ("AIza",),
        "SLACK_TOKEN": ("xox",),
        "STRIPE_KEY": ("sk_live_", "sk_test_", "rk_live_", "rk_test_"),
        "PRIVATE_KEY_BLOCK": ("PRIVATE KEY",),
        "JWT": ("eyJ",),
        "DB_CONNECTION_STRING": ("postgres", "mysql", "mongodb", "redis", "amqp"),
    }

    def __init__(self, detectors: Optional[List[_Detector]] = None,
                 extra_detectors: Optional[List[_Detector]] = None,
                 boundary_tolerant: bool = True):
        self._detectors = detectors if detectors is not None else _build_detectors()
        if extra_detectors:
            self._detectors = self._detectors + list(extra_detectors)
        self._boundary_tolerant = boundary_tolerant
        # Boundary-tolerant variants of the credential detectors: the same
        # patterns with \b anchors removed, so a glued-in key is still caught.
        # Only applied to the high-value CRITICAL/HIGH credential types, where a
        # miss is a real leak and the pattern is specific enough that dropping
        # \b does not add false positives.
        self._loose = _build_loose_detectors() if boundary_tolerant else []
        # Attach the cheap literal prefilter to each detector (frozen dataclass →
        # rebuild via replace()). Skips a detector's regex when its required marker
        # is absent — the core of the large-input performance guard.
        self._detectors = [replace(d, required_any=self._PREFILTER.get(d.type))
                           if d.required_any is None else d for d in self._detectors]
        self._loose = [replace(d, required_any=self._PREFILTER.get(d.type))
                       if d.required_any is None else d for d in self._loose]

    def _loose_zero_width_pass(self, text: str) -> List[Finding]:
        """Strip zero-width chars, run boundary-tolerant detectors, remap offsets."""
        out: List[Finding] = []
        stripped, index_map = _strip_zero_width(text)
        if stripped != text or self._loose:
            for det in self._loose:
                for f in det.scan(stripped):
                    try:
                        o_start = index_map[f.span[0]]
                        o_end = index_map[f.span[1] - 1] + 1
                    except (IndexError, KeyError):
                        o_start, o_end = f.span
                    out.append(Finding(
                        type=f.type, severity=f.severity, confidence=f.confidence,
                        span=(o_start, o_end), placeholder_label=f.placeholder_label))
        return out

    def scan(self, text: str, *, time_budget_s: Optional[float] = None,
             max_bytes: Optional[int] = None) -> ScanResult:
        if not isinstance(text, str) or not text:
            return ScanResult(findings=[], scanned=True, complete=True, bytes_scanned=0)
        deadline = time.monotonic() + (
            time_budget_s if time_budget_s is not None else _TIME_BUDGET_S)
        budget_bytes = max_bytes if max_bytes is not None else _MAX_SCAN_BYTES
        try:
            n = len(text)
            # ── fast path: small inputs scan in one pass (unchanged behavior) ──
            if n <= _SINGLE_PASS_BYTES:
                raw: List[Finding] = []
                for det in self._detectors:
                    raw.extend(det.scan(text))
                if self._boundary_tolerant:
                    raw.extend(self._loose_zero_width_pass(text))
                return ScanResult(findings=_dedupe_overlaps(raw), scanned=True,
                                  complete=True, bytes_scanned=n)

            # ── large inputs: chunked, overlapping, under size + time budgets ──
            # SECURITY INVARIANT: a present literal marker must be found no
            # matter WHERE it sits, even if the linear walk would abort first.
            # So before the linear scan, do a cheap whole-input search for each
            # detector's required literal and scan a bounded window around every
            # hit. A substring search over 16 MB is fast and position-blind;
            # this closes the gap where a key past the time-budget abort point
            # (observed on slower/Windows hardware) went undetected.
            limit = min(n, budget_bytes)        # input-size budget
            detectors = self._detectors + self._loose
            raw = []
            complete = True
            _WINDOW = 4096                       # around each marker hit
            seen_windows = []
            for det in detectors:
                lits = getattr(det, "required_any", None)
                if not lits:
                    continue                     # literal-less detectors: linear pass only
                for lit in lits:
                    start = 0
                    while True:
                        idx = text.find(lit, start, limit)
                        if idx < 0:
                            break
                        w0 = max(0, idx - 64)
                        w1 = min(limit, idx + _WINDOW)
                        window = text[w0:w1]
                        for f in det.scan(window):
                            raw.append(_shift_finding(f, w0))
                        start = idx + len(lit)
            scanned_to = 0
            pos = 0
            while pos < limit:
                if time.monotonic() > deadline or len(raw) >= _MAX_FINDINGS:
                    complete = False
                    break
                end = min(limit, pos + _CHUNK_BYTES)
                chunk = text[pos:end]
                for det in detectors:
                    for f in det.scan(chunk):
                        raw.append(_shift_finding(f, pos))
                    if len(raw) >= _MAX_FINDINGS or time.monotonic() > deadline:
                        complete = False
                        break
                scanned_to = end
                if end >= limit or not complete:
                    break
                pos = end - _CHUNK_OVERLAP        # overlap catches boundary-straddling secrets
            # if the input-size budget cut us short of the true end, it's incomplete
            if scanned_to < n:
                complete = False
            findings = _dedupe_overlaps(raw)[:_MAX_FINDINGS]
            return ScanResult(findings=findings, scanned=True,
                              complete=complete, bytes_scanned=scanned_to)
        except Exception as e:  # never raise from the scanner itself
            return ScanResult(findings=[], scanned=False, complete=False, error=str(e))


def _shift_finding(f: Finding, offset: int) -> Finding:
    """Return a finding with its span shifted into global coordinates."""
    if offset == 0:
        return f
    return Finding(type=f.type, severity=f.severity, confidence=f.confidence,
                   span=(f.span[0] + offset, f.span[1] + offset),
                   placeholder_label=f.placeholder_label)


def _dedupe_overlaps(findings: List[Finding]) -> List[Finding]:
    """When two detectors match overlapping spans, keep the higher-confidence
    (then higher-severity) one. Detectors are ordered most-specific first, so
    this resolves e.g. an Anthropic key also matching the generic OpenAI prefix.
    """
    if not findings:
        return []
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    ranked = sorted(
        findings,
        key=lambda f: (f.confidence, order.index(f.severity), f.span[1] - f.span[0]),
        reverse=True,
    )
    kept: List[Finding] = []
    for f in ranked:
        if not any(_overlaps(f.span, k.span) for k in kept):
            kept.append(f)
    # stable output: by position
    return sorted(kept, key=lambda f: f.span[0])


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ─────────────────────────────────────────────────────────────────────────────
# Graduated policy decision
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SecretPolicy:
    """Maps findings to an action, graduated by confidence and severity.

    Defaults are conservative-but-safe for the credential case: a CRITICAL
    secret (a live key) BLOCKS by default — a partial credential leak is still
    dangerous, so redaction is not enough. Lower-severity, checksum-validated
    identifiers MUTATE (redact + reversible placeholder) so the call proceeds.

    `block_critical=True` is the enterprise default. A team that prefers
    redact-and-proceed for everything can set it False.
    """
    block_critical: bool = True
    # confidence bands → action for non-critical findings
    auto_redact_threshold: float = 0.90     # ≥ → MUTATE
    escalate_threshold: float = 0.70        # ≥ → ESCALATE ; below → ALLOW+record
    fail_closed: bool = True                 # scan error in this posture → BLOCK

    def decide(self, f: Finding) -> Action:
        if f.severity == Severity.CRITICAL:
            return Action.BLOCK if self.block_critical else Action.MUTATE
        if f.confidence >= self.auto_redact_threshold:
            return Action.MUTATE
        if f.confidence >= self.escalate_threshold:
            return Action.ESCALATE
        return Action.ALLOW


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement — the firewall
# ─────────────────────────────────────────────────────────────────────────────
class SecretBlocked(RuntimeError):
    """Raised when policy BLOCKS a prompt for containing a critical secret.
    Carries only content-blind descriptors — never the secret value."""
    def __init__(self, descriptors: List[dict]):
        self.descriptors = descriptors
        types = ", ".join(sorted({d["type"] for d in descriptors}))
        super().__init__(f"prompt blocked: contains {types}")


class SecretApprovalRequired(RuntimeError):
    """Raised when policy ESCALATES a prompt to human approval."""
    def __init__(self, descriptors: List[dict]):
        self.descriptors = descriptors
        super().__init__("prompt requires approval for sensitive content")


@dataclass
class EnforcementResult:
    """The outcome of running the firewall on a prompt."""
    safe_text: str                          # text to send onward (redacted if MUTATE)
    actions: List[Tuple[Finding, Action]]   # per-finding decisions
    descriptors: List[dict]                 # content-blind verdicts (for the chain)
    placeholder_map: Dict[str, Tuple[int, int]]  # label → original span (in-process only)
    blocked: bool = False
    escalated: bool = False
    redaction_count: int = 0

    def restore(self, response_text: str, original_text: str) -> str:
        """Restore typed placeholders in a model response back to the originals,
        so MUTATE is invisible to the user. Restoration failure degrades to
        leaving the placeholder — never to leaking beyond what was already sent.
        """
        if not self.placeholder_map or not isinstance(response_text, str):
            return response_text
        out = response_text
        for label, (start, end) in self.placeholder_map.items():
            try:
                out = out.replace(f"[{label}]", original_text[start:end])
            except Exception:
                continue
        return out


class SecretFirewall:
    """The Increment-1 firewall: scan → decide → enforce → content-blind verdict.

    Usage (inside the engine, before a model call):

        fw = SecretFirewall()
        result = fw.enforce(prompt)            # raises SecretBlocked if BLOCK
        send_to_model(result.safe_text)        # redacted if any MUTATE
        # ... record result.descriptors into the audit chain (content-blind) ...
        answer = result.restore(answer, prompt)  # placeholders → originals
    """

    def __init__(self, scanner: Optional[SecretScanner] = None,
                 policy: Optional[SecretPolicy] = None):
        self._scanner = scanner or SecretScanner()
        self._policy = policy or SecretPolicy()

    def enforce(self, text: str) -> EnforcementResult:
        # Defense in depth: even though the scanner is built to never raise,
        # a custom or buggy scanner might. In fail-closed posture, ANY scanner
        # failure (returned OR raised) must stop the call, never leak.
        try:
            scan = self._scanner.scan(text)
        except Exception:
            if self._policy.fail_closed:
                raise SecretBlocked([{"type": "SCAN_ERROR", "severity": "critical",
                                      "confidence": 1.0, "offset": -1, "length": 0}])
            return EnforcementResult(safe_text=text, actions=[], descriptors=[],
                                     placeholder_map={})

        # Fail-closed: if scanning itself failed and policy demands it, stop.
        if not scan.scanned:
            if self._policy.fail_closed:
                raise SecretBlocked([{"type": "SCAN_ERROR", "severity": "critical",
                                      "confidence": 1.0, "offset": -1, "length": 0}])
            return EnforcementResult(safe_text=text, actions=[], descriptors=[],
                                     placeholder_map={})

        # Incomplete scan (input exceeded the size/time budget). In block posture
        # this is fail-closed: an unscanned region might carry a secret, so stop.
        # In measure posture we proceed with whatever partial findings we have.
        if not scan.complete and self._policy.fail_closed:
            raise SecretBlocked([{"type": "SCAN_INCOMPLETE", "severity": "critical",
                                  "confidence": 1.0, "offset": -1, "length": 0}])

        if not scan.has_secrets:
            return EnforcementResult(safe_text=text, actions=[], descriptors=[],
                                     placeholder_map={})

        actions: List[Tuple[Finding, Action]] = [
            (f, self._policy.decide(f)) for f in scan.findings
        ]
        descriptors = [f.redacted_descriptor() | {"action": a.value}
                       for f, a in actions]

        # BLOCK wins: if any finding blocks, the whole call stops, fail-closed.
        if any(a == Action.BLOCK for _, a in actions):
            raise SecretBlocked([d for d in descriptors if d["action"] == "block"])

        # ESCALATE next: route to approval before sending.
        if any(a == Action.ESCALATE for _, a in actions):
            raise SecretApprovalRequired(
                [d for d in descriptors if d["action"] == "escalate"])

        # Otherwise apply MUTATE redactions (ALLOW findings are recorded, not redacted).
        safe_text, pmap, n = _apply_redactions(
            text, [(f, a) for f, a in actions if a == Action.MUTATE])
        return EnforcementResult(
            safe_text=safe_text, actions=actions, descriptors=descriptors,
            placeholder_map=pmap, redaction_count=n,
        )

    def scan_only(self, text: str) -> ScanResult:
        """Detection without enforcement — for shadow mode / measurement."""
        return self._scanner.scan(text)


def _apply_redactions(
    text: str, to_redact: List[Tuple[Finding, Action]]
) -> Tuple[str, Dict[str, Tuple[int, int]], int]:
    """Replace each MUTATE finding with a typed, numbered placeholder, building
    the in-process restore map. Processed right-to-left so earlier offsets stay
    valid as we splice."""
    if not to_redact:
        return text, {}, 0
    findings = sorted((f for f, _ in to_redact), key=lambda f: f.span[0], reverse=True)
    pmap: Dict[str, Tuple[int, int]] = {}
    counters: Dict[str, int] = {}
    out = text
    n = 0
    for f in findings:
        counters[f.placeholder_label] = counters.get(f.placeholder_label, 0) + 1
        label = f"{f.placeholder_label}_{counters[f.placeholder_label]}"
        start, end = f.span
        out = out[:start] + f"[{label}]" + out[end:]
        pmap[label] = (start, end)
        n += 1
    return out, pmap, n


# Module-level convenience singletons
default_firewall = SecretFirewall()


def scan_for_secrets(text: str) -> ScanResult:
    """Convenience: scan a string for structured secrets (no enforcement)."""
    return default_firewall.scan_only(text)
