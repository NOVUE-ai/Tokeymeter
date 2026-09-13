"""
Pluggable PII redaction for cache writes.

Why this exists
---------------
When a cache is shared across users (single-pod or distributed), raw PII
inside prompts and responses becomes a compliance and security risk:
  - GDPR / CCPA / HIPAA: stored personal data carries legal obligations
  - Audit trails: cached PII appears in every backup / log dump
  - Cross-tenant contamination: a hit can return another user's identifiers

Tokeymeter lets you register a `Redactor` that runs before:
  - The exact-cache key is computed (so prompts differing only in PII
    hash to the same key — when that's the semantic you want)
  - The semantic embedding is computed (so embeddings don't carry PII)
  - The response is stored (so cache contents are clean on disk)
  - Prompt previews are emitted in events (so observability is clean)

Trade-off (read carefully)
--------------------------
Redaction makes prompts that differ ONLY in PII hash to the same cache
key. On a hit, the cached response is returned **as-is** — which means
if the response itself contains PII from the *original* user, that PII
leaks to the *current* user.

Practical guidance:
  - Use redaction for prompts where PII is **incidental** to the answer
    (e.g., "summarize the legal implications of this email about topic X")
  - Do NOT use redaction when PII is **material** to the answer
    (e.g., "what is alice@example.com's order status?")
  - When in doubt, apply the redactor to RESPONSES too (`redact_response=True`)
    so cached responses are PII-stripped regardless.

The default redactor is BEST EFFORT. For regulated workloads, write your
own subclass with domain-specific patterns (medical record numbers,
account numbers, etc).
"""
from __future__ import annotations

import re
from typing import List, Optional, Protocol


class Redactor(Protocol):
    """Anything callable that takes a string and returns a redacted string.

    Use either a function or a class implementing __call__.
    """
    def __call__(self, text: str) -> str: ...


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum verification for credit-card-like number strings."""
    s = 0
    alt = False
    for ch in reversed(digits):
        if not ch.isdigit():
            continue
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0 and len(digits) >= 13


# ---- Default patterns ----

# Email: standard, no false positives on normal text
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# US SSN with dashes (low FP)
_SSN = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
# India Aadhaar (DPDP): 12 digits grouped 4-4-4, standalone only (not a slice
# of a longer number such as a 16-digit card). Best-effort.
_AADHAAR = re.compile(r"(?<!\d)(?<!\d\s)\d{4}\s\d{4}\s\d{4}(?!\s?\d)(?!\d)")
# US phone formats: (555) 123-4567, 555-123-4567, 555.123.4567, +1 555 123 4567.
_PHONE_US = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
)

# Credit-card-like sequences (13-19 digits, optional separators).
# We Luhn-check before redacting to avoid false positives on long numbers.
_CC_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

# E.164 phone numbers (international, strict)
_PHONE_E164 = re.compile(r"(?<!\w)\+[1-9]\d{1,14}(?!\w)")

# IPv4
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)

# OpenAI / Anthropic / AWS API keys (high precision)
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")
_AWS_KEY = re.compile(r"\bAKIA[A-Z0-9]{16}\b")

# JWT (three base64 segments separated by dots, common in headers)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")


class DefaultRedactor:
    """Best-effort PII redaction for common patterns.

    Catches:
      - Emails               → [EMAIL]
      - US SSN (dashed)      → [SSN]
      - Credit cards (Luhn)  → [CC]
      - E.164 phone numbers  → [PHONE]
      - IPv4 addresses       → [IP]
      - API keys (OpenAI / Anthropic / AWS)  → [APIKEY]
      - JWTs                 → [JWT]

    Does NOT catch: names, addresses, account numbers, medical record IDs,
    free-form phone numbers, IPv6, or anything domain-specific. For those,
    extend this class or write your own.
    """

    def __init__(
        self,
        redact_emails: bool = True,
        redact_ssn: bool = True,
        redact_credit_cards: bool = True,
        redact_phones: bool = True,
        redact_ips: bool = True,
        redact_api_keys: bool = True,
        redact_jwts: bool = True,
        custom_patterns: Optional[List[tuple]] = None,
    ):
        """Configure which built-in patterns to apply.

        Args:
            custom_patterns: List of (compiled_regex_or_pattern_str, replacement_str).
                Applied AFTER the built-ins.
        """
        self._enabled = {
            "email": redact_emails,
            "ssn": redact_ssn,
            "cc": redact_credit_cards,
            "phone": redact_phones,
            "ip": redact_ips,
            "key": redact_api_keys,
            "jwt": redact_jwts,
        }
        self._custom: List[tuple] = []
        for pat, repl in (custom_patterns or []):
            if isinstance(pat, str):
                pat = re.compile(pat)
            self._custom.append((pat, repl))

    def __call__(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        s = text

        # Order matters: most specific patterns first.
        if self._enabled["jwt"]:
            s = _JWT.sub("[JWT]", s)
        if self._enabled["key"]:
            s = _ANTHROPIC_KEY.sub("[APIKEY]", s)
            s = _OPENAI_KEY.sub("[APIKEY]", s)
            s = _AWS_KEY.sub("[APIKEY]", s)
        if self._enabled["email"]:
            s = _EMAIL.sub("[EMAIL]", s)
        if self._enabled["ssn"]:
            s = _SSN.sub("[SSN]", s)
            s = _AADHAAR.sub("[AADHAAR]", s)
        if self._enabled["cc"]:
            def _cc_sub(m):
                digits = re.sub(r"[^\d]", "", m.group(0))
                return "[CC]" if _luhn_ok(digits) else m.group(0)
            s = _CC_CANDIDATE.sub(_cc_sub, s)
        if self._enabled["phone"]:
            s = _PHONE_E164.sub("[PHONE]", s)
            s = _PHONE_US.sub("[PHONE]", s)
        if self._enabled["ip"]:
            s = _IPV4.sub("[IP]", s)
        for pat, repl in self._custom:
            s = pat.sub(repl, s)
        return s

    def count_redactions(self, text: str) -> int:
        """Count how many PII matches WOULD be redacted in `text`.

        Used by the audit layer to record per-decision redaction counts as
        compliance evidence. Never raises; returns 0 on any error.

        Note: credit-card candidates are only counted if they pass the Luhn
        check, matching the behavior of __call__.
        """
        if not isinstance(text, str) or not text:
            return 0
        try:
            count = 0
            if self._enabled["jwt"]:
                count += len(_JWT.findall(text))
            if self._enabled["key"]:
                count += len(_ANTHROPIC_KEY.findall(text))
                count += len(_OPENAI_KEY.findall(text))
                count += len(_AWS_KEY.findall(text))
            if self._enabled["email"]:
                count += len(_EMAIL.findall(text))
            if self._enabled["ssn"]:
                count += len(_SSN.findall(text))
                count += len(_AADHAAR.findall(text))
            if self._enabled["cc"]:
                for m in _CC_CANDIDATE.finditer(text):
                    digits = re.sub(r"[^\d]", "", m.group(0))
                    if _luhn_ok(digits):
                        count += 1
            if self._enabled["phone"]:
                count += len(_PHONE_E164.findall(text))
                count += len(_PHONE_US.findall(text))
            if self._enabled["ip"]:
                count += len(_IPV4.findall(text))
            for pat, _repl in self._custom:
                count += len(pat.findall(text))
            return count
        except Exception:
            return 0


# Module-level singleton for convenience
default_redactor = DefaultRedactor()


def redact(text: str) -> str:
    """Convenience: apply the default redactor to a string."""
    return default_redactor(text)
