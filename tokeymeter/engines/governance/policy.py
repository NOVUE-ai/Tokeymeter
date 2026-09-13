"""Security policy — opt-in, enforceable trust defaults.

By default the policy is fully permissive: NOTHING changes for existing users
(this is required — the public surface is frozen and behavior must not break).

An organization that wants to *enforce* a trust baseline calls
`tokeymeter.set_security_policy(...)`. When a requirement is enabled, the unsafe
construction REFUSES to build (raises `SecurityPolicyError`) instead of merely
warning — so the safe path is not just easier, it is the only path that runs.

This is process-wide and additive: new requirements can be added in future
versions without changing existing call sites. Designed for current and future
systems — a single switch a platform team flips to standardize the safe path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


class SecurityPolicyError(RuntimeError):
    """Raised when a construction violates the active SecurityPolicy."""


@dataclass(frozen=True)
class SecurityPolicy:
    # Shared/distributed cache VALUES must be encrypted (a real cipher, not
    # None and not the explicit NoOpCipher passthrough).
    require_encryption: bool = False
    # RedisStore must HMAC its cache keys (key_secret set) so the keyspace holds
    # only opaque MACs, defeating prompt-confirmation attacks.
    require_keyed_cache: bool = False
    # @tokeymeter.cache calls must resolve a redactor (local or default), so PII is
    # stripped before it reaches the model, cache, or audit.
    require_redaction: bool = False
    # The audit ledger must use a NON-REPUDIABLE (asymmetric) signer such as
    # Ed25519 — HMAC provides integrity but a holder of the key can forge, so it
    # is insufficient where non-repudiation is required (regulated environments).
    require_nonrepudiable_audit: bool = False
    # Audit appends must not be silently dropped under burst: the ledger applies
    # bounded backpressure and, if the async queue is exhausted, writes the entry
    # through synchronously rather than dropping it. Makes "every decision is
    # recorded" a guarantee under saturation, at the cost of bounded latency.
    require_audit_durability: bool = False

    def describe(self) -> str:
        on = [k for k, v in vars(self).items() if v]
        return "SecurityPolicy(" + (", ".join(on) if on else "permissive (default)") + ")"


_policy = SecurityPolicy()


def set_security_policy(policy: Optional[SecurityPolicy] = None, **overrides) -> SecurityPolicy:
    """Install a process-wide security policy. Accepts a SecurityPolicy and/or
    keyword overrides:

        tokeymeter.set_security_policy(require_encryption=True, require_keyed_cache=True)

    Returns the active policy. Enforcement happens at CONSTRUCTION time (store /
    audit) and at first-call time (redaction), with a clear SecurityPolicyError.
    """
    global _policy
    base = policy if policy is not None else _policy
    if overrides:
        valid = set(vars(SecurityPolicy()).keys())
        bad = set(overrides) - valid
        if bad:
            raise ValueError(f"unknown policy options: {sorted(bad)}; valid: {sorted(valid)}")
        base = replace(base, **overrides)
    _policy = base
    return _policy


def get_security_policy() -> SecurityPolicy:
    """Return the active security policy."""
    return _policy


def reset_security_policy() -> SecurityPolicy:
    """Reset to the permissive default (used in tests / teardown)."""
    global _policy
    _policy = SecurityPolicy()
    return _policy
