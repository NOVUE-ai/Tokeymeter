"""
Signing primitives for the audit layer.

A signature is what makes a proof tamper-evident across trust boundaries:
the hash chain alone is NOT tamper-evident, because the entry hash is
unkeyed and an attacker can recompute the whole chain. So a verifiable
signature is required, not optional, for any proof leaving your process.

Two signer types, with DIFFERENT security properties:

  HMACSigner (symmetric, stdlib): the verifier holds the SAME secret used
    to sign. This gives integrity/authenticity for an INTERNAL verifier you
    trust (e.g., your own CFO tooling). It does NOT provide non-repudiation:
    because the verifier could itself have produced any signature it can
    verify, it cannot prove to a third party that YOU created the proof.

  Ed25519Signer (asymmetric, requires `cryptography`): you sign with a
    PRIVATE key and hand auditors only the PUBLIC key. They can verify but
    cannot forge. THIS is the one that provides non-repudiation and is the
    correct choice for external auditors / regulators.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import Optional, Protocol

log = logging.getLogger("tokeymeter.audit")


class Signer(Protocol):
    """Anything that signs bytes and verifies signatures.

    `algorithm` identifies the algorithm in the proof bundle so
    verifiers know what to use.
    """
    algorithm: str

    def sign(self, data: bytes) -> bytes: ...
    def verify(self, data: bytes, signature: bytes) -> bool: ...


class HMACSigner:
    """Default signer: HMAC-SHA256. Stdlib only.

    Symmetric: same secret signs and verifies. Suitable for internal use
    where the verifier (e.g., your CFO's spreadsheet) has the secret.
    """
    algorithm = "hmac-sha256"

    def __init__(self, secret: bytes):
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
            raise ValueError("HMACSigner secret must be ≥16 bytes")
        self._secret = bytes(secret)

    def sign(self, data: bytes) -> bytes:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        return hmac.new(self._secret, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            expected = self.sign(data)
            return hmac.compare_digest(expected, signature)
        except Exception:
            return False


class Ed25519Signer:
    """Asymmetric signer (Ed25519). Provides NON-REPUDIATION.

    Sign with the private key; distribute only the public key to verifiers.
    A verifier with the public key can confirm authenticity but CANNOT forge
    a new signature — unlike HMAC, where the verifier shares the signing key.

    Requires `cryptography` (lazy-imported so the core stays zero-dependency).
    """
    algorithm = "ed25519"

    def __init__(self, private_key: Optional[bytes] = None,
                 public_key: Optional[bytes] = None):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey, Ed25519PublicKey,
            )
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "Ed25519Signer requires the 'cryptography' package. "
                "Install with: pip install tokeymeter[audit]  (or `cryptography`)."
            ) from e
        self._priv = None
        self._pub = None
        if private_key is not None:
            self._priv = Ed25519PrivateKey.from_private_bytes(bytes(private_key))
            self._pub = self._priv.public_key()
        elif public_key is not None:
            self._pub = Ed25519PublicKey.from_public_bytes(bytes(public_key))
        else:
            raise ValueError("provide private_key, public_key, or use .generate()")

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        priv = Ed25519PrivateKey.generate()
        raw = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return cls(private_key=raw)

    def private_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        if self._priv is None:
            raise RuntimeError("this is a verify-only signer (no private key)")
        return self._priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return self._pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_only(self) -> "Ed25519Signer":
        """Return a verify-only signer (the thing you hand an auditor)."""
        return Ed25519Signer(public_key=self.public_bytes())

    def sign(self, data: bytes) -> bytes:
        if self._priv is None:
            raise RuntimeError("cannot sign with a verify-only signer")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        return self._priv.sign(bytes(data))

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            self._pub.verify(bytes(signature), bytes(data))
            return True
        except Exception:
            return False


def generate_install_secret() -> bytes:
    """Generate a fresh 32-byte secret. Use once per Tokeymeter install.

    Stored in ~/.tokeymeter/install-secret (mode 0600) by default.
    """
    return secrets.token_bytes(32)


def load_or_create_install_secret(path: str) -> bytes:
    """Read an install secret from disk, creating it on first call.

    On first call, generates and writes the secret with mode 0600.
    On subsequent calls, reads it back.

    Returns the secret as bytes. On any error, logs and returns a
    freshly-generated secret (which means audit chains won't link
    across processes — degraded but not broken).
    """
    import os
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            if len(data) >= 16:
                return data
            log.debug("audit: install secret file too short, regenerating")
        # Create
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        secret = generate_install_secret()
        # Atomic write with restrictive perms
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(secret)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        return secret
    except Exception as e:
        log.warning("audit: could not persist install secret (%s); "
                    "using ephemeral. Chains will not link across processes.", e)
        return generate_install_secret()
