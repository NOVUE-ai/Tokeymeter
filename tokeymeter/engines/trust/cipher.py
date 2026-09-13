"""
Encryption for the shared cache — the "zero-knowledge cache" wedge.

When a cache is shared across pods via Redis, the values (your LLM
responses) normally sit in Redis as plaintext. Anyone with Redis access —
a DBA, a compromised neighbor service, a misconfigured backup — can read
every cached completion.

Tokeymeter lets you encrypt cache values at rest with a key only your application
holds. Combined with the fact that cache KEYS are already HMAC hashes, the
shared store becomes zero-knowledge: it holds only opaque hashes and
ciphertext. Even full read access to Redis reveals nothing.

  cache key   = HMAC(secret, prompt)      ← already a hash, set by Tokeymeter
  cache value = Encrypt(app_key, response) ← optional, this module

No proxy-based competitor can offer this: a proxy must decrypt to route.
Tokeymeter encrypts in your process and the bytes never leave readable.

The Cipher protocol is injectable — bring your own (AWS KMS, HSM, Vault).
The shipped FernetCipher uses the `cryptography` library (optional).
"""
from __future__ import annotations

import base64
import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cipher(Protocol):
    """Anything that can encrypt and decrypt bytes.

    Implementations MUST be deterministic in availability (no network in the
    hot path unless you accept the latency) and MUST raise on decrypt failure
    so the store can fail-open to a cache miss rather than serve garbage.
    """
    def encrypt(self, data: bytes) -> bytes: ...
    def decrypt(self, token: bytes) -> bytes: ...


class FernetCipher:
    """Authenticated encryption using cryptography.fernet (AES-128-CBC + HMAC).

    Fernet provides confidentiality AND integrity: a tampered ciphertext
    fails to decrypt rather than returning corrupted plaintext. That matters
    for a cache — a flipped bit in Redis becomes a clean miss, not a poisoned
    response.

    Construct with a urlsafe-base64 32-byte key (Fernet.generate_key()), or
    derive one from a passphrase with FernetCipher.from_passphrase().
    """
    def __init__(self, key: bytes):
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise ImportError(
                "FernetCipher requires the 'cryptography' package. "
                "Install with: pip install tokeymeter[scale]  (or: pip install cryptography)"
            ) from e
        self._Fernet = Fernet
        self._f = Fernet(key)

    def encrypt(self, data: bytes) -> bytes:
        return self._f.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        # Raises cryptography.fernet.InvalidToken on tamper/wrong-key — caller
        # treats that as a cache miss (fail-open).
        return self._f.decrypt(token)

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a fresh random Fernet key (urlsafe base64, 32 bytes)."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    @classmethod
    def from_passphrase(cls, passphrase: str, *, salt: bytes,
                        iterations: int = 480_000) -> "FernetCipher":
        """Derive a Fernet key from a passphrase via PBKDF2-HMAC-SHA256.

        The salt should be stable per deployment (so all pods derive the same
        key) and stored alongside your config — it is not secret, but it must
        be consistent. iterations follows current OWASP guidance.
        """
        if not isinstance(salt, (bytes, bytearray)) or len(salt) < 16:
            raise ValueError("salt must be >= 16 bytes")
        dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"),
                                 bytes(salt), iterations, dklen=32)
        key = base64.urlsafe_b64encode(dk)
        return cls(key)


class NoOpCipher:
    """Identity cipher — stores plaintext. The explicit default.

    Exists so the store's code path is uniform (always calls a cipher) and so
    'no encryption' is a deliberate, visible choice rather than an absence.
    """
    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, token: bytes) -> bytes:
        return token
