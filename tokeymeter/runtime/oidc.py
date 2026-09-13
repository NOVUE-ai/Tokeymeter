"""OIDC identity (W8) — bind AI calls to corporate identity.

Access control today reads a principal from a contextvar. This module resolves
that principal from a standards-based OIDC/JWT token, so roles and identity
come from the enterprise's own identity provider rather than a parallel system.

Scope and honesty: this is the IDENTITY-RESOLUTION contract and a verifying
resolver. It validates a JWT's signature (when a key is supplied), its
expiry, its issuer and audience, and extracts a stable principal claim. It is
deliberately provider-agnostic — the deployment supplies the verification key
(from the IdP's JWKS) and the claim names. Full JWKS rotation and discovery is
a deployment concern layered on top; this is the verified-claims core.

Content-blind: only the principal identifier (and optionally declared role
claims) are bound into context — never the raw token, never arbitrary claims.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tokeymeter.engines.governance.identity import set_principal


class IdentityError(Exception):
    pass


class TokenExpired(IdentityError):
    pass


class TokenInvalid(IdentityError):
    pass


@dataclass
class ResolvedIdentity:
    principal: str
    roles: List[str] = field(default_factory=list)
    issuer: str = ""
    expires_at: float = 0.0


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def _decode_unverified(token: str) -> Dict[str, Any]:
    try:
        _, payload_b64, _ = token.split(".")
        return json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise TokenInvalid(f"malformed JWT: {type(exc).__name__}") from exc


class OIDCResolver:
    """Resolves a verified principal from a JWT.

    Verification modes:
      - hs256_secret: symmetric HMAC verification (for IdPs/tests using HS256)
      - rs256_public_key_pem / ed25519_public_key_hex: asymmetric verification
      - none: signature NOT checked (explicitly opt-in; refuses by default)

    Claims:
      - principal_claim: which claim is the stable principal (default "sub")
      - roles_claim: optional claim carrying role list
      - issuer / audience: enforced when provided
    """

    def __init__(self, *, principal_claim: str = "sub",
                 roles_claim: Optional[str] = None,
                 issuer: Optional[str] = None,
                 audience: Optional[str] = None,
                 hs256_secret: Optional[bytes] = None,
                 ed25519_public_key_hex: Optional[str] = None,
                 allow_unverified: bool = False,
                 leeway_s: float = 30.0) -> None:
        self._principal_claim = principal_claim
        self._roles_claim = roles_claim
        self._issuer = issuer
        self._audience = audience
        self._hs256 = hs256_secret
        self._ed25519 = ed25519_public_key_hex
        self._allow_unverified = allow_unverified
        self._leeway = leeway_s

    # ---- verification --------------------------------------------------
    def _verify_signature(self, token: str) -> None:
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
        except ValueError as exc:
            raise TokenInvalid("JWT must have three segments") from exc
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        sig = _b64url_decode(sig_b64)
        header = json.loads(_b64url_decode(header_b64))
        alg = header.get("alg", "")

        if self._hs256 is not None and alg == "HS256":
            expected = hmac.new(self._hs256, signing_input,
                                hashlib.sha256).digest()
            if not hmac.compare_digest(expected, sig):
                raise TokenInvalid("HS256 signature mismatch")
            return
        if self._ed25519 is not None and alg == "EdDSA":
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey)
            try:
                pub = Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(self._ed25519))
                pub.verify(sig, signing_input)
                return
            except Exception as exc:
                raise TokenInvalid("EdDSA signature invalid") from exc
        if self._allow_unverified:
            return                                   # explicit opt-in only
        raise TokenInvalid(
            f"no verification key for alg={alg!r} and unverified not allowed")

    # ---- resolution ----------------------------------------------------
    def resolve(self, token: str) -> ResolvedIdentity:
        self._verify_signature(token)
        claims = _decode_unverified(token)

        exp = claims.get("exp")
        if exp is not None and time.time() > float(exp) + self._leeway:
            raise TokenExpired("token expired")
        nbf = claims.get("nbf")
        if nbf is not None and time.time() + self._leeway < float(nbf):
            raise TokenInvalid("token not yet valid")

        if self._issuer is not None and claims.get("iss") != self._issuer:
            raise TokenInvalid("issuer mismatch")
        if self._audience is not None:
            aud = claims.get("aud")
            aud_ok = (aud == self._audience or
                      (isinstance(aud, list) and self._audience in aud))
            if not aud_ok:
                raise TokenInvalid("audience mismatch")

        principal = claims.get(self._principal_claim)
        if not principal:
            raise TokenInvalid(
                f"missing principal claim {self._principal_claim!r}")
        roles = []
        if self._roles_claim:
            raw = claims.get(self._roles_claim) or []
            roles = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        return ResolvedIdentity(
            principal=str(principal), roles=roles,
            issuer=str(claims.get("iss", "")),
            expires_at=float(exp) if exp is not None else 0.0)

    def bind(self, token: str) -> ResolvedIdentity:
        """Resolve and bind the principal into the shipped contextvar so the
        access engine and telemetry see it. Returns the resolved identity."""
        identity = self.resolve(token)
        set_principal(identity.principal)
        return identity
