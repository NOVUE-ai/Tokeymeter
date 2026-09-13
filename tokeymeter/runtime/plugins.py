"""Plugin loader (W8) — the runtime becomes an extensible platform.

A plugin is code the customer or a third party supplies to attach to the
pipeline's hooks. Two laws govern it:

1. SIGNED-OR-REFUSED. Every plugin ships with a manifest (identity, version,
   the hooks it attaches to, its declared class — observe or enforce, and its
   order) plus an Ed25519 signature over the canonical manifest bytes. The
   loader recomputes the manifest hash, verifies the signature against a
   TRUSTED public key, and refuses to load anything that does not verify.
   The private key never ships; deployments carry only the public key.

2. DUAL-SEMANTICS ISOLATION. An "observe" plugin runs on the observation path:
   if it raises, it is isolated and the request is unaffected (a faulty
   extension can never break a call). An "enforce" plugin runs on the
   enforcement path with veto power and MUST be declared as such in its
   signed manifest — you cannot smuggle enforcement authority through an
   observe declaration, because the manifest (and thus the class) is signed.

The loader reuses the SHIPPED Ed25519Signer (sign/verify) — no new crypto.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tokeymeter.engines.trust.audit.signers import Ed25519Signer

from .engine import Engine


class PluginError(Exception):
    pass


class PluginVerificationError(PluginError):
    """Manifest failed signature/hash verification — refused."""


class PluginDeclarationError(PluginError):
    """Manifest is malformed or declares something inconsistent."""


_VALID_CLASSES = ("observe", "enforce")
_VALID_HOOKS = ("before_request", "after_response", "on_error")


@dataclass
class PluginManifest:
    name: str
    version: str
    plugin_class: str                # "observe" | "enforce"
    hook: str                        # one of _VALID_HOOKS
    order: int = 100
    author: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.plugin_class not in _VALID_CLASSES:
            raise PluginDeclarationError(
                f"plugin_class must be one of {_VALID_CLASSES}")
        if self.hook not in _VALID_HOOKS:
            raise PluginDeclarationError(
                f"hook must be one of {_VALID_HOOKS}")
        if not self.name or not self.version:
            raise PluginDeclarationError("name and version are required")

    def canonical_bytes(self) -> bytes:
        """Deterministic manifest bytes — the thing that is signed and hashed.
        Sorted keys, no whitespace drift, so signer and verifier agree."""
        payload = {
            "name": self.name, "version": self.version,
            "plugin_class": self.plugin_class, "hook": self.hook,
            "order": self.order, "author": self.author,
            "description": self.description,
        }
        return json.dumps(payload, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass
class SignedPlugin:
    manifest: PluginManifest
    manifest_hash: str
    signature: str                   # hex
    public_key: str                  # hex — the key that must be trusted


def sign_plugin(manifest: PluginManifest, signer: Ed25519Signer
                ) -> SignedPlugin:
    """Author-side: produce a signed plugin record. Uses the shipped signer."""
    sig = signer.sign(manifest.canonical_bytes())
    return SignedPlugin(
        manifest=manifest,
        manifest_hash=manifest.manifest_hash(),
        signature=sig.hex(),
        public_key=signer.public_bytes().hex(),
    )


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    fn: Callable[[Dict[str, Any]], None]


class _PluginEngine(Engine):
    """Adapts a loaded plugin to the Engine contract. Observe plugins swallow
    exceptions (isolation); enforce plugins let them propagate (veto)."""

    handles_execution = False

    def __init__(self, loaded: "LoadedPlugin") -> None:
        self._loaded = loaded
        self.name = f"plugin:{loaded.manifest.name}"
        self._hook = loaded.manifest.hook
        self._enforce = (loaded.manifest.plugin_class == "enforce")

    def _run(self, ctx: Dict[str, Any]) -> None:
        if self._enforce:
            self._loaded.fn(ctx)                       # veto propagates
        else:
            try:
                self._loaded.fn(ctx)
            except Exception:
                pass                                   # observe isolation

    def before_request(self, ctx: Dict[str, Any]) -> None:
        if self._hook == "before_request":
            self._run(ctx)

    def after_response(self, ctx: Dict[str, Any]) -> None:
        if self._hook == "after_response":
            self._run(ctx)

    def on_error(self, ctx: Dict[str, Any], exc: BaseException) -> None:
        if self._hook == "on_error":
            self._run(ctx)


class PluginRegistry:
    """Loads verified plugins and attaches them to the kernel's hooks. The
    kernel already isolates observe-path subscribers; this registry enforces
    the SIGNED CLASS so an enforce plugin cannot masquerade as observe."""

    def __init__(self, *, trusted_keys: Optional[List[str]] = None) -> None:
        # hex public keys the deployment trusts. Empty = trust nothing (secure
        # default): every load must present a key on the trust list.
        self._trusted = set(trusted_keys or [])
        self._loaded: List[LoadedPlugin] = []

    def trust_key(self, public_key_hex: str) -> None:
        self._trusted.add(public_key_hex)

    def verify(self, signed: SignedPlugin) -> None:
        """Raise PluginVerificationError unless the plugin verifies AND its
        key is trusted."""
        # 1. hash integrity
        recomputed = signed.manifest.manifest_hash()
        if recomputed != signed.manifest_hash:
            raise PluginVerificationError(
                f"manifest hash mismatch for {signed.manifest.name}")
        # 2. key must be trusted (a valid signature from an untrusted key is
        # still refused — signing proves origin, trust is a separate decision)
        if signed.public_key not in self._trusted:
            raise PluginVerificationError(
                f"untrusted signing key for {signed.manifest.name}")
        # 3. signature over the canonical bytes (offline verify with the
        # presented public key — same primitive as the proof spine)
        if not _verify_signature(signed):
            raise PluginVerificationError(
                f"bad signature for {signed.manifest.name}")

    def load(self, signed: SignedPlugin,
             fn: Callable[[Dict[str, Any]], None]) -> LoadedPlugin:
        """Verify then register. Refuses on any failure."""
        self.verify(signed)
        loaded = LoadedPlugin(manifest=signed.manifest, fn=fn)
        self._loaded.append(loaded)
        return loaded

    def as_engines(self) -> "List[_PluginEngine]":
        """Materialize loaded plugins as first-class kernel engines, ordered
        by signed order. Enforce plugins raise on the pipeline (fail closed,
        veto); observe plugins are wrapped so a raise is isolated."""
        return [_PluginEngine(lp) for lp in sorted(
            self._loaded, key=lambda p: (p.manifest.hook, p.manifest.order))]

    def attach_all(self, kernel: Any) -> None:
        """Register every loaded plugin as a kernel engine so the signed
        `order` field is the actual FIRING order for that hook.

        Subtlety: after_response engines unwind in REVERSE registration order,
        so after-hook plugins are registered in reverse to fire in ascending
        signed order; before_request / on_error fire in registration order."""
        engines = self.as_engines()
        before = [e for e in engines if e._hook in ("before_request",
                                                    "on_error")]
        after = [e for e in engines if e._hook == "after_response"]
        for eng in before:
            kernel.register_engine(eng)
        for eng in reversed(after):                    # reverse -> fires ascending
            kernel.register_engine(eng)

    @property
    def loaded(self) -> List[LoadedPlugin]:
        return list(self._loaded)


def _verify_signature(signed: SignedPlugin) -> bool:
    """Verify the Ed25519 signature using the cryptography primitive directly
    (the shipped signer verifies with its own keypair; here we verify an
    arbitrary public key, which is the offline-verify pattern from the proof
    spine)."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        pub = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(signed.public_key))
        pub.verify(bytes.fromhex(signed.signature),
                   signed.manifest.canonical_bytes())
        return True
    except Exception:
        return False
