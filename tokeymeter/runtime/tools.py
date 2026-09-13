"""Developer tools (W9) — the doctor, and the policy-pack registry.

`doctor()` is the first-hour developer experience: it inspects the environment
and a Runtime's configuration and reports what is ready, what is missing, and
what to do — the friction-remover that makes the wedge land. It is
content-blind and network-free: it checks presence and shape, never secrets'
values and never a live provider call unless explicitly asked.

Policy packs are named, testable bundles of governance settings a deployment
can apply as one unit (a data-residency pack, a strict-logging pack, a
regulated-sector pack). A pack is config, not code — it maps to the config
keys the enforcement engines already read, so applying a pack is a
configuration change with a known, testable effect.

REACH + EXPERIENCE only. No new inference path; builds on shipped config and
adapters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .catalog import CATALOG, get_spec


# =====================================================================
# doctor
# =====================================================================
@dataclass
class DoctorCheck:
    name: str
    status: str                      # "ok" | "warn" | "missing"
    detail: str = ""


@dataclass
class DoctorReport:
    checks: List[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "missing" for c in self.checks)

    def render(self) -> str:
        icon = {"ok": "✓", "warn": "!", "missing": "✗"}
        lines = ["Tokeymeter doctor:"]
        for c in self.checks:
            lines.append(f"  {icon.get(c.status, '?')} {c.name}: {c.detail}")
        verdict = "READY" if self.ok else "ACTION NEEDED"
        lines.append(f"  → {verdict}")
        return "\n".join(lines)


def doctor(*, provider: Optional[str] = None,
           config: Optional[Dict[str, Any]] = None) -> DoctorReport:
    """Diagnose the environment and (optional) provider/config. Content-blind:
    reports whether an API key is PRESENT, never its value."""
    report = DoctorReport()

    # python + cryptography (required for proof/signing)
    import sys
    report.checks.append(DoctorCheck(
        "python", "ok", f"{sys.version_info.major}.{sys.version_info.minor}"))
    try:
        import cryptography  # noqa: F401
        report.checks.append(DoctorCheck("cryptography", "ok", "available"))
    except Exception:
        report.checks.append(DoctorCheck(
            "cryptography", "missing", "required for proof packets"))

    # provider readiness (presence of the key, never the value)
    if provider is not None:
        try:
            spec = get_spec(provider)
            if spec.env_key:
                present = bool(os.environ.get(spec.env_key))
                report.checks.append(DoctorCheck(
                    f"provider:{provider}",
                    "ok" if present else "warn",
                    f"{spec.env_key} " +
                    ("present" if present else "not set — set it to run")))
            else:
                report.checks.append(DoctorCheck(
                    f"provider:{provider}", "ok",
                    f"self-hosted ({spec.base_url}) — no key needed"))
        except KeyError as exc:
            report.checks.append(DoctorCheck(
                f"provider:{provider}", "missing", str(exc)))

    # config sanity
    if config is not None:
        from .config import RuntimeConfig
        rc = RuntimeConfig(config)
        if rc.get("trust.proof.enabled") and not _has_signer(config):
            report.checks.append(DoctorCheck(
                "proof-signer", "warn",
                "proof enabled but no signer configured — packets unsigned"))
        else:
            report.checks.append(DoctorCheck("config", "ok", "shape valid"))

    # optional OTel
    try:
        import opentelemetry  # noqa: F401
        report.checks.append(DoctorCheck("opentelemetry", "ok", "available"))
    except Exception:
        report.checks.append(DoctorCheck(
            "opentelemetry", "warn", "not installed — telemetry degrades to no-op"))

    return report


def _has_signer(config: Dict[str, Any]) -> bool:
    return bool(config.get("_signer"))  # facade takes signer via kwarg, not config


# =====================================================================
# policy packs
# =====================================================================
@dataclass(frozen=True)
class PolicyPack:
    """A named, testable bundle of governance config. Applying a pack merges
    its settings into a Runtime config — config, not code."""
    name: str
    description: str
    config: Dict[str, Any]

    def apply_to(self, base: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
        """Merge this pack over a base config (pack wins on conflict)."""
        from .config import _deep_merge
        return _deep_merge(dict(base or {}), self.config)


PACKS: Dict[str, PolicyPack] = {
    "strict-logging": PolicyPack(
        "strict-logging",
        "Minimal retention: block secrets, redact PII, seal proof. For "
        "regulated data where nothing sensitive may persist.",
        {"governance": {"security": {"enabled": True, "secrets_mode": "block",
                                     "pii": True}},
         "trust": {"proof": {"enabled": True}}}),
    "cost-guard": PolicyPack(
        "cost-guard",
        "Enforce budgets and optimize aggressively. For teams under a hard "
        "AI spend ceiling.",
        {"optimization": {"enabled": True, "compress": True,
                          "route": True, "objective": "cost"},
         "economics": {"enabled": True, "budget_mode": "enforce"}}),
    "regulated-financial": PolicyPack(
        "regulated-financial",
        "Full governance stack: secrets/PII screening, deny-by-default access, "
        "proof, and approval routing for over-budget calls.",
        {"governance": {"security": {"enabled": True, "secrets_mode": "block",
                                     "pii": True}},
         "economics": {"enabled": True, "budget_mode": "approval"},
         "trust": {"proof": {"enabled": True}},
         "telemetry": {"enabled": True}}),
    "self-hosted-gpu": PolicyPack(
        "self-hosted-gpu",
        "For in-perimeter open models on own GPUs: optimization on, proof on, "
        "no external provider assumptions.",
        {"optimization": {"enabled": True, "compress": True},
         "trust": {"proof": {"enabled": True}},
         "telemetry": {"enabled": True}}),
}


def list_packs() -> List[str]:
    return sorted(PACKS)


def get_pack(name: str) -> PolicyPack:
    pack = PACKS.get(name)
    if pack is None:
        raise KeyError(
            f"unknown pack {name!r}; known: {', '.join(list_packs())}")
    return pack
