"""Tokeymeter Runtime Kernel (K-track, Sprint 1+2).

The kernel is the structural spine mandated by the Universal Enterprise AI
Runtime architecture: config loading, dependency injection, an event/hook bus
with safe-failover, an ordered engine registry, and a graceful lifecycle.

Design contract (binding):
- COMPATIBILITY MODE: this package is purely additive. Existing entrypoints
  (@tokeymeter.cache, wrappers, TokeNet) are untouched; engines DELEGATE to
  shipped modules rather than duplicating them.
- CONTENT-BLIND AT THE KERNEL: request payloads pass through transiently for
  execution, but kernel telemetry (traces, trust log) carries only metadata
  and SHA-256 fingerprints — never prompt or response text.
- SAFE-FAILOVER: a failing hook/plugin is reported on the degraded bus and
  skipped; it can never crash the core (doc: "a bad plugin cannot crash the
  core").
- STAGE GATES: Knowledge/Evaluation slots exist as interfaces with no-op
  defaults; content-aware implementations remain gated to their locked stage.
"""
from .config import RuntimeConfig
from .container import Container
from .bus import HookBus
from .engine import Engine, EngineRegistry
from .kernel import Kernel, KernelRequest, KernelResponse, KernelStopped
from .providers import (
    ProviderAdapter,
    ProviderInfo,
    HealthStatus,
    CallableAdapter,
    ExecutionEngine,
)
from .engines import (
    GovernanceEngine,
    CacheEngine,
    CachePolicy,
    InMemoryCache,
    TrustEngine,
    ReliabilityEngine,
    KnowledgeEngine,
)

__all__ = [
    "RuntimeConfig", "Container", "HookBus", "Engine", "EngineRegistry",
    "Kernel", "KernelRequest", "KernelResponse", "KernelStopped",
    "ProviderAdapter", "ProviderInfo", "HealthStatus", "CallableAdapter",
    "ExecutionEngine", "GovernanceEngine", "CacheEngine", "CachePolicy",
    "InMemoryCache", "TrustEngine", "ReliabilityEngine", "KnowledgeEngine",
]

# W2 (K3+EXEC-4): real adapters + error taxonomy — lazily exported so that
# `import tokeymeter.runtime` stays feather-light (the adapters pull the
# full shipped optimization stack only when actually used).
_LAZY = {
    "SecurityEngine": "enforcement", "AccessEngine": "enforcement",
    "RateLimitEngine": "enforcement", "CheckpointHook": "enforcement",
    "LocalApprover": "enforcement", "SecretBlocked": "enforcement",
    "ContentPolicyViolation": "enforcement", "AccessDenied": "enforcement",
    "RateLimitExceeded": "enforcement", "CheckpointDenied": "enforcement",
    "CheckpointPending": "enforcement",
    "EconomicsEngine": "economics", "BudgetExceeded": "economics",
    "Runtime": "facade", "RuntimeConfigurationError": "facade",
    "ProviderSpec": "catalog", "CATALOG": "catalog",
    "list_providers": "catalog", "get_spec": "catalog",
    "adapter_for": "catalog", "async_adapter_for": "catalog",
    "check_adapter": "conformance", "ConformanceReport": "conformance",
    "ConformanceResult": "conformance", "ConformanceClient": "conformance",
    "wrap_callable": "frameworks", "wrap_langchain_llm": "frameworks",
    "wrap_llamaindex_llm": "frameworks", "governed_tool": "frameworks",
    "doctor": "tools", "DoctorReport": "tools", "DoctorCheck": "tools",
    "PolicyPack": "tools", "PACKS": "tools", "list_packs": "tools",
    "get_pack": "tools",
    "PluginManifest": "plugins", "SignedPlugin": "plugins",
    "PluginRegistry": "plugins", "sign_plugin": "plugins",
    "PluginVerificationError": "plugins", "PluginError": "plugins",
    "ReloadableConfig": "hotreload", "ConfigSnapshot": "hotreload",
    "ConfigValidationError": "hotreload",
    "OIDCResolver": "oidc", "ResolvedIdentity": "oidc",
    "TokenExpired": "oidc", "TokenInvalid": "oidc", "IdentityError": "oidc",
    "TelemetrySink": "telemetry", "InMemorySink": "telemetry",
    "OTelSink": "telemetry", "TelemetryEngine": "telemetry",
    "content_blind_record": "telemetry",
    "OptimizationEngine": "optimization",
    "Optimizer": "optimize", "TieredOptimizer": "optimize",
    "NoOpOptimizer": "optimize", "RoutePlanner": "optimize",
    "RoutePlan": "optimize",
    "CircuitBreaker": "resilience", "BreakerState": "resilience",
    "Bulkhead": "resilience", "BackoffPolicy": "resilience",
    "OutputValidator": "resilience", "OutputInvalid": "resilience",
    "ChaosInjector": "resilience", "ResilientExecution": "resilience",
    "CircuitOpen": "resilience", "BulkheadFull": "resilience",
    "ProofEngine": "proof", "ProofPacket": "proof",
    "verify_proof_packet": "proof", "FileAuditSink": "proof",
    "AuditSink": "proof", "merkle_root": "proof", "merkle_proof": "proof",
    "verify_merkle_proof": "proof",
    "OpenAIAdapter": "adapters", "AnthropicAdapter": "adapters",
    "AsyncOpenAIAdapter": "adapters",
    "ProviderError": "errors", "RateLimited": "errors", "AuthError": "errors",
    "TransientError": "errors", "MalformedRequest": "errors",
    "ProviderDown": "errors", "classify": "errors",
}
__all__ = __all__ + sorted(_LAZY)

def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
