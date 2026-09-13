"""
Tokeymeter — industrial-grade LLM cost optimization.

Two-tier cache (exact + semantic) with TTL, single-flight dedup, optional
sqlite-vec acceleration, sync + async + streaming. Plus a trust layer:
observable events, shadow mode, PII redaction, workload tagging.

Quick start:

    import tokeymeter

    @tokeymeter.cache(
        model="gpt-4o-mini",
        semantic=True,        # paraphrase matching
        ttl=3600,             # entries expire after 1 hour
        shadow=False,         # set True to measure without serving cached
        redactor=tokeymeter.privacy.default_redactor,  # strip PII before caching
        tag="support",        # workload label for reporting
    )
    async def ask(prompt: str) -> str:
        return await openai_call(prompt)

    # Wire observability
    @tokeymeter.events.on_event
    def to_datadog(event):
        statsd.increment("tokeymeter.lookup",
                         tags=[f"hit_type:{event.hit_type or 'miss'}"])

    print(tokeymeter.savings_report())
"""
from . import admin, audit, backends, compression, events, integrity, memory, metrics, privacy, warmup
from .cache_admin import clear_cache
from .compression import (
    Compressor,
    CompressionResult,
    LLMLinguaCompressor,
    StructuralCompressor,
    compose,
    safe_compress,
)
from .salience import SalienceCompressor
from .safe_compress import SafeCompressor, compression_stats, reset_compression_stats
from .router import Router, RouteDecision, Cascade, CascadeResult
from .cache_optimize import CacheOptimizer, CacheReport, VolatileFinding
from .decorator import (
    cache,
    cache_stream,
    lineage,
    tenant_scope,
    set_compression_max_reduction,
    set_fidelity_circuit_breaker,
    compression_breaker_state,
    set_event_preview_policy,
    set_runtime_guards,
    enterprise_defaults,
    compression_verification_log,
    memory_fidelity_log,
    set_default_redactor,
    set_default_semantic_cache,
    set_default_store,
    with_memory,
)
from .envelope import unwrap, wrap
from . import prompt_ir  # Phase 0 Prompt IR substrate (namespaced; consumes nothing yet)
from .policy import (
    SecurityPolicy,
    SecurityPolicyError,
    set_security_policy,
    get_security_policy,
    reset_security_policy,
)
from .decision import (
    DecisionRecord,
    on_decision,
    remove_decision_subscriber,
    clear_decision_subscribers,
)
from .memory import (
    CallableSummarizer,
    ConversationMemory,
    InMemoryMemoryStore,
    SQLiteMemoryStore,
    Summarizer,
    Turn,
    TruncationSummarizer,
)
from .metrics import PrometheusCollector
from .integrity import scan_environment, self_check, verify_self
from .backends import RedisStore, FernetCipher, NoOpCipher
from .privacy import DefaultRedactor, default_redactor, redact
from .savings import (reset_savings, savings_report, savings_ledger_health, set_home,
                      set_savings_path, set_buffered_savings, set_in_memory_savings, flush_savings,
                      capacity_report)
from .identity import set_principal, get_principal, principal
from .engines.governance.rules import (
    RuleSet, Rule, Condition, RulePolicyError,
    load_rules, load_rules_file, set_rules, get_rules, clear_rules,
    rules_version, set_env, current_env)
from .engines.execution.halts import (
    HaltEvent, on_halt, remove_halt_handler, clear_halt_handlers,
    halt_handler_count)
from .engines.governance.compliance import (
    data_class, set_data_class, current_data_class,
    region, set_region, current_region,
    PolicyViolation, ModelNotPermitted, EndpointNotPermitted,
    resolve_compliance,
    STARTER_POLICIES, starter_policy, list_starter_policies, policy_report)
from .engines.governance.simulate import simulate_rules
from .engines.governance.plan import (
    plan_report, coverage_report, render_plan)
from .engines.governance.agents import (
    agent_report, render_agents, compare_reports, render_comparison)
from .engines.governance.suggest import suggest_thresholds, render_suggestions
from .engines.governance.agents_html import render_agents_html, write_agents_html
# LangChain / LlamaIndex adapters. Importing this module never requires either
# package to be installed — the adapters duck-type whatever is handed to them.
from .runtime.frameworks import wrap_langchain_llm, wrap_llamaindex_llm
from .engines.execution.task import (
    task, bind_task, set_task, get_task, current_task_id, current_agent,
    task_snapshot,
    TaskLimitExceeded, TaskEnvelopeExceeded, TaskLoopDetected,
    TaskStalled, TaskCallLimitExceeded)
from .engines.execution.endpoint import set_endpoint, get_endpoint, endpoint
from .usage import set_queue_wait_ms
from .keys import (register_key, unregister_key, clear_keys, key,
                   key_status, leak_scan, get_current_key, KeyBudgetExceeded)
from . import context_passport
from .pricing import (register_pricing, unregister_pricing, clear_registered_pricing,
                      registered_pricing, pricing_info, estimate_cost_with_source,
                      derive_selfhost_rate, register_selfhost_pricing, capacity_reclaimed,
                      register_cluster_costs, derive_cluster_gpu_hour_rate)
from .engines.economics.capacity_report import capacity_recovery_report
from .engines.economics.chargeback import chargeback_report, chargeback_csv
from .engines.economics.hybrid import hybrid_placement_report, hybrid_placement_csv
from .engines.economics.close_packet import (
    close_packet, general_ledger_rows, general_ledger_csv)
from .storage import MemoryStore, SQLiteStore

# Semantic primitives are imported lazily to avoid hard-failing when
# extras aren't installed.
try:
    from .semantic import SemanticCache, is_available as semantic_available
    from .semantic import is_vec_index_available
except ImportError:
    SemanticCache = None  # type: ignore[assignment]

    def semantic_available() -> bool:  # type: ignore[no-redef]
        return False

    def is_vec_index_available() -> bool:  # type: ignore[no-redef]
        return False


__version__ = "0.31.1"

# ---------------------------------------------------------------------------
# THE PRIMARY SURFACE
#
# 164 public names is a complete toolbox and a poor first impression: someone
# opening this package has no way to tell which twenty things ARE the product
# and which hundred-and-forty are the machinery underneath.
#
# Nothing is removed — deleting public API from a shipped package breaks
# working code and buys nothing. Instead the product is NAMED here, so docs,
# tooling and a new reader all have one honest entry point. Everything absent
# from this list still works exactly as before.
# ---------------------------------------------------------------------------
PRIMARY_API = (
    # Install it: one decorator on the function that calls a model.
    "cache", "cache_stream",
    # Bound a unit of agent work. An agent task is 10-100 calls; this is the
    # boundary everything else keys on.
    "task", "bind_task", "TaskLimitExceeded",
    # Price what you use, on an API or on your own hardware.
    "register_pricing", "register_cluster_costs",
    # Declare the rules once; the node enforces them everywhere.
    "load_rules_file", "set_rules",
    # See what happened, and what a rule WOULD do before you apply it.
    "agent_report", "plan_report", "savings_report", "report",
    # Money, in the shape finance already works in.
    "chargeback_report", "close_packet", "general_ledger_csv",
    # Owned hardware.
    "capacity_recovery_report", "hybrid_placement_report",
    # Is it healthy, and is it what we shipped?
    "doctor", "verify_self",
)


def primary_api() -> "list[tuple[str, str]]":
    """The product's main surface, with the first line of each docstring.

    Written for a person opening this package for the first time and for the
    docs build — not as a substitute for `__all__`, which remains the complete
    public API.
    """
    out = []
    for name in PRIMARY_API:
        obj = globals().get(name)
        doc = (getattr(obj, "__doc__", "") or "").strip().split("\n")[0]
        out.append((name, doc))
    return out


__all__ = [
    "PRIMARY_API",
    "primary_api",
    # Core
    "cache",
    "no_optimize",
    "lineage",
    "tenant_scope",
    "set_compression_max_reduction",
    "set_fidelity_circuit_breaker",
    "compression_breaker_state",
    "set_event_preview_policy",
    "set_runtime_guards",
    "enterprise_defaults",
    "cache_stream",
    "savings_report",
    "capacity_report",
    # Identity binding (v0.14 keystone)
    "set_principal",
    "get_principal",
    "principal",
    "set_endpoint",
    "get_endpoint",
    "endpoint",
    "task",
    "bind_task",
    "RuleSet",
    "Rule",
    "Condition",
    "RulePolicyError",
    "load_rules",
    "load_rules_file",
    "set_rules",
    "get_rules",
    "clear_rules",
    "rules_version",
    "set_env",
    "current_env",
    "data_class",
    "set_data_class",
    "current_data_class",
    "region",
    "set_region",
    "current_region",
    "PolicyViolation",
    "ModelNotPermitted",
    "EndpointNotPermitted",
    "resolve_compliance",
    "STARTER_POLICIES",
    "starter_policy",
    "list_starter_policies",
    "policy_report",
    "HaltEvent",
    "on_halt",
    "remove_halt_handler",
    "clear_halt_handlers",
    "halt_handler_count",
    "simulate_rules",
    "plan_report",
    "coverage_report",
    "render_plan",
    "agent_report",
    "render_agents",
    "compare_reports",
    "render_comparison",
    "suggest_thresholds",
    "render_suggestions",
    "render_agents_html",
    "write_agents_html",
    "wrap_langchain_llm",
    "wrap_llamaindex_llm",
    "set_task",
    "get_task",
    "current_task_id",
    "current_agent",
    "task_snapshot",
    "TaskLimitExceeded",
    "TaskEnvelopeExceeded",
    "TaskLoopDetected",
    "TaskStalled",
    "TaskCallLimitExceeded",
    "set_queue_wait_ms",
    # Budget-Enforced Keys (T3.3)
    "register_key",
    "unregister_key",
    "clear_keys",
    "key",
    "key_status",
    "leak_scan",
    "get_current_key",
    "KeyBudgetExceeded",
    "context_passport",
    "savings_ledger_health",
    # Pricing registry (v0.13) — anti-fabrication layer
    "register_pricing",
    "unregister_pricing",
    "clear_registered_pricing",
    "registered_pricing",
    "pricing_info",
    "estimate_cost_with_source",
    "derive_selfhost_rate",
    "register_selfhost_pricing",
    "register_cluster_costs",
    "derive_cluster_gpu_hour_rate",
    "capacity_recovery_report",
    "chargeback_report",
    "chargeback_csv",
    "hybrid_placement_report",
    "hybrid_placement_csv",
    "close_packet",
    "general_ledger_rows",
    "general_ledger_csv",
    "capacity_reclaimed",
    "set_home",
    "set_savings_path",
    "set_buffered_savings",
    "set_in_memory_savings",
    "flush_savings",
    "doctor",
    "reset_savings",
    "clear_cache",
    # Configuration
    "set_default_store",
    "set_default_semantic_cache",
    "set_default_redactor",
    # Storage
    "MemoryStore",
    "SQLiteStore",
    "SemanticCache",
    "semantic_available",
    "is_vec_index_available",
    # Trust + ops layer
    "events",
    "privacy",
    "admin",
    "metrics",
    "warmup",
    "compression",
    "memory",
    "audit",
    "integrity",
    "verify_self",
    "scan_environment",
    "self_check",
    "backends",
    "RedisStore",
    "FernetCipher",
    "NoOpCipher",
    "DefaultRedactor",
    "default_redactor",
    "redact",
    "PrometheusCollector",
    # Compression (v0.6)
    "Compressor",
    "CompressionResult",
    "StructuralCompressor",
    "LLMLinguaCompressor",
    "SalienceCompressor",
    "SafeCompressor",
    "compression_stats",
    "reset_compression_stats",
    "Router",
    "RouteDecision",
    "Cascade",
    "CascadeResult",
    "CacheOptimizer",
    "CacheReport",
    "VolatileFinding",
    "compress_tools",
    "compose",
    "safe_compress",
    "compression_verification_log",
    # Memory (v0.7)
    "with_memory",
    "ConversationMemory",
    "Turn",
    "Summarizer",
    "TruncationSummarizer",
    "CallableSummarizer",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "memory_fidelity_log",
    # Envelope (advanced)
    "wrap",
    "unwrap",
    "prompt_ir",
    "SecurityPolicy",
    "SecurityPolicyError",
    "set_security_policy",
    "get_security_policy",
    "reset_security_policy",
    "DecisionRecord",
    "on_decision",
    "remove_decision_subscriber",
    "clear_decision_subscribers",
]

# ---------------------------------------------------------------------------
# Tokeymeter public product surface (the curated facade). Re-exported here so
# `from tokeymeter import meter, report, ...` works. Implemented in _api.py to
# keep this module focused on the engine's symbol table. The facade's
# compress_tools/no_optimize intentionally rebind the engine names above to the
# product-facing wrappers.
# ---------------------------------------------------------------------------
from ._api import (  # noqa: E402
    meter,
    report,
    reset,
    doctor,
    compressor,
    router,
    metered_route,
    cascade,
    optimize_cache,
    check_cache,
    compress_tools,
    no_optimize,
)

__all__ += [
    "meter",
    "report",
    "reset",
    "compressor",
    "router",
    "metered_route",
    "cascade",
    "optimize_cache",
    "check_cache",
]

# K4 (W3): the one-line Runtime facade — `Runtime(client=...).execute(prompt)`.
from .runtime.facade import Runtime, RuntimeConfigurationError  # noqa: E402
