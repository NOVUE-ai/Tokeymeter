"""Runtime Kernel battery (K-track Sprint 1+2).

Pins the doc's acceptance criteria and the NOVUE binding constraints:
- Sprint 1: kernel primitives work; a simple request flows through the
  kernel with NO LLM calls.
- Sprint 2: adapter contract compiles for multiple providers; health checks
  succeed; routing + streaming + fallback behave.
- Binding: kernel telemetry is content-blind; a failing hook can never crash
  the core; Knowledge stays a no-op slot.
"""
from __future__ import annotations

import threading
import time

import pytest

from tokeymeter.runtime import (
    CacheEngine,
    CallableAdapter,
    Container,
    Engine,
    EngineRegistry,
    ExecutionEngine,
    GovernanceEngine,
    HookBus,
    InMemoryCache,
    Kernel,
    KernelRequest,
    KernelStopped,
    KnowledgeEngine,
    ReliabilityEngine,
    RuntimeConfig,
    TrustEngine,
)
from tokeymeter.runtime.container import ServiceNotRegistered
from tokeymeter.runtime.engine import DuplicateEngine, NoExecutionEngine
from tokeymeter.runtime.providers import (
    AnthropicStubAdapter,
    OpenAIStubAdapter,
    UnknownModel,
)


SECRET_PAYLOAD = "TOP-SECRET quarterly numbers do not leak"


def make_kernel(**config):
    k = Kernel(RuntimeConfig(config))
    return k


def echo_adapter(tag="echo"):
    return CallableAdapter(lambda p: f"{tag}:{p}", provider=tag)


# ---------------------------------------------------------------- config ---
def test_config_defaults_and_override():
    cfg = RuntimeConfig({"cache": {"max_entries": 7}}, env={})
    assert cfg.get("cache.max_entries") == 7
    assert cfg.get("cache.enabled") is True            # default preserved
    assert cfg.get("kernel.drain_timeout_s") == 30.0   # untouched section


def test_config_env_layer_wins_and_coerces():
    env = {"TOKEYMETER_RELIABILITY__MAX_RETRIES": "3",
           "TOKEYMETER_CACHE__ENABLED": "false",
           "IGNORED_VAR": "1"}
    cfg = RuntimeConfig({"reliability": {"max_retries": 1}}, env=env)
    assert cfg.get("reliability.max_retries") == 3     # int, not "3"
    assert cfg.get("cache.enabled") is False           # bool, not "false"
    assert cfg.get("ignored_var") is None


def test_config_missing_path_returns_default():
    cfg = RuntimeConfig(env={})
    assert cfg.get("no.such.path", "fallback") == "fallback"


# ------------------------------------------------------------- container ---
def test_container_instance_and_lazy_factory_singleton():
    c = Container()
    c.register_instance("a", 42)
    calls = []
    c.register_factory("b", lambda _c: calls.append(1) or object())
    assert c.resolve("a") == 42
    b1, b2 = c.resolve("b"), c.resolve("b")
    assert b1 is b2 and len(calls) == 1                # factory ran once


def test_container_missing_raises_and_override_restores():
    c = Container()
    with pytest.raises(ServiceNotRegistered):
        c.resolve("ghost")
    c.register_instance("svc", "real")
    with c.override("svc", "fake"):
        assert c.resolve("svc") == "fake"
    assert c.resolve("svc") == "real"


# ------------------------------------------------------------------- bus ---
def test_bus_priority_order_is_deterministic():
    bus, order = HookBus(), []
    bus.subscribe("h", lambda ctx: order.append("late"), priority=200)
    bus.subscribe("h", lambda ctx: order.append("early"), priority=10)
    bus.subscribe("h", lambda ctx: order.append("mid_a"), priority=100)
    bus.subscribe("h", lambda ctx: order.append("mid_b"), priority=100)
    bus.emit("h", {})
    assert order == ["early", "mid_a", "mid_b", "late"]


def test_bus_safe_failover_bad_subscriber_never_crashes_core():
    bus, seen = HookBus(), []

    def bomb(ctx):
        raise RuntimeError("plugin gone wrong")

    bus.subscribe("h", bomb, priority=1)
    bus.subscribe("h", lambda ctx: seen.append("survivor"), priority=2)
    bus.emit("h", {})                                   # must not raise
    assert seen == ["survivor"]
    assert bus.error_count == 1


def test_bus_unsubscribe():
    bus, seen = HookBus(), []
    fn = bus.subscribe("h", lambda ctx: seen.append(1))
    bus.unsubscribe("h", fn)
    bus.emit("h", {})
    assert seen == [] and bus.subscriber_count("h") == 0


# -------------------------------------------------------- engine registry ---
def test_registry_rejects_duplicates_and_requires_execution_engine():
    reg = EngineRegistry()
    reg.register(GovernanceEngine())
    with pytest.raises(DuplicateEngine):
        reg.register(GovernanceEngine())
    with pytest.raises(NoExecutionEngine):
        reg.execution_engine()


# ------------------------------------------------- sprint-1 pipeline flow ---
def test_simple_request_flows_through_kernel_no_llm():
    """Doc Sprint-1 acceptance: simple request flows through kernel (no LLM)."""
    k = make_kernel().start()
    execution = ExecutionEngine()
    execution.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(GovernanceEngine())
    k.register_engine(execution)
    resp = k.process(KernelRequest(payload="hello kernel"))
    assert resp.payload == "echo:hello kernel"
    assert resp.request_id and resp.model == "default"
    phases = [(t["engine"], t["phase"]) for t in resp.trace]
    assert phases[0] == ("governance", "before_request")
    assert ("execution", "execute") in phases
    # after_response unwinds in REVERSE registration order
    afters = [e for e, p in phases if p == "after_response"]
    assert afters == ["execution", "governance"]


def test_kernel_hooks_fire_around_pipeline():
    k = make_kernel().start()
    execution = ExecutionEngine()
    execution.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(execution)
    events = []
    k.use("before_request", lambda ctx: events.append("before"))
    k.use("after_response", lambda ctx: events.append("after"))
    k.process(KernelRequest(payload="x"))
    assert events == ["before", "after"]


def test_on_error_hooks_fire_and_exception_propagates():
    k = make_kernel().start()

    class BoomEngine(Engine):
        name = "boom"
        handles_execution = True

        def execute(self, ctx):
            raise ValueError("provider exploded")

    k.register_engine(BoomEngine())
    errors = []
    k.use("on_error", lambda ctx: errors.append(ctx["error_type"]))
    with pytest.raises(ValueError):
        k.process(KernelRequest(payload="x"))
    assert errors == ["ValueError"]                    # hooks ran, then raise


def test_shutdown_drains_inflight_and_refuses_new_requests():
    k = make_kernel(kernel={"drain_timeout_s": 5.0}).start()
    release = threading.Event()
    started = threading.Event()

    def slow(payload):
        started.set()
        release.wait(timeout=5)
        return "done"

    execution = ExecutionEngine()
    execution.register_adapter(
        CallableAdapter(slow), models=["default"], default=True)
    k.register_engine(execution)

    result = {}
    t = threading.Thread(
        target=lambda: result.update(r=k.process(KernelRequest(payload="p"))))
    t.start()
    assert started.wait(timeout=5)
    stopper = threading.Thread(target=k.shutdown)
    stopper.start()
    time.sleep(0.1)                                    # shutdown is now draining
    with pytest.raises(KernelStopped):
        k.process(KernelRequest(payload="new"))        # new work refused
    release.set()
    t.join(timeout=5); stopper.join(timeout=5)
    assert result["r"].payload == "done"               # in-flight completed


def test_kernel_trace_is_content_blind():
    """BINDING: payload text appears nowhere in trace/metadata; only a
    sha256 fingerprint may reference it."""
    k = make_kernel().start()
    execution = ExecutionEngine()
    execution.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(GovernanceEngine())
    k.register_engine(TrustEngine())
    k.register_engine(execution)
    resp = k.process(KernelRequest(payload=SECRET_PAYLOAD))
    import json
    trace_blob = json.dumps(resp.trace) + json.dumps(resp.metadata)
    assert SECRET_PAYLOAD not in trace_blob
    assert "TOP-SECRET" not in trace_blob


# -------------------------------------------------- sprint-2: adapters -----
def test_multiple_adapters_compile_and_health_check_succeeds():
    """Doc Sprint-2 acceptance verbatim."""
    openai = OpenAIStubAdapter(lambda p: "o:" + p)
    anthropic = AnthropicStubAdapter(lambda p: "a:" + p)
    for adapter in (openai, anthropic):
        info = adapter.get_info()
        assert info.provider in ("openai", "anthropic") and info.models
        hs = adapter.health_check()
        assert hs.healthy and hs.latency_ms >= 0


def test_execution_routes_by_model_and_unknown_model_raises():
    ex = ExecutionEngine()
    ex.register_adapter(OpenAIStubAdapter(lambda p: "o:" + p), models=["gpt-x"])
    ex.register_adapter(AnthropicStubAdapter(lambda p: "a:" + p), models=["claude-x"])
    k = make_kernel().start()
    k.register_engine(ex)
    assert k.process(KernelRequest(payload="hi", model="gpt-x")).payload == "o:hi"
    assert k.process(KernelRequest(payload="hi", model="claude-x")).payload == "a:hi"
    with pytest.raises(UnknownModel):
        k.process(KernelRequest(payload="hi", model="nonexistent"))


def test_stream_infer_default_yields_chunks():
    adapter = CallableAdapter(lambda p: "chunky", provider="s")
    ctx = {"request": KernelRequest(payload="p")}
    assert list(adapter.stream_infer(ctx)) == ["chunky"]


# ---------------------------------------------------------- governance -----
def test_governance_stamps_principal_from_shipped_identity():
    import tokeymeter
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(GovernanceEngine())
    k.register_engine(ex)
    with tokeymeter.principal("dev-1"):
        resp = k.process(KernelRequest(payload="p"))
    assert resp.metadata["principal"] == "dev-1"
    resp2 = k.process(KernelRequest(payload="p"))
    assert "principal" not in resp2.metadata           # no ambient bleed


def test_governance_hook_can_veto_content_blind():
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(GovernanceEngine())
    k.register_engine(ex)

    def veto(ctx):
        if ctx["meta"].get("model", ctx["request"].model) == "default":
            raise PermissionError("model not allowed for this principal")

    k.use("governance_check", veto)
    with pytest.raises(PermissionError):
        k.process(KernelRequest(payload="p"))


# ---------------------------------------------------------------- cache ----
def test_cache_hit_short_circuits_second_identical_request():
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    cache = CacheEngine()
    k.register_engine(cache)
    k.register_engine(ex)
    r1 = k.process(KernelRequest(payload="same"))
    r2 = k.process(KernelRequest(payload="same"))
    r3 = k.process(KernelRequest(payload="different"))
    assert r1.payload == r2.payload == "echo:same"
    assert r2.metadata["cache"] == "hit"
    assert r3.metadata["cache"] == "miss"
    assert ex.calls == 2                               # hit never reached adapter
    assert (cache.hits, cache.misses) == (1, 2)


def test_cache_disabled_by_config():
    k = make_kernel(cache={"enabled": False}).start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(CacheEngine())
    k.register_engine(ex)
    k.process(KernelRequest(payload="same"))
    k.process(KernelRequest(payload="same"))
    assert ex.calls == 2                               # no caching happened


def test_inmemory_cache_lru_eviction_bounded():
    c = InMemoryCache(max_entries=2)
    c.set("a", 1); c.set("b", 2); c.get("a"); c.set("c", 3)
    assert c.get("a") == 1 and c.get("c") == 3
    assert c.get("b") is None                          # LRU evicted


# ---------------------------------------------------------------- trust ----
def test_trust_chain_links_and_verifies_and_detects_tamper():
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    trust = TrustEngine()
    k.register_engine(trust)
    k.register_engine(ex)
    for i in range(3):
        k.process(KernelRequest(payload=f"req {i}"))
    entries = trust.entries()
    assert len(entries) == 3
    assert entries[1]["prev"] == entries[0]["hash"]
    ok, bad = trust.verify()
    assert ok and bad == -1
    trust._entries[1]["body"] += "|tampered"           # forge in place
    ok, bad = trust.verify()
    assert not ok and bad == 1                         # detected at index 1


def test_trust_entries_carry_fingerprint_not_content():
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    trust = TrustEngine()
    k.register_engine(trust)
    k.register_engine(ex)
    k.process(KernelRequest(payload=SECRET_PAYLOAD))
    blob = str(trust.entries())
    assert SECRET_PAYLOAD not in blob and "TOP-SECRET" not in blob
    import hashlib
    fp = hashlib.sha256(SECRET_PAYLOAD.encode()).hexdigest()
    assert fp in blob                                  # fingerprint present


# ----------------------------------------------------------- reliability ---
def test_reliability_retries_transient_failure():
    k = make_kernel(reliability={"max_retries": 2}).start()
    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    ex = ExecutionEngine()
    ex.register_adapter(CallableAdapter(flaky), models=["default"], default=True)
    rel = ReliabilityEngine(ex)
    k.register_engine(rel)
    k.register_engine(ex)
    resp = k.process(KernelRequest(payload="p"))
    assert resp.payload == "recovered"
    assert rel.retries_used == 2 and attempts["n"] == 3


def test_reliability_falls_back_to_alternate_model():
    k = make_kernel(reliability={"max_retries": 0,
                                 "fallback_order": ["backup"]}).start()
    ex = ExecutionEngine()

    def dead(payload):
        raise ConnectionError("provider down")

    ex.register_adapter(CallableAdapter(dead, provider="primary"),
                        models=["default"], default=True)
    ex.register_adapter(CallableAdapter(lambda p: "saved by backup",
                                        provider="backup"),
                        models=["backup"])
    rel = ReliabilityEngine(ex)
    k.register_engine(rel)
    k.register_engine(ex)
    resp = k.process(KernelRequest(payload="p"))
    assert resp.payload == "saved by backup"
    assert rel.fallbacks_used == 1
    assert resp.metadata["fallback"] == "backup"
    assert resp.metadata["provider"] == "backup"


def test_reliability_exhaustion_raises_last_error():
    k = make_kernel(reliability={"max_retries": 1}).start()
    ex = ExecutionEngine()

    def dead(payload):
        raise TimeoutError("always down")

    ex.register_adapter(CallableAdapter(dead), models=["default"], default=True)
    k.register_engine(ReliabilityEngine(ex))
    k.register_engine(ex)
    with pytest.raises(TimeoutError):
        k.process(KernelRequest(payload="p"))


# ------------------------------------------------------------- knowledge ---
def test_knowledge_slot_is_noop_by_default_stage_gated():
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    k.register_engine(KnowledgeEngine())
    k.register_engine(ex)
    resp = k.process(KernelRequest(payload="p"))
    assert resp.metadata["knowledge"] == "noop"        # interface reserved


# -------------------------------------------- full-stack integration run ---
def test_full_pipeline_all_engines_together():
    """Governance → Knowledge(noop) → Cache → Reliability/Execution → Trust,
    with hits, principal stamping, chain verification — one coherent run."""
    import tokeymeter
    k = make_kernel().start()
    ex = ExecutionEngine()
    ex.register_adapter(echo_adapter(), models=["default"], default=True)
    trust = TrustEngine()
    cache = CacheEngine()
    k.register_engine(GovernanceEngine())
    k.register_engine(KnowledgeEngine())
    k.register_engine(cache)
    k.register_engine(ReliabilityEngine(ex))
    k.register_engine(trust)
    k.register_engine(ex)
    with tokeymeter.principal("team-a"):
        first = k.process(KernelRequest(payload="quarterly summary"))
        second = k.process(KernelRequest(payload="quarterly summary"))
    assert first.payload == second.payload
    assert second.metadata["cache"] == "hit" and ex.calls == 1
    assert first.metadata["principal"] == "team-a"
    ok, _ = trust.verify()
    assert ok and len(trust.entries()) == 2
    k.shutdown()


# ------------------------------------------------------- regression pins ---
def test_config_instances_are_isolated_no_defaults_pollution():
    """REGRESSION: env-layer mutation must never leak into module defaults
    or sibling configs (shallow-copy defect caught by this battery)."""
    poisoned = RuntimeConfig(env={"TOKEYMETER_CACHE__ENABLED": "false",
                                  "TOKEYMETER_KERNEL__DRAIN_TIMEOUT_S": "1"})
    assert poisoned.get("cache.enabled") is False
    clean = RuntimeConfig(env={})
    assert clean.get("cache.enabled") is True
    assert clean.get("kernel.drain_timeout_s") == 30.0


def test_enforcement_vs_observability_emit_semantics():
    """REGRESSION: emit() swallows (safe-failover); emit_strict() propagates
    (fail closed). Enforcement through the failover path fails OPEN."""
    bus = HookBus()

    def veto(ctx):
        raise PermissionError("blocked")

    bus.subscribe("check", veto)
    bus.emit("check", {})                              # swallowed, core survives
    assert bus.error_count == 1
    with pytest.raises(PermissionError):
        bus.emit_strict("check", {})                   # propagates
