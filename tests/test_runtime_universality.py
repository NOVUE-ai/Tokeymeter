"""W9 battery — universality: catalog, conformance, frameworks, tools, packs.

THE UNIVERSALITY GATE: test_every_openai_dialect_adapter_is_conformant — every
OpenAI-dialect provider in the catalog produces an adapter that passes the full
conformance kit. That is what "any model" means as an engineering fact: not a
list we maintain, but a contract every provider satisfies.
"""
from __future__ import annotations

import json

import pytest

from tokeymeter import Runtime
from tokeymeter.runtime import (
    CATALOG, ConformanceClient, ConformanceReport, DoctorReport, PolicyPack,
    adapter_for, async_adapter_for, check_adapter, doctor, get_pack, get_spec,
    governed_tool, list_packs, list_providers, wrap_callable,
    wrap_langchain_llm, wrap_llamaindex_llm,
)
from tokeymeter.runtime.adapters import AnthropicAdapter, OpenAIAdapter
from tokeymeter.runtime.catalog import ProviderSpec


# ============================================= CATALOG ==================
def test_catalog_is_populated_and_well_formed():
    providers = list_providers()
    assert len(providers) >= 15                        # the long tail, covered
    for name in providers:
        spec = get_spec(name)
        assert spec.dialect in ("openai", "anthropic", "callable")
        assert spec.name == name
        # OpenAI-dialect providers with a FIXED public endpoint declare a
        # base_url; openai/azure-openai/bedrock use a deployment-supplied
        # endpoint (SDK default, Azure deployment, or Bedrock gateway) that
        # the customer's client carries — NOVUE never hardcodes those.
        _deployment_supplied = ("openai", "azure-openai", "bedrock")
        if spec.dialect == "openai" and name not in _deployment_supplied:
            assert spec.base_url, f"{name} needs a base_url"


def test_unknown_provider_raises_helpful_error():
    with pytest.raises(KeyError) as ei:
        get_spec("nonexistent-provider")
    assert "known:" in str(ei.value)                   # lists alternatives


def test_self_hosted_providers_need_no_key():
    for name in ("vllm", "ollama", "lmstudio"):
        assert get_spec(name).env_key == ""            # local, no credentials


def test_adapter_for_builds_correct_dialect():
    oai = adapter_for("groq", ConformanceClient(), semantic=False)
    assert isinstance(oai, OpenAIAdapter)

    class FakeAnthropic:
        messages = type("M", (), {"create": staticmethod(lambda **k: None)})()
    ant = adapter_for("anthropic", FakeAnthropic(), semantic=False)
    assert isinstance(ant, AnthropicAdapter)


def test_async_adapter_for_openai_dialect_only():
    a = async_adapter_for("together", ConformanceClient(), semantic=False)
    assert a.provider == "openai"
    with pytest.raises(ValueError):
        async_adapter_for("anthropic", object())       # not OpenAI dialect


# ============================================= THE UNIVERSALITY GATE =====
def test_every_openai_dialect_adapter_is_conformant():
    """Every OpenAI-dialect provider in the catalog yields an adapter that
    passes the FULL conformance kit. This is 'any model' as a proof, not a
    promise: one contract, satisfied by every provider, verified offline."""
    dialect_providers = [n for n in list_providers()
                         if get_spec(n).dialect == "openai"]
    assert len(dialect_providers) >= 12                # the bulk of the catalog
    for name in dialect_providers:
        report = check_adapter(
            lambda n=name: adapter_for(n, ConformanceClient(),
                                       semantic=False),
            model="conf-model", provider_name=name)
        assert report.passed, (
            f"{name} FAILED conformance: "
            f"{[f.check for f in report.failures()]}")


def test_conformance_report_shape():
    report = check_adapter(
        lambda: OpenAIAdapter(ConformanceClient(), semantic=False),
        provider_name="openai")
    assert isinstance(report, ConformanceReport)
    assert report.passed and len(report.results) == 7
    assert "7/7" in report.summary


def test_conformance_catches_a_broken_adapter():
    """A non-conformant adapter must FAIL the kit — the gate has teeth."""
    from tokeymeter.runtime.providers import ProviderAdapter

    class BrokenAdapter(ProviderAdapter):
        provider = "broken"

        def get_info(self):
            raise RuntimeError("broken get_info")

        def infer(self, ctx):
            raise RuntimeError("broken infer")

    report = check_adapter(lambda: BrokenAdapter(), provider_name="broken")
    assert not report.passed
    assert len(report.failures()) >= 2                 # info + infer fail


# ============================================= FRAMEWORKS ================
def test_wrap_callable_is_the_universal_guarantee():
    """ANY callable is governed — the framework we've never heard of works."""
    g = wrap_callable(lambda p: "framework-result:" + p, receipt="never")
    assert g.execute("hello") == "framework-result:hello"
    assert g.last.trace                                # governed pipeline ran


def test_wrap_langchain_llm_adapts_invoke():
    class FakeLCMessage:
        def __init__(self, content):
            self.content = content

    class FakeChatModel:
        def invoke(self, prompt):
            return FakeLCMessage("lc:" + prompt)

    g = wrap_langchain_llm(FakeChatModel(), receipt="never")
    assert g.execute("hi") == "lc:hi"                  # .content unwrapped


def test_wrap_langchain_predict_fallback():
    class OldLLM:
        def predict(self, prompt):
            return "predicted:" + prompt
    g = wrap_langchain_llm(OldLLM(), receipt="never")
    assert g.execute("x") == "predicted:x"


def test_wrap_langchain_rejects_non_model():
    with pytest.raises(TypeError):
        wrap_langchain_llm(object())


def test_wrap_llamaindex_llm_adapts_complete():
    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeLLM:
        def complete(self, prompt):
            return FakeResponse("li:" + prompt)

    g = wrap_llamaindex_llm(FakeLLM(), receipt="never")
    assert g.execute("q") == "li:q"


def test_governed_tool_screens_model_facing_text():
    """An agent tool's primary string arg is governed before the tool runs."""
    received = {}

    def raw_tool(text, extra=None):
        received["text"] = text
        received["extra"] = extra
        return "tool-done"

    wrapped = governed_tool(raw_tool, tool_name="lookup")
    result = wrapped("search this", extra="meta")
    assert result == "tool-done"
    assert received["text"] == "search this"           # governed passthrough
    assert received["extra"] == "meta"                 # other args intact


def test_governed_tool_passes_non_string_first_arg():
    def numeric_tool(n):
        return n * 2
    wrapped = governed_tool(numeric_tool, tool_name="calc")
    assert wrapped(21) == 42                            # non-string: passthrough


# ============================================= DOCTOR ===================
def test_doctor_reports_ready_environment():
    rep = doctor()
    assert isinstance(rep, DoctorReport)
    names = [c.name for c in rep.checks]
    assert "python" in names and "cryptography" in names
    assert "✓" in rep.render() or "→" in rep.render()


def test_doctor_provider_key_presence(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rep = doctor(provider="groq")
    groq_check = next(c for c in rep.checks if c.name == "provider:groq")
    assert groq_check.status == "warn"                 # key not set
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    rep2 = doctor(provider="groq")
    groq2 = next(c for c in rep2.checks if c.name == "provider:groq")
    assert groq2.status == "ok"                        # key present
    # content-blind: the value is NEVER in the report
    assert "sk-test" not in rep2.render()


def test_doctor_self_hosted_needs_no_key():
    rep = doctor(provider="ollama")
    check = next(c for c in rep.checks if c.name == "provider:ollama")
    assert check.status == "ok" and "no key" in check.detail


def test_doctor_unknown_provider_flagged():
    rep = doctor(provider="does-not-exist")
    check = next(c for c in rep.checks if "does-not-exist" in c.name)
    assert check.status == "missing"


def test_doctor_warns_proof_without_signer():
    rep = doctor(config={"trust": {"proof": {"enabled": True}}})
    signer_check = next((c for c in rep.checks if c.name == "proof-signer"),
                        None)
    assert signer_check is not None and signer_check.status == "warn"


# ============================================= POLICY PACKS =============
def test_policy_packs_populated():
    packs = list_packs()
    assert len(packs) >= 4
    for name in packs:
        pack = get_pack(name)
        assert isinstance(pack, PolicyPack) and pack.config


def test_pack_apply_merges_over_base():
    merged = get_pack("cost-guard").apply_to({"model": "x",
                                              "optimization": {"tier": "salience"}})
    assert merged["model"] == "x"                      # base preserved
    assert merged["optimization"]["enabled"] is True   # pack applied
    assert merged["optimization"]["tier"] == "salience"  # deep-merged


def test_strict_logging_pack_enables_security_and_proof():
    cfg = get_pack("strict-logging").config
    assert cfg["governance"]["security"]["secrets_mode"] == "block"
    assert cfg["trust"]["proof"]["enabled"] is True


def test_pack_applied_to_runtime_actually_governs():
    """A pack is config with a real effect: applying strict-logging to a
    Runtime blocks secrets end to end."""
    from tokeymeter.runtime import SecretBlocked
    cfg = get_pack("strict-logging").apply_to()
    r = Runtime(config=cfg, call=lambda p: "ok", receipt="never")
    key = "sk-ant-api03-" + "".join(
        __import__("secrets").choice("abcdefghijklmnopqrstuvwxyz0123456789")
        for _ in range(88))
    with pytest.raises(SecretBlocked):
        r.execute(f"leak {key}")


def test_unknown_pack_raises_helpful():
    with pytest.raises(KeyError) as ei:
        get_pack("nonexistent")
    assert "known:" in str(ei.value)


# ============================================= FULL W9 INTEGRATION ======
def test_w9_catalog_adapter_runs_full_pipeline_via_facade():
    """A catalog-built adapter, dropped into a Runtime, runs the whole
    governed pipeline — universality meets the full stack."""
    from tokeymeter.runtime import Kernel, KernelRequest, RuntimeConfig
    from tokeymeter.runtime.providers import ExecutionEngine
    adapter = adapter_for("deepseek", ConformanceClient(), semantic=False)
    k = Kernel(RuntimeConfig({"cache": {"enabled": False}})).start()
    ex = ExecutionEngine()
    ex.register_adapter(adapter, models=["conf-model"], default=True)
    k.register_engine(ex)
    resp = k.process(KernelRequest(payload="hello deepseek", model="conf-model"))
    assert resp.payload is not None


# ============================================= RELEASE HARDENING =========
def test_release_manifest_is_deterministic_and_content_blind():
    """The runtime manifest is reproducible (same hash across runs) and
    content-blind (paths/hashes/sizes only, never file contents)."""
    import subprocess
    import sys as _sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    r1 = subprocess.run([_sys.executable, "scripts/release.py", "manifest"],
                        capture_output=True, text=True, cwd=str(root))
    r2 = subprocess.run([_sys.executable, "scripts/release.py", "manifest"],
                        capture_output=True, text=True, cwd=str(root))
    m1, m2 = json.loads(r1.stdout), json.loads(r2.stdout)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]   # deterministic
    assert m1["file_count"] >= 20
    # content-blind: no file body, only metadata fields
    for entry in m1["files"]:
        assert set(entry.keys()) == {"path", "sha256", "size"}


def test_release_deps_core_is_minimal():
    """The in-process moat requires a light core — assert the declared core
    dependency set stays small."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "_release", Path(__file__).parent.parent / "scripts" / "release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CORE_DEPS == {"cryptography"}            # stdlib + one lib
