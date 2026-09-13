"""The model catalog (W9) — TRUE universality through one dialect.

The insight that makes "any model" real rather than aspirational: the large
majority of model providers speak the OpenAI chat-completions dialect
(`client.chat.completions.create(model=..., messages=...)`). One
OpenAI-compatible adapter, pointed at a provider's base URL, therefore
reaches the entire long tail — Gemini (OpenAI-compat endpoint), Together,
Groq, Mistral, DeepSeek, Fireworks, Perplexity, vLLM, Ollama, LM Studio,
Anyscale, OpenRouter, and any future provider that ships an OpenAI-compatible
surface. Providers with a native dialect (Anthropic messages, Bedrock's
runtime) get a thin dedicated adapter; a model that does not exist yet is
supported the day it launches by pointing this adapter at its endpoint.

Every adapter here satisfies the SHIPPED ProviderAdapter contract and passes
the conformance kit (conformance.py). Nothing is reimplemented — the OpenAI
adapters wrap the shipped `OpenAIAdapter`, which wraps the shipped wrapper,
which carries the whole optimization/audit pipeline.

Design law: this module adds REACH, not behavior. It is a registry of
factories that configure the shipped adapters for named providers. No new
inference path, no new crypto, no new telemetry — reach only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapters import AnthropicAdapter, AsyncOpenAIAdapter, OpenAIAdapter
from .providers import CallableAdapter, ProviderAdapter


@dataclass(frozen=True)
class ProviderSpec:
    """A named provider: how to reach it and what dialect it speaks."""
    name: str
    dialect: str                     # "openai" | "anthropic" | "callable"
    base_url: Optional[str] = None   # for OpenAI-compatible providers
    default_models: tuple = ()
    env_key: str = ""                # the env var holding its API key
    notes: str = ""


# The catalog. OpenAI-dialect providers need only a base URL — proof that the
# long tail is one adapter, not one-per-vendor.
CATALOG: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai", "openai", None,
        ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o4-mini"),
        "OPENAI_API_KEY"),
    "anthropic": ProviderSpec(
        "anthropic", "anthropic", None,
        ("claude-opus-4", "claude-sonnet-4", "claude-3-5-haiku"),
        "ANTHROPIC_API_KEY"),
    "azure-openai": ProviderSpec(
        "azure-openai", "openai", None,
        ("gpt-4o", "gpt-4o-mini"), "AZURE_OPENAI_API_KEY",
        notes="base_url is the Azure deployment endpoint"),
    "gemini": ProviderSpec(
        "gemini", "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("gemini-2.0-flash", "gemini-1.5-pro"), "GEMINI_API_KEY",
        notes="Google's OpenAI-compatible endpoint"),
    "bedrock": ProviderSpec(
        "bedrock", "openai", None,
        ("anthropic.claude-3-5-sonnet", "meta.llama3-1-70b"),
        "AWS_BEDROCK_API_KEY",
        notes="via an OpenAI-compatible Bedrock gateway"),
    "together": ProviderSpec(
        "together", "openai", "https://api.together.xyz/v1",
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo",
         "Qwen/Qwen2.5-72B-Instruct-Turbo"), "TOGETHER_API_KEY"),
    "groq": ProviderSpec(
        "groq", "openai", "https://api.groq.com/openai/v1",
        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768"), "GROQ_API_KEY"),
    "mistral": ProviderSpec(
        "mistral", "openai", "https://api.mistral.ai/v1",
        ("mistral-large-latest", "mistral-small-latest"), "MISTRAL_API_KEY"),
    "deepseek": ProviderSpec(
        "deepseek", "openai", "https://api.deepseek.com/v1",
        ("deepseek-chat", "deepseek-reasoner"), "DEEPSEEK_API_KEY"),
    "fireworks": ProviderSpec(
        "fireworks", "openai",
        "https://api.fireworks.ai/inference/v1",
        ("accounts/fireworks/models/llama-v3p1-70b-instruct",),
        "FIREWORKS_API_KEY"),
    "perplexity": ProviderSpec(
        "perplexity", "openai", "https://api.perplexity.ai",
        ("sonar", "sonar-pro"), "PERPLEXITY_API_KEY"),
    "openrouter": ProviderSpec(
        "openrouter", "openai", "https://openrouter.ai/api/v1",
        ("openai/gpt-4o", "anthropic/claude-sonnet-4"), "OPENROUTER_API_KEY"),
    "xai": ProviderSpec(
        "xai", "openai", "https://api.x.ai/v1",
        ("grok-2", "grok-2-mini"), "XAI_API_KEY"),
    # self-hosted — no API key, local endpoints
    "vllm": ProviderSpec(
        "vllm", "openai", "http://localhost:8000/v1",
        ("meta-llama/Llama-3.1-8B-Instruct",), "",
        notes="self-hosted vLLM OpenAI-compatible server"),
    "ollama": ProviderSpec(
        "ollama", "openai", "http://localhost:11434/v1",
        ("llama3.2", "qwen2.5", "deepseek-r1"), "",
        notes="self-hosted Ollama OpenAI-compatible server"),
    "lmstudio": ProviderSpec(
        "lmstudio", "openai", "http://localhost:1234/v1",
        ("local-model",), "", notes="self-hosted LM Studio server"),
}


def list_providers() -> List[str]:
    return sorted(CATALOG)


def get_spec(name: str) -> ProviderSpec:
    spec = CATALOG.get(name)
    if spec is None:
        raise KeyError(
            f"unknown provider {name!r}; known: {', '.join(list_providers())}")
    return spec


def adapter_for(name: str, client: Any, *, models: Optional[List[str]] = None,
                **wrap_opts: Any) -> ProviderAdapter:
    """Build the correct SHIPPED adapter for a named provider around an
    already-constructed client. The client is the provider's SDK object (or
    any OpenAI-compatible client pointed at the provider's base_url) — NOVUE
    never holds credentials; the caller constructs the authenticated client.

    This is the single entry point that makes the catalog universal: pass a
    provider name and its client, get a conformant adapter wrapping the full
    pipeline.
    """
    spec = get_spec(name)
    picked = models or list(spec.default_models) or None
    if spec.dialect == "anthropic":
        return AnthropicAdapter(client, models=picked, **wrap_opts)
    if spec.dialect == "openai":
        # async clients are auto-routed by the facade's detector; here we
        # return the sync adapter — callers wanting async use the async
        # adapter explicitly (parity is proven in the W2 battery).
        return OpenAIAdapter(client, models=picked, **wrap_opts)
    if spec.dialect == "callable":
        return CallableAdapter(client, provider=name, models=picked)
    raise ValueError(f"unknown dialect {spec.dialect!r} for {name}")


def async_adapter_for(name: str, client: Any, *,
                      models: Optional[List[str]] = None,
                      **wrap_opts: Any) -> ProviderAdapter:
    """Async twin for OpenAI-dialect providers."""
    spec = get_spec(name)
    if spec.dialect != "openai":
        raise ValueError(
            f"async catalog adapter is OpenAI-dialect only; {name} is "
            f"{spec.dialect}")
    picked = models or list(spec.default_models) or None
    return AsyncOpenAIAdapter(client, models=picked, **wrap_opts)
