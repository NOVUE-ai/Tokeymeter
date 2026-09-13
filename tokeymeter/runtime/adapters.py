"""Real provider adapters (K3, W2) — the shipped pipeline behind the contract.

Design law: adapters wrap the WRAPPER (`integrations.openai.wrap` /
`integrations.anthropic.wrap` / `openai_async.wrap_async`), so every request
through the kernel path runs the exact shipped machinery — cache,
compression, routing, salience, audit sealing, and reported-usage truth —
by construction, not by re-implementation. The parity gate
(test_usage_truth_parity) pins this: the kernel path and the direct-wrapper
path emit identical usage/cost envelopes.

EXEC-6 (tool calls): the wrapper passes tool calls straight through
uncached (documented in integrations.openai); adapters surface only
`meta.tool_calls` = COUNT. Arguments/payloads never enter kernel telemetry
(L4 pin in the battery).

Errors: every provider exception is classified into the EXEC-4 taxonomy
before it leaves the adapter, so Reliability routes on type.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from .errors import classify
from .providers import HealthStatus, ProviderAdapter, ProviderInfo


def _messages_from_ctx(ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    """Callers may pass structured messages via metadata; otherwise the
    kernel payload becomes a single user message."""
    msgs = ctx["meta"].get("messages") or ctx["request"].metadata.get("messages")
    if msgs:
        return msgs
    return [{"role": "user", "content": ctx["request"].payload}]


def _surface_tool_calls(ctx: Dict[str, Any], response: Any) -> None:
    try:
        choice = response.choices[0]
        tcs = getattr(choice.message, "tool_calls", None)
        if tcs:
            ctx["meta"]["tool_calls"] = len(tcs)   # COUNT only — L4
    except (AttributeError, IndexError, TypeError):
        pass


class OpenAIAdapter(ProviderAdapter):
    # Routes through the shipped wrapper, which records. The runtime must not
    # record again for this adapter.
    emits_ledger_record = True

    """Wraps an OpenAI-style client through the shipped wrapper."""

    provider = "openai"

    def __init__(self, client: Any, *, models: Optional[List[str]] = None,
                 **wrap_opts: Any) -> None:
        from tokeymeter.engines.execution.integrations import openai as _oai
        self._wrapped = _oai.wrap(client, **wrap_opts)
        self._models = models or ["gpt-*"]

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(provider=self.provider, models=list(self._models))

    def infer(self, ctx: Dict[str, Any]) -> Any:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        try:
            resp = self._wrapped.chat.completions.create(
                model=model, messages=_messages_from_ctx(ctx), **kwargs)
        except Exception as exc:
            raise classify(exc, self.provider) from exc
        _surface_tool_calls(ctx, resp)
        return resp

    def stream_infer(self, ctx: Dict[str, Any]) -> Iterator[Any]:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        try:
            stream = self._wrapped.chat.completions.create(
                model=model, messages=_messages_from_ctx(ctx),
                stream=True, **kwargs)
            for chunk in stream:
                yield chunk
        except Exception as exc:
            raise classify(exc, self.provider) from exc

    def health_check(self) -> HealthStatus:
        try:
            ok = self._wrapped is not None and hasattr(
                self._wrapped, "chat")
            return HealthStatus(bool(ok), 0.0,
                                "wrapped client ready" if ok else "no chat surface")
        except Exception as exc:
            return HealthStatus(False, 0.0, type(exc).__name__)


class AnthropicAdapter(ProviderAdapter):
    # Routes through the shipped wrapper, which records. The runtime must not
    # record again for this adapter.
    emits_ledger_record = True

    provider = "anthropic"

    def __init__(self, client: Any, *, models: Optional[List[str]] = None,
                 max_tokens: int = 1024, **wrap_opts: Any) -> None:
        from tokeymeter.engines.execution.integrations import anthropic as _ant
        self._wrapped = _ant.wrap(client, **wrap_opts)
        self._models = models or ["claude-*"]
        self._max_tokens = max_tokens

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(provider=self.provider, models=list(self._models))

    def infer(self, ctx: Dict[str, Any]) -> Any:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        kwargs.setdefault("max_tokens", self._max_tokens)
        try:
            return self._wrapped.messages.create(
                model=model, messages=_messages_from_ctx(ctx), **kwargs)
        except Exception as exc:
            raise classify(exc, self.provider) from exc

    def stream_infer(self, ctx: Dict[str, Any]) -> Iterator[Any]:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        kwargs.setdefault("max_tokens", self._max_tokens)
        try:
            for chunk in self._wrapped.messages.create(
                    model=model, messages=_messages_from_ctx(ctx),
                    stream=True, **kwargs):
                yield chunk
        except Exception as exc:
            raise classify(exc, self.provider) from exc

    def health_check(self) -> HealthStatus:
        ok = hasattr(self._wrapped, "messages")
        return HealthStatus(bool(ok), 0.0,
                            "wrapped client ready" if ok else "no messages surface")


class AsyncOpenAIAdapter(ProviderAdapter):
    # Routes through the shipped wrapper, which records. The runtime must not
    # record again for this adapter.
    emits_ledger_record = True

    """Async twin (KA-1 lands the async kernel in W3; W2 pins adapter parity).
    Exposes `ainfer`/`astream_infer`; the sync methods raise a typed error
    to fail loud rather than block an event loop silently."""

    provider = "openai"

    def __init__(self, client: Any, *, models: Optional[List[str]] = None,
                 **wrap_opts: Any) -> None:
        # Route through the sync wrap() entry: it auto-detects async
        # clients (is_async_client) and fills ALL option defaults before
        # delegating to wrap_async — calling wrap_async directly with
        # partial opts violates its required-kwargs contract (found by
        # this wave's battery).
        from tokeymeter.engines.execution.integrations import openai as _oai
        self._wrapped = _oai.wrap(client, **wrap_opts)
        self._models = models or ["gpt-*"]

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(provider=self.provider, models=list(self._models))

    def infer(self, ctx: Dict[str, Any]) -> Any:  # pragma: no cover - guard
        raise RuntimeError(
            "AsyncOpenAIAdapter is async-only: use ainfer() "
            "(async kernel path lands in W3/KA-1)")

    async def ainfer(self, ctx: Dict[str, Any]) -> Any:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        try:
            resp = await self._wrapped.chat.completions.create(
                model=model, messages=_messages_from_ctx(ctx), **kwargs)
        except Exception as exc:
            raise classify(exc, self.provider) from exc
        _surface_tool_calls(ctx, resp)
        return resp

    async def astream_infer(self, ctx: Dict[str, Any]) -> Any:
        model = ctx["meta"].get("model", ctx["request"].model)
        kwargs = dict(ctx["request"].metadata.get("provider_kwargs", {}))
        try:
            stream = await self._wrapped.chat.completions.create(
                model=model, messages=_messages_from_ctx(ctx),
                stream=True, **kwargs)
            async for chunk in stream:
                yield chunk
        except Exception as exc:
            raise classify(exc, self.provider) from exc

    def health_check(self) -> HealthStatus:
        ok = hasattr(self._wrapped, "chat")
        return HealthStatus(bool(ok), 0.0,
                            "wrapped async client ready" if ok else "no chat surface")
