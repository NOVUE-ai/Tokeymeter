"""
Tokeymeter universal adapter — put ANY provider's API call through the engine.

`tokeymeter.integrations.openai.wrap()` is a zero-config drop-in for OpenAI-shaped
clients. `meter()` is the provider-agnostic path: wrap ANY callable that hits ANY
API (Bedrock, Gemini, Cohere, a raw requests/httpx call, your own gateway) and get
the same engine — exact cache, single-flight de-dup, content-blind metering,
fail-open — with no dedicated per-provider integration.

The honest shape of "works with every API":
  - SEEING/caching a call is universal. The cache key is a SHA-256 hash of the
    call's arguments, so exact cache + single-flight + fail-open work on ANY call,
    because the key is just a hash of its inputs. No payload is stored (content-blind).
  - UNDERSTANDING it is per-provider. SEMANTIC caching needs to know WHERE the
    prompt is, and token metering needs to know WHERE the usage numbers are — and
    those fields differ in every API's schema. So they are OPTIONAL one-line hints
    (`text_fn`, `usage_fn`), not automatic.

Universal floor (zero hints):  exact cache + single-flight + content-blind + fail-open.
Full treatment (a few hints):  + semantic cache (text_fn) + token metering (usage_fn).

Works on sync AND async callables — the underlying `tokeymeter.cache` engine handles
both.

Example — three different API shapes, same engine:

    from tokeymeter.engines.execution.integrations.universal import meter

    # AWS Bedrock (usage nested in a dict)
    invoke = meter(bedrock.invoke_model,
                   key_fields=["modelId", "body"],
                   usage_fn=lambda r: (r["usage"]["inputTokens"], r["usage"]["outputTokens"]))

    # Google Gemini (usage on a metadata object)
    gen = meter(model.generate_content,
                usage_fn=lambda r: (r.usage_metadata.prompt_token_count,
                                    r.usage_metadata.candidates_token_count))

    # your own async HTTP gateway (no hints -> exact cache + dedup still work)
    call = meter(my_async_client.post)
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
from typing import Any, Callable, Optional, Sequence

import tokeymeter as tk
from tokeymeter.engines.economics import reconcile


def _generic_key(key_fields: Optional[Sequence[str]], args: tuple, kwargs: dict) -> str:
    """Content-blind cache key: a stable SHA-256 of the call arguments.

    By default the WHOLE call signature (positional + keyword) is hashed, so
    identical calls collide regardless of provider. `key_fields` restricts the
    key to specific keyword arguments (e.g. ignore a client handle passed in).
    """
    if key_fields is not None:
        basis: Any = {k: kwargs.get(k) for k in key_fields}
    else:
        basis = {"args": args, "kwargs": kwargs}
    raw = json.dumps(basis, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _record_usage(usage_fn: Callable[[Any], Any], resp: Any, model: str) -> None:
    """Record token usage extracted by a provider-specific usage_fn. Best-effort.
    Called only on the MISS path (see meter), so cache hits never double-count."""
    try:
        u = usage_fn(resp)
        if not u:
            return
        if isinstance(u, (tuple, list)) and len(u) == 2:
            in_tok, out_tok = u
        elif isinstance(u, dict):
            in_tok = u.get("input", u.get("input_tokens", 0))
            out_tok = u.get("output", u.get("output_tokens", 0))
        else:
            return
        reconcile.record(model=model,
                         input_tokens=int(in_tok or 0),
                         output_tokens=int(out_tok or 0))
    except Exception:
        pass  # metering never breaks the call


def meter(call_fn: Optional[Callable] = None, *,
          key_fields: Optional[Sequence[str]] = None,
          text_fn: Optional[Callable[..., str]] = None,
          usage_fn: Optional[Callable[[Any], Any]] = None,
          model: str = "_default",
          semantic: Optional[bool] = None,
          semantic_threshold: float = 0.92,
          single_flight: bool = True,
          shadow: bool = False,
          tag: Optional[str] = None,
          namespace: Optional[str] = None) -> Callable:
    """Put any provider's API call through the Tokeymeter engine (see module docstring).

    Args:
        call_fn:     the provider call to wrap (any sync/async callable). If omitted,
                     `meter(...)` returns a decorator.
        key_fields:  keyword-arg names to build the cache key from. Default: all args.
        text_fn:     fn(*args, **kwargs) -> str giving the prompt text. Enables the
                     SEMANTIC (near-duplicate) cache. Omit -> exact-cache only.
        usage_fn:    fn(response) -> (input_tokens, output_tokens) | {"input":..,"output":..}.
                     Enables precise token/cost metering. Omit -> caching still works,
                     token counts just aren't reconciled.
        model:       label for cost/usage breakdowns.
        semantic:    force semantic on/off. Default: on iff text_fn is provided.
        shadow:      measure-only — record would-be hits without serving cached values.
        tag:         optional workload label.
    Returns the wrapped callable (sync or async, matching the input).
    """
    use_semantic = bool(text_fn) if semantic is None else bool(semantic)

    sem_cache = None
    if use_semantic:
        try:
            from tokeymeter.engines.optimization.semantic import SemanticCache
            sc = SemanticCache(path=":memory:", threshold=semantic_threshold)
            sem_cache = sc if getattr(sc, "is_functional", False) else None
        except Exception:
            sem_cache = None  # exact-only if embedding deps absent

    # NOTE: the engine calls key_fn / extract_text as fn(args, kwargs) — two
    # positional params (the call's arg-tuple and kwarg-dict), NOT *args/**kwargs.
    def _key(call_args: tuple, call_kwargs: dict) -> str:
        return _generic_key(key_fields, call_args, call_kwargs)

    def _text(call_args: tuple, call_kwargs: dict) -> str:
        if text_fn is None:
            return ""
        try:
            # text_fn is user-facing: call it with the natural call signature.
            return text_fn(*call_args, **call_kwargs) or ""
        except Exception:
            return ""

    def _decorate(fn: Callable) -> Callable:
        # Record usage INSIDE the cached function so it fires only on a miss
        # (when the real provider call actually runs) — never on a cache hit.
        if usage_fn is not None:
            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def instrumented(*a: Any, **k: Any) -> Any:
                    resp = await fn(*a, **k)
                    _record_usage(usage_fn, resp, model)
                    return resp
            else:
                @functools.wraps(fn)
                def instrumented(*a: Any, **k: Any) -> Any:
                    resp = fn(*a, **k)
                    _record_usage(usage_fn, resp, model)
                    return resp
        else:
            instrumented = fn

        ns = namespace or f"tokeymeter.universal.{getattr(fn, '__name__', 'fn')}"
        return tk.cache(
            instrumented,
            model=model,
            key_fn=_key,
            extract_text=_text,
            semantic=use_semantic and sem_cache is not None,
            semantic_cache=sem_cache,
            semantic_threshold=semantic_threshold,
            single_flight=single_flight,
            shadow=shadow,
            tag=tag,
            namespace=ns,
        )

    return _decorate(call_fn) if call_fn is not None else _decorate
