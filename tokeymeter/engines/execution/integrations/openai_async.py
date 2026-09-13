"""
Tokeymeter drop-in for the *async* OpenAI SDK (`AsyncOpenAI`).

The sync wrapper in `tokeymeter.integrations.openai` only intercepts the
synchronous `chat.completions.create`. An async client's `.create` is a
coroutine, so wrapping it with the sync proxy would mishandle the awaitable.
This module provides the async mirror so adoption stays one line for async
codebases too:

    from openai import AsyncOpenAI
    from tokeymeter.engines.execution.integrations.openai import wrap   # auto-detects async

    client = wrap(AsyncOpenAI())          # <-- the only change
    r = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

`wrap()` detects an async client and routes here automatically; you can also
call `wrap_async()` directly.

Semantics are IDENTICAL to the sync path — this is a faithful mirror, not a
new policy:
  - Content-blind: the cache key is a SHA-256 of the request (`_request_fingerprint`);
    no prompt text is ever stored. Reused verbatim from the sync module.
  - Exact + semantic cache: an in-process object store for exact hits, and the
    same `SemanticCache` (Stage-1 cosine + optional Stage-2 cross-encoder verify)
    used by the sync wrapper.
  - Single-flight: concurrent in-flight requests with the same fingerprint share
    one upstream call. Implemented with an asyncio future map (the sync path's
    threading single-flight does not help coroutines on one event loop).
  - Fail-open: if ANYTHING in the engine raises, the original client coroutine is
    awaited and returned. The wrapped call never breaks.
  - Streaming and positional/tool-result calls pass straight through, uncached
    (a streamed response is not a deterministic cacheable value) — matching the
    sync wrapper exactly. Nothing is silently broken.
  - Cost is measured from the provider's OWN reported `response.usage`, so the
    savings figure reconciles against the real bill.

Async-specific care:
  - The one potentially slow engine op (the MiniLM embedding for the semantic
    cache) is offloaded via `asyncio.to_thread`, so it does not block the event
    loop. All other engine work (hashing, exact-store get/set) is sub-millisecond
    and runs inline.

Scope of v1 (honest): cascade (verify-then-escalate) is NOT wired in the async
path yet. It is off by default and is a verification/governance feature, not a
cost saver, so omitting it here changes nothing for the default cost wedge. If
`cascade=True` is passed to an async client it is simply not applied (a degraded
event is emitted so it is visible, never silent).
"""
from __future__ import annotations
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.economics.usage import set_queue_wait_ms
from tokeymeter.engines.economics.usage import (
    extract_queue_wait_ms as _extract_queue_wait_ms,
    consume_queue_wait_ms as _consume_queue_wait_ms,
)
from tokeymeter.engines.execution.endpoint import get_endpoint as _get_endpoint

import asyncio
import time
from typing import Any, Dict

import tokeymeter as tk
from tokeymeter import pricing
from tokeymeter.engines.economics import reconcile
from tokeymeter.engines.economics.savings import CallRecord, _record
from tokeymeter.engines.economics.savings import build_call_record as _build_call_record
from tokeymeter.pricing import estimate_cost_with_source as _estimate_cost_with_source
from tokeymeter.identity import get_principal as _get_principal
from tokeymeter import keys as _keys

# Reuse the sync module's pure, content-blind helpers verbatim so the two paths
# can never drift on fingerprinting, prompt flattening, routing, or compression.
from tokeymeter.engines.execution.integrations.openai import (
    _request_fingerprint,
    _messages_text,
    _apply_router,
    _apply_compression,
    _seal_routing_verdict,
)

try:  # observability for fail-open paths (optional, never required)
    from tokeymeter.engines.reliability.degraded import emit_degraded
except Exception:  # pragma: no cover
    def emit_degraded(*_a: Any, **_k: Any) -> None:  # type: ignore[misc]
        pass


class _AsyncCachedCompletions:
    """Async-optimized `chat.completions`. Mirrors `_CachedCompletions` for await."""

    def __init__(self, inner, *, semantic, semantic_threshold, shadow, tag,
                 router=None, compressor=None, audit_log=None,
                 verify=False, verify_threshold=0.0,
                 verify_model="cross-encoder/quora-distilroberta-base",
                 eval_loop=None,
                 cascade=False, cascade_cheap=None, cascade_capable=None,
                 cascade_verify=False, cascade_verify_model=None,
                 cheap_model=None, capable_model=None,
                 compression_eval=None, compression_relevance_fn=None):
        self._inner = inner
        self._shadow = bool(shadow)
        self._tag = tag
        self._router = router
        self._compressor = compressor
        self._compression_eval = compression_eval
        if compression_eval is not None and compressor is not None:
            try:
                inner_c = getattr(compressor, "inner", None)
                if getattr(compression_eval, "_compressor", None) is None:
                    compression_eval._compressor = inner_c
            except Exception:
                pass
        self._audit_log = audit_log

        # async single-flight: fingerprint -> in-flight Future
        self._inflight: Dict[str, "asyncio.Future"] = {}

        # cascade is not supported in the async path v1 (off by default; it is a
        # verification feature, not a cost saver). Surface it rather than ignore.
        if cascade:
            emit_degraded(
                "async_cascade_unsupported",
                NotImplementedError("cascade is not yet wired for AsyncOpenAI; "
                                    "running normal single calls"),
                function_name="wrap_async",
            )

        # Exact-hit store: in-process object store so SDK response objects round-trip
        # intact (a JSON store would stringify them and break attribute access).
        self._store = tk.MemoryStore()

        # Semantic (near-duplicate) cache — the SAME class the sync path uses, with
        # the SAME optional Stage-2 cross-encoder verifier. Built only when requested
        # so the heavy import stays optional.
        self._sc = None
        if semantic:
            try:
                from tokeymeter.engines.optimization.semantic import SemanticCache
                _verifier = None
                if verify:
                    try:
                        from tokeymeter.engines.optimization.semantic_verify import SemanticVerifier
                        _verifier = SemanticVerifier(
                            accept_threshold=verify_threshold,
                            model_name=verify_model,
                        )
                    except Exception:
                        _verifier = None  # graceful: Stage-1-only if verifier deps absent
                sc = SemanticCache(path=":memory:", threshold=semantic_threshold,
                                   verifier=_verifier)
                if eval_loop is not None:
                    sc._monitor = eval_loop
                    if getattr(eval_loop, "_cache", None) is None:
                        eval_loop._cache = sc
                self._sc = sc if getattr(sc, "is_functional", False) else None
            except Exception:
                self._sc = None  # exact-only if deps absent

    # ---- recording (mirrors the sync path's savings + reconcile signals) ----

    def _record_usage(self, resp, model) -> None:
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                set_reported_usage(getattr(u, "prompt_tokens", 0),
                                   getattr(u, "completion_tokens", 0))
            qw = _extract_queue_wait_ms(resp)
            if qw is not None:
                set_queue_wait_ms(qw)
            if u is not None:
                reconcile.record(
                    model=model,
                    input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(u, "completion_tokens", 0) or 0,
                )
        except Exception:
            pass

    def _emit_record(self, resp, model, hit, hit_type, t0) -> None:
        """Append a CallRecord to savings.jsonl. Never raises into the caller.

        Builds through the shared factory (savings.build_call_record) so this
        path carries the SAME fields as the decorator path — including
        pricing_source, principal, and key_name, which this wrapper previously
        omitted. Resolution mirrors the decorator: reported usage beats the
        estimate; pricing provenance comes from estimate_cost_with_source;
        identity/endpoint/queue-wait read the same contextvars.
        """
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                set_reported_usage(getattr(u, "prompt_tokens", 0),
                                   getattr(u, "completion_tokens", 0))
            qw = _extract_queue_wait_ms(resp)
            if qw is not None:
                set_queue_wait_ms(qw)
            in_tok = getattr(u, "prompt_tokens", 0) or 0 if u is not None else 0
            out_tok = getattr(u, "completion_tokens", 0) or 0 if u is not None else 0
            # Pricing WITH provenance (not the bare estimate) — parity with the
            # decorator, so async-wrapper records no longer read as unpriced.
            try:
                cost, pricing_source = _estimate_cost_with_source(
                    model, in_tok, out_tok)
            except Exception:
                cost, pricing_source = 0.0, None
            _record(_build_call_record(
                model=model,
                hit=hit,
                hit_type=hit_type,
                input_tokens=in_tok,
                output_tokens=out_tok,
                estimated_cost=cost,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                shadow=self._shadow,
                tag=self._tag,
                pricing_source=pricing_source,
                principal=_get_principal(),
                token_source="reported" if u is not None else "estimated",
                key_name=_keys.get_current_key(),
                endpoint_identity=_get_endpoint(),
                queue_wait_ms=None if hit else _consume_queue_wait_ms(),
            ))
        except Exception:
            pass  # measurement never breaks a call

    # ---- the optimized async create ----

    async def create(self, *args, **kwargs):
        # Streaming / positional / tool-result calls pass through, uncached.
        if kwargs.get("stream") or args:
            return await self._inner.create(*args, **kwargs)

        model = kwargs.get("model", "_default")
        messages = kwargs.get("messages")

        # ---- opt-in router (sync, fast, fail-open) — applied before fingerprint ----
        if self._router is not None and messages:
            try:
                decision = _apply_router(self._router, model, messages)
                routed = getattr(decision, "model", None) if decision else None
                if routed is not None:
                    verdict = {
                        "routed_model": routed,
                        "tier": getattr(decision, "tier", None),
                        "win_rate": round(float(getattr(decision, "win_rate", 0.0)), 4),
                        "reason": getattr(decision, "reason", ""),
                        "est_saved_usd": round(float(getattr(decision, "est_saved_usd", 0.0)), 6),
                        "requested_model": model,
                    }
                    if routed != model:
                        model = routed
                        kwargs = {**kwargs, "model": routed}
                    if self._audit_log is not None:
                        try:
                            _seal_routing_verdict(self._audit_log, verdict,
                                                  _messages_text(messages), self._tag)
                        except Exception:
                            pass  # fail-open: sealing never breaks the call
            except Exception:
                pass  # fail-open: keep the caller's model

        # ---- opt-in compression (sync, fail-open) ----
        if self._compressor is not None and messages:
            try:
                new_messages = _apply_compression(self._compressor, messages,
                                                  self._compression_eval)
                if new_messages is not None:
                    messages = new_messages
                    kwargs = {**kwargs, "messages": new_messages}
            except Exception:
                pass  # fail-open: keep the original prompt

        text = _messages_text(messages)
        fp_kw = {k: v for k, v in kwargs.items() if k not in ("model", "messages")}
        fp = _request_fingerprint(model, messages, **fp_kw)

        try:
            return await self._lookup_or_call(fp, text, model, kwargs)
        except Exception:
            # The sacred contract: any engine error -> raw upstream call.
            return await self._inner.create(**kwargs)

    async def _lookup_or_call(self, fp, text, model, kwargs):
        t0 = time.perf_counter()

        # 1) exact cache (sub-ms, inline)
        cached = None
        hit_type = None
        try:
            cached = self._store.get(fp)
        except Exception:
            cached = None
        if cached is not None:
            hit_type = "exact"

        # 2) semantic cache (encode offloaded off the event loop)
        emb = None
        if cached is None and self._sc is not None:
            try:
                emb = await asyncio.to_thread(self._sc.encode, text)
            except Exception:
                emb = None
            if emb is not None:
                try:
                    cached = self._sc.lookup_by_embedding(emb, query_prompt=text)
                except Exception:
                    cached = None
                if cached is not None:
                    hit_type = "semantic"

        # serve from cache unless shadow (shadow = measure would-be hits, never serve)
        if hit_type is not None and not self._shadow:
            self._emit_record(cached, model, True, hit_type, t0)
            return cached
        if hit_type is not None and self._shadow:
            self._emit_record(cached, model, True, "shadow_" + hit_type, t0)
            # fall through to a live call; do not double-count as a miss below

        # 3) miss (or shadow) -> single-flight live call
        return await self._single_flight(fp, text, emb, model, kwargs, t0,
                                         already_measured=(hit_type is not None))

    async def _single_flight(self, fp, text, emb, model, kwargs, t0, already_measured):
        # de-dup concurrent identical in-flight requests on this event loop
        existing = self._inflight.get(fp)
        if existing is not None:
            resp = await existing  # may raise -> propagates to create()'s fail-open
            if not already_measured:
                self._emit_record(resp, model, True, "single_flight", t0)
            return resp

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._inflight[fp] = fut
        try:
            resp = await self._inner.create(**kwargs)
            # populate caches (best-effort; never break the call)
            try:
                self._store.set(fp, resp)
            except Exception:
                pass
            if self._sc is not None:
                try:
                    if emb is None:
                        emb = await asyncio.to_thread(self._sc.encode, text)
                    if emb is not None:
                        self._sc.store_by_embedding(text, emb, resp)
                except Exception:
                    pass
            self._record_usage(resp, model)
            if not already_measured:
                self._emit_record(resp, model, False, None, t0)
            if not fut.done():
                fut.set_result(resp)
            return resp
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            self._inflight.pop(fp, None)


class _AsyncChatProxy:
    def __init__(self, inner_chat: Any, **opts: Any) -> None:
        self._inner = inner_chat
        self._completions = _AsyncCachedCompletions(inner_chat.completions, **opts)

    @property
    def completions(self) -> _AsyncCachedCompletions:
        return self._completions

    def __getattr__(self, name: str) -> Any:  # passthrough for everything else
        return getattr(self._inner, name)


class _AsyncClientProxy:
    """Transparent proxy over an AsyncOpenAI client. Only
    `chat.completions.create` is intercepted; everything else passes straight
    through to the real async client (so `await client.embeddings.create(...)`,
    `.beta`, etc. keep working unchanged)."""

    def __init__(self, client: Any, **opts: Any) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_chat", _AsyncChatProxy(client.chat, **opts))

    @property
    def chat(self) -> _AsyncChatProxy:
        return object.__getattribute__(self, "_chat")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_client"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_client"), name, value)


def is_async_client(client: Any) -> bool:
    """True if `client.chat.completions.create` is a coroutine function
    (i.e. an AsyncOpenAI-shaped client). Fail-safe: returns False on any
    introspection error, so an odd client falls back to the sync path."""
    try:
        import inspect
        create = client.chat.completions.create
        if inspect.iscoroutinefunction(create):
            return True
    except Exception:
        pass
    try:
        return type(client).__name__.startswith("Async")
    except Exception:
        return False


def wrap_async(client: Any, **opts: Any) -> _AsyncClientProxy:
    """Wrap an AsyncOpenAI client. Normally reached via `wrap()` auto-detection;
    exposed directly for callers who want to be explicit."""
    return _AsyncClientProxy(client, **opts)
