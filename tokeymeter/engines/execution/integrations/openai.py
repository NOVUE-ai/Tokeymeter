"""
Tokeymeter drop-in for the OpenAI SDK.

Adoption is one line. Instead of restructuring your code into decorated
functions, wrap the client you already have:

    from openai import OpenAI
    from tokeymeter.engines.execution.integrations.openai import wrap

    client = wrap(OpenAI())          # <-- the only change
    # ...use client exactly as before...
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

Every `chat.completions.create` now passes through the engine: exact +
semantic cache, single-flight de-duplication, and a content-blind audit
record, with cost measured from the provider's OWN reported token usage
(`response.usage`) on misses — so the savings figure reconciles against the
real bill rather than a heuristic.

Design rules honored:
  - Provider-neutral core: this file only adapts shapes; all logic is the engine's.
  - Content-blind: the cache key is a hash of the request; no prompt text is stored.
  - Fail-open: if anything here raises, the original client method is called.
  - Streaming and tool calls are passed straight through UNCACHED (a streamed or
    tool-calling response is not a deterministic cacheable value); only plain
    completions are optimized. Nothing is ever silently broken.
"""
from __future__ import annotations
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.economics.usage import set_queue_wait_ms
from tokeymeter.engines.economics.usage import extract_queue_wait_ms as _extract_queue_wait_ms

import re
import threading
import hashlib
import json
from typing import Any

import tokeymeter as tk
from tokeymeter.engines.economics import reconcile


def _request_fingerprint(model: str, messages: Any, **kw: Any) -> str:
    """Stable, content-blind hash of the semantically meaningful request parts."""
    basis = {
        "model": model,
        "messages": messages,
        "temperature": kw.get("temperature"),
        "top_p": kw.get("top_p"),
        "max_tokens": kw.get("max_tokens") or kw.get("max_completion_tokens"),
        "tools": kw.get("tools"),
        "response_format": kw.get("response_format"),
    }
    raw = json.dumps(basis, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _messages_text(messages: Any) -> str:
    """Flatten chat messages to a single string for semantic embedding only.

    Used solely to drive the semantic cache's local embedding; it never leaves
    the process and is never stored in the audit record.
    """
    parts = []
    try:
        for m in messages or []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):  # content-parts form
                for p in c:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        parts.append(p["text"])
    except Exception:
        return ""
    return "\n".join(parts)


def _apply_router(router: Any, model: str, messages: Any):
    """Ask the router which model to use for this prompt. Returns the full
    RouteDecision (or None to leave the caller's model unchanged) so the wrapper
    can both re-route the call AND seal the routing verdict into the proof spine.
    Fail-open by design (the caller wraps this in try/except). The prompt text is
    the last user message — the part whose win-rate drives the routing decision.
    """
    last_user = ""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                last_user = c
                break
    if not last_user:
        return None
    return router.route(last_user)


def _seal_routing_verdict(audit_log: Any, verdict: dict, prompt_text: str,
                          tag: Any) -> None:
    """Seal a routing verdict into the signed, hash-chained audit log.

    Content-blind by construction: the prompt is HMAC-hashed by the ledger (we
    pass prompt_text only so the ledger can derive its per-install hash — it is
    never stored raw), and the win-rate / reason / requested-model go in
    `metadata`, which the ledger hashes rather than storing in clear. What ends
    up on the chain: decision_type="routing", the routed model, the saving, and
    a hash of the routing rationale — tamper-evident and auditable, with no
    prompt or response content exposed.
    """
    if not verdict:
        return
    audit_log.append(
        decision_type="routing",
        prompt_text=prompt_text or "",
        model=str(verdict.get("routed_model", "_default")),
        cost_saved_usd=float(verdict.get("est_saved_usd", 0.0) or 0.0),
        tag=tag,
        metadata={
            "tier": verdict.get("tier"),
            "win_rate": verdict.get("win_rate"),
            "reason": verdict.get("reason"),
            "requested_model": verdict.get("requested_model"),
        },
    )


def _seal_cascade_decision(audit_log: Any, decision: dict, prompt_text: str,
                           tag: Any) -> None:
    """Seal a cascade (verify-then-escalate) decision into the signed, hash-chained
    audit log. Content-blind: prompt is HMAC-hashed by the ledger; the escalation
    reason / served model / saving live in hashed metadata. Provable that no answer
    was served below the quality bar (escalations are recorded and verifiable)."""
    if not decision:
        return
    audit_log.append(
        decision_type="cascade",
        prompt_text=prompt_text or "",
        model=str(decision.get("served_model", "_default")),
        cost_saved_usd=float(decision.get("est_saved_usd", 0.0) or 0.0),
        tag=tag,
        metadata={
            "escalated": decision.get("escalated"),
            "reason": decision.get("reason"),
            "layer": decision.get("layer"),
            "cheap_model": decision.get("cheap_model"),
            "capable_model": decision.get("capable_model"),
        },
    )


from tokeymeter.engines.execution.response_text import (
    openai_response_text as _openai_response_text)


def _extract_question(content: str) -> str:
    """Pull the actual QUESTION from a (possibly long) user message for query-aware
    compression. Prefers the last sentence ending in '?'; falls back to the last
    non-empty line, then the whole content. Using just the question (not the whole
    context blob) makes relevance scoring sharp — the answer-bearing chunk wins."""
    if not isinstance(content, str) or not content.strip():
        return content or ""
    # last '?'-terminated sentence/line
    questions = re.findall(r"[^.?!\n]*\?", content)
    if questions:
        q = questions[-1].strip()
        if len(q) >= 8:
            return q
    # else: last non-empty line (often the instruction/ask)
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return content


def _apply_compression(compressor: Any, messages: Any, comp_eval: Any = None) -> Any:
    """Compress the long user message(s) in place, returning a new messages list,
    or None if nothing was compressed. The SafeCompressor's quality gate decides
    whether each message is actually compressed or shipped intact, so this never
    silently degrades a prompt. Only user-role string content is considered.
    """
    changed = False
    out = []
    # For a query-aware compressor, the question is the last user message. Find it
    # so context in earlier/long messages is pruned by relevance to that question.
    query_aware = getattr(compressor, "_query_aware", False)
    last_user_q = None
    if query_aware:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user_q = _extract_question(m["content"])
                break
    for m in messages or []:
        if (isinstance(m, dict) and m.get("role") == "user"
                and isinstance(m.get("content"), str) and len(m["content"]) > 200):
            comp = compressor
            if query_aware and last_user_q:
                # use the SafeCompressor's own query mechanism: a query-scoped copy
                # passes the question to the inner query-aware compressor AND adds
                # the query-fact survival gate (drops that lose the answer fall back).
                try:
                    from dataclasses import replace as _replace
                    comp = _replace(compressor, query=last_user_q)
                    comp._query_aware = True
                except Exception:
                    comp = compressor
            res = comp.compress(m["content"])
            after = getattr(res, "after", None)
            if after and after != m["content"]:
                # report to the compression eval loop (content-blind) — what was
                # dropped is the difference between original and kept.
                if comp_eval is not None:
                    try:
                        original = m["content"]
                        kept_lines = set(after.split("\n"))
                        dropped = "\n".join(ln for ln in original.split("\n")
                                            if ln and ln not in kept_lines)
                        q_for_eval = last_user_q if query_aware else _extract_question(original)
                        comp_eval.record_compression(
                            query=q_for_eval or "",
                            kept_text=after,
                            dropped_text=dropped,
                            ratio=getattr(res, "ratio", None),
                            tokens_saved=(getattr(res, "tokens_before", 0)
                                          - getattr(res, "tokens_after", 0)),
                        )
                    except Exception:
                        pass  # monitoring never breaks a call
                out.append({**m, "content": after})
                changed = True
                continue
        out.append(m)
    return out if changed else None


def _is_deliberate_refusal(exc: BaseException) -> bool:
    """Is this exception a decision we made on purpose?

    Fail-open exists so a bug in metering never breaks a caller's request. It
    must NOT apply to a refusal, because a refusal IS the feature: catching it
    and calling the unwrapped client makes the call anyway, off-meter.

    Measured before this guard existed: a task with a stall ceiling metered 4
    calls while 20 went upstream — the ceiling appeared to hold and 16 calls
    were billed invisibly. That breaks enforcement AND the reconciliation
    claim in one move.
    """
    try:
        from tokeymeter.engines.execution.task import TaskLimitExceeded
        if isinstance(exc, TaskLimitExceeded):
            return True
    except Exception:
        pass
    try:
        from tokeymeter.engines.governance.compliance import PolicyViolation
        if isinstance(exc, PolicyViolation):
            return True
    except Exception:
        pass
    try:
        from tokeymeter.engines.governance.identity import KeyBudgetExceeded
        if isinstance(exc, KeyBudgetExceeded):
            return True
    except Exception:
        pass
    return False


class _CachedCompletions:
    """Wraps `client.chat.completions` so `.create` is engine-optimized.

    The engine-cached function is built ONCE at construction with a stable
    namespace, so repeated requests share one cache. The actual request is
    passed via a thread-safe handoff and the cache key is the request
    fingerprint — that is what makes identical/near-identical requests collide.
    """

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
        self._local = threading.local()
        # Opt-in optimizers (off by default — they alter the request, so a caller
        # must explicitly enable them). Applied BEFORE fingerprinting so the
        # routed model and compressed prompt drive the cache key correctly.
        self._router = router
        self._compressor = compressor
        # compression harmful-drop monitor (optional). Connected to the compressor
        # so auto-backoff (opt-in) can raise its keep-ratio on a breach.
        self._compression_eval = compression_eval
        if compression_eval is not None and compressor is not None:
            try:
                inner_c = getattr(compressor, "inner", None)
                if getattr(compression_eval, "_compressor", None) is None:
                    compression_eval._compressor = inner_c
            except Exception:
                pass
        # Optional audit ledger: when present, routing verdicts are sealed into
        # its content-blind, hash-chained, signed log — turning routing into
        # PROVABLE cost governance (an auditor can verify no high-stakes call was
        # silently cheaped out, and the savings reconcile). This is the moat.
        self._audit_log = audit_log
        # Cascade (verify-then-escalate). Built when enabled. The call function is
        # the inner client's create; the cascade overrides `model` per attempt.
        self._cascade = None
        if cascade:
            try:
                from tokeymeter.engines.optimization.cascade import Cascade, QualityGate, make_self_verifier
                from tokeymeter.engines.economics.pricing import estimate_cost as _ec
                verifier_fn = None
                if cascade_verify:
                    verifier_fn = make_self_verifier(
                        inner.create, cascade_verify_model or (capable_model or "gpt-4o"))
                self._cascade = Cascade(
                    cheap_model=cascade_cheap or cheap_model or "gpt-4o-mini",
                    capable_model=cascade_capable or capable_model or "gpt-4o",
                    call_fn=inner.create,
                    gate=QualityGate(),
                    verifier_fn=verifier_fn,
                    estimate_cost_fn=_ec,
                )
            except Exception:
                self._cascade = None  # graceful: normal single calls if unavailable
        # One semantic cache per wrapped client (local MiniLM encoder, in-memory
        # vector index). Built only when requested so the import stays optional.
        # Pin an in-process object store so SDK response objects round-trip
        # intact (a JSON-serializing default store would return a string and
        # break attribute access like response.choices[0].message.content).
        import tokeymeter as _tk
        self._store = _tk.MemoryStore()
        _sc = None
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
                _sc = SemanticCache(path=':memory:', threshold=semantic_threshold,
                                    verifier=_verifier)
                # Eval loop (false-positive monitoring). Attached AFTER the cache
                # exists so it can auto-tighten that cache's threshold if opted in.
                if eval_loop is not None:
                    _sc._monitor = eval_loop
                    if getattr(eval_loop, "_cache", None) is None:
                        eval_loop._cache = _sc
                self._eval_loop = eval_loop
            except Exception:
                _sc = None  # fall back to exact-only if deps are absent

        @tk.cache(
            model="_default",  # real model is set per-call via the fingerprint + record
            store=self._store,
            semantic=semantic,
            semantic_cache=_sc,
            semantic_threshold=semantic_threshold,
            shadow=shadow,
            tag=tag,
            single_flight=True,
            namespace="tokeymeter.integrations.openai.chat",
            key_fn=lambda *_a, **_k: self._local.fingerprint,
            extract_text=lambda *_a, **_k: self._local.text,
            # The RESPONSE extractor, deliberately distinct from extract_text
            # above: that one yields the PROMPT for token estimation. Without
            # this the progress signal cannot score a provider object at all,
            # because its text form carries a per-call request id.
            extract_response_text=_openai_response_text,
        )
        def _cached():
            kwargs = self._local.kwargs
            model = kwargs.get("model", "_default")
            # ---- Cascade (verify-then-escalate) on the cache-miss path ----
            # When enabled, try cheap -> gate -> escalate to capable only if the
            # cheap answer fails the quality bar. Off by default. Fail-open: any
            # cascade error falls back to a normal single call.
            if self._cascade is not None:
                try:
                    prompt_text = self._local.text if hasattr(self._local, "text") else ""
                    resp, cdec = self._cascade.run(prompt_text, **kwargs)
                    model = getattr(cdec, "served_model", model)
                    # seal the cascade decision into the content-blind proof spine
                    self._local.cascade_decision = {
                        "served_model": cdec.served_model,
                        "escalated": cdec.escalated,
                        "reason": cdec.reason,
                        "layer": cdec.layer,
                        "est_saved_usd": round(float(cdec.est_saved_usd), 6),
                        "cheap_model": cdec.cheap_model,
                        "capable_model": cdec.capable_model,
                    }
                    if self._audit_log is not None:
                        try:
                            _seal_cascade_decision(
                                self._audit_log, self._local.cascade_decision,
                                prompt_text, self._tag if hasattr(self, "_tag") else None)
                        except Exception:
                            pass  # fail-open: sealing never breaks the call
                    self._record_usage(resp, model)
                    return resp
                except Exception:
                    pass  # fall through to a normal single call
            resp = self._inner.create(**kwargs)
            self._record_usage(resp, model)
            return resp

        self._cached = _cached

    def _record_usage(self, resp, model):
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                set_reported_usage(getattr(u, "prompt_tokens", 0),
                                   getattr(u, "completion_tokens", 0))
            # Self-hosted serving layers may surface per-request queue time on
            # the response; capture it where present (None otherwise — never
            # fabricated). Set alongside reported usage so the same miss record
            # carries both provider-truth signals.
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

    def create(self, *args, **kwargs):
        # Streaming / positional / tool-result calls are passed through, uncached.
        if kwargs.get("stream") or args:
            return self._inner.create(*args, **kwargs)
        model = kwargs.get("model", "_default")
        messages = kwargs.get("messages")

        # ---- Opt-in pre-processing (router, then compression) ----
        # Applied here, before fingerprinting, so the chosen model and the
        # pruned prompt become part of the cache key. Each is fail-open: any
        # error leaves the request untouched and the call proceeds normally.
        if self._router is not None and messages:
            try:
                decision = _apply_router(self._router, model, messages)
                routed = getattr(decision, "model", None) if decision else None
                if routed is not None:
                    # seal the routing verdict so it can be recorded in the
                    # content-blind proof spine (provable cost governance: an
                    # auditor can verify no high-stakes call was cheaped out).
                    self._local.routing_verdict = {
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
                    # Seal the verdict into the signed, hash-chained audit log
                    # (content-blind: prompt is HMAC'd, the win-rate/reason live
                    # in hashed metadata). This makes the routing decision
                    # tamper-evident and auditable — provable cost governance.
                    if self._audit_log is not None:
                        try:
                            _seal_routing_verdict(
                                self._audit_log, self._local.routing_verdict,
                                self._local.text if hasattr(self._local, "text") else "",
                                self._tag if hasattr(self, "_tag") else None)
                        except Exception:
                            pass  # fail-open: sealing never breaks the call
            except Exception:
                pass  # fail-open: keep the caller's model
        if self._compressor is not None and messages:
            try:
                new_messages = _apply_compression(self._compressor, messages, getattr(self, "_compression_eval", None))
                if new_messages is not None:
                    messages = new_messages
                    kwargs = {**kwargs, "messages": new_messages}
            except Exception:
                pass  # fail-open: keep the original prompt

        # stash per-call context for the stable cached function (thread-local)
        self._local.kwargs = kwargs
        self._local.text = _messages_text(messages)
        _fp_kw = {k: v for k, v in kwargs.items() if k not in ('model', 'messages')}
        self._local.fingerprint = _request_fingerprint(model, messages, **_fp_kw)
        # Announce the model for THIS call. The decorator was bound with a
        # placeholder because a wrapped client cannot know the model until the
        # caller picks one; without this the placeholder is what the allowlist,
        # the price and the record all see.
        from tokeymeter.decorator import set_per_call_model, reset_per_call_model
        _model_token = set_per_call_model(kwargs.get("model"))
        try:
            return self._cached()
        except Exception as exc:
            # A refusal is a decision, not a failure. Re-raise it: calling the
            # unwrapped client here would make the call anyway, off-meter.
            if _is_deliberate_refusal(exc):
                raise
            return self._inner.create(**kwargs)  # fail-open
        finally:
            reset_per_call_model(_model_token)


class _ChatProxy:
    def __init__(self, inner_chat: Any, **opts: Any) -> None:
        self._inner = inner_chat
        self._completions = _CachedCompletions(inner_chat.completions, **opts)

    @property
    def completions(self) -> _CachedCompletions:
        return self._completions

    def __getattr__(self, name: str) -> Any:  # passthrough for everything else
        return getattr(self._inner, name)


class _ClientProxy:
    """Transparent proxy over an OpenAI client. Only `.chat.completions.create`
    is intercepted; every other attribute and method passes straight through."""

    def __init__(self, client: Any, **opts: Any) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_chat", _ChatProxy(client.chat, **opts))

    @property
    def chat(self) -> _ChatProxy:
        return object.__getattribute__(self, "_chat")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_client"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_client"), name, value)


def wrap(client: Any, *, semantic: bool = True, semantic_threshold: float = 0.92,
         shadow: bool = False, tag: str | None = None,
         routing: bool = False, cheap_model: str | None = None,
         capable_model: str | None = None, routing_threshold: float = 0.5,
         compression: bool = False, audit_log: Any = None,
         compression_query: bool = True, compression_ratio: float = 0.6,
         compression_eval: Any = None, compression_relevance_fn: Any = None,
         verify: bool = True, verify_threshold: float = 0.5,
         verify_model: str = "cross-encoder/quora-distilroberta-base",
         eval_loop: Any = None,
         cascade: bool = False, cascade_cheap: str | None = None,
         cascade_capable: str | None = None, cascade_verify: bool = False,
         cascade_verify_model: str | None = None) -> Any:
    """Return a proxy over an OpenAI client with engine optimization on
    `chat.completions.create`.

    Args:
        semantic:  enable the semantic (near-duplicate) cache. Default on.
        semantic_threshold:  similarity cutoff for a semantic hit.
        shadow:  measure-only — compute what WOULD be saved without serving any
                 cached response. The honest way to quantify value on live
                 traffic before trusting the cache.
        tag:  optional label for cost/usage breakdowns (e.g. a team or feature).
        routing:  enable cheap->capable model routing. OFF by default because it
                  changes which model serves a call. When on, easy prompts go to
                  cheap_model and hard ones to capable_model.
        cheap_model / capable_model:  the two tiers for routing. If routing is on
                  and these are unset, sensible OpenAI defaults are used.
        routing_threshold:  complexity (0..1) above which a prompt routes up.
        compression:  enable quality-gated prompt compression. OFF by default. The
                  SafeCompressor gate ships the original whenever compression would
                  be too extreme or drop query terms, so it never silently degrades.
        verify:  enable STAGE-2 cross-encoder verification of semantic hits. OFF by
                  default. When on, a semantic (Stage-1 cosine) candidate is only
                  served if the cross-encoder confirms the two prompts are genuinely
                  the same question — eliminating near-miss false hits ("read a
                  file" vs "delete a file"). Requires sentence-transformers; if
                  unavailable, degrades safely to Stage-1-only. Adds ~10-50ms per
                  hit candidate (the cost of trustworthy semantic caching).
        verify_threshold:  cross-encoder score cutoff to accept a match. Higher =
                  stricter. Calibrate on real pairs (see the quality suite sweep).
        verify_model:  cross-encoder checkpoint for verification.
    """
    router = None
    if routing:
        try:
            from tokeymeter.engines.optimization.router import Router
            router = Router(
                cheap_model=cheap_model or "gpt-4o-mini",
                capable_model=capable_model or "gpt-4o",
                threshold=routing_threshold,
            )
        except Exception:
            router = None  # graceful degradation if the module is unavailable
    compressor = None
    if compression:
        try:
            from tokeymeter.engines.optimization.safe_compress import SafeCompressor
            if compression_query:
                # Query-aware (LongLLMLingua-style): prune context by relevance to
                # the question, with a per-role budget. Best for RAG / Q&A / long
                # context — exactly where compression pays off.
                from tokeymeter.engines.optimization.query_compress import QueryAwareCompressor
                inner_c = QueryAwareCompressor(
                    query=None,  # filled per-call from the last user message
                    context_target_ratio=compression_ratio,
                    relevance_fn=compression_relevance_fn,  # opt-in semantic tier
                )
                compressor = SafeCompressor(inner=inner_c)
                compressor._query_aware = True
            else:
                from tokeymeter.engines.optimization.salience import SalienceCompressor
                compressor = SafeCompressor(inner=SalienceCompressor())
        except Exception:
            compressor = None
    opts = dict(semantic=semantic, semantic_threshold=semantic_threshold,
                shadow=shadow, tag=tag, router=router, compressor=compressor,
                audit_log=audit_log, verify=verify,
                verify_threshold=verify_threshold, verify_model=verify_model,
                eval_loop=eval_loop,
                cascade=cascade, cascade_cheap=cascade_cheap,
                cascade_capable=cascade_capable, cascade_verify=cascade_verify,
                cascade_verify_model=cascade_verify_model,
                cheap_model=cheap_model, capable_model=capable_model,
                compression_eval=compression_eval,
                compression_relevance_fn=compression_relevance_fn)
    # Auto-detect an async client (AsyncOpenAI) and route to the async proxy so
    # adoption stays one line either way. Lazy import avoids an import cycle and
    # keeps the async module off the import path for sync-only users.
    try:
        from tokeymeter.engines.execution.integrations.openai_async import is_async_client, _AsyncClientProxy
        if is_async_client(client):
            return _AsyncClientProxy(client, **opts)
    except Exception:
        pass  # fail-safe: fall back to the sync proxy
    return _ClientProxy(client, **opts)
