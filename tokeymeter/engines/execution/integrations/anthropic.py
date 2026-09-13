"""
Tokeymeter drop-in for the Anthropic SDK.

One line to adopt:

    import anthropic
    from tokeymeter.engines.execution.integrations.anthropic import wrap

    client = wrap(anthropic.Anthropic())     # <-- the only change
    r = client.messages.create(
        model="claude-haiku-4",
        max_tokens=200,
        messages=[{"role": "user", "content": "hello"}],
    )

Same engine, same content-blind audit, same reconciliation against the
provider's own reported usage (`response.usage.input_tokens/output_tokens`).
Streaming and tool-use responses pass straight through, uncached.
"""
from __future__ import annotations
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.economics.usage import (
    set_queue_wait_ms, extract_queue_wait_ms as _extract_queue_wait_ms,
)

import hashlib
import threading
import json
from typing import Any

import tokeymeter as tk
from tokeymeter.engines.economics import reconcile


def _fingerprint(model: str, messages: Any, system: Any, **kw: Any) -> str:
    basis = {
        "model": model, "messages": messages, "system": system,
        "temperature": kw.get("temperature"), "top_p": kw.get("top_p"),
        "max_tokens": kw.get("max_tokens"), "tools": kw.get("tools"),
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


from tokeymeter.engines.execution.response_text import (
    anthropic_response_text as _anthropic_response_text)


def _messages_text(messages: Any, system: Any) -> str:
    parts = []
    if isinstance(system, str):
        parts.append(system)
    try:
        for m in messages or []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        parts.append(p["text"])
    except Exception:
        return ""
    return "\n".join(parts)


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


class _CachedMessages:
    """Wraps `client.messages` so `.create` is engine-optimized (stable cache)."""

    def __init__(self, inner, *, semantic, semantic_threshold, shadow, tag):
        self._inner = inner
        self._local = threading.local()
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
                _sc = SemanticCache(path=':memory:', threshold=semantic_threshold)
            except Exception:
                _sc = None  # fall back to exact-only if deps are absent

        @tk.cache(
            model="_default", store=self._store, semantic=semantic, semantic_cache=_sc,
            semantic_threshold=semantic_threshold,
            shadow=shadow, tag=tag, single_flight=True,
            namespace="tokeymeter.integrations.anthropic.messages",
            key_fn=lambda *_a, **_k: self._local.fingerprint,
            extract_text=lambda *_a, **_k: self._local.text,
            # The RESPONSE extractor, deliberately distinct from extract_text
            # above: that one yields the PROMPT for token estimation. Without
            # this the progress signal cannot score a provider object at all,
            # because its text form carries a per-call request id.
            extract_response_text=_anthropic_response_text,
        )
        def _cached():
            kwargs = self._local.kwargs
            model = kwargs.get("model", "_default")
            resp = self._inner.create(**kwargs)
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    set_reported_usage(getattr(u, "input_tokens", 0),
                                       getattr(u, "output_tokens", 0))
                qw = _extract_queue_wait_ms(resp)
                if qw is not None:
                    set_queue_wait_ms(qw)
                if u is not None:
                    reconcile.record(
                        model=model,
                        input_tokens=getattr(u, "input_tokens", 0) or 0,
                        output_tokens=getattr(u, "output_tokens", 0) or 0,
                    )
            except Exception:
                pass
            return resp

        self._cached = _cached

    def create(self, *args, **kwargs):
        if kwargs.get("stream") or args:
            return self._inner.create(*args, **kwargs)
        model = kwargs.get("model", "_default")
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        self._local.kwargs = kwargs
        self._local.text = _messages_text(messages, system)
        _fp_kw = {k: v for k, v in kwargs.items() if k not in ('model', 'messages', 'system')}
        self._local.fingerprint = _fingerprint(model, messages, system, **_fp_kw)
        # Announce the model for THIS call. The decorator was bound with a
        # placeholder because a wrapped client cannot know the model until the
        # caller picks one; without this the placeholder is what the allowlist,
        # the price and the record all see.
        from tokeymeter.decorator import set_per_call_model, reset_per_call_model
        _model_token = set_per_call_model(kwargs.get("model"))
        try:
            return self._cached()
        except Exception as exc:
            if _is_deliberate_refusal(exc):
                raise
            return self._inner.create(**kwargs)
        finally:
            reset_per_call_model(_model_token)


class _ClientProxy:
    def __init__(self, client: Any, **opts: Any) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_messages", _CachedMessages(client.messages, **opts))

    @property
    def messages(self) -> _CachedMessages:
        return object.__getattribute__(self, "_messages")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_client"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_client"), name, value)


def wrap(client: Any, *, semantic: bool = True, semantic_threshold: float = 0.92,
         shadow: bool = False, tag: str | None = None) -> Any:
    """Wrap an Anthropic client so `messages.create` is engine-optimized."""
    return _ClientProxy(client, semantic=semantic, semantic_threshold=semantic_threshold,
                        shadow=shadow, tag=tag)
