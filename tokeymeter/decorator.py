"""
The Tokeymeter decorators (v0.5).

@tokeymeter.cache(...)         — wraps sync OR async functions. Auto-detects.
@tokeymeter.cache_stream(...)  — wraps async generator functions.

Pipeline on every call:
  1. Exact-match cache (Method 5).      ~10 µs.
  2. Semantic cache (opt-in, Method 1). ~5-10 ms.
  3. Real function call.                The only "real" cost.

v0.5 additions on top of v0.4:
  - **Events**: every lookup emits a CacheEvent to subscribers.
  - **Shadow mode**: cache logic runs and records would-be hits, but
    the real function ALWAYS executes. Lets cautious teams adopt with
    zero behavior change for a measurement period.
  - **Redactor**: a callable that scrubs PII from prompts before they
    hit the cache key, the semantic embedding, the storage backend,
    and the event preview.
  - **Tag**: a workload label that flows into events and the savings
    report (`tag="support"`, `tag="rag"`, etc).

Every layer fails open: subscribers, redactors, encoders, stores — any
of them can raise and the wrapped call still runs correctly.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import os
import time
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

from . import events as _events
from .compression import Compressor, CompressionResult, safe_compress
from .envelope import unwrap, wrap
from .envelope import meta as _env_meta
from .memory import ConversationMemory
from .pricing import estimate_cost, estimate_cost_with_source, estimate_tokens
from .identity import get_principal as _get_principal
from .engines.execution.endpoint import resolve as _resolve_endpoint
from .engines.execution.endpoint import _validate as _validate_endpoint
from . import keys as _keys
from .usage import consume_reported_usage as _consume_reported
from .usage import peek_reported_usage as _peek_reported
from .usage import consume_queue_wait_ms as _consume_queue_wait
from .savings import CallRecord, _record
from .savings import build_call_record as _build_call_record
from .engines.execution import task as _task
from . import overhead as _overhead
from .storage import MemoryStore, SQLiteStore, _SF_MISSING
from .policy import get_security_policy, SecurityPolicyError
from .utils import make_cache_key

log = logging.getLogger("tokeymeter")

# ============ Module-level defaults ============

_default_store: Optional[Any] = None
_default_semantic_cache: Optional[Any] = None
_warned_semantic_missing = False
# Guards lazy creation of the default singletons. Without it, a concurrent
# first-access burst constructs multiple store instances (each with its own
# in-memory single-flight state), so the first stampede fails to collapse.
_default_init_lock = __import__("threading").Lock()

# --- High-stakes / "do not optimize" mode ---
# When active, EVERY optimization is bypassed for the call: no cache serve
# (exact or semantic), no single-flight, no compression. The wrapped function
# always executes fresh with the prompt UNALTERED, and the result is NOT written
# to any cache (so a high-stakes answer can never later be served as a normal
# hit). The call is STILL recorded to the audit ledger — you keep provability
# without optimization. Use for irreversible / high-consequence calls.
import contextvars as _contextvars
import contextlib as _contextlib

_no_optimize_flag: "_contextvars.ContextVar[bool]" = _contextvars.ContextVar(
    "tokeymeter_no_optimize", default=False
)


@_contextlib.contextmanager
def no_optimize():
    """Context manager: bypass all Tokeymeter optimization for calls made inside.

        with tokeymeter.no_optimize():
            answer = ask(critical_prompt)   # fresh model call, prompt untouched

    Composes with @tokeymeter.cache(...) and is contextvar-based, so it is safe across
    threads and asyncio tasks. Still records to the audit ledger.
    """
    token = _no_optimize_flag.set(True)
    try:
        yield
    finally:
        _no_optimize_flag.reset(token)


def _is_high_stakes(high_stakes: bool) -> bool:
    return bool(high_stakes) or _no_optimize_flag.get()


# --- Context-lineage protection for long tasks ---
# A "lineage" is a logical task/conversation boundary. When set, cache entries
# are partitioned by lineage: a lookup in lineage A can never return a value
# produced in lineage B, and fuzzy semantic serving is disabled (the semantic
# store is not lineage-partitioned, so cross-lineage bleed is conservatively
# prevented). Prevents context from one task silently leaking into another.
_lineage_var: "_contextvars.ContextVar[Optional[str]]" = _contextvars.ContextVar(
    "tokeymeter_lineage", default=None
)


@_contextlib.contextmanager
def lineage(lineage_id: str):
    """Context manager: isolate all cached/semantic serving to this lineage.

        with tokeymeter.lineage(task_id):
            step1 = ask(...)   # cannot receive any answer from another lineage

    Contextvar-based (thread/async safe). Composes with @tokeymeter.cache(...).
    """
    if lineage_id is None:
        raise ValueError("lineage_id must not be None")
    token = _lineage_var.set(str(lineage_id))
    try:
        yield
    finally:
        _lineage_var.reset(token)


def _resolve_lineage(static_lineage: Optional[str]) -> Optional[str]:
    return static_lineage if static_lineage is not None else _lineage_var.get()


# --- Tenant isolation (multi-tenant safety) ---
# A "tenant" is a hard customer/organization boundary. When set, the cache key is
# partitioned by tenant so one tenant can NEVER receive another tenant's cached
# answer, even for an identical prompt+model on the same function and store. Like
# lineage, an active tenant also disables fuzzy semantic serving, because the
# semantic store is not tenant-partitioned and cross-tenant fuzzy hits would leak.
# Resolution precedence: explicit tenant= on @cache  >  tenant_scope() contextvar.
_tenant_var: "_contextvars.ContextVar[Optional[str]]" = _contextvars.ContextVar(
    "tokeymeter_tenant", default=None
)


@_contextlib.contextmanager
def tenant_scope(tenant_id: str):
    """Context manager: bind all cached/semantic serving to this tenant.

        with tokeymeter.tenant_scope(request.tenant_id):
            answer = ask(prompt)   # can never return another tenant's cached value

    The typical multi-tenant pattern is to enter this scope per request (e.g. in
    middleware) so every @cache call inside is automatically tenant-isolated.
    """
    if tenant_id is None:
        raise ValueError("tenant_id must not be None")
    token = _tenant_var.set(str(tenant_id))
    try:
        yield
    finally:
        _tenant_var.reset(token)


def _resolve_tenant(static_tenant: Optional[str]) -> Optional[str]:
    return static_tenant if static_tenant is not None else _tenant_var.get()


# Cache-namespace isolation. The default namespace is the decorated function's
# identity (module.qualname), which is stable across processes so the SAME
# function deployed on many pods can share a cache. But dynamically-generated
# functions (e.g. a factory that decorates a handler in a loop, or per-tenant
# handlers) share one qualname — naively they would collide and serve each
# other's cached answers. So we DETECT that collision: the first function object
# to claim a qualname keeps the clean, cross-process-stable namespace; a
# DIFFERENT object claiming the same qualname is auto-disambiguated, which
# preserves isolation (no silent bleed) at the cost of cross-process sharing for
# those generated functions — the correct trade, since such functions have no
# stable shared identity anyway. For intentional sharing or multi-tenant keys,
# callers pass an explicit namespace=.
_auto_ns_lock = __import__("threading").Lock()
_auto_ns_owner: dict = {}  # base_namespace -> id() of the first claiming function
_entry_point_id_cache: Optional[str] = None


def _entry_point_id() -> str:
    """A short, stable identifier for the running entry point (script), used to
    disambiguate "__main__"-defined functions across different scripts that share
    the default on-disk cache. Stable across restarts of the same script; differs
    between different scripts. Best-effort and never raises."""
    global _entry_point_id_cache
    if _entry_point_id_cache is not None:
        return _entry_point_id_cache
    ident = "anon"
    try:
        import sys as _sys
        main_mod = _sys.modules.get("__main__")
        path = getattr(main_mod, "__file__", None) or (_sys.argv[0] if _sys.argv else None)
        if path:
            ident = os.path.abspath(path)
        else:
            # interactive/REPL/notebook: no stable file — fall back to a per-process
            # id so two REPLs don't share, accepting no cross-restart persistence here.
            ident = f"pid{os.getpid()}"
    except Exception:
        ident = "anon"
    digest = hashlib.sha256(ident.encode("utf-8", "replace")).hexdigest()[:12]
    _entry_point_id_cache = digest
    return digest


def _resolve_namespace(func: Callable, namespace: Optional[str], shared_namespace: bool) -> Optional[str]:
    if namespace is not None:
        return str(namespace)
    if shared_namespace:
        return None
    qual = getattr(func, "__qualname__", getattr(func, "__name__", "fn"))
    module = getattr(func, "__module__", "?")
    # Functions defined at a script's top level all live in module "__main__", so
    # two DIFFERENT scripts sharing the default on-disk cache.db would collide on
    # e.g. "__main__.ask" and serve each other's answers. Salt "__main__" with a
    # stable entry-point identity: distinct scripts get distinct namespaces, while
    # the SAME script across restarts stays stable (persistence preserved).
    if module == "__main__":
        module = f"__main__[{_entry_point_id()}]"
    base = f"{module}.{qual}"
    fid = id(func)
    with _auto_ns_lock:
        owner = _auto_ns_owner.get(base)
        if owner is None:
            _auto_ns_owner[base] = fid
            return base
        if owner == fid:
            return base  # same object re-decorated -> keep stable namespace
        return f"{base}#{fid:x}"  # different object, same qualname -> disambiguate


from . import _fidelity  # extracted compression-fidelity unit (live-state accessors)
from ._fidelity import (  # re-exported for public API + back-compat imports
    set_compression_max_reduction,
    set_fidelity_circuit_breaker,
    compression_breaker_state,
)


# --- Event prompt-preview policy ---
# Events carry a short prompt preview for observability. By default this is the
# first ~200 chars ("full"), which can contain sensitive text. Policy lets you
# tighten it globally; high-stakes calls NEVER emit plaintext regardless.
#   "full"    -> first ~200 chars (opt-in; for local debugging only)
#   "hashed"  -> non-reversible "sha256:<12hex>" correlation token, no content
#   "omit"    -> None (no preview at all)
# Default is "hashed": event subscribers get a stable correlation token, never
# raw prompt content — consistent with the content-blind/local-first guarantee.
# Opt into "full" explicitly via set_event_preview_policy("full") for local dev.
_PREVIEW_POLICY: str = "hashed"


def set_event_preview_policy(policy: str) -> None:
    """Set how much of the prompt appears in emitted events: 'full' | 'hashed'
    | 'omit'. High-stakes / no_optimize calls are always capped at 'hashed'
    (never plaintext), so the sensitive mode cannot leak content via events."""
    global _PREVIEW_POLICY
    if policy not in ("full", "hashed", "omit"):
        raise ValueError("policy must be 'full', 'hashed', or 'omit'")
    _PREVIEW_POLICY = policy


# --- Runtime guards ---
# Cheap, fail-safe boundary checks that ENFORCE the core invariants at runtime
# (not just in tests). On a violation they take the SAFE action and log loudly
# via log.error — they never raise, so a guard firing degrades gracefully rather
# than crashing a production call. Toggle off only if you must shave the checks.
_RUNTIME_GUARDS: bool = True


def set_runtime_guards(enabled: bool) -> None:
    """Enable/disable runtime invariant guards (default on). When on, the core
    guarantees are enforced at the code boundary, not merely tested."""
    global _RUNTIME_GUARDS
    _RUNTIME_GUARDS = bool(enabled)


def enterprise_defaults(
    *,
    redactor: "Optional[Callable[[str], str]]" = None,
    require_encryption: bool = False,
    require_keyed_cache: bool = False,
    require_nonrepudiable_audit: bool = False,
    require_audit_durability: bool = False,
) -> dict:
    """Flip Tokeymeter to a safe-by-default ENTERPRISE posture in one call.

    The defaults are tuned for developer ergonomics; enterprises want the safe
    posture without wiring each knob by hand. This bundles them, and — crucially
    — is immediately *satisfiable*: it never leaves you in a state where every
    call fails closed for lack of setup.

    Always applied (safe and immediately satisfiable):
      - event previews -> "hashed": telemetry carries a correlation token, never
        raw prompt content.
      - runtime invariant guards -> on.
      - a redactor is ensured: uses `redactor=` if given, else the built-in
        DefaultRedactor when none is configured — then SecurityPolicy.require_redaction
        is enabled, so PII is stripped or the call fails closed (never leaks).

    Opt-in (enable once the corresponding material is configured; otherwise
    construction will correctly refuse rather than run unsafely):
      - require_encryption: shared/persistent cache VALUES must be encrypted.
      - require_keyed_cache: distributed cache keys must be HMAC'd.
      - require_nonrepudiable_audit: the audit ledger must use an asymmetric signer.

    Note: cache namespacing is isolation-by-default already (per-function); no
    flag needed. For MULTI-TENANT services, wrap each request in
    tokeymeter.tenant_scope(tenant_id) (or pass tenant= on @cache) so one tenant
    can never receive another's cached answer. Idempotent. Returns the active posture.
    """
    global _default_redactor
    set_event_preview_policy("hashed")
    set_runtime_guards(True)
    # Make require_redaction satisfiable: ensure a redactor is present.
    if redactor is not None:
        _default_redactor = redactor
    elif _default_redactor is None:
        try:
            from .privacy import DefaultRedactor
            _default_redactor = DefaultRedactor()
        except Exception as e:
            log.debug("tokeymeter: enterprise_defaults could not install a default redactor: %s", e)
    from .policy import set_security_policy
    policy_kwargs = {"require_redaction": True}
    if require_encryption:
        policy_kwargs["require_encryption"] = True
    if require_keyed_cache:
        policy_kwargs["require_keyed_cache"] = True
    if require_nonrepudiable_audit:
        policy_kwargs["require_nonrepudiable_audit"] = True
    if require_audit_durability:
        policy_kwargs["require_audit_durability"] = True
    policy = set_security_policy(**policy_kwargs)
    return {
        "event_preview_policy": _PREVIEW_POLICY,
        "runtime_guards": _RUNTIME_GUARDS,
        "redactor_configured": _default_redactor is not None,
        "security_policy": policy.describe(),
    }


def _make_preview(prompt_text: Optional[str], high_stakes: bool = False) -> Optional[str]:
    """Build the event preview honoring policy. High-stakes never returns
    plaintext (capped at 'hashed'); 'omit'/empty returns None.

    Runtime guard: regardless of policy logic above, a high-stakes preview that
    is not a hash token is scrubbed — an independent last line of defense against
    a future refactor reintroducing a leak."""
    if not prompt_text:
        return None
    policy = _PREVIEW_POLICY
    if high_stakes and policy == "full":
        policy = "hashed"        # the sensitive mode must not emit plaintext
    if policy == "omit":
        return None
    if policy == "hashed":
        try:
            digest = hashlib.sha256(prompt_text.encode("utf-8", "replace")).hexdigest()[:12]
            return f"sha256:{digest}"
        except Exception:
            return None
    # Runtime guard: an independent last line of defense. If we are about to
    # return plaintext for a high-stakes call, scrub it to a hash and log loudly.
    if _RUNTIME_GUARDS and high_stakes:
        log.error("tokeymeter GUARD: high_stakes preview reached plaintext path; "
                  "scrubbing to hash (invariant: no_sensitive_preview_leak).")
        try:
            digest = hashlib.sha256(prompt_text.encode("utf-8", "replace")).hexdigest()[:12]
            return f"sha256:{digest}"
        except Exception:
            return None
    return prompt_text[:200]
_default_redactor: Optional[Callable[[str], str]] = None  # global redactor override

# Single-flight: cache_key -> Future. Per-process, per-event-loop in practice.
_inflight_async: dict = {}

# Compression fidelity verification log. Populated only when verify_rate>0.
# Each entry: {"timestamp", "tag", "similarity", "tokens_before", "tokens_after"}
_compression_verifications: list = []
_compression_verifications_lock = __import__("threading").Lock()


def _record_verification(rec: dict) -> None:
    """Append a verification record. Never raises."""
    try:
        with _compression_verifications_lock:
            _compression_verifications.append(rec)
            # Cap memory: keep the most recent 10k entries
            if len(_compression_verifications) > 10_000:
                del _compression_verifications[: len(_compression_verifications) - 10_000]
    except Exception:
        pass


def compression_verification_log() -> list:
    """Return a copy of the in-memory verification log."""
    with _compression_verifications_lock:
        return list(_compression_verifications)


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]. Zero deps, deterministic.

    Crude but effective for "did these two outputs say roughly the same thing?".
    For high-stakes uses, override with a semantic similarity callback.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    if a == b:
        return 1.0
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _get_default_store() -> Any:
    global _default_store
    if _default_store is None:
        with _default_init_lock:
            if _default_store is None:  # double-checked under the lock
                try:
                    _default_store = SQLiteStore()
                except Exception as e:
                    log.debug("tokeymeter: SQLite unavailable, falling back to memory: %s", e)
                    _default_store = MemoryStore()
    return _default_store


def _get_default_semantic_cache(threshold: float = 0.92) -> Optional[Any]:
    global _default_semantic_cache, _warned_semantic_missing
    if _default_semantic_cache is not None:
        return _default_semantic_cache
    try:
        from .semantic import SemanticCache, is_available
    except ImportError as e:
        log.debug("tokeymeter: semantic module import failed: %s", e)
        return None
    if not is_available():
        if not _warned_semantic_missing:
            warnings.warn(
                "tokeymeter: semantic caching requested but dependencies not installed. "
                "Falling back to exact-match only. "
                "Install with: pip install tokeymeter[semantic]",
                RuntimeWarning,
                stacklevel=3,
            )
            _warned_semantic_missing = True
        return None
    try:
        with _default_init_lock:
            if _default_semantic_cache is None:  # double-checked
                _default_semantic_cache = SemanticCache(threshold=threshold)
        return _default_semantic_cache
    except Exception as e:
        log.debug("tokeymeter: default semantic cache init failed: %s", e)
        return None


def set_default_store(store: Any) -> None:
    global _default_store
    _default_store = store


def set_default_semantic_cache(cache: Any) -> None:
    global _default_semantic_cache
    _default_semantic_cache = cache


def set_default_redactor(redactor: Optional[Callable[[str], str]]) -> None:
    """Apply a redactor to ALL @tokeymeter.cache calls unless they override locally.

    Set to None to disable.
    """
    global _default_redactor
    _default_redactor = redactor


# ============ Internal helpers ============

def _resolve_redactor(local: Optional[Callable]) -> Optional[Callable]:
    return local if local is not None else _default_redactor


def _resolve_redactor_enforced(local: Optional[Callable]) -> Optional[Callable]:
    """Resolve the redactor and enforce SecurityPolicy.require_redaction.

    If the policy requires redaction and none resolves, refuse to proceed — PII
    would otherwise reach the model/cache/audit unredacted."""
    red = _resolve_redactor(local)
    if red is None:
        try:
            from .policy import get_security_policy, SecurityPolicyError
            if get_security_policy().require_redaction:
                raise SecurityPolicyError(
                    "SecurityPolicy.require_redaction is enabled but no redactor is "
                    "configured for this call. Pass redactor=... or set a default via "
                    "tokeymeter.set_default_redactor(...), so PII is stripped before egress.")
        except SecurityPolicyError:
            raise
        except Exception:
            pass
    return red


def _on_redactor_failure(text: str, exc: Exception) -> str:
    """Decide what to do when a redactor cannot produce redacted text.

    Strict policy (SecurityPolicy.require_redaction): refuse to proceed — raising
    SecurityPolicyError is the ONLY safe outcome, because returning the original
    text would leak exactly the PII the operator demanded be stripped. This
    mirrors the resolve-time contract (a missing required redactor already
    raises), now extended to runtime failure.

    Permissive policy (the default): fail OPEN — return the original text so a
    broken redactor never takes down the app. Unchanged behavior; only triggers
    on the exceptional path where the redactor actually fails, so a working
    redactor leaves prompt/output quality entirely untouched.
    """
    try:
        if get_security_policy().require_redaction:
            raise SecurityPolicyError(
                "SecurityPolicy.require_redaction is enabled but the redactor failed "
                "to return redacted text; refusing to send unredacted content "
                "(fail-closed). Fix or replace the redactor."
            ) from exc
    except SecurityPolicyError:
        raise
    except Exception:
        pass  # policy lookup itself failed — fall through to permissive behavior
    log.debug("tokeymeter: redactor failed, falling open (permissive policy): %s", exc)
    try:
        from .degraded import emit_degraded
        emit_degraded("redactor", exc)
    except Exception:
        pass
    return text


def _safe_redact(redactor: Optional[Callable], text: str) -> str:
    """Run the redactor on one string.

    On success, returns the redacted string (no behavior change — quality is
    preserved whenever the redactor works). On failure (raises OR returns a
    non-string), defers to _on_redactor_failure, which fails CLOSED under strict
    policy and OPEN otherwise.
    """
    if redactor is None or not isinstance(text, str) or not text:
        return text
    try:
        out = redactor(text)
    except SecurityPolicyError:
        raise  # never swallow a policy violation
    except Exception as e:
        return _on_redactor_failure(text, e)
    if isinstance(out, str):
        return out
    # A redactor that returns a non-string has not redacted anything — under
    # strict policy that is just as unsafe as a crash.
    return _on_redactor_failure(text, TypeError("redactor returned a non-string"))


_REDACT_MAX_DEPTH = 25  # guards against pathological / cyclic nesting


def _redact_deep(value: Any, redactor: Optional[Callable], _depth: int = 0) -> Any:
    """Recursively redact string leaves inside dict / list / tuple structures.

    Single source of truth for both argument redaction (before keying, semantic
    embedding, the model call, events, and storage) and response redaction. It:
      - redacts only string *values*; mapping keys are preserved verbatim so the
        structure's shape is identical;
      - preserves container types (dict stays dict, tuple stays tuple);
      - is bounded by _REDACT_MAX_DEPTH so malformed/deeply nested or cyclic input
        can never blow the stack;
      - never raises — on any error the original value is returned (fail-open at
        the value level; strict-policy enforcement is layered above this).
    """
    if redactor is None:
        return value
    if _depth > _REDACT_MAX_DEPTH:
        return value  # refuse to recurse further; leave the subtree untouched
    try:
        if isinstance(value, str):
            return _safe_redact(redactor, value)
        if isinstance(value, list):
            return [_redact_deep(v, redactor, _depth + 1) for v in value]
        if isinstance(value, tuple):
            return tuple(_redact_deep(v, redactor, _depth + 1) for v in value)
        if isinstance(value, dict):
            return {k: _redact_deep(v, redactor, _depth + 1) for k, v in value.items()}
    except SecurityPolicyError:
        # Strict-policy fail-closed must propagate — never degrade to returning
        # the unredacted value.
        raise
    except Exception:
        return value
    return value  # non-text scalars (int/float/bool/None/objects) pass through


def _redact_args(args, kwargs, redactor: Optional[Callable]) -> Tuple[tuple, dict]:
    """Apply the redactor to every arg / kwarg, recursing into nested structures.

    Realistic LLM inputs are nested (e.g. messages=[{"role":..,"content":..}]),
    so redaction MUST descend into dict/list/tuple — not just top-level strings —
    before the content is keyed, embedded, sent to the model, emitted to events,
    or stored. Redaction falls open per-value (the original value is kept).
    """
    if redactor is None:
        return args, kwargs
    new_args = tuple(_redact_deep(a, redactor) for a in args)
    new_kwargs = {k: _redact_deep(v, redactor) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _redact_result(value: Any, redactor: Optional[Callable]) -> Any:
    """Best-effort response redaction. Never raises.

    Redacts string values within nested mappings/sequences while preserving keys
    and structure. Delegates to the shared recursive walker so request and
    response redaction can never diverge.
    """
    return _redact_deep(value, redactor)


def _count_redactions(redactor: Optional[Callable], args, kwargs) -> int:
    """Count PII redactions across all args/kwargs, recursing into nested
    structures (for audit evidence).

    Only works if the redactor exposes a `count_redactions(text) -> int`
    method (DefaultRedactor does). For arbitrary callables we can't know
    the count, so we return 0. Mirrors _redact_deep's traversal so the count
    matches what is actually redacted on nested inputs. Never raises.
    """
    if redactor is None:
        return 0
    counter = getattr(redactor, "count_redactions", None)
    if not callable(counter):
        return 0

    def _count_deep(value, _depth=0):
        if _depth > _REDACT_MAX_DEPTH:
            return 0
        if isinstance(value, str):
            return int(counter(value))
        if isinstance(value, (list, tuple)):
            return sum(_count_deep(v, _depth + 1) for v in value)
        if isinstance(value, dict):
            return sum(_count_deep(v, _depth + 1) for v in value.values())
        return 0

    try:
        total = sum(_count_deep(a) for a in args)
        total += sum(_count_deep(v) for v in kwargs.values())
        return total
    except (TypeError, ValueError, AttributeError):
        return 0  # exotic / uncountable arg types — expected, count as 0
    except Exception as e:
        # Unexpected: a bug in token counting would silently skew savings; surface
        # it while still failing safe (cost just isn't counted for this call).
        _internal_error("token_count", e)
        return 0


def _extract_prompt_text(
    args, kwargs, prompt_arg: Optional[Union[str, int]], extract_text: Optional[Callable]
) -> str:
    if extract_text is not None:
        try:
            return str(extract_text(args, kwargs) or "")
        except Exception:
            return ""
    if isinstance(prompt_arg, str) and prompt_arg in kwargs:
        v = kwargs[prompt_arg]
        if isinstance(v, str):
            return v
    if isinstance(prompt_arg, int) and 0 <= prompt_arg < len(args):
        v = args[prompt_arg]
        if isinstance(v, str):
            return v
    if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
        return kwargs["prompt"]
    for a in args:
        if isinstance(a, str):
            return a
    return ""


def _text_for_token_estimate(args, kwargs, extract_text):
    if extract_text is not None:
        try:
            return str(extract_text(args, kwargs) or "")
        except Exception:
            return ""
    parts = [str(a) for a in args]
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    return " ".join(parts)


def _try_exact_lookup(args, kwargs, key_fn, model, backend, lineage=None, namespace=None, tenant=None) -> Tuple[Optional[str], Any, Optional[dict]]:
    """Return (cache_key, unwrapped_value, envelope_meta).

    value is None on miss/expired. envelope_meta is the token metadata the
    ORIGINAL computation stamped into the envelope ({"in", "out", "src"}), or
    None for legacy entries / misses — the S1.1 contract that lets a hit
    record report the true avoided token volume instead of re-estimating.

    Partitioning, applied as key prefixes so they compose:
      - `namespace`: the workload identity (by default the decorated function's
        module+qualname). Prevents one function/tool/tenant from ever receiving
        another's cached answer just because the prompt+model happen to match.
        Pass shared_namespace=True (or an explicit namespace=) to opt into
        intentional cross-function sharing.
      - `lineage`: a logical task/conversation boundary — a lookup in one lineage
        can never return a value stored under another.
    """
    try:
        if key_fn:
            key = key_fn(args, kwargs)
            if not isinstance(key, str):
                key = str(key)
        else:
            key = make_cache_key(args, kwargs, model=model)
        # Compose partitions as prefixes (applied to custom key_fn too, so it is
        # namespaced by default — opt out with shared_namespace=True).
        if namespace is not None:
            key = f"ns={namespace}::{key}"
        if lineage is not None:
            key = f"lin={lineage}::{key}"
        if tenant is not None:
            # Outermost prefix: tenant is the hardest boundary, so it dominates
            # the composed key and can never collide across tenants.
            key = f"t={tenant}::{key}"
    except Exception as e:
        # Key composition is pure string logic — a failure here is almost always
        # a Tokeymeter bug, not a runtime condition. Fail SAFE (treat as a miss so
        # we recompute, never serving a wrong hit) AND surface it so a silent
        # caching outage can't hide.
        log.debug("tokeymeter: key generation failed: %s", e)
        _internal_error("key_generation", e)
        return None, None, None
    try:
        raw = backend.get(key)
        return key, unwrap(raw), _env_meta(raw)
    except Exception as e:
        log.debug("tokeymeter: exact get failed: %s", e)
        try:
            from .degraded import emit_degraded
            emit_degraded("store.get", e)
        except Exception:
            pass
        return key, None, None


def _safe_exact_set(backend, key: Optional[str], value: Any) -> None:
    if key is None:
        return
    # Retention check at the single write point, so every path that could
    # persist a response is covered by one gate rather than by remembering to
    # check at each call site. A `never_cache` rule means the answer is not
    # written at all — not written-then-hidden.
    try:
        from tokeymeter.engines.governance import compliance as _c
        if _c.resolve_compliance().never_cache:
            return
    except Exception:
        pass
    try:
        backend.set(key, value)
    except Exception as e:
        log.debug("tokeymeter: exact set failed: %s", e)
        try:
            from .degraded import emit_degraded
            emit_degraded("store.set", e)
        except Exception:
            pass


def _dsf_begin(backend, cache_key, single_flight, shadow):
    """Begin distributed single-flight if the backend supports it.

    Returns (leader_token, follower_envelope):
      - If we acquired the lock (leader): (token, None) — caller computes.
      - If a peer already computed it (follower): (None, envelope) — caller
        serves the peer's result.
      - Otherwise (no DSF, disabled, or fail-open): (None, None) — caller
        computes normally.

    Fail-open everywhere: any error degrades to "compute it yourself", which
    is correct (occasionally redundant) rather than wrong.
    """
    if (not single_flight or shadow or cache_key is None
            or not hasattr(backend, "acquire_compute_lock")):
        return None, None
    # A `never_cache` context forbids response REUSE, not merely persistence.
    # Single-flight retains nothing — the in-flight entry dies with the call —
    # but it does hand caller B an answer generated for caller A, and an
    # officer who wrote "never cache PHI" meant responses here are not shared.
    # Declining costs one redundant upstream call; getting it wrong costs the
    # control, so this errs conservative deliberately.
    try:
        from tokeymeter.engines.governance import compliance as _c
        if _c.resolve_compliance().never_cache:
            return None, None
    except Exception:
        pass
    try:
        token = backend.acquire_compute_lock(cache_key)
        if token is not None:
            return token, None  # we are the leader
        # Follower: wait for the leader's stored result.
        envelope = backend.wait_for_result(cache_key)
        return None, envelope   # envelope may be None → caller computes
    except Exception as e:
        log.debug("tokeymeter: distributed single-flight begin failed (fail-open): %s", e)
        return None, None


def _dsf_release(backend, cache_key, token, value=_SF_MISSING) -> None:
    if token is None:
        return
    try:
        try:
            # In-process stores accept the computed value and publish it in
            # memory for waiting/late peers (avoids a store re-read race).
            backend.release_compute_lock(cache_key, token, value)
        except TypeError:
            # Backends without value support (e.g. Redis) coordinate via their
            # own shared store, so the value isn't needed here.
            backend.release_compute_lock(cache_key, token)
    except Exception as e:
        log.debug("tokeymeter: distributed single-flight release failed: %s", e)


def _resolve_semantic_layer(semantic, semantic_cache, semantic_threshold):
    if not semantic:
        return None
    if semantic_cache is not None:
        return semantic_cache
    return _get_default_semantic_cache(threshold=semantic_threshold)


def _internal_error(where: str, error: BaseException) -> None:
    """Surface an UNEXPECTED error from Tokeymeter's own logic.

    Exception policy (see docs/EXCEPTION_POLICY): I/O / backend / untrusted-input
    boundaries fail OPEN and emit a source-specific degraded event (store.get,
    redis_write, ...). Pure-logic paths handle their EXPECTED exceptions
    explicitly and fail SAFE (e.g. treat as a cache miss and recompute — never
    serve a wrong answer); an UNEXPECTED exception there is a programmer bug, so
    instead of vanishing at debug level it is surfaced as an `internal_error`
    degraded event. That lets a Tokeymeter bug be told apart from a backend/ops
    failure, while the call still succeeds. Never raises."""
    try:
        from .degraded import emit_degraded
        emit_degraded(f"internal_error:{where}", error)
    except Exception:
        pass


def _emit_compression_fallback(reduction: float, cap: float) -> None:
    """Surface a compression reduction-cap fallback as a degraded event, so the
    loss of compression savings (and a possibly mis-tuned compressor) is visible
    rather than silent."""
    try:
        from .degraded import emit_degraded
        emit_degraded(
            "compression_fallback",
            RuntimeError(f"reduction {reduction:.2f} exceeded cap {cap:.2f}; used original prompt"),
        )
    except Exception:
        pass


def _apply_compressor(args, kwargs, prompt_arg, extract_text, compressor, workload_key=None):
    """Apply a compressor to the prompt text in args/kwargs. Fail-open.

    Returns (new_args, new_kwargs, compression_result_or_none).

    If the fidelity circuit breaker is OPEN for this workload (measured fidelity
    has dropped), compression is withheld and the original prompt is used.
    """
    if compressor is None:
        return args, kwargs, None

    # Measured-fidelity circuit breaker: withhold compression for workloads whose
    # observed compressed-vs-original fidelity has fallen below threshold.
    if workload_key is not None and not _fidelity.get_breaker().should_compress(workload_key):
        return args, kwargs, None

    # Identify WHERE the prompt lives so we can write back the compressed form
    target: Optional[tuple] = None  # ("kwarg", name) or ("arg", index)
    prompt_text: Optional[str] = None

    if isinstance(prompt_arg, str) and prompt_arg in kwargs and isinstance(kwargs[prompt_arg], str):
        target = ("kwarg", prompt_arg)
        prompt_text = kwargs[prompt_arg]
    elif isinstance(prompt_arg, int) and 0 <= prompt_arg < len(args) and isinstance(args[prompt_arg], str):
        target = ("arg", prompt_arg)
        prompt_text = args[prompt_arg]
    elif "prompt" in kwargs and isinstance(kwargs["prompt"], str):
        target = ("kwarg", "prompt")
        prompt_text = kwargs["prompt"]
    else:
        # Try the first positional str arg
        for i, a in enumerate(args):
            if isinstance(a, str):
                target = ("arg", i)
                prompt_text = a
                break

    if target is None or prompt_text is None:
        # Can't safely identify the prompt — skip compression
        return args, kwargs, None

    cresult = safe_compress(compressor, prompt_text)

    # Only apply compression if it was safe AND it actually reduced size
    if not cresult.safe or cresult.after == cresult.before:
        return args, kwargs, cresult

    # Conservative-by-default fidelity floor: if compression removed an extreme
    # fraction of the prompt, that is a strong signal of content loss, so fall
    # back to the ORIGINAL prompt rather than risk a degraded answer. Tunable
    # via tokeymeter.set_compression_max_reduction(); set to 1.0 to disable the guard.
    try:
        before_tok = max(1, cresult.tokens_before)
        reduction = 1.0 - (cresult.tokens_after / before_tok)
    except (TypeError, AttributeError, ZeroDivisionError):
        reduction = 0.0  # malformed cresult — expected; treat as no reduction
    except Exception as e:
        _internal_error("reduction_calc", e)
        reduction = 0.0
    if reduction > _fidelity.get_max_reduction():
        log.debug("tokeymeter: compression reduction %.2f exceeds conservative cap %.2f; "
                  "falling back to original prompt.", reduction, _fidelity.get_max_reduction())
        _emit_compression_fallback(reduction, _fidelity.get_max_reduction())
        return args, kwargs, cresult  # withheld: original prompt used

    # Runtime guard: independent re-check at the point of no return. If the
    # reduction somehow exceeds the cap here (e.g. a future edit removed the
    # check above), withhold rather than apply over-aggressive compression.
    if _RUNTIME_GUARDS and reduction > _fidelity.get_max_reduction():
        log.error("tokeymeter GUARD: compression reduction %.2f exceeds cap %.2f at apply; "
                  "withholding (invariant: no_over_aggressive_compression).",
                  reduction, _fidelity.get_max_reduction())
        return args, kwargs, cresult

    if target[0] == "kwarg":
        new_kwargs = dict(kwargs)
        new_kwargs[target[1]] = cresult.after
        return args, new_kwargs, cresult
    else:  # ("arg", idx)
        new_args = list(args)
        new_args[target[1]] = cresult.after
        return tuple(new_args), kwargs, cresult


def _verify_should_sample(verify_rate: float) -> bool:
    """Decide whether to fire a compression-fidelity verification on this call.

    Uses random sampling. Cheap import (stdlib random).
    """
    if verify_rate <= 0.0:
        return False
    if verify_rate >= 1.0:
        return True
    import random as _r
    return _r.random() < verify_rate


def _resolve_miss_meta(args, kwargs, extract_text, result, model):
    """Compute the token meta to stamp into a miss's cache envelope so a later
    hit can report the TRUE avoided token volume (S1.1).

    Mirrors _record_and_emit's own resolution EXACTLY — reported usage if the
    wrapper set it (peeked, never consumed here; the record still consumes),
    else the chars/4 estimate on the same text/result. Divergence between this
    and the record path would reintroduce the very inconsistency S1.1 fixes, so
    both read the same primitives. Returns {"in", "out", "src"}, or None if the
    counts can't be resolved (e.g. an uncountable arg/result) — a None meta
    means the later hit falls back to estimation, exactly as before S1.1. This
    must NEVER raise: it runs on the hot miss-write path and a failure here
    (including text extraction on an exotic arg) would break the user's call for
    a measurement nicety. Text extraction is therefore done INSIDE the guard.
    """
    try:
        reported = _peek_reported()      # peek: the record consumes, not us
        if reported is not None:
            return {"in": int(reported[0]), "out": int(reported[1]),
                    "src": "reported"}
        text = _text_for_token_estimate(args, kwargs, extract_text)
        return {"in": estimate_tokens(text),
                "out": estimate_tokens(str(result)), "src": "estimated"}
    except Exception:
        return None


def _response_fp_safe(result, extract_response_text=None):
    """Non-reversible digest of the response CONTENT, or None. Never raises.

    Deliberately NOT given the decorator's `extract_text`: that extracts the
    PROMPT for token estimation (the shipped SDK integrations set it to the
    request messages), and hashing a growing prompt while calling it a response
    fingerprint made every identical failure look novel.
    """
    try:
        return _task.fingerprint_response(result, extract_response_text)
    except Exception:
        return None


def _compliance_fields_safe():
    """(data_class, region, policy_rules) for the record, or Nones.

    The DECLARED context is read straight from its context vars, never through
    the resolved policy decision. Reading it through the decision meant these
    fields were only recorded when a compliance policy already existed — so a
    platform owner could not plan a residency rule against ungoverned history,
    which is precisely when they need to. What the application declared is a
    fact about the call and is recorded whether or not anything governs it.

    `policy_rules` is different: it names the rules that actually applied, so
    it is empty when nothing did.
    """
    data_class = region = None
    rules = None
    try:
        from tokeymeter.engines.governance import compliance as _c
        data_class = _c.current_data_class()
        region = _c.current_region()
    except Exception:
        pass
    try:
        from tokeymeter.engines.governance import compliance as _c
        d = _c.resolve_compliance()
        if d.rules and (d.record_outcome or d.restricts_models):
            rules = ",".join(d.rules)
    except Exception:
        pass
    return (data_class, region, rules)


def _agent_safe():
    """The bound agent, or None. Never raises."""
    try:
        return _task.current_agent()
    except Exception:
        return None


def _task_id_safe():
    """The bound task id, or None. Never raises — the record path must not be
    breakable by task bookkeeping."""
    try:
        return _task.current_task_id()
    except Exception:
        return None


def _fingerprint_safe(cache_key):
    """Non-reversible digest of the cache key, or None. Never raises."""
    try:
        return _task.fingerprint_of(cache_key)
    except Exception:
        return None


_PER_CALL_MODEL = _contextvars.ContextVar(
    "tokeymeter_per_call_model", default=None)


def set_per_call_model(name):
    """Announce the model THIS call is using, for wrappers that cannot know it
    at decoration time.

    A wrapped SDK client is decorated once, but the caller picks a model per
    request, so the decorator is bound with a placeholder. Without this channel
    the placeholder is what everything downstream sees, and the consequences
    are not cosmetic: cost is computed from a generic fallback price (measured
    6.5x high against a real OpenAI invoice), chargeback collapses every model
    into one bucket, and a model allowlist compares the placeholder against the
    permitted names and refuses EVERY call, including the approved one.

    A ContextVar rather than an attribute: concurrent tasks in one process must
    not see each other's model, and asyncio propagates context automatically.

    This mechanism existed once, was lost in a refactor, and returned as a
    regression because nothing tested it. tests/test_per_call_model.py now does.
    """
    return _PER_CALL_MODEL.set(name if isinstance(name, str) and name else None)


def reset_per_call_model(token):
    try:
        _PER_CALL_MODEL.reset(token)
    except Exception:
        pass


def _effective_model(bound_model):
    """The per-call model when one was announced, else the decorated value."""
    return _PER_CALL_MODEL.get() or bound_model


def _compliance_gate(model):
    """Refuse the call if policy does not permit this model.

    Runs BEFORE the cache is consulted, deliberately. A cache hit still
    RETURNS data derived from that model, so serving one for a model the policy
    forbids would satisfy the letter of "we did not call it" and none of the
    intent. Returns the decision so the caller can honour `never_cache` without
    resolving twice.

    This is the one gate in this module that fails CLOSED — see
    engines/governance/compliance for why the inversion is kept this narrow.
    """
    try:
        from tokeymeter.engines.governance import compliance as _c
    except Exception:
        return None
    try:
        return _c.check_model(model)
    except _c.PolicyViolation:
        raise
    except Exception:
        return None


def _task_preflight(cache_key, is_hit=False):
    """Evaluate declared task ceilings BEFORE this call proceeds.

    Placed after the cache key is known but BEFORE a hit is served or a miss
    executes, deliberately: a loop served entirely from cache is still a loop,
    and halting only on the upstream path would never catch it. Checking
    afterwards would mean the spend already happened, which is the failure
    mode of every alert-based tool.

    Raises TaskLimitExceeded only on a real breach with enforcement on;
    everything else is swallowed.
    """
    try:
        _task.before_call(cache_key, is_hit)
    except _task.TaskLimitExceeded:
        raise
    except Exception:
        pass


def _record_and_emit(
    args, kwargs, result, hit, hit_type, model, extract_text, latency_ms,
    cache_key, prompt_text, shadow, tag, function_name=None, compression=None,
    pii_redactions=0, high_stakes=False,
    reported_override=None, pending_shadow_hits=None, endpoint=None,
    meta_override=None, extract_response_text=None,
):
    """Build a CallRecord (savings.jsonl) AND emit a CacheEvent (subscribers).

    Never raises.
    """
    try:
        # On a hit, no wrapped function ran, so latency_ms is pure meter overhead.
        # Record it for the doctor's p50/p95/p99 (near-zero-cost deque append).
        if hit:
            _overhead.record(latency_ms)
        # Runtime guard: a high-stakes call never caches, so it must not carry a
        # cache_key into the record/event. Scrub if it somehow does.
        if _RUNTIME_GUARDS and high_stakes and cache_key is not None:
            log.error("tokeymeter GUARD: high_stakes call carried a cache_key; scrubbing "
                      "(invariant: no_unbounded_optimization).")
            cache_key = None
        text = _text_for_token_estimate(args, kwargs, extract_text)
        endpoint_identity = _resolve_endpoint(endpoint)
        # Queue wait is a MISS-only measurement, consume-once. Hits didn't
        # queue (nothing executed), so they carry None by construction. We
        # consume on the real-call/miss record only — NOT inside the deferred
        # shadow-hit flush below (those are hits), so the value survives intact
        # for the miss record that follows.
        queue_wait_ms = None if hit else _consume_queue_wait()
        if reported_override is not None:
            reported = reported_override
        else:
            reported = None if hit else _consume_reported()
        # Deferred shadow-hit records (S0 provenance fix): flushed here, BEFORE
        # this real-call record, so ledger order stays hit-then-miss — and
        # stamped with the SAME provider-reported counts this invocation just
        # produced. The tokens a shadow hit would have avoided ARE the tokens
        # the real call actually spent; the chars/4 estimate has no business on
        # these records when the provider's truth exists in-invocation. The
        # endpoint is the same for the hit and its real call, so it threads
        # through too.
        if pending_shadow_hits:
            for _p in list(pending_shadow_hits):
                _record_and_emit(
                    args, kwargs, _p["result"], True, _p["hit_type"], model,
                    extract_text, _p["latency_ms"], cache_key, prompt_text,
                    True, tag, function_name=function_name,
                    compression=compression, pii_redactions=pii_redactions,
                    reported_override=reported, endpoint=endpoint, extract_response_text=extract_response_text,
                    meta_override=_p.get("meta"),
                )
            pending_shadow_hits.clear()
        if reported_override is not None:
            # S0-1: a shadow hit carries the REAL call's reported tokens (the
            # true avoided volume for shadow mode). This wins over any envelope
            # meta — the two agree for shadow-exact, and for shadow-semantic the
            # real call's own reported count is the authoritative measure.
            in_tok, out_tok = reported_override
            token_source = "reported"
        elif meta_override is not None:
            # S1.1: a LIVE hit recovers the original miss's true token counts
            # from the cache envelope. Report the ACTUAL avoided volume, not a
            # re-estimate of the cached value's size. token_source reflects how
            # the ORIGINAL computation measured those tokens.
            in_tok = int(meta_override.get("in", 0))
            out_tok = int(meta_override.get("out", 0))
            token_source = meta_override.get("src", "estimated")
        elif reported is not None:
            in_tok, out_tok = reported
            token_source = "reported"
        else:
            in_tok = estimate_tokens(text)
            out_tok = estimate_tokens(str(result))
            token_source = "estimated"
        cost, pricing_source = estimate_cost_with_source(model, in_tok, out_tok)
        principal = _get_principal()
        key_name = _keys.get_current_key()
        _keys.on_spend(key_name, 0.0 if hit else cost, hit, shadow)

        compression_ratio = None
        tokens_saved_via_compression = 0
        compression_method = None
        if compression is not None and compression.safe:
            compression_ratio = compression.ratio
            tokens_saved_via_compression = max(
                0, compression.tokens_before - compression.tokens_after
            )
            compression_method = compression.method
            try:    # T3.4: compression is a real context-provenance signal
                from . import context_passport as _cp
                _cp.emit(context_id=_cp.fingerprint(text),
                         source_type="prompt",
                         tokens_before=compression.tokens_before,
                         tokens_after=compression.tokens_after,
                         method=compression.method, model=model)
            except Exception:
                pass

        _record(_build_call_record(
            model=model,
            hit=hit,
            hit_type=hit_type,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost=cost,
            latency_ms=latency_ms,
            shadow=shadow,
            tag=tag,
            compression_ratio=compression_ratio,
            tokens_saved_via_compression=tokens_saved_via_compression,
            compression_method=compression_method,
            pricing_source=pricing_source,
            principal=principal,
            token_source=token_source,
            key_name=key_name,
            endpoint_identity=endpoint_identity,
            queue_wait_ms=queue_wait_ms,
            # Task boundary: WHICH unit of work this call belonged to, and a
            # non-reversible digest of the cache key so repeats inside one task
            # are detectable without ever reading a prompt. Both resolve to
            # None outside a task() block, exactly as principal/endpoint do.
            task_id=_task_id_safe(),
            agent=_agent_safe(),
            **dict(zip(('data_class', 'region', 'policy_rules'),
                       _compliance_fields_safe())),
            response_fingerprint=_response_fp_safe(result, extract_response_text),
            prompt_fingerprint=_fingerprint_safe(cache_key),
        ))
        # Fold this call into the task's running accounting. Cache hits count
        # toward call and repeat totals (a loop served from cache is still a
        # loop) but add no spend, because a hit cost nothing.
        _task.after_call(_fingerprint_safe(cache_key), cost, executed=not hit,
                         response_fp=_response_fp_safe(result, extract_response_text),
                         input_tokens=in_tok)

        extra = {}
        if compression_ratio is not None:
            extra["compression_ratio"] = compression_ratio
            extra["tokens_saved_via_compression"] = tokens_saved_via_compression
            extra["compression_method"] = compression_method
        if pii_redactions:
            extra["pii_redactions"] = int(pii_redactions)

        _evt = _events.CacheEvent(
            timestamp=time.time(),
            event_type="lookup_hit" if hit else "lookup_miss",
            hit=hit,
            hit_type=hit_type,
            model=model,
            cache_key=cache_key,
            prompt_preview=_make_preview(prompt_text, high_stakes),
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
            shadow=shadow,
            tag=tag,
            function_name=function_name,
            principal=principal,
            token_source=token_source,
            endpoint_identity=endpoint_identity,
            extra=extra,
        )
        _events.emit(_evt)
        # Private audit bus: unlike public events, this can receive the full
        # prompt text in-process so the audit ledger can HMAC it with its own
        # install secret without exposing plaintext to observability subscribers.
        try:
            from .audit import log as _audit_log
            _audit_log._emit_private(_evt, prompt_text)
        except Exception:
            pass
        # Control-plane: derive and dispatch the first-class DecisionRecord.
        try:
            from . import decision as _decision
            _decision._dispatch(_decision.DecisionRecord.from_event(_evt))
        except Exception as e:
            try:
                from .degraded import emit_degraded
                emit_degraded("decision_stream", e)
            except Exception:
                pass  # fail-open: the decision stream must never break a call
    except Exception:
        pass


def _shadow_flush_fallback(pending, args, kwargs, model, extract_text,
                           cache_key, prompt_text, tag, function_name,
                           compression, pii_redactions, endpoint=None,
                           extract_response_text=None):
    """Emit deferred shadow-hit records when the real call never produced a
    record (exception, hard-cap stop, abandoned stream). Never raises.

    Drains any provider-reported usage the upstream call set before failing:
    (1) it is the truest measure of the avoided cost we can still attach, and
    (2) leaving it in the contextvar would bleed into the NEXT call's record.
    """
    if not pending:
        return
    try:
        rep = _consume_reported()
        for _p in list(pending):
            _record_and_emit(
                args, kwargs, _p["result"], True, _p["hit_type"], model,
                extract_text, _p["latency_ms"], cache_key, prompt_text,
                True, tag, function_name=function_name, compression=compression,
                pii_redactions=pii_redactions, reported_override=rep,
                endpoint=endpoint, extract_response_text=extract_response_text,
            )
        pending.clear()
    except Exception:  # pragma: no cover - defensive; record path never raises
        pass


# ============ @tokeymeter.cache (sync + async) ============

def cache(
    fn: Optional[Callable] = None,
    *,
    model: str = "_default",
    store: Optional[Any] = None,
    key_fn: Optional[Callable] = None,
    extract_text: Optional[Callable] = None,
    enabled: bool = True,
    semantic: bool = False,
    semantic_cache: Optional[Any] = None,
    semantic_threshold: float = 0.92,
    prompt_arg: Optional[Union[str, int]] = None,
    ttl: Optional[float] = None,
    single_flight: bool = True,
    shadow: bool = False,
    redactor: Optional[Callable[[str], str]] = None,
    redact_response: bool = False,
    tag: Optional[str] = None,
    # ---- v0.6: prompt compression ----
    compressor: Optional[Compressor] = None,
    verify_rate: float = 0.0,
    verify_similarity_fn: Optional[Callable[[Any, Any], float]] = None,
    # ---- quality preservation: high-stakes / do-not-optimize ----
    high_stakes: bool = False,
    lineage: Optional[str] = None,
    # ---- isolation: cache namespace (prevents cross-function/tenant collision) ----
    namespace: Optional[str] = None,
    shared_namespace: bool = False,
    tenant: Optional[str] = None,
    # ---- self-host: operator-declared endpoint id for this callable ----
    endpoint: Optional[str] = None,
    # Appended at the END on purpose: inserting a parameter mid-signature
    # shifts every POSITIONAL argument after it and silently breaks any
    # caller that passed them by position.
    extract_response_text: Optional[Callable] = None,
):
    """Two-tier cache decorator. Works on sync and async functions.

    See module docstring for the full pipeline.

    Compression (v0.6):
        compressor: A Compressor instance applied to the prompt BEFORE
            cache lookup. Two prompts that compress to the same form
            share a cache entry. Compression failures fall open — the
            original prompt is used.
        verify_rate: Fraction in [0, 1] of calls to fidelity-audit.
            When triggered, the wrapped function is invoked with BOTH
            the original AND the compressed prompt; the two outputs are
            compared via `verify_similarity_fn` (default: Jaccard) and
            recorded to `tokeymeter.compression_verification_log()`.
            Default 0.0 (no verification).
        verify_similarity_fn: (orig_result, compressed_result) -> float
            in [0, 1]. Custom similarity for audited calls. Default
            uses token-set Jaccard.

    Trust-layer args (v0.5): shadow, redactor, tag.
    Production-polish args (v0.4): ttl, single_flight.
    Core args: model, semantic, store, etc.
    """

    def decorator(func: Callable) -> Callable:
        # Fail loudly on a malformed endpoint literal (developer error) HERE,
        # not silently in the non-raising record path. A None/valid id passes.
        _validate_endpoint(endpoint)
        # Resolve the cache namespace (workload identity). Default: the function's
        # module+qualname, with collision-safe disambiguation for dynamically
        # generated functions. Opt into sharing via shared_namespace / namespace=.
        eff_ns = _resolve_namespace(func, namespace, shared_namespace)
        if inspect.iscoroutinefunction(func):
            return _make_async_wrapper(
                func, model, store, key_fn, extract_text, enabled,
                semantic, semantic_cache, semantic_threshold, prompt_arg,
                ttl, single_flight, shadow, redactor, redact_response, tag,
                compressor, verify_rate, verify_similarity_fn, high_stakes, lineage,
                namespace=eff_ns, tenant=tenant, endpoint=endpoint, extract_response_text=extract_response_text,
            )
        return _make_sync_wrapper(
            func, model, store, key_fn, extract_text, enabled,
            semantic, semantic_cache, semantic_threshold, prompt_arg,
            ttl, single_flight, shadow, redactor, redact_response, tag,
            compressor, verify_rate, verify_similarity_fn, high_stakes, lineage,
            namespace=eff_ns, tenant=tenant, endpoint=endpoint, extract_response_text=extract_response_text,
        )

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


def _make_sync_wrapper(
    func, model, store, key_fn, extract_text, enabled,
    semantic, semantic_cache, semantic_threshold, prompt_arg,
    ttl, single_flight, shadow, redactor, redact_response, tag,
    compressor=None, verify_rate=0.0, verify_similarity_fn=None, high_stakes=False,
    lineage=None, namespace=None, tenant=None, endpoint=None,
    extract_response_text=None,
):
    # The model as DECORATED. The wrapper resolves the per-call
    # value from this; assigning to `model` inside the wrapper
    # would shadow the closure and raise UnboundLocalError.
    _bound_model = model
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # The model THIS call used, resolved FIRST — before any branch
        # reads it. The high-stakes bypass skips the compliance gate
        # entirely, so resolving at the gate left it unassigned there.
        model = _effective_model(_bound_model)
        start = time.perf_counter()
        if not enabled:
            return func(*args, **kwargs)

        # High-stakes / do-not-optimize: fresh call, prompt untouched, no cache
        # serve or store, no compression — but STILL recorded to audit.
        if _is_high_stakes(high_stakes):
            # Redaction is an egress control, not an optimization: honor it even
            # in high-stakes (strip PII before the verbatim model call). Bypass
            # only caching/compression/semantic.
            _hs_red = _resolve_redactor_enforced(redactor)
            _hs_pii = _count_redactions(_hs_red, args, kwargs)
            args, kwargs = _redact_args(args, kwargs, _hs_red)
            _keys.check_current()
            result = func(*args, **kwargs)
            if redact_response:
                result = _redact_result(result, _hs_red)
            latency_ms = (time.perf_counter() - start) * 1000
            prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
            _record_and_emit(args, kwargs, result, False, "high_stakes",
                             model, extract_text, latency_ms,
                             None, prompt_text, False, tag,
                             function_name=func.__name__,
                             compression=None, pii_redactions=_hs_pii,
                             high_stakes=True, endpoint=endpoint, extract_response_text=extract_response_text)
            return result

        red = _resolve_redactor_enforced(redactor)
        # Redact args BEFORE keying. Keeps the cache surface PII-free.
        pii_count = _count_redactions(red, args, kwargs)
        args, kwargs = _redact_args(args, kwargs, red)

        # --- Apply compression (if any) BEFORE keying. Fail-open. ---
        compression_record: Optional[CompressionResult] = None
        original_args = args
        original_kwargs = kwargs
        if compressor is not None:
            args, kwargs, compression_record = _apply_compressor(
                args, kwargs, prompt_arg, extract_text, compressor,
                f"{func.__name__}::{tag}"
            )

        exact_backend = store if store is not None else _get_default_store()
        sem_layer = _resolve_semantic_layer(semantic, semantic_cache, semantic_threshold)

        eff_lineage = _resolve_lineage(lineage)
        eff_tenant = _resolve_tenant(tenant)
        # Compliance first: a cache hit still returns data derived from a
        # model the policy may forbid.
        _compliance = _compliance_gate(model)
        cache_key, cached, cached_meta = _try_exact_lookup(args, kwargs, key_fn, model, exact_backend, eff_lineage, namespace, eff_tenant)
        if _compliance is not None and _compliance.never_cache:
            # Retention is forbidden for this context: neither serve from the
            # cache nor write to it. Suppressing only the WRITE would still
            # serve a previously-retained answer, which is the same violation
            # one request later.
            cached, cached_meta = None, None
        _task_preflight(cache_key, cached is not None)
        if eff_lineage is not None or eff_tenant is not None:
            sem_layer = None  # no cross-lineage/cross-tenant fuzzy serving (store isn't partitioned)
        prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
        pending_shadow_hits: List[dict] = []

        # ----- L1: exact -----
        if cached is not None:
            if shadow:
                latency_ms = (time.perf_counter() - start) * 1000
                pending_shadow_hits.append({
                    "result": cached, "hit_type": "shadow_exact",
                    "latency_ms": latency_ms,
                })
                # fall through to real call (shadow == measurement only). The
                # record is DEFERRED so it can carry the real call's reported
                # token counts — flushed by _record_and_emit on the real-call
                # record, or by _shadow_flush_fallback if the call raises.
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                _record_and_emit(args, kwargs, cached, True, "exact",
                                 model, extract_text, latency_ms,
                                 cache_key, prompt_text, False, tag,
                                 function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=cached_meta)
                return cached

        # ----- L2: semantic -----
        sem_embedding = None
        sem_value = None
        if sem_layer is not None and prompt_text:
            try:
                sem_embedding = sem_layer.encode(prompt_text)
            except Exception as e:
                log.debug("tokeymeter: semantic encode failed: %s", e)
            if sem_embedding is not None:
                try:
                    sem_raw = sem_layer.lookup_by_embedding(sem_embedding, query_prompt=prompt_text)
                except TypeError:
                    sem_raw = sem_layer.lookup_by_embedding(sem_embedding)  # older signature
                except Exception as e:
                    log.debug("tokeymeter: semantic lookup failed: %s", e)
                    sem_raw = None
                sem_value = unwrap(sem_raw)
                sem_meta = _env_meta(sem_raw)
                if sem_value is not None:
                    if shadow:
                        latency_ms = (time.perf_counter() - start) * 1000
                        pending_shadow_hits.append({
                            "result": sem_value, "hit_type": "shadow_semantic",
                            "latency_ms": latency_ms, "meta": sem_meta,
                        })
                        # fall through (deferred — see shadow_exact above)
                    else:
                        # promote into exact cache CARRYING the original meta,
                        # so the next exact hit recovers the true miss volume too
                        _safe_exact_set(exact_backend, cache_key,
                                        wrap(sem_value, ttl, meta=sem_meta))
                        latency_ms = (time.perf_counter() - start) * 1000
                        _record_and_emit(args, kwargs, sem_value, True, "semantic",
                                         model, extract_text, latency_ms,
                                         cache_key, prompt_text, False, tag,
                                         function_name=func.__name__,
                                         compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                         meta_override=sem_meta)
                        return sem_value

        # ----- L3: real call (with compressed prompt) -----
        # Distributed single-flight: if the backend supports it, one process
        # computes while peers wait for the result (cross-pod stampede
        # protection). Fail-open: on any issue we just compute locally.
        _dsf_token, _dsf_envelope = _dsf_begin(
            exact_backend, cache_key, single_flight, shadow)
        if _dsf_envelope is not None:
            follower_value = unwrap(_dsf_envelope)
            follower_meta = _env_meta(_dsf_envelope)
            if follower_value is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                _record_and_emit(args, kwargs, follower_value, True, "single_flight",
                                 model, extract_text, latency_ms,
                                 cache_key, prompt_text, False, tag,
                                 function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=follower_meta)
                return follower_value

        try:
            _keys.check_current()   # hard key-budget stop BEFORE spend (T3.3)
            result = func(*args, **kwargs)
            if redact_response:
                result = _redact_result(result, red)
            latency_ms = (time.perf_counter() - start) * 1000

            # S1.1: stamp the miss's true token counts into the envelope so a
            # later hit reports the actual avoided volume. Peeked (not consumed)
            # so the record path below still consumes normally.
            _miss_meta = _resolve_miss_meta(args, kwargs, extract_text, result, model)
            envelope = wrap(result, ttl, meta=_miss_meta)
            _safe_exact_set(exact_backend, cache_key, envelope)
            if sem_layer is not None and prompt_text and sem_embedding is not None:
                try:
                    sem_layer.store_by_embedding(prompt_text, sem_embedding, envelope)
                except Exception as e:
                    log.debug("tokeymeter: semantic store failed: %s", e)
        except BaseException:
            # Real call failed AFTER a shadow hit was deferred: the hit still
            # happened — emit it (with drained reported usage, if the upstream
            # got far enough to set it) before propagating.
            _shadow_flush_fallback(pending_shadow_hits, args, kwargs, model,
                                   extract_text, cache_key, prompt_text, tag,
                                   func.__name__, compression_record, pii_count,
                                   endpoint=endpoint, extract_response_text=extract_response_text)
            raise
        finally:
            # Release the compute lock AFTER storing, so waiting peers find the
            # value. On exception, release too (peers fall back to computing).
            _dsf_release(exact_backend, cache_key, _dsf_token, locals().get("envelope", _SF_MISSING))

        _record_and_emit(args, kwargs, result, False, None,
                         model, extract_text, latency_ms,
                         cache_key, prompt_text, shadow, tag,
                         function_name=func.__name__,
                         compression=compression_record, pii_redactions=pii_count,
                         pending_shadow_hits=pending_shadow_hits, endpoint=endpoint, extract_response_text=extract_response_text)

        # ----- Fidelity verification (sampled) -----
        if (compressor is not None and verify_rate > 0
                and compression_record is not None
                and compression_record.safe
                and compression_record.after != compression_record.before
                and _verify_should_sample(verify_rate)):
            try:
                orig_result = func(*original_args, **original_kwargs)
                sim_fn = verify_similarity_fn or _jaccard_similarity
                sim = float(sim_fn(orig_result, result))
                _record_verification({
                    "timestamp": time.time(),
                    "tag": tag,
                    "function_name": func.__name__,
                    "similarity": sim,
                    "tokens_before": compression_record.tokens_before,
                    "tokens_after": compression_record.tokens_after,
                    "method": compression_record.method,
                })
                _fidelity.get_breaker().record(f"{func.__name__}::{tag}", sim)
            except Exception as e:
                log.debug("tokeymeter: compression verify failed: %s", e)

        return result

    return wrapper


def _make_async_wrapper(
    func, model, store, key_fn, extract_text, enabled,
    semantic, semantic_cache, semantic_threshold, prompt_arg,
    ttl, single_flight, shadow, redactor, redact_response, tag,
    compressor=None, verify_rate=0.0, verify_similarity_fn=None, high_stakes=False,
    lineage=None, namespace=None, tenant=None, endpoint=None,
    extract_response_text=None,
):
    # The model as DECORATED. The wrapper resolves the per-call
    # value from this; assigning to `model` inside the wrapper
    # would shadow the closure and raise UnboundLocalError.
    _bound_model = model
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # The model THIS call used, resolved FIRST — before any branch
        # reads it. The high-stakes bypass skips the compliance gate
        # entirely, so resolving at the gate left it unassigned there.
        model = _effective_model(_bound_model)
        start = time.perf_counter()
        if not enabled:
            return await func(*args, **kwargs)

        # High-stakes / do-not-optimize: fresh call, prompt untouched, no cache
        # serve or store, no compression — but STILL recorded to audit.
        if _is_high_stakes(high_stakes):
            _hs_red = _resolve_redactor_enforced(redactor)
            _hs_pii = _count_redactions(_hs_red, args, kwargs)
            args, kwargs = _redact_args(args, kwargs, _hs_red)
            _keys.check_current()
            result = await func(*args, **kwargs)
            if redact_response:
                result = _redact_result(result, _hs_red)
            latency_ms = (time.perf_counter() - start) * 1000
            prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
            _record_and_emit(args, kwargs, result, False, "high_stakes",
                             model, extract_text, latency_ms,
                             None, prompt_text, False, tag,
                             function_name=func.__name__,
                             compression=None, pii_redactions=_hs_pii,
                             high_stakes=True, endpoint=endpoint, extract_response_text=extract_response_text)
            return result

        red = _resolve_redactor_enforced(redactor)
        pii_count = _count_redactions(red, args, kwargs)
        args, kwargs = _redact_args(args, kwargs, red)

        # Compression BEFORE keying. Fail-open.
        compression_record: Optional[CompressionResult] = None
        original_args = args
        original_kwargs = kwargs
        if compressor is not None:
            args, kwargs, compression_record = await asyncio.to_thread(
                _apply_compressor, args, kwargs, prompt_arg, extract_text, compressor,
                f"{func.__name__}::{tag}"
            )

        exact_backend = store if store is not None else _get_default_store()
        sem_layer = _resolve_semantic_layer(semantic, semantic_cache, semantic_threshold)

        eff_lineage = _resolve_lineage(lineage)
        eff_tenant = _resolve_tenant(tenant)
        # Compliance first: a cache hit still returns data derived from a
        # model the policy may forbid.
        _compliance = _compliance_gate(model)
        cache_key, cached, cached_meta = _try_exact_lookup(args, kwargs, key_fn, model, exact_backend, eff_lineage, namespace, eff_tenant)
        if _compliance is not None and _compliance.never_cache:
            # Retention is forbidden for this context: neither serve from the
            # cache nor write to it. Suppressing only the WRITE would still
            # serve a previously-retained answer, which is the same violation
            # one request later.
            cached, cached_meta = None, None
        _task_preflight(cache_key, cached is not None)
        if eff_lineage is not None or eff_tenant is not None:
            sem_layer = None  # no cross-lineage/cross-tenant fuzzy serving (store isn't partitioned)
        prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
        pending_shadow_hits: List[dict] = []

        # ----- L1: exact -----
        if cached is not None:
            latency_ms = (time.perf_counter() - start) * 1000
            if shadow:
                pending_shadow_hits.append({
                    "result": cached, "hit_type": "shadow_exact",
                    "latency_ms": latency_ms,
                })
                # fall through (deferred so the record carries the real call's
                # reported counts — flushed on the real-call record, or by
                # _shadow_flush_fallback if the call raises)
            else:
                _record_and_emit(args, kwargs, cached, True, "exact",
                                 model, extract_text, latency_ms,
                                 cache_key, prompt_text, False, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=cached_meta)
                return cached

        # ----- Single-flight gate (disabled in shadow mode) -----
        leader_future: Optional[asyncio.Future] = None
        if single_flight and not shadow and cache_key is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                leader_future = loop.create_future()
                existing = _inflight_async.setdefault(cache_key, leader_future)
                if existing is not leader_future:
                    try:
                        follower_result = await existing
                        latency_ms = (time.perf_counter() - start) * 1000
                        _record_and_emit(args, kwargs, follower_result, True,
                                         "single_flight", model, extract_text,
                                         latency_ms, cache_key, prompt_text,
                                         False, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text)
                        return follower_result
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        _inflight_async[cache_key] = leader_future

        try:
            # ----- L2: semantic -----
            sem_embedding = None
            sem_value = None
            if sem_layer is not None and prompt_text:
                try:
                    sem_embedding = await asyncio.to_thread(sem_layer.encode, prompt_text)
                except Exception as e:
                    log.debug("tokeymeter: semantic encode failed: %s", e)
                if sem_embedding is not None:
                    try:
                        sem_raw = await asyncio.to_thread(
                            sem_layer.lookup_by_embedding, sem_embedding, prompt_text
                        )
                    except TypeError:
                        sem_raw = await asyncio.to_thread(
                            sem_layer.lookup_by_embedding, sem_embedding
                        )
                    except Exception as e:
                        log.debug("tokeymeter: semantic lookup failed: %s", e)
                        sem_raw = None
                    sem_value = unwrap(sem_raw)
                    sem_meta = _env_meta(sem_raw)
                    if sem_value is not None:
                        latency_ms = (time.perf_counter() - start) * 1000
                        if shadow:
                            pending_shadow_hits.append({
                                "result": sem_value,
                                "hit_type": "shadow_semantic",
                                "latency_ms": latency_ms, "meta": sem_meta,
                            })
                            # fall through (deferred — see shadow_exact above)
                        else:
                            envelope = wrap(sem_value, ttl, meta=sem_meta)
                            _safe_exact_set(exact_backend, cache_key, envelope)
                            if leader_future is not None and not leader_future.done():
                                leader_future.set_result(sem_value)
                            _record_and_emit(args, kwargs, sem_value, True, "semantic",
                                             model, extract_text, latency_ms,
                                             cache_key, prompt_text, False, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=sem_meta)
                            return sem_value

            # ----- Distributed single-flight gate (cross-pod) -----
            _dsf_token = None
            try:
                _dsf_token, _dsf_envelope = await asyncio.to_thread(
                    _dsf_begin, exact_backend, cache_key, single_flight, shadow
                )
            except Exception:
                _dsf_token, _dsf_envelope = None, None
            if _dsf_envelope is not None:
                follower_value = unwrap(_dsf_envelope)
                follower_meta = _env_meta(_dsf_envelope)
                if follower_value is not None:
                    latency_ms = (time.perf_counter() - start) * 1000
                    if leader_future is not None and not leader_future.done():
                        leader_future.set_result(follower_value)
                    _record_and_emit(args, kwargs, follower_value, True,
                                     "single_flight", model, extract_text,
                                     latency_ms, cache_key, prompt_text,
                                     False, tag, function_name=func.__name__,
                                     compression=compression_record,
                                     pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                     meta_override=follower_meta)
                    return follower_value

            # ----- L3: real call -----
            _keys.check_current()   # hard key-budget stop BEFORE spend (T3.3)
            result = await func(*args, **kwargs)
            if redact_response:
                result = _redact_result(result, red)
            latency_ms = (time.perf_counter() - start) * 1000

            _miss_meta = _resolve_miss_meta(args, kwargs, extract_text, result, model)
            envelope = wrap(result, ttl, meta=_miss_meta)
            _safe_exact_set(exact_backend, cache_key, envelope)
            if sem_layer is not None and prompt_text and sem_embedding is not None:
                try:
                    await asyncio.to_thread(
                        sem_layer.store_by_embedding, prompt_text, sem_embedding, envelope
                    )
                except Exception as e:
                    log.debug("tokeymeter: semantic store failed: %s", e)

            if leader_future is not None and not leader_future.done():
                leader_future.set_result(result)

            _record_and_emit(args, kwargs, result, False, None,
                             model, extract_text, latency_ms,
                             cache_key, prompt_text, shadow, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count,
                                 pending_shadow_hits=pending_shadow_hits, endpoint=endpoint, extract_response_text=extract_response_text)

            # ----- Fidelity verification (sampled, async) -----
            if (compressor is not None and verify_rate > 0
                    and compression_record is not None
                    and compression_record.safe
                    and compression_record.after != compression_record.before
                    and _verify_should_sample(verify_rate)):
                try:
                    orig_result = await func(*original_args, **original_kwargs)
                    sim_fn = verify_similarity_fn or _jaccard_similarity
                    sim = float(sim_fn(orig_result, result))
                    _record_verification({
                        "timestamp": time.time(),
                        "tag": tag,
                        "function_name": func.__name__,
                        "similarity": sim,
                        "tokens_before": compression_record.tokens_before,
                        "tokens_after": compression_record.tokens_after,
                        "method": compression_record.method,
                    })
                    _fidelity.get_breaker().record(f"{func.__name__}::{tag}", sim)
                except Exception as e:
                    log.debug("tokeymeter: compression verify failed: %s", e)

            return result

        except BaseException as e:
            # Real call failed AFTER a shadow hit was deferred: the hit still
            # happened — emit it (with drained reported usage, if the upstream
            # got far enough to set it) before propagating.
            _shadow_flush_fallback(pending_shadow_hits, args, kwargs, model,
                                   extract_text, cache_key, prompt_text, tag,
                                   func.__name__, compression_record, pii_count,
                                   endpoint=endpoint, extract_response_text=extract_response_text)
            if leader_future is not None and not leader_future.done():
                leader_future.set_exception(
                    e if isinstance(e, Exception) else RuntimeError(repr(e))
                )
                # Mark the exception retrieved. Awaiting a done future still
                # re-raises for any follower, but a leader that fails with NO
                # followers must not leave an unretrieved-exception future —
                # asyncio warns on those at GC, which pollutes logs and hard-
                # crashes deployments running with warnings-as-errors.
                leader_future.exception()
            raise
        finally:
            try:
                await asyncio.to_thread(_dsf_release, exact_backend, cache_key, locals().get("_dsf_token"), locals().get("envelope", _SF_MISSING))
            except Exception:
                pass
            if cache_key is not None and leader_future is not None:
                if _inflight_async.get(cache_key) is leader_future:
                    _inflight_async.pop(cache_key, None)

    return wrapper


# ============ @tokeymeter.cache_stream ============

def cache_stream(
    fn: Optional[Callable] = None,
    *,
    model: str = "_default",
    store: Optional[Any] = None,
    key_fn: Optional[Callable] = None,
    extract_text: Optional[Callable] = None,
    enabled: bool = True,
    semantic: bool = False,
    semantic_cache: Optional[Any] = None,
    semantic_threshold: float = 0.92,
    prompt_arg: Optional[Union[str, int]] = None,
    ttl: Optional[float] = None,
    shadow: bool = False,
    redactor: Optional[Callable[[str], str]] = None,
    redact_response: bool = False,
    tag: Optional[str] = None,
    compressor: Optional[Compressor] = None,
    high_stakes: bool = False,
    lineage: Optional[str] = None,
    namespace: Optional[str] = None,
    shared_namespace: bool = False,
    tenant: Optional[str] = None,
    endpoint: Optional[str] = None,
    # Appended at the END on purpose: inserting a parameter mid-signature
    # shifts every POSITIONAL argument after it and silently breaks any
    # caller that passed them by position.
    extract_response_text: Optional[Callable] = None,
):
    """Cache async generators that yield streaming chunks.

    Shadow / redactor / tag work the same way as for @tokeymeter.cache.
    Compression is applied before keying, same as @tokeymeter.cache.

    Streaming does not use single-flight or verify_rate in v0.6.
    """

    def decorator(func: Callable) -> Callable:
        # Fail loudly on a malformed endpoint literal (developer error) at
        # decoration, mirroring @cache.
        _validate_endpoint(endpoint)
        # Same namespace resolution as @cache (workload-identity isolation,
        # collision-safe for dynamically generated functions).
        eff_ns = _resolve_namespace(func, namespace, shared_namespace)
        if not inspect.isasyncgenfunction(func):
            raise TypeError(
                f"@cache_stream requires an async generator function "
                f"(an `async def` with `yield`), got {func!r}"
            )

        # The model as DECORATED. The wrapper resolves the per-call
        # value from this; assigning to `model` inside the wrapper
        # would shadow the closure and raise UnboundLocalError.
        _bound_model = model
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # The model THIS call used, resolved FIRST — before any branch
            # reads it. The high-stakes bypass skips the compliance gate
            # entirely, so resolving at the gate left it unassigned there.
            model = _effective_model(_bound_model)
            start = time.perf_counter()

            if not enabled:
                async for chunk in func(*args, **kwargs):
                    yield chunk
                return

            red = _resolve_redactor_enforced(redactor)
            pii_count = _count_redactions(red, args, kwargs)
            args, kwargs = _redact_args(args, kwargs, red)

            if _is_high_stakes(high_stakes):
                chunks: List[Any] = []
                completed = False
                try:
                    _keys.check_current()   # hard key-budget stop BEFORE first chunk (T3.3)
                    async for chunk in func(*args, **kwargs):
                        out_chunk = _redact_result(chunk, red) if redact_response else chunk
                        chunks.append(out_chunk)
                        yield out_chunk
                    completed = True
                finally:
                    if completed:
                        latency_ms = (time.perf_counter() - start) * 1000
                        prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
                        _record_and_emit(args, kwargs, chunks, False, "high_stakes",
                                         model, extract_text, latency_ms,
                                         None, prompt_text, False, tag,
                                         function_name=func.__name__,
                                         compression=None,
                                         pii_redactions=pii_count,
                                         high_stakes=True, endpoint=endpoint, extract_response_text=extract_response_text)
                return

            # Compression BEFORE keying. Fail-open.
            compression_record: Optional[CompressionResult] = None
            if compressor is not None:
                args, kwargs, compression_record = await asyncio.to_thread(
                    _apply_compressor, args, kwargs, prompt_arg, extract_text, compressor,
                    f"{func.__name__}::{tag}"
                )

            exact_backend = store if store is not None else _get_default_store()
            sem_layer = _resolve_semantic_layer(semantic, semantic_cache, semantic_threshold)

            eff_lineage = _resolve_lineage(lineage)
            eff_tenant = _resolve_tenant(tenant)
            # Compliance first: a cache hit still returns data derived from a
            # model the policy may forbid.
            _compliance = _compliance_gate(model)
            cache_key, cached, cached_meta = _try_exact_lookup(args, kwargs, key_fn, model, exact_backend, eff_lineage, eff_ns, eff_tenant)
            if _compliance is not None and _compliance.never_cache:
                cached, cached_meta = None, None
            _task_preflight(cache_key, cached is not None)
            if eff_lineage is not None or eff_tenant is not None:
                sem_layer = None
            prompt_text = _extract_prompt_text(args, kwargs, prompt_arg, extract_text)
            pending_shadow_hits: List[dict] = []

            # ----- L1: exact -----
            if isinstance(cached, list):
                latency_ms = (time.perf_counter() - start) * 1000
                if shadow:
                    pending_shadow_hits.append({
                        "result": cached, "hit_type": "shadow_exact",
                        "latency_ms": latency_ms,
                    })
                    # fall through (deferred so the record carries the real
                    # stream's reported counts — flushed on the real-call
                    # record, or by _shadow_flush_fallback on abandon/raise)
                else:
                    _record_and_emit(args, kwargs, cached, True, "exact",
                                     model, extract_text, latency_ms,
                                     cache_key, prompt_text, False, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=cached_meta)
                    for chunk in cached:
                        yield chunk
                    return

            # ----- L2: semantic -----
            sem_embedding = None
            sem_value = None
            if sem_layer is not None and prompt_text:
                try:
                    sem_embedding = await asyncio.to_thread(sem_layer.encode, prompt_text)
                except Exception:
                    sem_embedding = None
                if sem_embedding is not None:
                    try:
                        sem_raw = await asyncio.to_thread(
                            sem_layer.lookup_by_embedding, sem_embedding
                        )
                    except Exception:
                        sem_raw = None
                    sem_value = unwrap(sem_raw)
                    sem_meta = _env_meta(sem_raw)
                    if isinstance(sem_value, list):
                        latency_ms = (time.perf_counter() - start) * 1000
                        if shadow:
                            pending_shadow_hits.append({
                                "result": sem_value,
                                "hit_type": "shadow_semantic",
                                "latency_ms": latency_ms,
                            })
                            # fall through (deferred — see shadow_exact above)
                        else:
                            _safe_exact_set(exact_backend, cache_key, wrap(sem_value, ttl, meta=sem_meta))
                            _record_and_emit(args, kwargs, sem_value, True, "semantic",
                                             model, extract_text, latency_ms,
                                             cache_key, prompt_text, False, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count, endpoint=endpoint, extract_response_text=extract_response_text,
                                 meta_override=sem_meta)
                            for chunk in sem_value:
                                yield chunk
                            return

            # ----- L3: tee the real generator -----
            chunks: List[Any] = []
            completed = False
            try:
                _keys.check_current()   # hard key-budget stop BEFORE first chunk (T3.3)
                async for chunk in func(*args, **kwargs):
                    out_chunk = _redact_result(chunk, red) if redact_response else chunk
                    chunks.append(out_chunk)
                    yield out_chunk
                completed = True
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                if completed:
                    _miss_meta = _resolve_miss_meta(args, kwargs, extract_text, chunks, model)
                    envelope = wrap(chunks, ttl, meta=_miss_meta)
                    _safe_exact_set(exact_backend, cache_key, envelope)
                    if sem_layer is not None and prompt_text and sem_embedding is not None:
                        try:
                            await asyncio.to_thread(
                                sem_layer.store_by_embedding,
                                prompt_text, sem_embedding, envelope,
                            )
                        except Exception:
                            pass
                    _record_and_emit(args, kwargs, chunks, False, None,
                                     model, extract_text, latency_ms,
                                     cache_key, prompt_text, shadow, tag, function_name=func.__name__,
                                 compression=compression_record, pii_redactions=pii_count,
                                 pending_shadow_hits=pending_shadow_hits, endpoint=endpoint, extract_response_text=extract_response_text)
                else:
                    # Stream abandoned or raised AFTER a shadow hit was
                    # deferred: the hit still happened — emit it (with drained
                    # reported usage, if the upstream set it) before exiting.
                    _shadow_flush_fallback(pending_shadow_hits, args, kwargs,
                                           model, extract_text, cache_key,
                                           prompt_text, tag, func.__name__,
                                           compression_record, pii_count,
                                           endpoint=endpoint, extract_response_text=extract_response_text)

        return wrapper

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


# ============ Diagnostics ============

def _inflight_size() -> int:
    """Internal: number of in-flight single-flight entries. For tests."""
    return len(_inflight_async)


# ============================================================
#                  with_memory decorator (v0.7)
# ============================================================

# Memory fidelity log. Populated when fidelity_rate > 0.
# Each entry: {timestamp, similarity, tokens_full, tokens_with_memory, ...}
_memory_fidelity_log: list = []
_memory_fidelity_lock = __import__("threading").Lock()


def _record_memory_fidelity(rec: dict) -> None:
    """Append a memory-fidelity audit record. Never raises."""
    try:
        with _memory_fidelity_lock:
            _memory_fidelity_log.append(rec)
            if len(_memory_fidelity_log) > 10_000:
                del _memory_fidelity_log[: len(_memory_fidelity_log) - 10_000]
    except Exception:
        pass


def memory_fidelity_log() -> list:
    """Return a copy of the in-memory fidelity audit log."""
    with _memory_fidelity_lock:
        return list(_memory_fidelity_log)


def with_memory(
    memory: ConversationMemory,
    *,
    session_arg: str = "session_id",
    prompt_arg: str = "prompt",
    context_arg: Optional[str] = None,
    fidelity_rate: float = 0.0,
    fidelity_similarity_fn: Optional[Callable[[Any, Any], float]] = None,
    tag: Optional[str] = None,
):
    """Wrap an async function with conversation memory.

    On every call:
      1. Pull session_id from kwargs[session_arg].
      2. Get context (summarized older turns + recent full turns) from memory.
      3. Inject context into the function call. Two modes:
         - Default: prepend context.text to kwargs[prompt_arg]
           ("Earlier context:\\n...\\n\\nUser: <your prompt>")
         - context_arg mode: pass context.messages as kwargs[context_arg]
           (no rewriting of the prompt arg — your function knows what to do)
      4. Call the wrapped function.
      5. Record (prompt, response) as a new turn.

    Compose with @tokeymeter.cache by stacking decorators:
        @tokeymeter.with_memory(memory=..., session_arg="session_id")
        @tokeymeter.cache(model="gpt-4o-mini")
        async def ask(prompt, session_id): ...

    Fidelity audit:
        fidelity_rate=R samples R fraction of memory-using calls and runs
        the same query AGAIN with the FULL un-summarized history. Compares
        outputs via Jaccard (or fidelity_similarity_fn). Recorded to
        tokeymeter.memory_fidelity_log(). Use this to verify your summarization
        is faithful before trusting it in production.

    Args:
        memory: a ConversationMemory instance.
        session_arg: kwarg name that carries the session id.
        prompt_arg: kwarg name that carries the user's prompt (string).
        context_arg: if set, pass List[dict] messages here instead of
            rewriting prompt_arg. Useful for OpenAI/Anthropic message APIs.
        fidelity_rate: in [0, 1]. Default 0 (no audit).
        fidelity_similarity_fn: (full_resp, mem_resp) -> float in [0, 1].
            Default: token-set Jaccard.
        tag: workload label flowed into savings + events for this function.
    """

    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                "@tokeymeter.with_memory requires an async function. "
                "Wrap sync functions with asyncio.to_thread in your handler."
            )

        # The model as DECORATED. The wrapper resolves the per-call
        # value from this; assigning to `model` inside the wrapper
        # would shadow the closure and raise UnboundLocalError.
        _bound_model = model
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # The model THIS call used, resolved FIRST — before any branch
            # reads it. The high-stakes bypass skips the compliance gate
            # entirely, so resolving at the gate left it unassigned there.
            model = _effective_model(_bound_model)
            session_id = kwargs.get(session_arg)
            user_prompt = kwargs.get(prompt_arg)

            # If we can't identify session or prompt, pass through without memory
            if (not isinstance(session_id, str)
                    or not isinstance(user_prompt, str)
                    or not session_id
                    or not user_prompt):
                return await func(*args, **kwargs)

            # ----- Get context -----
            try:
                ctx = await memory.get_context(session_id)
            except Exception as e:
                log.debug("with_memory: get_context failed: %s", e)
                # Fail open — call without memory
                response = await func(*args, **kwargs)
                try:
                    await memory.add_turn(session_id, user_prompt, str(response))
                except Exception:
                    pass
                return response

            # ----- Build augmented call kwargs -----
            mem_kwargs = dict(kwargs)
            if context_arg is not None:
                mem_kwargs[context_arg] = ctx.messages
            else:
                if ctx.text:
                    mem_kwargs[prompt_arg] = f"{ctx.text}\n\nUser: {user_prompt}"

            # ----- Call wrapped function -----
            response = await func(*args, **mem_kwargs)

            # ----- Record turn -----
            try:
                await memory.add_turn(session_id, user_prompt, str(response))
            except Exception:
                pass

            # ----- Record savings tracking (linear-vs-quadratic story) -----
            try:
                _record(CallRecord(
                    timestamp=time.time(),
                    model="_memory",
                    hit=ctx.used_summary,  # used_summary = memory served you
                    hit_type="memory_summary" if ctx.used_summary else "memory_buffer",
                    input_tokens=ctx.tokens_with_memory,
                    output_tokens=estimate_tokens(str(response)),
                    estimated_cost=0.0,    # memory itself has no API cost
                    latency_ms=0.0,
                    shadow=False,
                    tag=tag,
                ))
            except Exception:
                pass

            # ----- Fidelity audit (sampled) -----
            if (fidelity_rate > 0
                    and ctx.used_summary
                    and _verify_should_sample(fidelity_rate)):
                try:
                    # Build FULL history (no summary)
                    all_turns = await asyncio.to_thread(
                        memory._store.get_turns, session_id
                    )
                    # The just-added turn is now in there; drop it for the audit
                    audit_turns = all_turns[:-1] if all_turns else []
                    full_text_parts = []
                    for t in audit_turns:
                        full_text_parts.append(f"User: {t.user}")
                        full_text_parts.append(f"Assistant: {t.assistant}")
                    full_text = "\n".join(full_text_parts)

                    full_kwargs = dict(kwargs)
                    if context_arg is not None:
                        msgs = []
                        for t in audit_turns:
                            msgs.append({"role": "user", "content": t.user})
                            msgs.append({"role": "assistant", "content": t.assistant})
                        full_kwargs[context_arg] = msgs
                    else:
                        full_kwargs[prompt_arg] = (
                            f"{full_text}\n\nUser: {user_prompt}" if full_text else user_prompt
                        )

                    full_response = await func(*args, **full_kwargs)
                    sim_fn = fidelity_similarity_fn or _jaccard_similarity
                    sim = float(sim_fn(full_response, response))

                    _record_memory_fidelity({
                        "timestamp": time.time(),
                        "tag": tag,
                        "function_name": func.__name__,
                        "session_id_hash": hashlib.sha256(
                            session_id.encode("utf-8")
                        ).hexdigest()[:16],  # don't leak raw session id
                        "similarity": sim,
                        "turns_audited": len(audit_turns),
                        "tokens_full": ctx.tokens_full_history,
                        "tokens_with_memory": ctx.tokens_with_memory,
                    })
                except Exception as e:
                    log.debug("with_memory: fidelity audit failed: %s", e)

            return response

        return wrapper

    return decorator