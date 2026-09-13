"""Endpoint identity binding — WHERE every self-hosted call executed, as a
first-class property of the record.

The self-hosted sibling of `identity.principal`. A self-hoster runs the same
model behind many serving endpoints (a vLLM pool on one node class, a TGI
deployment on another, an Ollama box for spillover). To answer the questions
that matter — unit cost per endpoint, duplicate deployments to consolidate,
which endpoint's queue is the expensive one — every record must know which
endpoint produced it. Bound from a contextvar so it composes across threads
and async tasks exactly like `principal`, and stamped at record time.

Content-blind by design: an endpoint id is an OPERATOR-DECLARED IDENTIFIER
(a stable name like "vllm-a100-pool" or "tgi-spillover"), never a URL, host,
or credential. We deliberately do NOT derive it from base_url: a raw URL can
carry auth (`https://user:pass@host`), leaks topology into the record stream,
and is not stable across redeploys. The operator names their endpoints; we
bind the name.

Usage:
    tokeymeter.set_endpoint("vllm-a100-pool")          # process/task default
    with tokeymeter.endpoint("tgi-spillover"):
        answer = ask(prompt)                            # scoped, restores prior

Precedence (resolved in the decorator): an explicit `endpoint=` argument on
`@cache` wins for that call; otherwise the bound contextvar value is used;
otherwise None. Unset (None) is always legal — records carry endpoint=None
and per-endpoint views simply group those as unattributed, the same posture
`principal=None` already uses. Nothing here blocks or enforces; it binds and
propagates only.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Iterator, Optional

_ENDPOINT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tokeymeter_endpoint", default=None)

# Same identifier grammar as principal/ontology ids — enforced HERE so a
# malformed (or content-smuggling, or URL-shaped) endpoint id never even
# enters the record stream. A URL fails this by construction (":" after the
# scheme is fine, but "/" is not permitted), which is intentional.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")


def _validate(eid: Optional[str]) -> Optional[str]:
    if eid is None:
        return None
    if not isinstance(eid, str) or not _ID_RE.match(eid):
        raise ValueError(
            "endpoint must be 1-128 chars of [A-Za-z0-9._:-] starting "
            "alphanumeric (an operator-declared identifier, never a URL or "
            f"content); got {eid!r}")
    return eid


def set_endpoint(eid: Optional[str]) -> Optional[str]:
    """Set the current endpoint id (None to clear). Returns the PREVIOUS value
    so callers can restore it. Prefer the `endpoint()` context manager."""
    prev = _ENDPOINT.get()
    _ENDPOINT.set(_validate(eid))
    return prev


def get_endpoint() -> Optional[str]:
    """The endpoint id bound to the current context, or None."""
    return _ENDPOINT.get()


@contextlib.contextmanager
def endpoint(eid: Optional[str]) -> Iterator[None]:
    """Bind an endpoint id for a scope; always restores the prior binding,
    including on exception. Contextvar semantics: threads and async tasks each
    see their own binding, so concurrent calls to different endpoints stamp
    correctly."""
    token = _ENDPOINT.set(_validate(eid))
    try:
        yield
    finally:
        _ENDPOINT.reset(token)


def resolve(explicit: Optional[str]) -> Optional[str]:
    """Precedence resolver used by the record path: an explicit per-call
    endpoint (from `@cache(endpoint=...)`) wins; otherwise the bound
    contextvar; otherwise None. `explicit` is validated so a bad per-call
    value fails loudly at decoration/first-call, not silently."""
    if explicit is not None:
        return _validate(explicit)
    return _ENDPOINT.get()
