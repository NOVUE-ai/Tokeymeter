"""Identity binding — WHO is behind every call, as a first-class property.

The keystone of L2 governance: every metered call can carry a `principal`
(a registered person or agent id from the TokeNet Org Registry), stamped at
record time from a contextvar so it composes correctly across threads and
async tasks. With it, spend-per-person / per-team / per-department stops
being a dashboard feature and becomes a property of every record at ingest.

Content-blind by design: a principal is an IDENTIFIER (validated to the same
grammar as ontology ids), never a name, email, or payload.

Usage:
    tokeymeter.set_principal("person:priya")          # process/task default
    with tokeymeter.principal("agent:evals-pipeline"):
        run_batch()                                    # scoped, restores prior

Unset (None) is always legal — records simply carry principal=None and the
control plane surfaces them as unattributed. Enforcement postures (flag /
block unregistered principals) live in the control plane and, later, the
Agent Passport; this module only binds and propagates.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Iterator, Optional

_PRINCIPAL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tokeymeter_principal", default=None)

# Same id grammar as the TokeNet ontology — enforced HERE so a malformed
# (or content-smuggling) principal never even enters the record stream.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")


def _validate(pid: Optional[str]) -> Optional[str]:
    if pid is None:
        return None
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        raise ValueError(
            "principal must be 1-128 chars of [A-Za-z0-9._:-] starting "
            f"alphanumeric (an identifier, never content); got {pid!r}")
    return pid


def set_principal(pid: Optional[str]) -> Optional[str]:
    """Set the current principal (None to clear). Returns the PREVIOUS value
    so callers can restore it. Prefer the `principal()` context manager."""
    prev = _PRINCIPAL.get()
    _PRINCIPAL.set(_validate(pid))
    return prev


def get_principal() -> Optional[str]:
    """The principal bound to the current context, or None."""
    return _PRINCIPAL.get()


@contextlib.contextmanager
def principal(pid: Optional[str]) -> Iterator[None]:
    """Bind a principal for a scope; always restores the prior binding,
    including on exception. Contextvar semantics: threads and async tasks
    each see their own binding."""
    token = _PRINCIPAL.set(_validate(pid))
    try:
        yield
    finally:
        _PRINCIPAL.reset(token)
