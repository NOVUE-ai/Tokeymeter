"""
Central resolution of where Tokeymeter writes local state.

By default everything lives under ``~/.tokeymeter``. Two override layers let
Tokeymeter run inside CI, containers, sandboxes, and locked-down machines where
``$HOME`` may be read-only or absent:

  - ``TOKEYMETER_HOME``          base directory for all local state
  - ``TOKEYMETER_SAVINGS_PATH``  the savings ledger file specifically
  - ``set_home(path)`` / ``set_savings_path(path)``   programmatic overrides

Resolution order for the savings ledger:
    explicit ``set_savings_path`` > ``TOKEYMETER_SAVINGS_PATH``
        > ``<home>/savings.jsonl``
where ``<home>`` = ``set_home`` > ``TOKEYMETER_HOME`` > ``~/.tokeymeter``.

Resolution is performed on every call (not cached) so an override set at startup
always takes effect, and tests can point state at a temp dir without reimporting.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

_lock = threading.RLock()
_home_override: Optional[str] = None
_savings_override: Optional[str] = None
_DEFAULT_HOME = os.path.join("~", ".tokeymeter")


def home() -> str:
    """Resolve the Tokeymeter home directory (expanded to an absolute path)."""
    with _lock:
        ov = _home_override
    raw = ov or os.environ.get("TOKEYMETER_HOME") or _DEFAULT_HOME
    return os.path.expanduser(raw)


def set_home(path: Optional[str]) -> None:
    """Override the home dir for all local state. Pass ``None`` to clear."""
    global _home_override
    with _lock:
        _home_override = path


def savings_path() -> str:
    """Resolve the savings ledger file path."""
    with _lock:
        ov = _savings_override
    if ov:
        return os.path.expanduser(ov)
    env = os.environ.get("TOKEYMETER_SAVINGS_PATH")
    if env:
        return os.path.expanduser(env)
    return os.path.join(home(), "savings.jsonl")


def state_path(filename: str) -> str:
    """Resolve a local-state file inside the Tokeymeter home directory.

    Every persistent artifact (cache, semantic index, memory, audit ledger and
    its keys) must resolve through here rather than hardcoding
    ``~/.tokeymeter/<name>``. A hardcoded home ignores both ``TOKEYMETER_HOME``
    and ``set_home()``, which breaks exactly the environments that need the
    redirect most — containers, CI, and locked-down service accounts where
    ``~`` is unwritable, ephemeral, or shared between tenants.

    Resolution is deliberately LAZY: callers pass ``None`` and resolve at use
    time, because the home may be set (via env or ``set_home``) after the
    module defining the default was imported."""
    return os.path.join(home(), filename)


def set_savings_path(path: Optional[str]) -> None:
    """Override the savings ledger path specifically. Pass ``None`` to clear."""
    global _savings_override
    with _lock:
        _savings_override = path
