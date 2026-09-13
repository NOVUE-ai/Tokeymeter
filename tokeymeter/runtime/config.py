"""Hierarchical runtime configuration.

Layering (later wins): built-in defaults ← dict/JSON file ← environment
(TOKEYMETER_* variables, dots encoded as double underscores).

Stdlib only. Values from env are JSON-decoded when possible so that
TOKEYMETER_RELIABILITY__MAX_RETRIES=3 arrives as int(3), not "3".
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

_DEFAULTS: Dict[str, Any] = {
    "kernel": {
        "drain_timeout_s": 30.0,
    },
    "reliability": {
        "max_retries": 1,
        "fallback_order": [],
    },
    "cache": {
        "enabled": True,
        "max_entries": 1024,
    },
}

_ENV_PREFIX = "TOKEYMETER_"


def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


class RuntimeConfig:
    """Immutable-by-convention layered config with dotted-path access."""

    def __init__(
        self,
        overrides: Optional[Mapping[str, Any]] = None,
        *,
        file_path: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        # DEEP copy: the env layer below mutates nested dicts in place; a
        # shallow copy here would silently rewrite module-level _DEFAULTS for
        # every future RuntimeConfig (real defect caught by the battery).
        data = json.loads(json.dumps(_DEFAULTS))
        if file_path:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = _deep_merge(data, json.load(fh))
        if overrides:
            data = _deep_merge(data, overrides)
        env_map = os.environ if env is None else env
        for key, raw in env_map.items():
            if not key.startswith(_ENV_PREFIX):
                continue
            path = key[len(_ENV_PREFIX):].lower().split("__")
            node: Dict[str, Any] = data
            for part in path[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[path[-1]] = _coerce(raw)
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))  # deep copy, JSON-able proof
