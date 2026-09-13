"""
Deterministic cache key generation.

Design notes:
- We use SHA-256, not because we need cryptographic strength but because
  it's collision-resistant enough that two different prompts will never
  produce the same key in practice.
- We canonicalize via json.dumps(sort_keys=True) so the order of kwargs
  doesn't change the key.
- For non-JSON-serializable objects (rare in LLM calls, but possible),
  we fall back to repr(). This is intentionally lossy — if you pass
  an unserializable object you should provide a custom key_fn.
"""
import hashlib
import json
from typing import Any, Tuple


def make_cache_key(
    args: Tuple[Any, ...],
    kwargs: dict,
    model: str = "_default",
    extra: Any = None,
) -> str:
    """Build a deterministic SHA-256 hex digest from call arguments.

    The key incorporates: positional args, keyword args, model name,
    and any extra context. Identical inputs always produce identical
    keys; different inputs essentially never collide.
    """
    payload = {
        "args": args,
        "kwargs": kwargs,
        "model": model,
        "extra": extra,
    }
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Last-resort fallback. Less deterministic but still hashable.
        serialized = repr(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
