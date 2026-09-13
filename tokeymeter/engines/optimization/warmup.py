"""
Cache pre-warming from production logs.

The novel value proposition: every cache starts cold. On day 1 of a
deployment, hit rates are near zero. With Tokeymeter warmup, you can replay
the last N days of your existing API logs into the cache before
deployment — so Day 1 looks like Day 30. No other LLM caching library
ships this.

Sources supported:
  - JSONL files (the standard logging format from openai-python, anthropic,
    LangSmith, Helicone exports, custom NDJSON)
  - Any iterable of (prompt, response) tuples (database rows, CSV, S3, etc)

For semantic caching, warmup is bulk-encoded. Encoding 10,000 prompts
one-by-one with sentence-transformers takes ~50 seconds. In a single
batch encode it's ~2 seconds — a 25× speedup. This matters because
warmup is the only "I'll wait for this" operation in the library.

Privacy note: warmup reads YOUR logs and stores YOUR responses LOCALLY.
Nothing is sent anywhere. The cache file lives at ~/.tokeymeter/cache.db
unless you override.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

from tokeymeter.engines.optimization.envelope import wrap
from tokeymeter.utils import make_cache_key

log = logging.getLogger("tokeymeter.warmup")


def warm_from_iterable(
    source: Iterable[Tuple[str, Any]],
    *,
    model: str = "_default",
    store: Optional[Any] = None,
    semantic_cache: Optional[Any] = None,
    ttl: Optional[float] = None,
    batch_size: int = 100,
    progress: Optional[Callable[[int, int], None]] = None,
    namespace: Optional[str] = None,
) -> dict:
    """Warm the cache from any iterable of (prompt, response) pairs.

    Args:
        source: Iterable yielding (prompt: str, response: Any) tuples.
        model: Model name to associate with cached entries (for cache key
            generation and cost reports).
        store: Exact-match store (defaults to global SQLiteStore).
        semantic_cache: Optional SemanticCache instance. If provided,
            entries are bulk-encoded and stored there too. The biggest
            payoff per minute of warmup time.
        ttl: Time-to-live in seconds. None = never expires.
        batch_size: Batch size for semantic encoding. 100-500 is typical;
            adjust based on RAM.
        progress: Optional callback (entries_loaded, total_in_batch) →
            called once per batch. Use for progress bars.
        namespace: Cache namespace to warm into. Must match the consuming
            decorator: pass the same `namespace=` you give @tokeymeter.cache,
            or leave both as the shared space (namespace=None here +
            shared_namespace=True on the decorator). Since @cache defaults to a
            per-function namespace, warmed entries are only found if this
            matches — otherwise the function looks under a different partition.

    Returns:
        Stats dict: total_loaded, exact_stored, semantic_stored, duration_s.
    """
    if store is None:
        from tokeymeter.decorator import _get_default_store
        store = _get_default_store()

    stats = {
        "total_loaded": 0,
        "exact_stored": 0,
        "semantic_stored": 0,
        "errors": 0,
        "duration_s": 0.0,
    }
    t0 = time.perf_counter()
    batch_prompts: list = []
    batch_responses: list = []

    def _flush_batch():
        """Process one batch: store exact + (optionally) bulk-encode semantic."""
        if not batch_prompts:
            return

        # ---- Exact cache: one insert per entry ----
        for prompt, response in zip(batch_prompts, batch_responses):
            try:
                # Key must match what the decorator generates: bare make_cache_key,
                # then the same `ns=...::` namespace prefix the decorator applies.
                key = make_cache_key((prompt,), {}, model=model)
                if namespace is not None:
                    key = f"ns={namespace}::{key}"
                envelope = wrap(response, ttl)
                store.set(key, envelope)
                stats["exact_stored"] += 1
            except Exception as e:
                log.debug("warmup: exact store failed: %s", e)
                stats["errors"] += 1

        # ---- Semantic cache: BULK encode for ~25x speedup ----
        if semantic_cache is not None and semantic_cache.is_functional:
            try:
                embeddings = _bulk_encode(semantic_cache, batch_prompts)
                if embeddings is not None:
                    for prompt, response, emb in zip(batch_prompts, batch_responses, embeddings):
                        try:
                            envelope = wrap(response, ttl)
                            semantic_cache.store_by_embedding(prompt, emb, envelope)
                            stats["semantic_stored"] += 1
                        except Exception:
                            stats["errors"] += 1
            except Exception as e:
                log.debug("warmup: bulk encode failed, falling back to sequential: %s", e)
                # Sequential fallback (slower but always works)
                for prompt, response in zip(batch_prompts, batch_responses):
                    try:
                        envelope = wrap(response, ttl)
                        semantic_cache.store(prompt, envelope)
                        stats["semantic_stored"] += 1
                    except Exception:
                        stats["errors"] += 1

        if progress is not None:
            try:
                progress(stats["total_loaded"], len(batch_prompts))
            except Exception:
                pass

        batch_prompts.clear()
        batch_responses.clear()

    for entry in source:
        try:
            prompt, response = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            stats["errors"] += 1
            continue
        if not isinstance(prompt, str) or not prompt:
            stats["errors"] += 1
            continue

        batch_prompts.append(prompt)
        batch_responses.append(response)
        stats["total_loaded"] += 1

        if len(batch_prompts) >= batch_size:
            _flush_batch()

    _flush_batch()
    stats["duration_s"] = round(time.perf_counter() - t0, 2)
    return stats


def warm_from_jsonl(
    path: str,
    *,
    prompt_extract: Callable[[dict], Optional[str]] = lambda r: r.get("prompt"),
    response_extract: Callable[[dict], Any] = lambda r: r.get("response"),
    model: str = "_default",
    store: Optional[Any] = None,
    semantic_cache: Optional[Any] = None,
    ttl: Optional[float] = None,
    batch_size: int = 100,
    progress: Optional[Callable[[int, int], None]] = None,
    max_entries: Optional[int] = None,
) -> dict:
    """Warm the cache by reading a JSONL log file.

    Each line is a JSON object. Use prompt_extract / response_extract to
    pull the relevant fields out of YOUR log format.

    Args:
        path: Path to a JSONL file.
        prompt_extract: dict -> str, returning the prompt text. Default
            extracts r["prompt"]. For OpenAI format:
                prompt_extract=lambda r: r["messages"][-1]["content"]
        response_extract: dict -> Any. Default extracts r["response"].
            For OpenAI format:
                response_extract=lambda r: r["choices"][0]["message"]["content"]
        max_entries: Cap the warm-up at this many entries.
        (rest: see warm_from_iterable)

    Returns:
        Stats dict (see warm_from_iterable).

    Example (OpenAI logs):
        from tokeymeter.engines.optimization.warmup import warm_from_jsonl
        warm_from_jsonl(
            "openai_logs.jsonl",
            prompt_extract=lambda r: r["messages"][-1]["content"],
            response_extract=lambda r: r["choices"][0]["message"]["content"],
            model="gpt-4o-mini",
        )
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"warmup: file not found: {path}")

    def _iterate():
        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    prompt = prompt_extract(rec)
                    response = response_extract(rec)
                except (KeyError, IndexError, TypeError):
                    continue
                if prompt is None or response is None:
                    continue
                yield (prompt, response)
                count += 1
                if max_entries is not None and count >= max_entries:
                    return

    return warm_from_iterable(
        _iterate(),
        model=model,
        store=store,
        semantic_cache=semantic_cache,
        ttl=ttl,
        batch_size=batch_size,
        progress=progress,
    )


# ---------- Internal: bulk encoding ----------

def _bulk_encode(semantic_cache, prompts: list):
    """Encode many prompts in one batch call. Returns a list of embeddings.

    sentence-transformers supports batch encoding natively and gives a
    ~25x speedup over sequential encoding. If the user's encoder doesn't
    support batching, fall back to None and the caller goes sequential.
    """
    if not prompts:
        return []

    encoder = getattr(semantic_cache, "_encoder", None)
    if encoder is None:
        return None

    # The default sentence-transformers encoder works on lists natively.
    # User-supplied encoders may not. We probe by calling with a list and
    # catching errors.
    try:
        result = encoder(prompts)
        # numpy: result.shape == (N, dim); we want a list of length N
        if hasattr(result, "shape") and len(result.shape) == 2 and result.shape[0] == len(prompts):
            return [result[i] for i in range(len(prompts))]
        # If it returned a list/tuple, hope each element is an embedding
        if isinstance(result, (list, tuple)) and len(result) == len(prompts):
            return list(result)
    except Exception:
        pass
    return None
