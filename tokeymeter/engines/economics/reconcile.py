"""
Reconciliation ledger — the bridge from "estimated" to "verified".

Tokeymeter's savings_report() estimates cost with a portable ~4-chars-per-token
heuristic (no tokenizer dependency). That is honest for a free local estimate,
but the company's central claim is a number that *reconciles against the
provider's bill*. This ledger closes that gap WITHOUT touching the locked engine:
the SDK wrappers feed it the provider's OWN reported token counts
(`response.usage`) on every real (miss) call, and it computes the exact cost the
provider will charge from the public price table.

The published number is then a comparison the reader can trust:
    engine-estimated spend   vs   provider-actual spend (reconcilable to invoice)
    engine-estimated saved   vs   provider-actual saved

Content-blind by construction: this ledger stores only model names and integer
token counts — never a prompt, never a response. It is process-local and resets
with the process, exactly like the engine's own savings counters.
"""
from __future__ import annotations

import threading
from typing import Dict

from tokeymeter import pricing

_lock = threading.Lock()
# model -> {"calls", "input_tokens", "output_tokens", "actual_cost_usd"}
_actual: Dict[str, Dict[str, float]] = {}


def record(*, model: str, input_tokens: int, output_tokens: int) -> None:
    """Record one real provider call's actual token usage (content-blind)."""
    cost = pricing.estimate_cost(model, input_tokens, output_tokens)
    with _lock:
        b = _actual.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "actual_cost_usd": 0.0}
        )
        b["calls"] += 1
        b["input_tokens"] += input_tokens
        b["output_tokens"] += output_tokens
        b["actual_cost_usd"] += cost


def report() -> dict:
    """Aggregate the actual-usage ledger. Pairs with tokeymeter.savings_report()."""
    with _lock:
        models = {m: dict(v) for m, v in _actual.items()}
    total_calls = sum(v["calls"] for v in models.values())
    total_in = sum(v["input_tokens"] for v in models.values())
    total_out = sum(v["output_tokens"] for v in models.values())
    total_cost = sum(v["actual_cost_usd"] for v in models.values())
    return {
        "provider_actual": {
            "real_calls": total_calls,             # calls that actually hit a provider
            "input_tokens": total_in,
            "output_tokens": total_out,
            "actual_spent_usd": round(total_cost, 6),
            "by_model": {m: {**v, "actual_cost_usd": round(v["actual_cost_usd"], 6)}
                         for m, v in models.items()},
        }
    }


def reconcile(savings: dict) -> dict:
    """Combine engine savings with provider-actual usage into one verdict.

    Args:
        savings: the dict returned by tokeymeter.savings_report().

    Returns a structure suitable for the report artifact: the engine's estimate,
    the provider-actual spend on the calls that really ran, and the implied
    saved figure (actual per-miss cost × the number of hits the engine served).
    """
    pa = report()["provider_actual"]
    real_calls = pa["real_calls"] or 0
    actual_spent = pa["actual_spent_usd"]
    avg_real_call = (actual_spent / real_calls) if real_calls else 0.0

    total_calls = savings.get("total_calls", 0)
    hits = savings.get("cache_hits", 0)
    # Each hit avoided a call that, on average, would have cost avg_real_call.
    actual_saved = round(avg_real_call * hits, 6)

    return {
        "engine_estimate": {
            "total_calls": total_calls,
            "cache_hits": hits,
            "hit_rate_pct": savings.get("hit_rate_pct", 0.0),
            "estimated_spent_usd": savings.get("estimated_spent_usd", 0.0),
            "estimated_saved_usd": savings.get("estimated_saved_usd", 0.0),
        },
        "provider_actual": pa,
        "reconciled": {
            "avg_real_call_usd": round(avg_real_call, 6),
            "actual_spent_usd": actual_spent,
            "actual_saved_usd": actual_saved,
            "actual_total_without_cache_usd": round(actual_spent + actual_saved, 6),
            "savings_pct": round(100.0 * actual_saved / (actual_spent + actual_saved), 2)
            if (actual_spent + actual_saved) > 0 else 0.0,
        },
        "note": (
            "actual_* figures derive from the provider's own reported token usage "
            "(response.usage) priced at public list rates; reconcile against your "
            "provider invoice for the authoritative bill."
        ),
    }


def reset() -> None:
    with _lock:
        _actual.clear()
