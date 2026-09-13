"""
Public list-price data for major LLM providers, used ONLY for savings estimation.

Important caveats:
  - These are public list prices and change frequently. Update quarterly.
  - We never charge anyone; this is for the "you saved $X" report only.
  - Prices are in USD per 1 million tokens.
  - When pricing is missing for a model, we fall back to a generic average
    rather than refusing to estimate.
  - These are best-effort public list prices for a "you saved $X" estimate, not
    a billing source of truth. Verify against the provider's pricing page for
    exact figures, and run shadow mode on your own workload for real numbers.
"""
from typing import Dict

# USD per 1M tokens. Snapshot: 2026-Q2 public list prices.
# This will go stale — refresh quarterly or pull from a hosted endpoint.
PRICING: Dict[str, Dict[str, float]] = {
    # ---- OpenAI ----
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # ---- Anthropic ----
    # opus-4 = legacy Opus 4/4.1 launch price; current Opus 4.6/4.7/4.8 are $5/$25.
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    # haiku-4 generation = Haiku 4.5 ($1/$5). ($0.80/$4 was legacy Haiku 3.5.)
    "claude-haiku-4": {"input": 1.00, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    # ---- Google ----
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.00},
    # gemini-3.5-flash verified 2026-06 against public list pricing (was
    # previously a copy of 2.5-flash's rate — corrected). Re-verify quarterly.
    "gemini-3.5-flash":  {"input": 1.50, "output": 9.00},
    # ---- Generic fallback ----
    "_default": {"input": 1.00, "output": 3.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost estimate for one call. Falls back to _default pricing."""
    p = _resolve_pricing(model)
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


def _resolve_pricing(model: str) -> Dict[str, float]:
    """Resolve a model name to its price entry, tolerating provider version
    suffixes. The real APIs often return dated/versioned strings like
    'gpt-4o-mini-2024-07-18' or 'claude-haiku-4-5-20251001'; an exact-match-only
    lookup would silently fall back to the generic default and distort the
    savings figure. We match exact first, then the longest registered prefix.
    """
    if not model:
        return PRICING["_default"]
    if model in PRICING:
        return PRICING[model]
    # longest-prefix match: 'gpt-4o-mini-2024-07-18' → 'gpt-4o-mini'
    best = None
    for known in PRICING:
        if known != "_default" and model.startswith(known):
            if best is None or len(known) > len(best):
                best = known
    return PRICING[best] if best else PRICING["_default"]


def estimate_tokens(text: str) -> int:
    """Crude token estimate: ~4 characters per token.

    This is rough by design — accurate tokenization requires the model's
    tokenizer, which we don't want as a hard dependency. The error is
    fine for savings estimation since misses and hits use the same heuristic.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)

# ═════════════════════════════════════════════════════════════════════════
PRICING_AS_OF = "2026-06-15"   # date the static list-price table was last
                               # verified against provider pricing pages.

def pricing_age_days(now=None) -> int:
    """Days since the static LIST table was verified. Registered (runtime)
    rates are always current by definition — this measures the table only."""
    import datetime as _dt
    ref = _dt.datetime.strptime(PRICING_AS_OF, "%Y-%m-%d")
    cur = (_dt.datetime.fromtimestamp(now, _dt.timezone.utc).replace(tzinfo=None)
           if now else _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None))
    return max(0, (cur - ref).days)


# Runtime pricing registry (v0.13) — the anti-fabrication layer.
#
# The static PRICING table above holds public list prices for hosted APIs.
# Self-hosted OSS models (vLLM/Ollama/TGI on the customer's own GPUs) have NO
# list price — before this registry, they silently fell through to `_default`
# ($1/$3 per 1M), producing a savings figure derived from prices that do not
# exist for that customer. That is fabrication, and it dies here:
#
#   1. `register_pricing()` lets an operator install true per-token rates for
#      any model at runtime (thread-safe; registered rates take precedence
#      over the static table).
#   2. `register_selfhost_pricing()` DERIVES those rates from two numbers the
#      operator can measure on their own cluster — GPU-hour cost and measured
#      throughput — and returns the full derivation so every figure is
#      auditable back to its inputs. Nothing is assumed.
#   3. `pricing_info()` / `estimate_cost_with_source()` expose the PROVENANCE
#      of every rate ("registered" / "list" / "default"), so the savings
#      report can label any USD figure that rests on the generic fallback
#      instead of passing it off as real.
#
# Invariant: a number derived from `_default` must never be presentable as a
# customer-specific truth without being flagged as such.
# ═════════════════════════════════════════════════════════════════════════
import math as _math
import threading as _threading

_RUNTIME_PRICING: Dict[str, Dict[str, float]] = {}
_pricing_lock = _threading.Lock()


def _validate_rate(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not _math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def register_pricing(model: str, *, input_per_1m, output_per_1m) -> Dict[str, float]:
    """Install (or overwrite) a runtime price for `model`, USD per 1M tokens.

    Registered prices take precedence over the static list-price table and
    participate in the same longest-prefix matching (so registering
    "kimi-k2" also covers "kimi-k2-instruct-0905"). Thread-safe. Returns the
    stored entry. Raises ValueError on non-finite or negative rates — a bad
    rate must fail loudly at registration, never silently at report time.
    """
    if not model or not isinstance(model, str):
        raise ValueError(f"model must be a non-empty string, got {model!r}")
    entry = {
        "input": _validate_rate("input_per_1m", input_per_1m),
        "output": _validate_rate("output_per_1m", output_per_1m),
    }
    with _pricing_lock:
        _RUNTIME_PRICING[model] = entry
    return dict(entry)


def unregister_pricing(model: str) -> bool:
    """Remove a runtime-registered price. Returns True if one was removed."""
    with _pricing_lock:
        return _RUNTIME_PRICING.pop(model, None) is not None


def clear_registered_pricing() -> None:
    """Remove ALL runtime-registered prices (test/reset hook)."""
    with _pricing_lock:
        _RUNTIME_PRICING.clear()


def registered_pricing() -> Dict[str, Dict[str, float]]:
    """Snapshot copy of the runtime registry (read-only view)."""
    with _pricing_lock:
        return {k: dict(v) for k, v in _RUNTIME_PRICING.items()}


def _longest_prefix(table: Dict[str, Dict[str, float]], model: str):
    best = None
    for known in table:
        if known != "_default" and model.startswith(known):
            if best is None or len(known) > len(best):
                best = known
    return best


def pricing_info(model: str) -> Dict[str, object]:
    """Resolve `model` to its rate AND its provenance.

    source is one of:
      "registered"        exact match in the runtime registry
      "registered_prefix" longest-prefix match in the runtime registry
      "list"              exact match in the static list-price table
      "list_prefix"       longest-prefix match in the static table
      "default"           generic fallback — NOT a real price for this model
    """
    if model:
        with _pricing_lock:
            if model in _RUNTIME_PRICING:
                e = _RUNTIME_PRICING[model]
                return {"model": model, "input_per_1m": e["input"],
                        "output_per_1m": e["output"], "source": "registered"}
            pfx = _longest_prefix(_RUNTIME_PRICING, model)
            if pfx is not None:
                e = _RUNTIME_PRICING[pfx]
                return {"model": model, "input_per_1m": e["input"],
                        "output_per_1m": e["output"], "source": "registered_prefix",
                        "matched": pfx}
        if model in PRICING:
            e = PRICING[model]
            return {"model": model, "input_per_1m": e["input"],
                    "output_per_1m": e["output"], "source": "list"}
        pfx = _longest_prefix(PRICING, model)
        if pfx is not None:
            e = PRICING[pfx]
            return {"model": model, "input_per_1m": e["input"],
                    "output_per_1m": e["output"], "source": "list_prefix",
                    "matched": pfx}
    e = PRICING["_default"]
    return {"model": model or "", "input_per_1m": e["input"],
            "output_per_1m": e["output"], "source": "default"}


def estimate_cost_with_source(model: str, input_tokens: int, output_tokens: int):
    """(cost_usd, source) — cost identical to estimate_cost(), plus provenance."""
    info = pricing_info(model)
    cost = ((input_tokens / 1_000_000) * info["input_per_1m"]
            + (output_tokens / 1_000_000) * info["output_per_1m"])
    return cost, info["source"]


def derive_selfhost_rate(*, gpu_hour_rate_usd, measured_tokens_per_second) -> Dict[str, object]:
    """Derive true USD-per-1M-token rates from measured cluster economics.

    Two inputs, both measurable by the operator — nothing assumed:
      gpu_hour_rate_usd:          fully-amortized cost of one GPU-hour
                                  (hardware amortization + power + hosting/ops,
                                  divided by hours; or the cloud on-demand rate)
      measured_tokens_per_second: sustained aggregate throughput of the serving
                                  stack on that GPU (prefill + decode combined,
                                  as measured on the real workload)

    Derivation (shown in the returned dict so it is auditable):
      tokens_per_gpu_hour = measured_tokens_per_second * 3600
      usd_per_1m_tokens   = gpu_hour_rate_usd / tokens_per_gpu_hour * 1_000_000

    Input and output tokens are priced identically: on your own hardware a
    token-second is a token-second at the throughput you measured. If prefill
    and decode throughput differ materially on your stack, measure them
    separately and register split rates via register_pricing().
    """
    rate = _validate_rate("gpu_hour_rate_usd", gpu_hour_rate_usd)
    tps = _validate_rate("measured_tokens_per_second", measured_tokens_per_second)
    if tps <= 0:
        raise ValueError("measured_tokens_per_second must be > 0")
    tokens_per_gpu_hour = tps * 3600.0
    per_1m = rate / tokens_per_gpu_hour * 1_000_000.0
    return {
        "gpu_hour_rate_usd": rate,
        "measured_tokens_per_second": tps,
        "tokens_per_gpu_hour": tokens_per_gpu_hour,
        "usd_per_1m_tokens": per_1m,
        "derivation": ("usd_per_1m = gpu_hour_rate_usd / "
                       "(measured_tokens_per_second * 3600) * 1e6"),
    }


def derive_cluster_gpu_hour_rate(
    *,
    gpu_count: int,
    gpu_capex_usd=None,
    depreciation_months=None,
    lease_usd_per_month=None,
    power_kw_per_gpu=None,
    power_usd_per_kwh=None,
    facility_overhead_factor=None,
    staff_usd_per_month=None,
    hours_per_month: float = 730.0,
) -> Dict[str, object]:
    """Derive a fully-amortized $/GPU-hour from SEPARATED, operator-measured
    cost components — the enterprise inputs a CFO audits line by line.

    Exactly one capital path is required:
      - OWNED: gpu_capex_usd + depreciation_months (straight-line), OR
      - LEASED: lease_usd_per_month
    Optional operating components, each added only if supplied (never assumed):
      - power:    power_kw_per_gpu * power_usd_per_kwh   (per GPU, per hour)
      - facility: facility_overhead_factor applied to the hardware+power base
                  (e.g. 1.15 = +15% for cooling/space/network)
      - staff:    staff_usd_per_month spread across gpu_count

    Every component appears in the returned dict, and `derivation` names the
    exact arithmetic, so the resulting $/GPU-hour is checkable by hand. Nothing
    is blended into an opaque rate — that transparency IS the enterprise value.

    Returns a dict with per-component monthly and per-GPU-hour figures plus the
    final `gpu_hour_rate_usd`.
    """
    gpus = int(gpu_count)
    if gpus <= 0:
        raise ValueError("gpu_count must be > 0")
    if hours_per_month <= 0:
        raise ValueError("hours_per_month must be > 0")

    owned = gpu_capex_usd is not None or depreciation_months is not None
    leased = lease_usd_per_month is not None
    if owned and leased:
        raise ValueError(
            "specify EITHER an owned path (gpu_capex_usd + depreciation_months) "
            "OR a leased path (lease_usd_per_month), not both")
    if not owned and not leased:
        raise ValueError(
            "capital cost required: either gpu_capex_usd + depreciation_months "
            "(owned) or lease_usd_per_month (leased)")

    components: Dict[str, float] = {}

    # ── capital: $/GPU-hour ──────────────────────────────────────────────
    if owned:
        capex = _validate_rate("gpu_capex_usd", gpu_capex_usd)
        months = _validate_rate("depreciation_months", depreciation_months)
        if months <= 0:
            raise ValueError("depreciation_months must be > 0")
        # straight-line: capex is for the WHOLE fleet of gpu_count GPUs
        capital_per_gpu_hour = (capex / months) / gpus / hours_per_month
        capital_basis = "owned_straight_line"
    else:
        lease = _validate_rate("lease_usd_per_month", lease_usd_per_month)
        # lease figure is for the WHOLE fleet
        capital_per_gpu_hour = lease / gpus / hours_per_month
        capital_basis = "leased"
    components["capital_per_gpu_hour"] = capital_per_gpu_hour

    # ── power: $/GPU-hour ────────────────────────────────────────────────
    power_per_gpu_hour = 0.0
    if power_kw_per_gpu is not None or power_usd_per_kwh is not None:
        if power_kw_per_gpu is None or power_usd_per_kwh is None:
            raise ValueError(
                "power requires BOTH power_kw_per_gpu and power_usd_per_kwh")
        kw = _validate_rate("power_kw_per_gpu", power_kw_per_gpu)
        kwh_rate = _validate_rate("power_usd_per_kwh", power_usd_per_kwh)
        power_per_gpu_hour = kw * kwh_rate  # kW * $/kWh over one hour
        components["power_per_gpu_hour"] = power_per_gpu_hour

    base_per_gpu_hour = capital_per_gpu_hour + power_per_gpu_hour

    # ── facility overhead: multiplicative on the hardware+power base ─────
    overhead_added = 0.0
    if facility_overhead_factor is not None:
        factor = _validate_rate("facility_overhead_factor", facility_overhead_factor)
        if factor < 1.0:
            raise ValueError(
                "facility_overhead_factor is a multiplier >= 1.0 "
                "(e.g. 1.15 = +15%); got a value below 1.0")
        overhead_added = base_per_gpu_hour * (factor - 1.0)
        components["facility_overhead_per_gpu_hour"] = overhead_added

    # ── staff: $/GPU-hour, spread across the fleet ───────────────────────
    staff_per_gpu_hour = 0.0
    if staff_usd_per_month is not None:
        staff = _validate_rate("staff_usd_per_month", staff_usd_per_month)
        staff_per_gpu_hour = staff / gpus / hours_per_month
        components["staff_per_gpu_hour"] = staff_per_gpu_hour

    gpu_hour_rate = (base_per_gpu_hour + overhead_added + staff_per_gpu_hour)

    # Display reconciliation: round each component to 6 places for display, then
    # derive the stated total by SUMMING those same rounded components. This
    # guarantees a finance analyst who adds the displayed column gets exactly
    # the stated gpu_hour_rate — full-precision rounding could leave a
    # last-digit mismatch (the column summing to a hair more/less than the
    # total), which reads as a broken report even at sub-cent scale. The
    # unrounded gpu_hour_rate is preserved as gpu_hour_rate_usd_full_precision
    # for any downstream math that wants it.
    displayed_components = {k: round(v, 6) for k, v in components.items()}
    displayed_rate = round(sum(displayed_components.values()), 6)

    return {
        "gpu_count": gpus,
        "hours_per_month": hours_per_month,
        "capital_basis": capital_basis,
        "components_per_gpu_hour": displayed_components,
        "gpu_hour_rate_usd": displayed_rate,
        "gpu_hour_rate_usd_full_precision": gpu_hour_rate,
        "derivation": (
            "gpu_hour_rate = (capital_per_gpu_hour + power_per_gpu_hour) "
            "* facility_overhead_factor + staff_per_gpu_hour; where "
            "capital_per_gpu_hour = "
            "(owned: (gpu_capex_usd / depreciation_months) | leased: "
            "lease_usd_per_month) / gpu_count / hours_per_month; "
            "power_per_gpu_hour = power_kw_per_gpu * power_usd_per_kwh; "
            "staff_per_gpu_hour = staff_usd_per_month / gpu_count / hours_per_month; "
            "the stated gpu_hour_rate_usd is the sum of the displayed "
            "(6-dp rounded) components, so the column reconciles exactly"),
    }


def register_cluster_costs(
    model: str,
    *,
    measured_tokens_per_second,
    gpu_count: int,
    gpu_capex_usd=None,
    depreciation_months=None,
    lease_usd_per_month=None,
    power_kw_per_gpu=None,
    power_usd_per_kwh=None,
    facility_overhead_factor=None,
    staff_usd_per_month=None,
    hours_per_month: float = 730.0,
) -> Dict[str, object]:
    """Enterprise self-host pricing: derive a $/GPU-hour from SEPARATED cost
    components, then a true $/1M-token rate, and register it — all in one call.

    This is the auditable path a CFO signs off on: capital, power, facility,
    and staff each stated and each shown in the derivation, rather than a single
    pre-blended gpu-hour number. The simpler two-number entry point
    (register_selfhost_pricing) is a thin wrapper over the same token
    derivation for operators who already have a blended $/GPU-hour.

    Returns the full component breakdown, the token-rate derivation, and the
    registered pricing entry — everything needed to paste into a report where
    every figure traces to a measured input.
    """
    cluster = derive_cluster_gpu_hour_rate(
        gpu_count=gpu_count,
        gpu_capex_usd=gpu_capex_usd,
        depreciation_months=depreciation_months,
        lease_usd_per_month=lease_usd_per_month,
        power_kw_per_gpu=power_kw_per_gpu,
        power_usd_per_kwh=power_usd_per_kwh,
        facility_overhead_factor=facility_overhead_factor,
        staff_usd_per_month=staff_usd_per_month,
        hours_per_month=hours_per_month,
    )
    rate = derive_selfhost_rate(
        gpu_hour_rate_usd=cluster.get("gpu_hour_rate_usd_full_precision",
                                      cluster["gpu_hour_rate_usd"]),
        measured_tokens_per_second=measured_tokens_per_second)
    entry = register_pricing(model, input_per_1m=rate["usd_per_1m_tokens"],
                             output_per_1m=rate["usd_per_1m_tokens"])
    return {
        "model": model,
        "registered": entry,
        "cluster_costs": cluster,
        "token_rate": rate,
    }


def register_selfhost_pricing(model: str, *, gpu_hour_rate_usd,
                              measured_tokens_per_second) -> Dict[str, object]:
    """Derive AND register a self-hosted model's true rate in one call.

    The SIMPLE entry point: for operators who already have a blended
    $/GPU-hour. For the auditable component breakdown (capital / power /
    facility / staff shown separately) use register_cluster_costs — this
    function is a thin wrapper over the same token-rate derivation.

    Returns the full derivation plus the registered entry, so the operator
    can paste it into a report and every number traces to a measured input.
    """
    d = derive_selfhost_rate(gpu_hour_rate_usd=gpu_hour_rate_usd,
                             measured_tokens_per_second=measured_tokens_per_second)
    entry = register_pricing(model, input_per_1m=d["usd_per_1m_tokens"],
                             output_per_1m=d["usd_per_1m_tokens"])
    return {"model": model, "registered": entry, **d}


def capacity_reclaimed(*, saved_input_tokens: int, saved_output_tokens: int,
                       measured_tokens_per_second,
                       gpu_hour_rate_usd=None) -> Dict[str, object]:
    """Convert tokens NOT generated (cache/single-flight hits) into GPU time.

    For a self-hoster the honest savings unit is not dollars — it is GPU
    capacity returned to the cluster. Every figure here derives from the
    caller's own measured inputs:
      gpu_seconds_reclaimed = saved_tokens / measured_tokens_per_second
    USD equivalence is included ONLY if the caller supplies their own
    gpu_hour_rate_usd — never from an assumed price.
    """
    tps = _validate_rate("measured_tokens_per_second", measured_tokens_per_second)
    if tps <= 0:
        raise ValueError("measured_tokens_per_second must be > 0")
    si = max(0, int(saved_input_tokens)); so = max(0, int(saved_output_tokens))
    total = si + so
    gpu_seconds = total / tps
    out: Dict[str, object] = {
        "saved_input_tokens": si,
        "saved_output_tokens": so,
        "saved_tokens_total": total,
        "measured_tokens_per_second": tps,
        "gpu_seconds_reclaimed": round(gpu_seconds, 4),
        "gpu_hours_reclaimed": round(gpu_seconds / 3600.0, 6),
        "derivation": "gpu_seconds = saved_tokens_total / measured_tokens_per_second",
    }
    if gpu_hour_rate_usd is not None:
        rate = _validate_rate("gpu_hour_rate_usd", gpu_hour_rate_usd)
        out["gpu_hour_rate_usd"] = rate
        out["equivalent_usd"] = round(gpu_seconds / 3600.0 * rate, 6)
    return out
