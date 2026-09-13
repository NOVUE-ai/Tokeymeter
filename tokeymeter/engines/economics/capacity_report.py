"""Capacity Recovery Report (S1) — the first sellable self-host artifact.

For a self-hoster the unit that matters is not dollars, it is GPU CAPACITY
returned to the cluster: inference the GPUs did NOT have to run. This report
enumerates that recovered capacity by MECHANISM, states each in GPU-seconds /
GPU-hours derived from the operator's own measured throughput, and — where a
$/GPU-hour is registered — a USD equivalent.

HONESTY BOUNDARY (non-negotiable): this report contains ONLY mechanisms that
are shipped AND measured in the ledger today — exact cache, semantic cache,
single-flight collapse. Mechanisms named in the roadmap but not yet
instrumented (batching, off-peak deferral, cross-team consolidation) are NOT
given fabricated numbers. They are listed under `not_yet_measured` so the
report is transparent about its own scope: a recovered-capacity figure here is
always a MEASURED recovery, never a projection. This is the same discipline
that made the token reconciliation tie to the provider's own dashboard.

Every figure carries a derivation string. Pricing provenance from the ledger is
surfaced verbatim: if any USD rests on the generic fallback rate, the report
says so and does not launder it as customer-specific truth.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# The SAME sanitizer S2 (chargeback) and S3 (hybrid) use. Imported rather than
# reimplemented so all three artifacts make byte-identical exclusion decisions:
# if capacity dropped a different set of corrupt records than chargeback did,
# the close packet's cross-artifact checks would compare different populations
# and "reconciled" would stop meaning anything.
from .chargeback import _record_numbers, _record_ts

# Mechanisms whose recovery is MEASURED, in report order.
_PERIOD_SCOPED_FIELDS = (
    "exact_hits", "semantic_hits", "single_flight_hits",
    "saved_input_tokens", "saved_output_tokens",
)


def _validate_capacity_period(period_start, period_end):
    """Validate period bounds the same way chargeback_report does: finite
    numbers, half-open [start, end), start strictly before end. Identical rules
    across the three artifacts mean a period that is legal for one is legal for
    all — a close packet can never be assembled from reports that disagreed
    about what the window even was."""
    import math as _m

    def _coerce(name, v):
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be epoch seconds or None, got {v!r}")
        if not _m.isfinite(f):
            raise ValueError(f"{name} must be a finite epoch timestamp, got {v!r}")
        return f

    start = _coerce("period_start", period_start)
    end = _coerce("period_end", period_end)
    if start is not None and end is not None and not (start < end):
        raise ValueError(
            f"period_start ({start}) must be strictly before period_end ({end}); "
            "the window is half-open [start, end)")
    return start, end


def _period_scoped_aggregate(
    records: Iterable[dict],
    start: Optional[float],
    end: Optional[float],
) -> Tuple[Dict[str, int], int, int]:
    """Rebuild the capacity-relevant slice of savings_report() over one period.

    Returns (aggregate, excluded_malformed, excluded_shadow_hits).

    WHY THIS EXISTS: savings_report() aggregates the WHOLE ledger and carries no
    period, so a "monthly close" was placing an all-time recovery figure beside
    one month's spend. This rebuilds only the fields the capacity report
    consumes, bounded to the period, on the same footing as S2/S3.

    RECOVERED-CAPACITY RULE: a record counts as recovered capacity iff
    `hit AND NOT shadow`. That rule is forced from two directions and satisfies
    both exactly:
      * it matches savings.py's saved-token accumulation (`not is_shadow and
        hit`), so an unbounded period reproduces the all-time numbers; and
      * it matches S2/S3's shadow-hit exclusion, so all three artifacts count
        the same population.
    A shadow hit is a measurement projection — the real call still executed, so
    no capacity was recovered. It is excluded and counted, never silently
    dropped.
    """
    bounded = start is not None or end is not None
    agg = {f: 0 for f in _PERIOD_SCOPED_FIELDS}
    excluded_malformed = 0
    excluded_shadow_hits = 0

    for rec in records:
        ts, ts_ok = _record_ts(rec, bounded)
        nums = _record_numbers(rec)
        if not ts_ok or nums is None:
            excluded_malformed += 1
            continue
        in_tok, out_tok, _cost = nums
        if start is not None and (ts is None or ts < start):
            continue
        if end is not None and (ts is None or ts >= end):
            continue

        hit = bool(rec.get("hit"))
        shadow = bool(rec.get("shadow"))
        if hit and shadow:
            excluded_shadow_hits += 1
            continue
        if not hit:
            continue

        agg["saved_input_tokens"] += in_tok
        agg["saved_output_tokens"] += out_tok
        hit_type = rec.get("hit_type")
        if hit_type == "exact":
            agg["exact_hits"] += 1
        elif hit_type == "semantic":
            agg["semantic_hits"] += 1
        elif hit_type == "single_flight":
            agg["single_flight_hits"] += 1
        # A hit with an unrecognised or absent hit_type still contributes its
        # recovered TOKENS (the capacity was genuinely recovered) but belongs to
        # no named mechanism. The apportionment below distributes tokens by hit
        # share, so such tokens are attributed across the known mechanisms
        # rather than vanishing — and if there are no known-mechanism hits at
        # all, the mechanism rows stay zero while the total still reports the
        # recovery. Nothing is invented; nothing is lost.

    return agg, excluded_malformed, excluded_shadow_hits

# Mechanisms that are instrumented and measured in the ledger today. Each maps a
# hit_type family to the recovered-capacity line it produces.
_MEASURED_MECHANISMS = ("exact_cache", "semantic_cache", "single_flight")

# Roadmap mechanisms that are NOT yet instrumented. Named here so the report is
# explicit that they are out of measured scope — never assigned a number.
_NOT_YET_MEASURED = (
    ("batching", "Requests coalesced into larger serving batches."),
    ("off_peak_deferral", "Deferrable work shifted out of peak windows."),
    ("consolidation", "Duplicate deployments merged across teams/endpoints."),
)


def _gpu_seconds(tokens: int, tps: float) -> float:
    return tokens / tps if tps > 0 else 0.0


def capacity_recovery_report(
    *,
    measured_tokens_per_second,
    gpu_hour_rate_usd=None,
    savings_report: Optional[dict] = None,
    period_start=None,
    period_end=None,
    records: Optional[Iterable[dict]] = None,
) -> Dict[str, object]:
    """Build the Capacity Recovery Report from the live savings ledger.

    Args:
      measured_tokens_per_second: sustained aggregate throughput of the serving
        stack, as the operator measured it. The single input that converts
        avoided tokens into avoided GPU time. Required.
      gpu_hour_rate_usd: optional. If supplied, each recovered-capacity line
        also carries a USD equivalent. Omit for a capacity-only report — no USD
        is ever invented from an assumed price.
      savings_report: optional pre-fetched savings_report() dict (mainly for
        tests / callers that already have one). Defaults to a live fetch.
        Mutually exclusive with period bounds — see below.
      period_start / period_end: optional epoch-second bounds, half-open
        [start, end), identical semantics to chargeback_report and
        hybrid_placement_report. Supplying either makes this a PERIOD report
        whose figures can sit beside period spend in a close packet.
      records: optional explicit record iterable (tests / callers holding a
        ledger slice). Defaults to the live ledger when period bounds are used.

    Scope: with no period bounds this reads the ENTIRE ledger — correct for a
    lifetime-recovery view, wrong beside one month's spend. The returned
    `period` and `scope` fields state which of the two you are holding, so a
    consumer (the close packet) can verify rather than assume.

    Returns a dict with: per-mechanism recovered capacity (GPU-seconds/hours,
    optional USD, derivation each), a total, the pricing-provenance block from
    the ledger, the explicit `not_yet_measured` list, and — for period reports —
    the exclusion counts that let it be reconciled against S2/S3.
    """
    tps = float(measured_tokens_per_second)
    # NOTE: nan compares False to everything, so `tps <= 0` alone would let a
    # NaN through and poison every derived figure. isfinite() closes that hole
    # (and rejects inf, which would silently zero all GPU-seconds).
    import math as _math
    if not _math.isfinite(tps) or tps <= 0:
        raise ValueError(
            "measured_tokens_per_second must be a finite number > 0, "
            f"got {measured_tokens_per_second!r}")
    usd_rate = None
    if gpu_hour_rate_usd is not None:
        usd_rate = float(gpu_hour_rate_usd)
        # same nan hole as tps: `< 0` alone lets nan through and every
        # equivalent_usd in the artifact becomes nan.
        if not _math.isfinite(usd_rate) or usd_rate < 0:
            raise ValueError(
                "gpu_hour_rate_usd must be a finite number >= 0, "
                f"got {gpu_hour_rate_usd!r}")

    bounded = period_start is not None or period_end is not None

    # A pre-aggregated savings_report has no timestamps, so it CANNOT be
    # period-filtered. Accepting both silently would return all-time numbers
    # under a period label — precisely the mislabelling this commit exists to
    # remove. Refuse the contradiction instead of resolving it arbitrarily.
    if savings_report is not None and (bounded or records is not None):
        raise ValueError(
            "savings_report cannot be combined with period_start/period_end or "
            "records: a pre-aggregated report carries no timestamps and cannot "
            "be period-filtered. Pass records (or leave both out for a live "
            "read) when you want a period-bounded capacity report.")

    start, end = _validate_capacity_period(period_start, period_end)

    excluded_malformed = 0
    excluded_shadow_hits = 0
    pricing_block: Dict[str, object] = {}

    if bounded or records is not None:
        # PERIOD-SCOPED PATH — rebuild the aggregate from raw records on the
        # same sanitizer/exclusion footing as S2 and S3.
        if records is None:
            from tokeymeter.engines.economics import savings as _sv
            records = list(_sv._tracker._iter_records())
        agg, excluded_malformed, excluded_shadow_hits = _period_scoped_aggregate(
            records, start, end)
        # Pricing provenance is a ledger-wide property (which models resolved
        # against which price source); it is surfaced verbatim and is not a
        # period figure, so it is fetched, never synthesised.
        try:
            from tokeymeter.engines.economics import savings as _sv2
            pricing_block = (_sv2.savings_report() or {}).get("pricing", {})
        except Exception:
            pricing_block = {}
        rep = agg
        scope = "period_bounded"
    else:
        # ALL-TIME PATH — unchanged behaviour for existing callers.
        if savings_report is None:
            # local import to avoid a module cycle at import time
            from tokeymeter.engines.economics import savings as _sv
            savings_report = _sv.savings_report()
        rep = savings_report
        pricing_block = rep.get("pricing", {})
        scope = "all_time_entire_ledger"

    # Recovered capacity is the token volume of LIVE hits (each hit is inference
    # the cluster did not run). The ledger already splits saved tokens by
    # input/output; the report sums them (a token-second is a token-second on
    # owned hardware). We attribute the recovered tokens across the three
    # measured mechanisms in proportion to their hit counts, since the ledger
    # aggregates saved tokens across all live hits rather than per-hit-type.
    exact = int(rep.get("exact_hits", 0) or 0)
    semantic = int(rep.get("semantic_hits", 0) or 0)
    single = int(rep.get("single_flight_hits", 0) or 0)
    total_hits = exact + semantic + single

    saved_in = int(rep.get("saved_input_tokens", 0) or 0)
    saved_out = int(rep.get("saved_output_tokens", 0) or 0)
    saved_tokens = saved_in + saved_out

    counts = {
        "exact_cache": exact,
        "semantic_cache": semantic,
        "single_flight": single,
    }
    labels = {
        "exact_cache": "Exact-cache hits — identical request served from cache.",
        "semantic_cache": "Semantic-cache hits — near-duplicate request served from cache.",
        "single_flight": "Single-flight collapses — concurrent duplicate computed once.",
    }

    # Largest-remainder apportionment of saved tokens across mechanisms by hit
    # share. Chosen over independent round() precisely so the mechanism rows
    # SUM EXACTLY to the total — a report whose columns don't add up fails the
    # first finance review it meets. Floors are assigned first; the remaining
    # tokens go to the largest fractional remainders (ties broken by mechanism
    # order, deterministically).
    tokens_by_mech = {name: 0 for name in _MEASURED_MECHANISMS}
    if total_hits > 0 and saved_tokens > 0:
        shares = {n: saved_tokens * (counts[n] / total_hits)
                  for n in _MEASURED_MECHANISMS}
        floors = {n: int(shares[n]) for n in _MEASURED_MECHANISMS}
        remainder = saved_tokens - sum(floors.values())
        by_frac = sorted(_MEASURED_MECHANISMS,
                         key=lambda n: shares[n] - floors[n], reverse=True)
        for n in by_frac[:remainder]:
            floors[n] += 1
        tokens_by_mech = floors

    mechanisms: List[dict] = []
    total_gpu_seconds = _gpu_seconds(saved_tokens, tps)
    for name in _MEASURED_MECHANISMS:
        hits = counts[name]
        tokens = tokens_by_mech[name]
        gpu_seconds = _gpu_seconds(tokens, tps)
        line: Dict[str, object] = {
            "mechanism": name,
            "label": labels[name],
            "hits": hits,
            "recovered_tokens": tokens,
            "gpu_seconds_recovered": round(gpu_seconds, 4),
            "gpu_hours_recovered": round(gpu_seconds / 3600.0, 6),
            "derivation": (
                "recovered_tokens = largest-remainder apportionment of "
                "saved_tokens_total by (mechanism_hits / total_live_hits) — "
                "mechanism rows sum exactly to the total; gpu_seconds = "
                "recovered_tokens / measured_tokens_per_second"),
        }
        if usd_rate is not None:
            line["equivalent_usd"] = round(gpu_seconds / 3600.0 * usd_rate, 6)
        mechanisms.append(line)

    total: Dict[str, object] = {
        "recovered_tokens_total": saved_tokens,
        "gpu_seconds_recovered_total": round(total_gpu_seconds, 4),
        "gpu_hours_recovered_total": round(total_gpu_seconds / 3600.0, 6),
        "measured_tokens_per_second": tps,
    }
    if usd_rate is not None:
        total["equivalent_usd_total"] = round(total_gpu_seconds / 3600.0 * usd_rate, 6)

    # Surface the ledger's pricing-provenance block verbatim so any fallback
    # USD is flagged, not laundered. Present under the same shape savings_report
    # uses.
    pricing_block = rep.get("pricing", {})

    out: Dict[str, object] = {
        "report": "capacity_recovery",
        "scope": scope,
        "period": {"start": start, "end": end},
        "measured_mechanisms": mechanisms,
        "total_recovered": total,
        "pricing": pricing_block,
        "not_yet_measured": [
            {"mechanism": name, "note": note,
             "status": "not instrumented — no measured recovery available"}
            for name, note in _NOT_YET_MEASURED
        ],
        "scope_note": (
            "Only shipped, measured recovery mechanisms are quantified here. "
            "Mechanisms under 'not_yet_measured' are roadmap items with no "
            "fabricated figures — a recovered-capacity number in this report is "
            "always a measured recovery."
            + ("" if scope == "period_bounded" else
               " SCOPE: this report covers the ENTIRE ledger, not a period — do "
               "not read it as period recovery beside period spend.")),
    }
    if scope == "period_bounded":
        # Only meaningful on the period path, where this report performed its
        # own record-level exclusion. Emitting them lets the close packet check
        # that capacity dropped exactly what chargeback dropped.
        out["excluded_shadow_hits"] = excluded_shadow_hits
        out["excluded_malformed_records"] = excluded_malformed
    return out
