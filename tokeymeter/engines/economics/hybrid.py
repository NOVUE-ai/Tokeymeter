"""Hybrid Placement Intelligence (S3) — the crown jewel.

Answers the question every self-hosting CTO must defend to the board and no
other tool can answer, because no other tool holds both sides in one ledger:

    "At current utilization, is self-hosting this workload cheaper than the
     API alternative — and at what utilization does the answer flip?"

For each workload group this report puts side by side, from the SAME ledger:
  - what the workload actually cost on owned hardware (booked, from records)
  - what the SAME executed token volume would have cost on each declared API
    alternative (registry-priced, blended by the workload's OWN input:output
    mix — never a generic blend)
  - the ratio, the verdict at current state, and the UTILIZATION FLIP
    THRESHOLD: the utilization above which self-hosting wins

EVIDENCE, NEVER ACTION. Placement changes are the customer's call through
their own change process; this module produces the document that makes the
call defensible. (Boundary rule: Tokeymeter never touches supply.)

DESIGN RULES (every one inherited from a prior audit):

  CROSS-REPORT CONSISTENCY. "Executed volume" here uses EXACTLY the S2
  chargeback semantics (hit=False records; cache hits consumed no compute;
  shadow hits excluded). A CTO who shows the board this report next to the
  chargeback statement must see the same executed numbers — two artifacts
  from one ledger that disagree kill both.

  RECOMPUTABLE BY HAND. Every ratio and threshold derives from displayed
  numbers with the formula printed. flip_utilization = selfhost_$per1M_at_
  full_utilization / api_$per1M_blended — an analyst can verify each cell.

  NO FABRICATED PRICES. An alternative with no registered/known price is
  reported as unpriced and excluded from verdicts — never guessed. A
  utilization above 1.0 (served volume exceeding declared capacity) is
  reported AS IS with an inconsistency flag — the declared capacity is wrong,
  and hiding that with a clamp would launder bad inputs into a clean report.

  NOTHING DROPPED. Missing dimensions land in "(unattributed)"; totals equal
  the ledger's executed totals for the period.

  CONTENT-BLIND. Dimensions and alternatives are operator-declared
  identifiers only.
"""
from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chargeback import (
    ALLOWED_DIMENSIONS, _csv_safe, _validate_period,
    _record_numbers, _record_ts,
)

_UNATTRIBUTED = "(unattributed)"


def _finite_pos(name, value, *, integer=False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    if integer:
        iv = int(v)
        if iv != v:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return iv
    return v


def _api_price_per_1m(model: str):
    """Resolve an alternative's (input,output) $/1M from the pricing registry
    WITH provenance. Returns (in_rate, out_rate, source) or (None, None,
    'unpriced...') — never a guessed number.

    CRITICAL: the registry returns a generic DEFAULT rate for unknown models
    (source='default'). A board-level build-vs-buy verdict must never rest on
    that fabricated generic — default/fallback sources are treated as
    UNPRICED here and excluded from verdicts, with the reason stated."""
    try:
        from tokeymeter.pricing import pricing_info
        info = pricing_info(model)
    except Exception:
        info = None
    if not info:
        return None, None, "unpriced"
    src = info.get("source") or "unknown"
    if src in ("default", "fallback", "unknown", "_default"):
        return None, None, f"unpriced (registry returned {src} rate; refusing to verdict on a generic price — register_pricing('{model}', ...) to enable)"
    in_rate = info.get("input_per_1m")
    out_rate = info.get("output_per_1m")
    if in_rate is None or out_rate is None:
        return None, None, "unpriced"
    try:
        in_rate = float(in_rate)
        out_rate = float(out_rate)
    except (TypeError, ValueError):
        return None, None, "unpriced"
    if not (math.isfinite(in_rate) and math.isfinite(out_rate)) or \
            in_rate < 0 or out_rate < 0:
        return None, None, "unpriced"
    return in_rate, out_rate, src


def hybrid_placement_report(
    *,
    gpu_count,
    gpu_hour_rate_usd,
    per_gpu_tokens_per_second,
    alternatives: Sequence[str] = (),
    selfhosted_models: Optional[Sequence[str]] = None,
    group_by: Sequence[str] = ("model",),
    period_start=None,
    period_end=None,
    records: Optional[Iterable[dict]] = None,
) -> Dict[str, object]:
    """Build the build-vs-buy placement analysis from the live ledger.

    Cluster inputs (operator-measured, all validated finite > 0):
      gpu_count: GPUs in the serving fleet for this estate.
      gpu_hour_rate_usd: fully-amortized $/GPU-hour (from
        register_cluster_costs / derive_cluster_gpu_hour_rate — use the
        full-precision value for exactness).
      per_gpu_tokens_per_second: sustained throughput of ONE GPU at full
        utilization. Explicitly per-GPU so fleet capacity = per_gpu_tps *
        gpu_count * seconds is unambiguous.

    alternatives: model names to price the same volume against (must be
      priced in the registry; unpriced ones are flagged, never guessed).

    selfhosted_models: the models that run on THIS estate's own hardware. A
      hybrid ledger contains both self-hosted and API-served records; only the
      operator knows which is which (pricing_source can't tell them apart —
      both are 'registered'). Records for models NOT in this set are API-served
      and are NOT counted as self-host booked cost — they are summarized
      separately under api_served_workloads so the estate view is complete
      without mislabeling API spend as GPU spend. If None (the default), ALL
      executed records are treated as self-hosted (backward-compatible with a
      single-estate ledger); the report flags this assumption.

    group_by / period / records: same semantics as chargeback_report.

    Returns rows per group with booked self-host cost, per-alternative
    API-equivalent cost, ratios, verdicts, flip thresholds; a separate
    api_served_workloads summary; fleet-level utilization (when the period is
    bounded); provenance throughout.
    """
    gpus = _finite_pos("gpu_count", gpu_count, integer=True)
    rate = _finite_pos("gpu_hour_rate_usd", gpu_hour_rate_usd)
    tps = _finite_pos("per_gpu_tokens_per_second", per_gpu_tokens_per_second)

    sh_set = None if selfhosted_models is None else set(selfhosted_models)

    dims = tuple(group_by)
    if not dims:
        raise ValueError("group_by needs at least one dimension")
    bad = [d for d in dims if d not in ALLOWED_DIMENSIONS]
    if bad:
        raise ValueError(
            f"unknown group_by dimension(s) {bad!r}; allowed: {ALLOWED_DIMENSIONS}")
    start = _validate_period("period_start", period_start)
    end = _validate_period("period_end", period_end)
    if start is not None and end is not None and end < start:
        raise ValueError("period_end must be >= period_start")

    # self-host $/1M at FULL utilization: one GPU-hour costs `rate` and serves
    # tps*3600 tokens. The flip-curve anchor; per-GPU consistent by definition.
    selfhost_full_util_per_1m = rate / (tps * 3600.0) * 1e6

    # resolve alternative prices once, with provenance
    alt_prices: Dict[str, Tuple[Optional[float], Optional[float], str]] = {}
    for alt in alternatives:
        alt_prices[alt] = _api_price_per_1m(alt)

    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = _sv._tracker._iter_records()

    rows: Dict[Tuple[str, ...], dict] = {}
    shadow_hits_excluded = 0
    # API-served workloads in the same ledger — summarized, NOT counted as
    # self-host booked cost (a hybrid ledger holds both; mislabeling API spend
    # as GPU spend is the failure this partition prevents).
    api_served: Dict[str, dict] = {}
    assume_all_selfhosted = sh_set is None
    malformed_excluded = 0
    bounded = start is not None or end is not None

    for rec in records:
        ts, ts_ok = _record_ts(rec, bounded)
        nums = _record_numbers(rec)
        if not ts_ok or nums is None:
            # identical exclusion rule to chargeback — the two board artifacts
            # must drop exactly the same records or they stop reconciling
            malformed_excluded += 1
            continue
        in_tok, out_tok, cost = nums
        if start is not None and (ts is None or ts < start):
            continue
        if end is not None and (ts is None or ts >= end):
            continue
        hit = bool(rec.get("hit"))
        shadow = bool(rec.get("shadow"))
        if hit and shadow:
            shadow_hits_excluded += 1
            continue
        if hit:
            # cache hits consumed no compute — identical S2 semantics; the
            # comparable placement volume is what EXECUTED
            continue

        model = rec.get("model")

        # Estate partition: a record is self-host ONLY if its model is declared
        # self-hosted (or if the caller declared nothing → single-estate mode).
        if not assume_all_selfhosted and model not in sh_set:
            mkey = str(model) if model is not None else _UNATTRIBUTED
            ab = api_served.get(mkey)
            if ab is None:
                ab = {"requests": 0, "in": 0, "out": 0, "cost_raw": 0.0}
                api_served[mkey] = ab
            ab["requests"] += 1
            ab["in"] += in_tok
            ab["out"] += out_tok
            ab["cost_raw"] += cost
            continue

        key = tuple((str(rec.get(d)) if rec.get(d) not in (None, "") else _UNATTRIBUTED)
                    for d in dims)
        b = rows.get(key)
        if b is None:
            b = {"in": 0, "out": 0, "n": 0, "booked_raw": 0.0,
                 "reported": 0, "estimated": 0,
                 "lat_sum": 0.0, "lat_n": 0, "qw_sum": 0.0, "qw_n": 0}
            rows[key] = b
        b["in"] += in_tok
        b["out"] += out_tok
        b["n"] += 1
        b["booked_raw"] += cost
        if rec.get("token_source") == "reported":
            b["reported"] += in_tok + out_tok
        else:
            b["estimated"] += in_tok + out_tok
        lat = rec.get("latency_ms")
        if isinstance(lat, (int, float)) and math.isfinite(lat):
            b["lat_sum"] += float(lat)
            b["lat_n"] += 1
        qw = rec.get("queue_wait_ms")
        if isinstance(qw, (int, float)) and math.isfinite(qw):
            b["qw_sum"] += float(qw)
            b["qw_n"] += 1

    out_rows: List[dict] = []
    for key in sorted(rows):
        b = rows[key]
        total_tok = b["in"] + b["out"]
        row: Dict[str, object] = {dims[i]: key[i] for i in range(len(dims))}
        row.update({
            "executed_requests": b["n"],
            "executed_input_tokens": b["in"],
            "executed_output_tokens": b["out"],
            "selfhost_booked_usd": round(b["booked_raw"], 6),
            "mean_latency_ms": round(b["lat_sum"] / b["lat_n"], 3) if b["lat_n"] else None,
            "mean_queue_wait_ms": round(b["qw_sum"] / b["qw_n"], 3) if b["qw_n"] else None,
            "provenance": {
                "reported_tokens": b["reported"],
                "estimated_tokens": b["estimated"],
            },
        })
        alts_out: List[dict] = []
        for alt in alternatives:
            in_rate, out_rate, src = alt_prices[alt]
            entry: Dict[str, object] = {"model": alt, "pricing_source": src}
            if in_rate is None:
                entry["status"] = "unpriced — excluded from verdicts, no price fabricated"
                alts_out.append(entry)
                continue
            api_cost = (b["in"] * in_rate + b["out"] * out_rate) / 1e6
            # blended $/1M at THIS row's own input:output mix (stated, so the
            # threshold is honest for this workload, not a generic blend)
            blended = ((b["in"] * in_rate + b["out"] * out_rate) / total_tok
                       if total_tok > 0 else None)
            entry["api_equivalent_usd"] = round(api_cost, 6)
            if b["booked_raw"] > 0 and api_cost > 0:
                entry["selfhost_over_api_ratio"] = round(b["booked_raw"] / api_cost, 4)
                entry["verdict_at_booked_cost"] = (
                    "selfhost_cheaper" if b["booked_raw"] < api_cost
                    else "api_cheaper")
            else:
                entry["selfhost_over_api_ratio"] = None
                entry["verdict_at_booked_cost"] = "insufficient_data"
            if blended and blended > 0:
                flip = selfhost_full_util_per_1m / blended
                entry["api_blended_usd_per_1m"] = round(blended, 6)
                entry["flip_utilization"] = round(flip, 6)
                entry["flip_meaning"] = (
                    "self-hosting is cheaper than this alternative when fleet "
                    "utilization exceeds flip_utilization"
                    if flip <= 1.0 else
                    "this alternative is cheaper than self-hosting at ANY "
                    "utilization up to 100% (flip_utilization > 1)")
            alts_out.append(entry)
        row["alternatives"] = alts_out
        out_rows.append(row)

    totals = {
        "executed_requests": sum(r["executed_requests"] for r in out_rows),
        "executed_input_tokens": sum(r["executed_input_tokens"] for r in out_rows),
        "executed_output_tokens": sum(r["executed_output_tokens"] for r in out_rows),
        "selfhost_booked_usd": round(
            sum(r["selfhost_booked_usd"] for r in out_rows), 6),
    }

    # fleet utilization needs a bounded period (capacity = tps*gpus*seconds)
    utilization = None
    utilization_note = None
    if start is not None and end is not None and end > start:
        capacity_tokens = tps * gpus * (end - start)
        served = totals["executed_input_tokens"] + totals["executed_output_tokens"]
        utilization = round(served / capacity_tokens, 6) if capacity_tokens > 0 else None
        if utilization is not None and utilization > 1.0:
            utilization_note = (
                "utilization > 1.0: observed served volume exceeds the declared "
                "capacity (per_gpu_tokens_per_second * gpu_count * period). The "
                "declared inputs are inconsistent with the ledger — reported AS "
                "IS, not clamped; re-measure throughput or check gpu_count.")
    else:
        utilization_note = (
            "utilization requires a bounded period (period_start and "
            "period_end); flip thresholds do not depend on it and are computed")

    # API-served workloads summary (separate from self-host booked cost)
    api_served_out = []
    for model in sorted(api_served):
        ab = api_served[model]
        api_served_out.append({
            "model": model,
            "requests": ab["requests"],
            "input_tokens": ab["in"],
            "output_tokens": ab["out"],
            "api_spend_usd": round(ab["cost_raw"], 6),
        })
    # total is the sum of the DISPLAYED api rows (matches the self-host totals'
    # construction) so it reconciles to its own rows exactly.
    api_served_total = round(sum(r["api_spend_usd"] for r in api_served_out), 6)

    # Whole-estate executed spend, from raw sums, rounded once — reconciles
    # per-model exactly with S2; the estate total is this single figure.
    selfhost_booked_raw = sum(b["booked_raw"] for b in rows.values())
    api_served_raw = sum(ab["cost_raw"] for ab in api_served.values())
    combined_executed_spend = round(selfhost_booked_raw + api_served_raw, 6)

    return {
        "report": "hybrid_placement",
        "group_by": list(dims),
        "period": {"start": start, "end": end},
        "estate_mode": ("all_records_assumed_selfhosted"
                        if assume_all_selfhosted else "declared_selfhosted_models"),
        "cluster": {
            "gpu_count": gpus,
            "gpu_hour_rate_usd": rate,
            "per_gpu_tokens_per_second": tps,
            "selfhost_usd_per_1m_at_full_utilization": round(
                selfhost_full_util_per_1m, 6),
        },
        "fleet_utilization": utilization,
        "utilization_note": utilization_note,
        "rows": out_rows,
        "totals": totals,
        "api_served_workloads": {
            "rows": api_served_out,
            "total_api_spend_usd": api_served_total,
            "note": ("models in the ledger NOT declared self-hosted — their "
                     "spend is API spend, shown separately and never counted as "
                     "self-host booked cost. Empty when estate_mode is "
                     "all_records_assumed_selfhosted."),
        },
        "total_executed_spend_usd": combined_executed_spend,
        "excluded_shadow_hits": shadow_hits_excluded,
        "excluded_malformed_records": malformed_excluded,
        "derivation": (
            "selfhost_usd_per_1m_at_full_utilization = gpu_hour_rate_usd / "
            "(per_gpu_tokens_per_second * 3600) * 1e6; "
            "api_equivalent_usd = (in_tokens*in_rate + out_tokens*out_rate)/1e6 "
            "at the row's own token mix; "
            "flip_utilization = selfhost_usd_per_1m_at_full_utilization / "
            "api_blended_usd_per_1m; fleet_utilization = served_tokens / "
            "(per_gpu_tokens_per_second * gpu_count * period_seconds). "
            "Executed volume uses the SAME semantics as chargeback_report "
            "(hit=False records; shadow hits excluded), so the two artifacts "
            "reconcile against each other."),
        "notes": (
            "EVIDENCE ONLY: placement changes are the operator's call through "
            "their own change process. Quality/eval dimensions are out of "
            "scope unless the operator supplies their own scores — no quality "
            "claim is fabricated here. Unpriced alternatives are flagged and "
            "excluded from verdicts."),
    }


def hybrid_placement_csv(report: Dict[str, object]) -> str:
    """Flatten the placement analysis to CSV (one line per row x alternative),
    with the same spreadsheet-injection defense as the chargeback export."""
    dims: List[str] = list(report["group_by"])  # type: ignore[index]
    headers = dims + [
        "executed_requests", "executed_input_tokens", "executed_output_tokens",
        "selfhost_booked_usd", "alternative", "alt_pricing_source",
        "api_equivalent_usd", "selfhost_over_api_ratio",
        "verdict_at_booked_cost", "api_blended_usd_per_1m", "flip_utilization",
    ]
    buf = io.StringIO()
    buf.write(",".join(_csv_safe(h) for h in headers) + "\r\n")
    for row in report["rows"]:  # type: ignore[union-attr]
        base = [row[d] for d in dims] + [
            row["executed_requests"], row["executed_input_tokens"],
            row["executed_output_tokens"], row["selfhost_booked_usd"],
        ]
        alts = row["alternatives"] or [{}]
        for a in alts:
            cells = base + [
                a.get("model"), a.get("pricing_source"),
                a.get("api_equivalent_usd"), a.get("selfhost_over_api_ratio"),
                a.get("verdict_at_booked_cost"),
                a.get("api_blended_usd_per_1m"), a.get("flip_utilization"),
            ]
            buf.write(",".join(_csv_safe(c) for c in cells) + "\r\n")
    return buf.getvalue()
