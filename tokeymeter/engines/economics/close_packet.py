"""Monthly Close Packet & General Ledger export (Phase S1.5).

The §3.12 export layer. Phase S1–S3 produced three artifacts; this module is
what lets them *reach a customer ritual* — the gate the build doc sets:
"value that cannot be delivered into a meeting is not yet value."

TWO OUTPUTS:

  close_packet(...) — composes the shipped reports into one ritual-ready
    object AND **proves they reconcile before finance sees them**. The packet
    is not a bundle; it is a bundle plus its own audit. If the components
    disagree, the packet says so loudly in `reconciliation.status` rather than
    shipping numbers that do not tie. A packet that quietly ships inconsistent
    artifacts is worse than no packet: it launders a defect into a filing.

  general_ledger_rows / general_ledger_csv — GL-shaped journal rows finance
    can import into its system of record, with the chargeback total preserved
    exactly.

DESIGN RULES (each inherited from a defect a prior audit found):

  ACCOUNTS ARE DECLARED, NEVER GUESSED. We cannot know a customer's chart of
  accounts. `account_mapping` is operator-supplied; anything unmapped posts to
  a declared suspense account and is FLAGGED in `unmapped_cost_centers`.
  Inventing an account code would put a fabricated identifier into a system of
  record — the finance equivalent of a fabricated number (§1.4).

  RECONCILIATION BY CONSTRUCTION. GL rows are rounded for display and the
  stated total is their sum, so the column an accountant adds equals the
  stated total. The GL total is then checked against the chargeback total it
  came from, and any difference is reported rather than absorbed.

  EXCLUSIONS TRAVEL WITH THE PACKET. Shadow hits and malformed records
  excluded by the underlying reports are surfaced at packet level. A close
  packet that hides its exclusions is not auditable.

  PERIOD COHERENCE IS VERIFIED. Components covering different periods is a
  real and silent failure mode (someone regenerates one report with a
  different window). The packet checks it and refuses to claim coherence it
  does not have.

  NO NEW NUMBERS. This module composes and verifies. It computes no cost, no
  token count, and no rate of its own — every figure traces to a component
  report that already carries provenance.

  CONTENT-BLIND. Cost centers are the operator-declared dimension values
  already in the ledger.
"""
from __future__ import annotations

import copy
import io
import math
import time
from typing import Dict, List, Optional, Sequence

from .chargeback import _csv_safe

# Cross-artifact totals can differ by display-ROUNDING ORDER (sum-of-rounded vs
# round-of-sum) — the documented S3 limit: per-row figures are exact, totals are
# not bit-identical across two independently-rounded reports.
#
# The tolerance MUST scale with row count. Each displayed row carries up to half
# a display unit (0.5e-6) of rounding, so n rows accumulate ~n*0.5e-6 worst case
# (~0.29e-6*sqrt(n) typically). A FIXED tolerance false-alarms on large estates:
# measured, a healthy 50,000-row close drifts 2.3e-4 and would have been reported
# as DISCREPANCY. A false alarm is worse than no check — it teaches finance to
# ignore the status field.
_BASE_TOLERANCE_USD = 1e-4        # floor, for small row counts
# Per-row rounding errors are independent and signed, so they accumulate as a
# RANDOM WALK (~sqrt(n)), not linearly. Measured across 10 → 200,000 rows the
# ratio drift/sqrt(n) is stable at 6e-7–9.5e-7, so this coefficient carries ~6x
# headroom over the worst observed drift at every scale.
#
# Linear scaling was rejected deliberately: at 200,000 rows it would allow $0.20
# of "rounding", which is ~90x the real drift and wide enough to HIDE a genuine
# error. A tolerance that is too loose is as much a defect as one that is too
# tight — the first hides errors, the second cries wolf.
_TOLERANCE_SQRT_COEFF_USD = 5e-6


def _tolerance_for(*row_counts: int) -> float:
    """Reconciliation tolerance for a comparison spanning the given row counts.

    Scales as sqrt(rows) because display-rounding error is a random walk. The
    figure is reported alongside every check so a reader can see exactly how
    much slack a comparison was given."""
    n = max([c for c in row_counts if isinstance(c, int)] or [0])
    return max(_BASE_TOLERANCE_USD, _TOLERANCE_SQRT_COEFF_USD * math.sqrt(n))


_SUSPENSE_DEFAULT = "UNMAPPED-SUSPENSE"

_REQUIRED_CHARGEBACK_KEYS = ("rows", "totals", "group_by")
_REQUIRED_HYBRID_KEYS = ("rows", "totals")


def _require_structure(report: dict, keys, label: str) -> None:
    """Fail with an explanatory error rather than a KeyError deep in the build.
    A truncated or hand-assembled report dict is a real input (someone loads a
    partial JSON), and 'KeyError: rows' tells an operator nothing."""
    missing = [k for k in keys if k not in report]
    if missing:
        raise ValueError(
            f"{label} report is missing required key(s) {missing}; it does not "
            f"look like a complete {label} report — regenerate it rather than "
            f"filing a packet built from a partial artifact")


def _period_of(report: Optional[dict]):
    if not report:
        return None
    p = report.get("period") or {}
    return (p.get("start"), p.get("end"))


def close_packet(
    *,
    chargeback: dict,
    hybrid: Optional[dict] = None,
    capacity: Optional[dict] = None,
    period_label: Optional[str] = None,
    generated_at: Optional[float] = None,
) -> Dict[str, object]:
    """Compose the monthly close packet and verify its components agree.

    Args:
      chargeback: a chargeback_report() result (required — it is the spine of
        the close).
      hybrid: an optional hybrid_placement_report() result.
      capacity: an optional capacity_recovery_report() result.
      period_label: human label for the ritual ("July 2026"). Free text; the
        authoritative period is the epoch window on the chargeback report.
      generated_at: epoch seconds; defaults to now.

    Returns a packet dict whose `reconciliation` block states plainly whether
    the components tie, and lists every check performed with its result.
    """
    if not isinstance(chargeback, dict) or chargeback.get("report") != "chargeback":
        raise ValueError(
            "chargeback must be a chargeback_report() result "
            "(got something else — the close packet composes real reports, "
            "it does not accept arbitrary dicts)")
    _require_structure(chargeback, _REQUIRED_CHARGEBACK_KEYS, "chargeback")
    if hybrid is not None:
        if hybrid.get("report") != "hybrid_placement":
            raise ValueError("hybrid must be a hybrid_placement_report() result")
        _require_structure(hybrid, _REQUIRED_HYBRID_KEYS, "hybrid_placement")
    if generated_at is not None:
        try:
            generated_at = float(generated_at)
        except (TypeError, ValueError):
            raise ValueError(
                f"generated_at must be epoch seconds or None, got {generated_at!r}")
        if not math.isfinite(generated_at):
            raise ValueError("generated_at must be a finite epoch timestamp")

    checks: List[dict] = []
    warnings: List[dict] = []

    def _check(name: str, ok: bool, detail: str) -> bool:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    def _warn(name: str, detail: str) -> None:
        # A warning is NOT a disagreement — it is a caveat a reader must know
        # (differing scope, degraded pricing). Keeping these out of `checks`
        # preserves the meaning of DISCREPANCY: the numbers do not tie.
        warnings.append({"warning": name, "detail": detail})

    # ── C0: every headline figure must be finite ────────────────────────
    # A corrupt or hand-built report can carry NaN/inf; without this the packet
    # would ship a nan headline while its checks "passed" vacuously (nan
    # comparisons are all False).
    cb_total = chargeback["totals"]["spend_usd"]
    _check("chargeback_total_is_finite",
           isinstance(cb_total, (int, float)) and math.isfinite(cb_total),
           f"chargeback total spend = {cb_total!r}")

    # ── C1: chargeback totals tie to its own displayed rows ─────────────
    cb_rows = chargeback["rows"]
    cb_rows_sum = round(sum(r["spend_usd"] for r in cb_rows), 6)
    tol_cb = _tolerance_for(len(cb_rows))
    _check("chargeback_total_equals_its_rows",
           math.isfinite(cb_total) and abs(cb_rows_sum - cb_total) <= tol_cb,
           f"rows sum {cb_rows_sum} vs stated total {cb_total} "
           f"(tolerance {tol_cb:.2e} for {len(cb_rows)} rows)")

    # ── C1b: pricing quality is a caveat, not a disagreement ────────────
    prov = chargeback.get("provenance") or {}
    if prov.get("all_rows_fully_priced") is False:
        _warn("not_all_rows_fully_priced",
              "at least one chargeback row rests on fallback/default pricing "
              "rather than a registered rate — the spend figure is directionally "
              "right but not customer-specific truth for those rows")

    # ── C2: hybrid totals tie to their own rows, and the estate total to
    #        the chargeback total (the cross-artifact promise) ───────────
    if hybrid is not None:
        hp_rows = hybrid["rows"]
        hp_rows_sum = round(
            sum(r["selfhost_booked_usd"] for r in hp_rows), 6)
        hp_total = hybrid["totals"]["selfhost_booked_usd"]
        tol_hp = _tolerance_for(len(hp_rows))
        _check("hybrid_selfhost_total_equals_its_rows",
               math.isfinite(hp_total) and abs(hp_rows_sum - hp_total) <= tol_hp,
               f"rows sum {hp_rows_sum} vs stated total {hp_total} "
               f"(tolerance {tol_hp:.2e} for {len(hp_rows)} rows)")

        # api_served has its own rows and its own stated total — unchecked, a
        # mismatch here (e.g. total 999 against rows summing to 5) passed as
        # "reconciled". Every stated total in the packet must tie to its rows.
        api_block = hybrid.get("api_served_workloads") or {}
        api_rows = api_block.get("rows") or []
        if "total_api_spend_usd" in api_block:
            api_sum = round(sum(r.get("api_spend_usd", 0.0) for r in api_rows), 6)
            api_total = api_block["total_api_spend_usd"]
            tol_api = _tolerance_for(len(api_rows))
            _check("hybrid_api_served_total_equals_its_rows",
                   isinstance(api_total, (int, float)) and math.isfinite(api_total)
                   and abs(api_sum - api_total) <= tol_api,
                   f"api rows sum {api_sum} vs stated total {api_total} "
                   f"(tolerance {tol_api:.2e} for {len(api_rows)} rows)")

        estate = hybrid.get("total_executed_spend_usd")
        if estate is not None:
            tol_estate = _tolerance_for(len(cb_rows), len(hp_rows), len(api_rows))
            diff = (abs(estate - cb_total)
                    if math.isfinite(estate) and math.isfinite(cb_total)
                    else float("inf"))
            _check("hybrid_estate_total_matches_chargeback_spend",
                   diff <= tol_estate,
                   f"hybrid estate {estate} vs chargeback {cb_total} "
                   f"(diff {diff:.2e}; tolerance {tol_estate:.2e})")

        # executed_requests is a count, not a rounded figure — it must match
        # EXACTLY. Both reports claim to cover the same executed population.
        cb_reqs = (chargeback.get("totals") or {}).get("executed_requests")
        hp_reqs = (hybrid.get("totals") or {}).get("executed_requests")
        api_reqs = sum(r.get("requests", 0) for r in api_rows)
        if cb_reqs is not None and hp_reqs is not None:
            _check("executed_request_counts_match",
                   cb_reqs == hp_reqs + api_reqs,
                   f"chargeback {cb_reqs} vs hybrid selfhost {hp_reqs} + "
                   f"api-served {api_reqs} = {hp_reqs + api_reqs}")

        # exclusions must match — divergent exclusion decisions between the two
        # artifacts silently break per-model reconciliation
        _check("exclusion_decisions_identical",
               (chargeback.get("excluded_shadow_hits")
                == hybrid.get("excluded_shadow_hits")
                and chargeback.get("excluded_malformed_records")
                == hybrid.get("excluded_malformed_records")),
               f"chargeback shadow/malformed "
               f"{chargeback.get('excluded_shadow_hits')}/"
               f"{chargeback.get('excluded_malformed_records')} vs hybrid "
               f"{hybrid.get('excluded_shadow_hits')}/"
               f"{hybrid.get('excluded_malformed_records')}")

        # ── C3: period coherence ────────────────────────────────────────
        _check("components_cover_same_period",
               _period_of(chargeback) == _period_of(hybrid),
               f"chargeback {_period_of(chargeback)} vs hybrid {_period_of(hybrid)}")

    # ── C4: capacity report internal consistency (rows sum to total) ────
    if capacity is not None:
        mech = capacity.get("measured_mechanisms") or []
        mech_sum = sum(m["recovered_tokens"] for m in mech)
        cap_total = (capacity.get("total_recovered") or {}).get(
            "recovered_tokens_total", 0)
        _check("capacity_mechanisms_sum_to_total",
               mech_sum == cap_total,
               f"mechanism rows {mech_sum} vs total {cap_total}")

        # PERIOD COHERENCE FOR CAPACITY. Previously this could only be a
        # warning: capacity_recovery_report carried no period, so an all-time
        # recovery figure sat beside one month's spend and all the packet could
        # do was caution the reader. The report is now period-aware, so this is
        # a real CHECK — the packet verifies the scopes match instead of
        # disclaiming that they might not.
        cap_scope = capacity.get("scope")
        if cap_scope == "period_bounded":
            _check("capacity_period_matches",
                   _period_of(capacity) == _period_of(chargeback),
                   f"capacity {_period_of(capacity)} vs "
                   f"chargeback {_period_of(chargeback)}")
            # exclusion parity: capacity now performs its own record-level
            # exclusion on the shared sanitizer, so it must have dropped
            # exactly what the chargeback spine dropped
            if "excluded_malformed_records" in capacity:
                _check("capacity_exclusions_match_chargeback",
                       (capacity.get("excluded_malformed_records")
                        == chargeback.get("excluded_malformed_records")
                        and capacity.get("excluded_shadow_hits")
                        == chargeback.get("excluded_shadow_hits")),
                       f"capacity shadow/malformed "
                       f"{capacity.get('excluded_shadow_hits')}/"
                       f"{capacity.get('excluded_malformed_records')} vs "
                       f"chargeback {chargeback.get('excluded_shadow_hits')}/"
                       f"{chargeback.get('excluded_malformed_records')}")
            # recovered capacity and chargeback's avoided side are the same
            # population counted two ways; they must agree exactly
            cb_avoided = (chargeback.get("totals") or {}).get("avoided_tokens")
            cap_tokens = (capacity.get("total_recovered") or {}).get(
                "recovered_tokens_total")
            if cb_avoided is not None and cap_tokens is not None:
                _check("capacity_recovered_matches_chargeback_avoided",
                       cb_avoided == cap_tokens,
                       f"chargeback avoided_tokens {cb_avoided} vs capacity "
                       f"recovered_tokens {cap_tokens}")
        elif (chargeback.get("period") or {}).get("start") is not None or (
                chargeback.get("period") or {}).get("end") is not None:
            # Still supported, still honest: an all-time capacity report inside
            # a period packet is a scope caveat, not a disagreement.
            _warn("capacity_scope_is_all_time",
                  "the capacity recovery figures cover the ENTIRE ledger, not "
                  "the packet's period — do not read them as period recovery "
                  "beside the period spend. Pass period_start/period_end to "
                  "capacity_recovery_report to make this a verified check.")

    all_passed = all(c["passed"] for c in checks)

    packet: Dict[str, object] = {
        "report": "close_packet",
        "period_label": period_label,
        "period": chargeback.get("period"),
        "generated_at": float(generated_at if generated_at is not None
                              else time.time()),
        "reconciliation": {
            "status": "reconciled" if all_passed else "DISCREPANCY",
            "checks": checks,
            "warnings": warnings,
            "tolerance_policy": (
                f"max({_BASE_TOLERANCE_USD}, {_TOLERANCE_SQRT_COEFF_USD} * sqrt(rows)) — "
                f"display-rounding error accumulates as a random walk, so a fixed "
                f"tolerance false-alarms on large estates and a linear one is "
                f"loose enough to hide real errors"),
            "note": (
                "Every check above was run against the component reports in "
                "this packet. status='DISCREPANCY' means at least one artifact "
                "does not tie — investigate before the close; do not file. "
                "Cross-artifact totals use a row-count-scaled display-rounding "
                "tolerance because two independently-rounded reports can differ "
                "by rounding order; per-row and per-group figures are exact. "
                "`warnings` are caveats a reader must know (differing scope, "
                "degraded pricing) — they do NOT mean the numbers disagree."),
        },
        "spend": {
            "total_usd": cb_total,
            "executed_requests": chargeback["totals"]["executed_requests"],
            "group_by": chargeback["group_by"],
            "rows": copy.deepcopy(chargeback["rows"]),
        },
        "exclusions": {
            "shadow_hits": chargeback.get("excluded_shadow_hits", 0),
            "malformed_records": chargeback.get("excluded_malformed_records", 0),
            "note": ("Records excluded by the underlying reports, surfaced here "
                     "so the packet is auditable. Malformed records are corrupt "
                     "ledger lines excluded whole, never partially counted."),
        },
        "provenance": copy.deepcopy(chargeback.get("provenance")),
        "components": {
            "chargeback": True,
            "hybrid_placement": hybrid is not None,
            "capacity_recovery": capacity is not None,
        },
    }

    if capacity is not None:
        packet["capacity_recovery"] = copy.deepcopy({
            "scope": capacity.get("scope", "all_time_entire_ledger"),
            "period": capacity.get("period"),
            "total_recovered": capacity.get("total_recovered"),
            "measured_mechanisms": capacity.get("measured_mechanisms"),
            "not_yet_measured": capacity.get("not_yet_measured"),
        })
    if hybrid is not None:
        packet["placement"] = copy.deepcopy({
            "estate_mode": hybrid.get("estate_mode"),
            "cluster": hybrid.get("cluster"),
            "fleet_utilization": hybrid.get("fleet_utilization"),
            "utilization_note": hybrid.get("utilization_note"),
            "rows": hybrid.get("rows"),
            "api_served_workloads": hybrid.get("api_served_workloads"),
            "total_executed_spend_usd": hybrid.get("total_executed_spend_usd"),
        })

    return packet


# ── General Ledger export ───────────────────────────────────────────────

def general_ledger_rows(
    *,
    chargeback: dict,
    account_mapping: Optional[Dict[str, str]] = None,
    suspense_account: str = _SUSPENSE_DEFAULT,
    cost_center_dimension: Optional[str] = None,
    currency: str = "USD",
    reference_prefix: str = "TOKEYMETER",
) -> Dict[str, object]:
    """Build GL-shaped journal rows from a chargeback report.

    account_mapping maps a cost-center value (e.g. the team tag) to the
    customer's own account code. **Nothing is guessed:** an unmapped cost
    centre posts to `suspense_account` and is listed in `unmapped_cost_centers`
    so finance can complete the mapping rather than discover a wrong posting
    later.

    cost_center_dimension selects which grouping dimension is the cost centre;
    defaults to the first dimension the chargeback report grouped by.

    Returns {"rows": [...], "total_usd": ..., "unmapped_cost_centers": [...],
             "currency": ..., "columns": [...]} where total_usd is the sum of
    the DISPLAYED row amounts.
    """
    if not isinstance(chargeback, dict) or chargeback.get("report") != "chargeback":
        raise ValueError("chargeback must be a chargeback_report() result")
    dims: List[str] = list(chargeback["group_by"])
    if cost_center_dimension is None:
        cost_center_dimension = dims[0]
    if cost_center_dimension not in dims:
        raise ValueError(
            f"cost_center_dimension {cost_center_dimension!r} is not one of the "
            f"report's group_by dimensions {dims}")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must be a non-empty string")

    mapping = dict(account_mapping or {})
    period = chargeback.get("period") or {}
    p_start, p_end = period.get("start"), period.get("end")
    ref = f"{reference_prefix}:CHARGEBACK"

    rows: List[dict] = []
    unmapped: List[str] = []
    for r in chargeback["rows"]:
        cc = str(r[cost_center_dimension])
        account = mapping.get(cc)
        if account is None:
            account = suspense_account
            if cc not in unmapped:
                unmapped.append(cc)
        amount = r["spend_usd"]
        # A non-finite amount must never reach a system of record. Chargeback
        # sanitizes its inputs, but a GL can also be built from a hand-loaded or
        # partial report — refuse loudly rather than post NaN to an account.
        if not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError(
                f"cost centre {cc!r} has a non-finite amount ({amount!r}); "
                f"refusing to emit a GL row — regenerate the chargeback report")
        prov = r.get("provenance") or {}
        rows.append({
            "period_start": p_start,
            "period_end": p_end,
            "account": account,
            "cost_center": cc,
            "description": f"AI execution spend — {cc}",
            "amount_usd": amount,                  # already display-rounded
            "currency": currency,
            "reference": ref,
            "executed_requests": r["executed_requests"],
            "fully_priced": prov.get("fully_priced", True),
            "reported_tokens": prov.get("reported_tokens", 0),
            "estimated_tokens": prov.get("estimated_tokens", 0),
        })

    total = round(sum(r["amount_usd"] for r in rows), 6)
    return {
        "rows": rows,
        "total_usd": total,
        "currency": currency,
        "cost_center_dimension": cost_center_dimension,
        "unmapped_cost_centers": unmapped,
        "suspense_account": suspense_account,
        "columns": ["period_start", "period_end", "account", "cost_center",
                    "description", "amount_usd", "currency", "reference",
                    "executed_requests", "fully_priced", "reported_tokens",
                    "estimated_tokens"],
        "note": (
            "GL-shaped journal rows for import into the customer's system of "
            "record. Account codes are OPERATOR-DECLARED via account_mapping; "
            "unmapped cost centres post to the suspense account and are listed "
            "in unmapped_cost_centers — no account code is ever invented. "
            "total_usd is the sum of the displayed row amounts and ties to the "
            "chargeback report's spend total."),
    }


def general_ledger_csv(gl: Dict[str, object]) -> str:
    """Flatten general_ledger_rows() to CSV with the same spreadsheet
    formula-injection defense as the chargeback export (a cost-centre value is
    operator-supplied text and lands in Excel)."""
    cols: List[str] = list(gl["columns"])  # type: ignore[index]
    buf = io.StringIO()
    buf.write(",".join(_csv_safe(c) for c in cols) + "\r\n")
    for row in gl["rows"]:  # type: ignore[union-attr]
        buf.write(",".join(_csv_safe(row.get(c)) for c in cols) + "\r\n")
    total_row = ["" for _ in cols]
    total_row[cols.index("account")] = "TOTAL"
    total_row[cols.index("amount_usd")] = gl["total_usd"]  # type: ignore[index]
    total_row[cols.index("currency")] = gl["currency"]     # type: ignore[index]
    buf.write(",".join(_csv_safe(c) for c in total_row) + "\r\n")
    return buf.getvalue()
