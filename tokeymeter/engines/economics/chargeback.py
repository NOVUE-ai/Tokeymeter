"""Unit-Cost & Chargeback Ledger (S2) — the monthly-close artifact.

Solves the pain finance is actively demanding an answer to: a shared AI estate
(API + owned GPUs) with no way to allocate cost to the teams consuming it.
This module turns the execution ledger into a chargeback statement: spend by
team/model/endpoint/principal for a period, in the same dollar units as the
API bill, exportable to CSV for the close.

DESIGN RULES (each one is a lesson a prior audit taught):

  RECONCILIATION BY CONSTRUCTION. Every displayed row dollar is rounded to
  6dp, and the displayed total is the SUM OF THE ROUNDED ROWS — an analyst
  who adds the column gets the stated total, always. (The S1 re-audit found
  a last-digit mismatch when totals came from unrounded values; that class
  of bug is structurally excluded here.)

  SPEND = WHAT ACTUALLY EXECUTED. A record with hit=False executed on real
  hardware/API and is billable — including shadow-mode real calls (shadow
  measures, but the wrapped call still ran and still cost). Cache hits are
  NOT billable: they consumed no upstream compute. They appear separately as
  the AVOIDED columns (what caching saved each group) — informational credit,
  never netted into spend. Shadow HITS are measurement projections and are
  excluded from both spend and avoided.

  NOTHING IS SILENTLY DROPPED. Records with a missing dimension value fall
  into an explicit "(unattributed)" bucket. The report's totals equal the
  ledger's totals for the period — a chargeback that quietly loses records
  is a chargeback finance cannot trust.

  HONESTY IS PER-ROW. Each row carries its provenance mix: how many of its
  tokens are provider-reported vs estimated, and which pricing sources priced
  it (registered / list / fallback). A row resting on fallback pricing is
  visibly flagged, never laundered as customer-specific truth.

  CONTENT-BLIND. Dimensions are operator-declared identifiers (tag, model,
  endpoint, principal, key name). No prompt or response content exists in the
  ledger, so none can exist here.

  EXCEL IS THE REAL CONSUMER. The CSV export neutralizes spreadsheet formula
  injection: any cell beginning with = + - @ (or tab/CR) is prefixed with a
  quote. Tags are user-supplied strings; a tag named "=HYPERLINK(...)" must
  land in a finance workbook as text, not as an executing formula.
"""
from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Dimensions a chargeback statement may group by — exactly the attribution
# fields that exist on CallRecord. Validated loudly: a typo'd dimension must
# fail at call time, not produce an empty report.
ALLOWED_DIMENSIONS: Tuple[str, ...] = (
    "tag", "model", "endpoint_identity", "principal", "key_name", "task_id",
    # `agent` is the KIND of task. task_id answers "what did this ticket cost";
    # agent answers "what does the diagnostic agent cost us" — the aggregate
    # finance actually budgets against.
    "agent",
)

_UNATTRIBUTED = "(unattributed)"


def _record_numbers(rec: dict):
    """Extract (input_tokens, output_tokens, estimated_cost) from a record with
    every value verified finite and non-negative, or None if ANY is corrupt.

    A deployed ledger will eventually contain a damaged line that still parses
    (bit-flip, partial write, buggy external producer): NaN poisons every total
    it touches, inf propagates, a NEGATIVE cost silently REDUCES the bill, and
    a non-numeric token count crashes the close. A record whose numbers can't
    be trusted can't be half-trusted either — partial inclusion would break the
    cost/token cross-checks — so the caller excludes the WHOLE record and
    counts it visibly (excluded_malformed_records). Shared by chargeback and
    hybrid so the two board artifacts always make the SAME exclusion decision
    and keep reconciling."""
    try:
        in_f = float(rec.get("input_tokens") or 0)
        out_f = float(rec.get("output_tokens") or 0)
        c_f = float(rec.get("estimated_cost") or 0.0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(in_f) and math.isfinite(out_f) and math.isfinite(c_f)):
        return None
    if in_f < 0 or out_f < 0 or c_f < 0:
        return None
    return int(in_f), int(out_f), c_f


def _record_ts(rec: dict, bounded: bool):
    """Return (ts, ok). When the report is period-bounded, a timestamp that
    isn't a finite number makes the record malformed (ok=False) instead of
    crashing the comparison; unbounded reports never read ts, so anything
    passes."""
    ts = rec.get("timestamp")
    if not bounded:
        return ts, True
    try:
        ts_f = float(ts)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(ts_f):
        return None, False
    return ts_f, True


def _validate_period(name, value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an epoch timestamp (seconds) or None, "
                         f"got {value!r}")
    # nan compares False to everything — an unguarded nan period would silently
    # exclude every record (nan comparisons all False). Same hole class as the
    # capacity report's tps guard; closed the same way.
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _row_key(rec: dict, dims: Sequence[str]) -> Tuple[str, ...]:
    # str() coercion: a non-string dimension value (e.g. an int model id from an
    # external producer) must not crash sorted() on mixed-type tuples or the
    # CSV; it becomes its string form, deterministically.
    return tuple(
        (str(rec.get(d)) if rec.get(d) not in (None, "") else _UNATTRIBUTED)
        for d in dims
    )


def chargeback_report(
    *,
    group_by: Sequence[str] = ("tag",),
    period_start=None,
    period_end=None,
    records: Optional[Iterable[dict]] = None,
) -> Dict[str, object]:
    """Build the chargeback statement from the live ledger.

    Args:
      group_by: attribution dimensions, in order (e.g. ("tag",) for per-team,
        ("tag", "model") for per-team-per-model). Each must be one of
        ALLOWED_DIMENSIONS; anything else raises.
      period_start / period_end: optional epoch-second bounds, half-open
        [start, end). Epoch-based so the statement is timezone-independent —
        the caller decides what "July" means in their books.
      records: optional pre-fetched record iterable (tests / callers holding
        one). Defaults to the live ledger.

    Returns a dict:
      rows: one per group — spend_usd, executed request/token counts, the
        avoided (cache-credit) columns, and a per-row provenance block
      totals: sums that reconcile to the rows BY CONSTRUCTION
      provenance: report-level honesty block
      excluded_shadow_hits: count of shadow-hit records excluded (visibility,
        never silent)
    """
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

    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = _sv._tracker._iter_records()

    rows: Dict[Tuple[str, ...], dict] = {}
    shadow_hits_excluded = 0
    malformed_excluded = 0
    bounded = start is not None or end is not None

    def _bucket(key: Tuple[str, ...]) -> dict:
        b = rows.get(key)
        if b is None:
            b = {
                "spend_usd_raw": 0.0,
                "executed_requests": 0,
                "executed_input_tokens": 0,
                "executed_output_tokens": 0,
                "reported_tokens": 0,
                "estimated_tokens": 0,
                "avoided_requests": 0,
                "avoided_tokens": 0,
                "avoided_usd_raw": 0.0,
                "pricing_sources": {},
            }
            rows[key] = b
        return b

    for rec in records:
        ts, ts_ok = _record_ts(rec, bounded)
        nums = _record_numbers(rec)
        if not ts_ok or nums is None:
            # corrupt numerics or (when bounded) corrupt timestamp: exclude the
            # whole record, visibly — never poison, never crash, never silent
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
            # measurement projection, not a real execution and not a real
            # avoidance — excluded from the statement, counted for visibility
            shadow_hits_excluded += 1
            continue

        b = _bucket(_row_key(rec, dims))

        if not hit:
            # executed on real hardware/API — billable (shadow real calls too)
            b["spend_usd_raw"] += cost
            b["executed_requests"] += 1
            b["executed_input_tokens"] += in_tok
            b["executed_output_tokens"] += out_tok
            if rec.get("token_source") == "reported":
                b["reported_tokens"] += in_tok + out_tok
            else:
                b["estimated_tokens"] += in_tok + out_tok
            ps = rec.get("pricing_source") or "unknown"
            b["pricing_sources"][ps] = b["pricing_sources"].get(ps, 0) + 1
        else:
            # live cache hit: consumed no upstream compute — credit columns
            b["avoided_requests"] += 1
            b["avoided_tokens"] += in_tok + out_tok
            b["avoided_usd_raw"] += cost

    # ── assemble rows with reconciliation by construction ────────────────
    out_rows: List[dict] = []
    for key in sorted(rows):
        b = rows[key]
        row: Dict[str, object] = {dims[i]: key[i] for i in range(len(dims))}
        row.update({
            "spend_usd": round(b["spend_usd_raw"], 6),
            "executed_requests": b["executed_requests"],
            "executed_input_tokens": b["executed_input_tokens"],
            "executed_output_tokens": b["executed_output_tokens"],
            "avoided_requests": b["avoided_requests"],
            "avoided_tokens": b["avoided_tokens"],
            "avoided_usd": round(b["avoided_usd_raw"], 6),
            "provenance": {
                "reported_tokens": b["reported_tokens"],
                "estimated_tokens": b["estimated_tokens"],
                "pricing_sources": dict(sorted(b["pricing_sources"].items())),
                # a row priced off fallback/default is flagged, not laundered
                "fully_priced": all(
                    ps not in ("fallback", "default", "unknown", "_default")
                    for ps in b["pricing_sources"]
                ) if b["pricing_sources"] else True,
            },
        })
        out_rows.append(row)

    # totals = sums of the DISPLAYED (rounded) row values — the column an
    # analyst adds is the total the report states, always
    totals = {
        "spend_usd": round(sum(r["spend_usd"] for r in out_rows), 6),
        "executed_requests": sum(r["executed_requests"] for r in out_rows),
        "executed_input_tokens": sum(r["executed_input_tokens"] for r in out_rows),
        "executed_output_tokens": sum(r["executed_output_tokens"] for r in out_rows),
        "avoided_requests": sum(r["avoided_requests"] for r in out_rows),
        "avoided_tokens": sum(r["avoided_tokens"] for r in out_rows),
        "avoided_usd": round(sum(r["avoided_usd"] for r in out_rows), 6),
    }
    total_reported = sum(r["provenance"]["reported_tokens"] for r in out_rows)
    total_estimated = sum(r["provenance"]["estimated_tokens"] for r in out_rows)

    return {
        "report": "chargeback",
        "group_by": list(dims),
        "period": {"start": start, "end": end},
        "rows": out_rows,
        "totals": totals,
        "provenance": {
            "reported_tokens": total_reported,
            "estimated_tokens": total_estimated,
            "reported_token_share": (
                round(total_reported / (total_reported + total_estimated), 6)
                if (total_reported + total_estimated) > 0 else None),
            "all_rows_fully_priced": all(
                r["provenance"]["fully_priced"] for r in out_rows) if out_rows else True,
        },
        "excluded_shadow_hits": shadow_hits_excluded,
        "excluded_malformed_records": malformed_excluded,
        "notes": (
            "spend_usd covers records that EXECUTED (cache misses, including "
            "shadow-mode real calls). Cache hits consumed no upstream compute "
            "and appear only in the avoided_* credit columns — never netted "
            "into spend. Shadow hits (measurement projections) are excluded "
            "entirely and counted in excluded_shadow_hits. Records with corrupt "
            "numeric fields (NaN/inf/negative/non-numeric) are excluded whole "
            "and counted in excluded_malformed_records — a damaged ledger line "
            "can reduce coverage but can never poison a total. Totals are sums "
            "of the displayed rows and reconcile by construction."),
    }


# ── CSV export (what finance actually opens) ────────────────────────────

def _csv_safe(value) -> str:
    """Render a cell, neutralizing spreadsheet formula injection.

    Tags/principals are user-supplied strings; a value beginning with = + - @
    (or tab/CR) executes as a formula when the CSV is opened in Excel/Sheets.
    Such cells are prefixed with a single quote — the standard neutralization —
    so they land as text. Quotes/commas/newlines are RFC-4180-escaped.
    """
    s = "" if value is None else str(value)
    if s[:1] in ("=", "+", "-", "@", "\t", "\r"):
        s = "'" + s
    if any(c in s for c in (",", '"', "\n", "\r")):
        s = '"' + s.replace('"', '""') + '"'
    return s


def chargeback_csv(report: Dict[str, object]) -> str:
    """Flatten a chargeback_report() into CSV for the finance close.

    One row per group plus a TOTAL row that equals the column sums (the report
    guarantees this by construction). Provenance is flattened into columns so
    the honesty travels with the export.
    """
    dims: List[str] = list(report["group_by"])  # type: ignore[index]
    headers = dims + [
        "spend_usd", "executed_requests",
        "executed_input_tokens", "executed_output_tokens",
        "avoided_requests", "avoided_tokens", "avoided_usd",
        "reported_tokens", "estimated_tokens", "fully_priced",
    ]
    buf = io.StringIO()
    buf.write(",".join(_csv_safe(h) for h in headers) + "\r\n")
    for row in report["rows"]:  # type: ignore[union-attr]
        prov = row["provenance"]
        cells = [row[d] for d in dims] + [
            row["spend_usd"], row["executed_requests"],
            row["executed_input_tokens"], row["executed_output_tokens"],
            row["avoided_requests"], row["avoided_tokens"], row["avoided_usd"],
            prov["reported_tokens"], prov["estimated_tokens"],
            prov["fully_priced"],
        ]
        buf.write(",".join(_csv_safe(c) for c in cells) + "\r\n")
    t = report["totals"]  # type: ignore[index]
    p = report["provenance"]  # type: ignore[index]
    total_cells = (["TOTAL"] + [""] * (len(dims) - 1)) + [
        t["spend_usd"], t["executed_requests"],
        t["executed_input_tokens"], t["executed_output_tokens"],
        t["avoided_requests"], t["avoided_tokens"], t["avoided_usd"],
        p["reported_tokens"], p["estimated_tokens"],
        p["all_rows_fully_priced"],
    ]
    buf.write(",".join(_csv_safe(c) for c in total_cells) + "\r\n")
    return buf.getvalue()
