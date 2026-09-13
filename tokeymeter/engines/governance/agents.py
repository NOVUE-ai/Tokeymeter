"""`tokeymeter agents` — the report card for agent behaviour.

THE SAME NUMBER, TWICE
----------------------
Stall detection asks one question in flight: are the recent responses still
novel while the input keeps growing? This asks the same question over history,
per agent, over time.

That is deliberate. One primitive, two surfaces — a stall that halts a task at
3am and the number a platform owner reads on Monday are computed the same way,
so the report can never disagree with the enforcement.

WHAT PROGRESS MEANS, AND WHY COST CANNOT TELL YOU
-------------------------------------------------
Cost does not reveal a stuck agent: a repeat is served from cache, so a
twenty-call loop can cost one call. The prompt does not reveal it either: a
real agent carries its conversation history, so every prompt hash is unique
even on the thirtieth identical failure.

The RESPONSE reveals it. An agent making progress produces new information; a
stuck one produces the same answer again. So:

    progress = distinct response fingerprints / executed calls

WHY NOVELTY ALONE WOULD BE A FALSE-POSITIVE MACHINE
----------------------------------------------------
A batch classifier that returns "APPROVED" for five hundred documents has a
progress score near zero and is working perfectly. What separates it from a
stall is that its input does NOT grow — each document is independent, while a
stuck conversational agent accumulates history and pays more every turn.

So this report never calls low progress a stall on its own. It reports
progress, it reports input growth, and it names a stall only where both hold.
Anything else would train a platform owner to ignore the column.

WHAT IT REFUSES TO DO
---------------------
No projections, no percentages of savings, no scores. Only what these records
say. Cache hits are excluded from progress entirely — a hit returns a
byte-identical response by definition, so counting them would make every
well-cached workload look stalled.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["agent_report", "render_agents", "compare_reports",
           "render_comparison"]

# A task needs enough executed calls before its progress score means anything.
# Two calls that happen to agree are noise, not evidence.
_MIN_CALLS_FOR_PROGRESS = 4
# Below this, responses have stopped being novel.
_LOW_PROGRESS = 0.25
# Input must have grown by at least this much across the task for low progress
# to indicate a stall rather than legitimate repetition.
_GROWTH_THRESHOLD = 0.25


def _usable(rec: dict):
    """(cost, hit, input_tokens, response_fp) or None for an unusable record.
    Mirrors the ledger sanitizer: a corrupt record is excluded whole and
    counted, never partially folded into a total."""
    try:
        cost = float(rec.get("estimated_cost") or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    try:
        tok = rec.get("input_tokens")
        tok = int(tok) if tok is not None else None
    except (TypeError, ValueError):
        tok = None
    if tok is not None and (tok < 0 or tok > 10_000_000):
        tok = None
    return cost, bool(rec.get("hit")), tok, rec.get("response_fingerprint")


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _growth(tokens: List[int]) -> Optional[float]:
    """Fractional change in input size from the first half of a task to the
    second. A conversational agent accumulates context and grows; independent
    work does not. Returns None when there is not enough to compare."""
    vals = [t for t in tokens if t is not None]
    if len(vals) < 4:
        return None
    half = len(vals) // 2
    first = sum(vals[:half]) / half
    second = sum(vals[half:]) / (len(vals) - half)
    if first <= 0:
        return None
    return (second - first) / first


def agent_report(records: Optional[Iterable[dict]] = None) -> Dict[str, Any]:
    """Per-agent behaviour over the supplied records (default: the ledger)."""
    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = list(_sv._tracker._iter_records())

    tasks: Dict[str, Dict[str, Any]] = {}
    excluded = 0
    untasked_spend = 0.0
    for rec in records:
        parsed = _usable(rec)
        if parsed is None:
            excluded += 1
            continue
        cost, hit, tok, rfp = parsed
        tid = rec.get("task_id")
        if not tid:
            if not hit:
                untasked_spend += cost
            continue
        t = tasks.setdefault(str(tid), {
            "agent": None, "spend": 0.0, "calls": 0, "executed": 0,
            "fps": [], "tokens": [], "ts": rec.get("timestamp") or 0.0,
            "seq": [],
        })
        if t["agent"] is None and rec.get("agent"):
            t["agent"] = str(rec["agent"])
        t["calls"] += 1
        if not hit:
            t["executed"] += 1
            t["spend"] += cost
            # Progress is measured on EXECUTED calls only: a cache hit returns
            # a byte-identical response by definition. The timestamp rides
            # along so the sequence can be restored below.
            try:
                ts = float(rec.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            t["seq"].append((ts, rfp, tok))

    # ORDER MATTERS AND FILE ORDER IS NOT GUARANTEED. The progress signal reads
    # the FIRST half of a task against the SECOND, so a task whose records
    # arrived interleaved — several processes appending to one ledger, or two
    # ledgers concatenated — would otherwise be judged on a sequence that never
    # happened. `simulate` already sorts for this reason and `suggest` now does
    # too; all three have to answer from the same sequence, or the report, the
    # advice and the enforcement can disagree.
    for _t in tasks.values():
        _t["seq"].sort(key=lambda x: x[0])
        _t["fps"] = [x[1] for x in _t["seq"] if x[1]]
        _t["tokens"] = [x[2] for x in _t["seq"] if x[2] is not None]

    agents: Dict[str, Dict[str, Any]] = {}
    for tid, t in tasks.items():
        name = t["agent"] or "(unnamed)"
        a = agents.setdefault(name, {
            "tasks": 0, "costs": [], "calls": [], "progress": [],
            "stalled_tasks": [], "spend_usd": 0.0,
            "responses_scored": 0, "responses_executed": 0,
        })
        a["tasks"] += 1
        a["spend_usd"] += t["spend"]
        a["costs"].append(t["spend"])
        a["calls"].append(t["calls"])

        # Score on the number of SCOREABLE responses, not executed calls. A task
        # with 20 calls of which 2 could be measured must not have a progress
        # figure presented as fact — that is a two-sample opinion wearing a
        # number's clothes.
        a["responses_scored"] = a.get("responses_scored", 0) + len(t["fps"])
        a["responses_executed"] = a.get("responses_executed", 0) + t["executed"]
        if len(t["fps"]) >= _MIN_CALLS_FOR_PROGRESS:
            prog = len(set(t["fps"])) / float(len(t["fps"]))
            a["progress"].append(prog)
            growth = _growth(t["tokens"])
            # BOTH signals required. Low progress with flat input is a batch
            # classifier doing its job, not a stall.
            if prog <= _LOW_PROGRESS and growth is not None and growth >= _GROWTH_THRESHOLD:
                a["stalled_tasks"].append({
                    "task_id": tid, "progress": round(prog, 4),
                    "input_growth": round(growth, 4),
                    "calls": t["calls"], "spend_usd": round(t["spend"], 6),
                })

    rows = []
    for name, a in agents.items():
        stalled = a["stalled_tasks"]
        stalled.sort(key=lambda s: s["spend_usd"], reverse=True)
        rows.append({
            "agent": name,
            "tasks": a["tasks"],
            "spend_usd": round(a["spend_usd"], 6),
            "median_cost_per_task_usd": round(statistics.median(a["costs"]), 6)
            if a["costs"] else 0.0,
            "p95_cost_per_task_usd": round(_pct(a["costs"], 0.95), 6),
            "median_calls_per_task": round(statistics.median(a["calls"]), 2)
            if a["calls"] else 0,
            "median_progress": round(statistics.median(a["progress"]), 4)
            if a["progress"] else None,
            "tasks_scored": len(a["progress"]),
            "responses_scored": a["responses_scored"],
            "responses_executed": a["responses_executed"],
            "scoring_coverage_pct": (
                round(a["responses_scored"] / a["responses_executed"] * 100.0, 1)
                if a["responses_executed"] else 0.0),
            "stalled_tasks": len(stalled),
            "stalled_spend_usd": round(sum(s["spend_usd"] for s in stalled), 6),
            "worst_stalls": stalled[:5],
        })
    rows.sort(key=lambda r: r["spend_usd"], reverse=True)

    return {
        "report": "agents",
        "agents": rows,
        "total_tasks": len(tasks),
        "total_spend_usd": round(sum(r["spend_usd"] for r in rows), 6),
        "untasked_spend_usd": round(untasked_spend, 6),
        "excluded_malformed_records": excluded,
        "thresholds": {
            "low_progress": _LOW_PROGRESS,
            "input_growth": _GROWTH_THRESHOLD,
            "min_calls_scored": _MIN_CALLS_FOR_PROGRESS,
        },
        "note": (
            "progress = distinct response fingerprints / executed calls. Cache "
            "hits are excluded: a hit returns a byte-identical response by "
            "definition. A task is called stalled only when progress is low AND "
            "its input grew - low progress with flat input is legitimate "
            "repetitive work, such as a classifier returning the same label."),
    }


def render_agents(report: Dict[str, Any]) -> str:
    """Human-readable. ASCII only: this runs on a Windows console under a
    legacy code page as often as it runs in a Linux CI job."""
    out: List[str] = []
    rows = report["agents"]
    if not rows:
        out.append("No agent tasks found - nothing has been recorded yet.")
        # NAME THE FILE. "Nothing was recorded" and "you are pointing at the
        # wrong ledger" produce an identical empty report, and the second is
        # far more common — a mistyped TOKEYMETER_HOME, or cmd.exe `set`
        # syntax pasted into PowerShell where it silently does nothing.
        try:
            from tokeymeter import paths as _paths
            out.append(f"  (looked in: {_paths.savings_path()})")
        except Exception:
            pass
        out.append("")
        out.append("TWO steps are needed, and the first is easy to miss:")
        out.append("")
        out.append("  1. wrap your client, so calls reach the ledger at all")
        out.append("       from tokeymeter.engines.execution.integrations \\")
        out.append("           import openai as tokeymeter_openai")
        out.append("       client = tokeymeter_openai.wrap(OpenAI())")
        out.append("")
        out.append("  2. mark one unit of work")
        out.append("       with tokeymeter.task('ticket-1', agent='support'):")
        out.append("           agent.run(ticket)")
        out.append("")
        out.append("Or run `tokeymeter firstrun` for an offline demo.")
        if report["untasked_spend_usd"] > 0:
            out.append(f"({report['untasked_spend_usd']:.4f} USD of spend belongs "
                       f"to no task and is invisible here.)")
        return "\n".join(out)

    out.append(f"{'agent':<18}{'tasks':>7}{'med $/task':>12}{'p95':>10}"
               f"{'calls':>7}{'progress':>10}{'stalled':>9}")
    out.append("-" * 73)
    for r in rows:
        prog = ("-" if r["median_progress"] is None
                else f"{r['median_progress']:.2f}")
        stalled = (f"{r['stalled_tasks']}" if not r["stalled_tasks"]
                   else f"{r['stalled_tasks']} !")
        out.append(f"{r['agent']:<18}{r['tasks']:>7}"
                   f"{r['median_cost_per_task_usd']:>12.4f}"
                   f"{r['p95_cost_per_task_usd']:>10.4f}"
                   f"{r['median_calls_per_task']:>7.1f}{prog:>10}{stalled:>9}")

    flagged = [r for r in rows if r["stalled_tasks"]]
    if flagged:
        out.append("")
        for r in flagged:
            out.append(f"{r['agent']}: {r['stalled_tasks']} task(s) stopped making "
                       f"progress while their input kept growing "
                       f"(${r['stalled_spend_usd']:.4f})")
            for s in r["worst_stalls"][:3]:
                out.append(f"   {s['task_id']:<22} progress {s['progress']:.2f}  "
                           f"input +{s['input_growth']:.0%}  "
                           f"{s['calls']} calls  ${s['spend_usd']:.4f}")

    unscored = [r for r in rows
                if r["median_progress"] is None or r["scoring_coverage_pct"] < 100.0]
    if unscored:
        out.append("")
        for r in unscored:
            if r["responses_executed"] == 0:
                continue
            if r["median_progress"] is None:
                # TWO DIFFERENT REASONS, and conflating them produced a line
                # that contradicted itself: "progress not scored - 13 of 13
                # responses could be measured". Every response WAS measurable;
                # the tasks were simply one call long, and progress across a
                # single call is not a thing. Say which it is.
                if r["responses_scored"] >= r["responses_executed"]:
                    out.append(f"{r['agent']}: progress not scored - its tasks "
                               f"are too short to measure progress across "
                               f"(progress compares responses WITHIN one task, "
                               f"so a single-call task has nothing to compare). "
                               f"Nothing is wrong here.")
                else:
                    out.append(f"{r['agent']}: progress not scored - "
                           f"{r['responses_scored']} of {r['responses_executed']} "
                           f"responses could be measured. Only text, bytes and "
                           f"chunk lists are scored on their own; a provider "
                           f"response object usually carries a per-call request "
                           f"id, so scoring it would invent novelty. Pass "
                           f"extract_response_text=... to score these.")
            else:
                out.append(f"{r['agent']}: progress scored on "
                           f"{r['scoring_coverage_pct']:.0f}% of responses "
                           f"({r['responses_scored']}/{r['responses_executed']})")

    out.append("")
    if report["untasked_spend_usd"] > 0:
        out.append(f"${report['untasked_spend_usd']:.4f} of spend belongs to no "
                   f"task and cannot be scored - wrap those entry points to see them")
    if report["excluded_malformed_records"]:
        out.append(f"excluded {report['excluded_malformed_records']} malformed record(s)")
    out.append("progress = distinct responses / executed calls. Low progress with "
               "FLAT input is normal (a classifier); low progress with GROWING "
               "input is a stall.")
    return "\n".join(out)


_USAGE = """tokeymeter agents - how your agents are actually behaving

  tokeymeter agents                 the report card
  tokeymeter agents --suggest       what to set, backtested against your history
  tokeymeter agents --since 7d --compare    what changed vs the window before
  tokeymeter agents --html [FILE]   one self-contained file you can send someone
  tokeymeter agents --json

progress = distinct response fingerprints / executed calls. An agent making
progress returns new information; a stuck one returns the same answer again.
Responses are hashed, never read."""


def _window(spec):
    from tokeymeter.engines.governance.plan import parse_window
    return parse_window(spec)


def _in(rec, start, end):
    ts = rec.get("timestamp")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return False
    return ts == ts and start <= ts < end


def main(argv: Optional[List[str]] = None) -> int:
    import json
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    as_json = "--json" in argv
    as_html = "--html" in argv
    html_path = "tokeymeter-agents.html"
    if as_html:
        i = argv.index("--html")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            html_path = argv[i + 1]
    suggest = "--suggest" in argv
    compare = "--compare" in argv
    since = None
    if "--since" in argv:
        try:
            since = argv[argv.index("--since") + 1]
        except IndexError:
            print("tokeymeter agents: --since needs a window like 7d")
            return 1
    known = {"--json", "--suggest", "--compare", "--since", "--html",
             since, html_path if as_html else None}
    unknown = [a for a in argv if a not in known]
    if unknown:
        print(f"tokeymeter agents: unrecognized argument {unknown[0]!r}\n")
        print(_USAGE)
        return 1
    if compare and not since:
        print("tokeymeter agents: --compare needs --since, e.g. --since 7d\n"
              "  (it compares that window against the one immediately before)")
        return 1

    from tokeymeter.engines.economics import savings as _sv
    records = list(_sv._tracker._iter_records())

    if suggest:
        from tokeymeter.engines.governance.suggest import (
            suggest_thresholds, render_suggestions)
        rep = suggest_thresholds(records)
        print(json.dumps(rep, indent=2, default=str) if as_json
              else render_suggestions(rep))
        return 0

    if compare:
        try:
            start, end = _window(since)
        except ValueError as e:
            print(f"tokeymeter agents: {e}")
            return 1
        span = end - start
        cur = [r for r in records if _in(r, start, end)]
        prev = [r for r in records if _in(r, start - span, start)]
        if not prev:
            print(f"No traffic in the {since} window before this one - "
                  f"nothing to compare against yet.")
            return 0
        cmp_rep = compare_reports(agent_report(cur), agent_report(prev))
        print(json.dumps(cmp_rep, indent=2, default=str) if as_json
              else render_comparison(cmp_rep))
        return 0

    if since:
        try:
            start, end = _window(since)
        except ValueError as e:
            print(f"tokeymeter agents: {e}")
            return 1
        records = [r for r in records if _in(r, start, end)]

    report = agent_report(records)
    if as_html:
        from tokeymeter.engines.governance.agents_html import write_agents_html
        try:
            written = write_agents_html(report, html_path)
        except OSError as exc:
            print(f"tokeymeter agents: could not write {html_path}: {exc}")
            return 1
        print(f"wrote {written}")
        print("  one self-contained file - no server, no account. Open it, or")
        print("  send it to whoever owns the agents.")
        return 0
    print(json.dumps(report, indent=2, default=str) if as_json
          else render_agents(report))
    return 0


# ── drift ───────────────────────────────────────────────────────────────

def compare_reports(current: Dict[str, Any],
                    previous: Dict[str, Any]) -> Dict[str, Any]:
    """What changed between two windows, per agent.

    A single report answers "is my agent healthy". This answers "is my agent
    getting worse", which is the question that makes someone look every week
    rather than once. Regression is the failure that never announces itself: an
    agent quietly going from 6 calls a task to 11 costs real money and looks
    completely normal in any single snapshot.

    Only agents present in BOTH windows are compared — a new agent has nothing
    to drift from, and one that stopped running has not improved.
    """
    cur = {a["agent"]: a for a in current.get("agents", [])}
    prev = {a["agent"]: a for a in previous.get("agents", [])}
    rows = []
    for name in sorted(set(cur) & set(prev)):
        c, p = cur[name], prev[name]
        row: Dict[str, Any] = {"agent": name, "metrics": {},
                               "new": False, "gone": False}
        for key, label, worse_when_up in (
                ("median_calls_per_task", "calls/task", True),
                ("median_cost_per_task_usd", "$/task", True),
                ("median_progress", "progress", False),
                ("p95_cost_per_task_usd", "p95 $/task", True)):
            a, b = p.get(key), c.get(key)
            if a is None or b is None:
                continue
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                continue
            # A percentage change from zero is undefined, not infinite.
            pct = ((b - a) / a) if a else None
            row["metrics"][label] = {
                "from": round(a, 6), "to": round(b, 6),
                "change_pct": round(pct, 4) if pct is not None else None,
                "worse": (b > a) if worse_when_up else (b < a),
            }
        before = p.get("stalled_tasks", 0)
        now = c.get("stalled_tasks", 0)
        row["stalled"] = {"from": before, "to": now, "worse": now > before}
        row["regressed"] = (any(m["worse"] for m in row["metrics"].values())
                            or row["stalled"]["worse"])
        rows.append(row)
    for name in sorted(set(cur) - set(prev)):
        rows.append({"agent": name, "metrics": {}, "new": True, "gone": False,
                     "regressed": False,
                     "stalled": {"from": 0,
                                 "to": cur[name].get("stalled_tasks", 0),
                                 "worse": False}})
    for name in sorted(set(prev) - set(cur)):
        rows.append({"agent": name, "metrics": {}, "new": False, "gone": True,
                     "regressed": False,
                     "stalled": {"from": prev[name].get("stalled_tasks", 0),
                                 "to": 0, "worse": False}})
    return {
        "report": "comparison",
        "agents": rows,
        "regressed": sum(1 for r in rows if r["regressed"]),
        "note": ("Compares two windows of the same ledger. Only agents present "
                 "in both are measured — a new agent has nothing to drift "
                 "from. A percentage change from zero is reported as undefined "
                 "rather than infinite."),
    }


def render_comparison(cmp: Dict[str, Any]) -> str:
    """ASCII only — this runs on a legacy console as often as a modern one."""
    rows = cmp.get("agents", [])
    if not rows:
        return "  No agent appears in both windows - nothing to compare."
    out = []
    for r in rows:
        if r["new"]:
            out.append(f"  {r['agent']:<20} NEW in this window")
            continue
        if r["gone"]:
            out.append(f"  {r['agent']:<20} no traffic in this window")
            continue
        flag = "  <-- regressed" if r["regressed"] else ""
        out.append(f"  {r['agent']:<20}{flag}")
        for label, m in r["metrics"].items():
            pct = ("" if m["change_pct"] is None
                   else f"  {'+' if m['change_pct'] >= 0 else ''}"
                        f"{m['change_pct']:.0%}")
            mark = " !" if m["worse"] else ""
            out.append(f"      {label:<12} {m['from']:.4g} -> {m['to']:.4g}"
                       f"{pct}{mark}")
        s = r["stalled"]
        if s["from"] or s["to"]:
            out.append(f"      {'stalled':<12} {s['from']} -> {s['to']}"
                       f"{' !' if s['worse'] else ''}")
        out.append("")
    n = cmp["regressed"]
    out.append(f"  {n} agent{'' if n == 1 else 's'} regressed")
    return "\n".join(out)
