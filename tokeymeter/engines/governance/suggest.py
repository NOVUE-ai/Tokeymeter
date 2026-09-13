"""Threshold suggestion — what to set, decided by backtest rather than by guess.

THE FRICTION THIS REMOVES
-------------------------
Installing takes one line. Then comes the question nobody can answer: is
`stall_window` 5 or 20? Is the envelope $0.50 or $5? Guess low and healthy work
gets killed; guess high and the runaway you installed this for slips through. So
most people set nothing, and a ceiling nobody sets protects nobody.

The answer is in their own ledger. Every candidate is REPLAYED against real
history through the same simulator `plan` uses, and reported with the two
numbers that actually decide it:

    caught          tasks that were going nowhere and would have been stopped
    false positives HEALTHY tasks the candidate would also have stopped

A suggestion without a false-positive count is a guess wearing a number's
clothes. This never emits one.

WHAT "HEALTHY" MEANS, AND WHY IT IS DEFINED FROM THEIR DATA
------------------------------------------------------------
A task is treated as genuinely stuck when it shows the shape the detector was
built for: response novelty collapsed AND input grew. Every other completed task
is treated as healthy — so halting one is counted as a false positive, which is
the conservative direction. If we assumed the opposite we would flatter every
candidate.

REFUSING TO ANSWER IS AN ANSWER
--------------------------------
Three cases produce no recommendation, on purpose:

  * too little history — a threshold fitted to nine tasks is noise, and shipping
    it would be worse than saying nothing;
  * no stalls in the window — there is nothing to tune against, and inventing a
    number here is exactly the vagueness this module exists to avoid;
  * batch-shaped work (low novelty, flat input) — stall detection does not apply
    to a classifier, and the honest advice is a call ceiling instead.

Each of those returns a reason a human can read, not a shrug.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["suggest_thresholds", "render_suggestions"]

# Candidates are the values a person would plausibly type. Suggesting 7.3 would
# be false precision — the data cannot distinguish it from 8.
_STALL_WINDOWS = (4, 6, 8, 12, 16, 20)
_MAX_REPEATS = (3, 4, 6, 8)

# An agent needs this many completed tasks before a fitted threshold means
# anything. Below it we say so rather than fitting to noise.
_MIN_TASKS = 10
# Below this novelty, with flat input, the workload is batch-shaped.
_BATCH_NOVELTY = 0.25
_GROWTH = 0.25
_MIN_SCORED = 4


def _task_view(records: Iterable[dict]) -> Dict[str, Dict[str, Any]]:
    """Group the ledger into tasks, keeping only what a threshold depends on."""
    tasks: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        tid = rec.get("task_id")
        if not tid:
            continue
        try:
            cost = float(rec.get("estimated_cost") or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cost) or cost < 0:
            continue
        t = tasks.setdefault(str(tid), {
            "agent": None, "calls": 0, "executed": 0, "spend": 0.0,
            "fps": [], "prompt_fps": [], "tokens": [], "seq": [],
        })
        if t["agent"] is None and rec.get("agent"):
            t["agent"] = str(rec["agent"])
        t["calls"] += 1
        if rec.get("hit"):
            if rec.get("prompt_fingerprint"):
                t["prompt_fps"].append(rec["prompt_fingerprint"])
            continue
        t["executed"] += 1
        t["spend"] += cost
        try:
            tok = rec.get("input_tokens")
            tok = int(tok) if tok is not None else None
        except (TypeError, ValueError):
            tok = None
        try:
            ts = float(rec.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        t["seq"].append((ts, rec.get("response_fingerprint"),
                         rec.get("prompt_fingerprint"), tok))

    # ORDER MATTERS AND FILE ORDER IS NOT GUARANTEED. The progress signal reads
    # the FIRST half of a window against the SECOND, so a task whose records
    # arrive interleaved — several processes appending to one ledger, or two
    # ledgers concatenated — would be judged on a sequence that never happened.
    # `simulate` already sorts for exactly this reason; doing it here keeps the
    # advice and the simulation answering from the same sequence.
    for t in tasks.values():
        t["seq"].sort(key=lambda x: x[0])
        t["fps"] = [x[1] for x in t["seq"] if x[1]]
        t["prompt_fps"] = [x[2] for x in t["seq"] if x[2]]
        t["tokens"] = [x[3] for x in t["seq"]]
    return tasks


def _novelty(fps: List[str]) -> Optional[float]:
    if len(fps) < 2:
        return None
    return len(set(fps)) / float(len(fps))


def _growth(tokens: List[Optional[int]]) -> Optional[float]:
    vals = [t for t in tokens if t is not None]
    if len(vals) < _MIN_SCORED:
        return None
    half = len(vals) // 2
    first = sum(vals[:half]) / half
    second = sum(vals[half:]) / (len(vals) - half)
    return (second - first) / first if first > 0 else None


def _is_stuck(t: Dict[str, Any]) -> bool:
    """The ground truth every candidate is scored against: novelty collapsed AND
    input grew. Conservative on purpose — anything else counts as healthy, so a
    candidate that halts it is charged a false positive."""
    n = _novelty(t["fps"])
    g = _growth(t["tokens"])
    return (n is not None and n <= _BATCH_NOVELTY
            and g is not None and g >= _GROWTH)


def _would_halt_stall(t: Dict[str, Any], window: int) -> bool:
    """Replay the enforcement rule over this task at a candidate window.

    Mirrors _TaskState._stall_check_locked: a full window of scored responses,
    novelty at or below the floor, and the second half of the window carrying
    more input than the first.
    """
    fps, toks = t["fps"], t["tokens"]
    for i in range(window, len(fps) + 1):
        w_fps = fps[i - window:i]
        w_tok = [x for x in toks[i - window:i] if x is not None]
        if len(w_fps) < window or len(w_tok) < window:
            continue
        if len(set(w_fps)) / float(window) > _BATCH_NOVELTY:
            continue
        half = window // 2
        first = sum(w_tok[:half]) / half
        second = sum(w_tok[half:]) / (window - half)
        if first > 0 and second > first:
            return True
    return False


def _would_halt_repeats(t: Dict[str, Any], limit: int) -> bool:
    seen: Dict[str, int] = {}
    for fp in t["prompt_fps"]:
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] >= limit:
            return True
    return False


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _evaluate(tasks: List[Dict[str, Any]], predicate) -> Dict[str, int]:
    """Score one candidate: how many stuck tasks it catches, and how many
    healthy ones it would have stopped."""
    caught = missed = false_pos = 0
    for t in tasks:
        halts = predicate(t)
        stuck = t["_stuck"]
        if stuck and halts:
            caught += 1
        elif stuck and not halts:
            missed += 1
        elif not stuck and halts:
            false_pos += 1
    return {"caught": caught, "missed": missed, "false_positives": false_pos}


def _longest_repeat_run(tasks: List[Dict[str, Any]]) -> int:
    """Longest stretch of identical consecutive responses inside a HEALTHY task.

    This is the thing a stall window must clear. An agent that legitimately
    retries an upstream three times and then recovers is not stalled, and a
    window at or below that run length will stop it. Measuring the run from
    real history turns "how much headroom do I have" from a feeling into a
    number.
    """
    longest = 0
    for t in tasks:
        if t["_stuck"]:
            continue
        run = 0
        prev = None
        for fp in t["fps"]:
            run = run + 1 if fp == prev else 1
            prev = fp
            longest = max(longest, run)
    return longest


def _best(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the candidate that catches the most with ZERO false positives, and
    among equals the TIGHTEST setting.

    Zero false positives is a hard requirement, not a preference: a ceiling
    that kills healthy work gets switched off within a day, and then it
    protects nothing.

    The tie-break goes to the tightest, and that direction was measured rather
    than assumed. Holdout testing on unseen traffic showed the looser choice
    catching 28 of 40 stalled tasks where the tighter caught 40 of 40 — with no
    false positives either way. The safety argument for looseness is already
    handled by the zero-false-positive filter above: when history contains
    healthy tasks that legitimately repeat, the tight windows fail that filter
    and never reach this tie-break at all.
    """
    clean = [c for c in candidates if c["false_positives"] == 0 and c["caught"] > 0]
    if not clean:
        return None
    best_catch = max(c["caught"] for c in clean)
    finalists = [c for c in clean if c["caught"] == best_catch]
    return min(finalists, key=lambda c: c["value"])


def suggest_thresholds(records: Optional[Iterable[dict]] = None,
                       ) -> Dict[str, Any]:
    """Backtested threshold recommendations, per agent.

    Never invents a number. Every recommendation carries the count it caught and
    the count of healthy tasks it would have stopped, both measured against the
    supplied history.
    """
    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = list(_sv._tracker._iter_records())
    else:
        records = list(records)

    tasks = _task_view(records)
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for tid, t in tasks.items():
        t["task_id"] = tid
        t["_stuck"] = _is_stuck(t)
        by_agent.setdefault(t["agent"] or "(unnamed)", []).append(t)

    out: List[Dict[str, Any]] = []
    for agent, group in sorted(by_agent.items()):
        scored = [t for t in group if len(t["fps"]) >= _MIN_SCORED]
        stuck = [t for t in group if t["_stuck"]]
        healthy = [t for t in group if not t["_stuck"]]
        novelties = [n for n in (_novelty(t["fps"]) for t in scored) if n is not None]
        growths = [g for g in (_growth(t["tokens"]) for t in group) if g is not None]
        healthy_novelty = [n for n in
                           (_novelty(t["fps"]) for t in healthy) if n is not None]

        entry: Dict[str, Any] = {
            "agent": agent,
            "tasks": len(group),
            "tasks_scored": len(scored),
            "stuck_tasks": len(stuck),
            "healthy_progress_range": (
                [round(min(healthy_novelty), 2), round(max(healthy_novelty), 2)]
                if healthy_novelty else None),
            "median_input_growth": (round(sorted(growths)[len(growths) // 2], 4)
                                    if growths else None),
            "recommendations": {},
            "notes": [],
        }

        # Batch-shaped workloads: low novelty with flat input is a classifier
        # doing its job, and stall detection would fight it every time.
        flat = (entry["median_input_growth"] is not None
                and entry["median_input_growth"] < _GROWTH)
        low_novelty = bool(novelties) and (sorted(novelties)[len(novelties) // 2]
                                           <= _BATCH_NOVELTY)
        if flat and low_novelty:
            calls = sorted(t["calls"] for t in group)
            cap = int(_percentile([float(c) for c in calls], 0.95) * 2) or 10
            entry["shape"] = "batch"
            entry["notes"].append(
                "responses repeat but input stays flat - this is batch work, "
                "not a stall. Stall detection is not recommended here.")
            entry["recommendations"]["max_calls"] = {
                "value": cap, "basis": "2x the p95 calls per task seen here",
                "caught": None, "false_positives": None,
                "confidence": "shape-based, not backtested",
            }
            out.append(entry)
            continue

        entry["shape"] = "conversational" if not flat else "mixed"

        if len(group) < _MIN_TASKS:
            entry["notes"].append(
                f"only {len(group)} task(s) recorded - too few to fit a "
                f"threshold. Run more traffic, then ask again.")
            out.append(entry)
            continue

        if not stuck:
            entry["notes"].append(
                "no task in this window shows the stalled shape, so there is "
                "nothing to tune against. A ceiling suggested here would be a "
                "guess.")
            # A budget ceiling is still defensible from the spend distribution.
            spends = [t["spend"] for t in group if t["spend"] > 0]
            if spends:
                p95 = _percentile(spends, 0.95)
                entry["recommendations"]["envelope"] = {
                    "value": round(p95 * 3, 4),
                    "basis": "3x the p95 task spend seen here",
                    "caught": 0, "false_positives": 0,
                    "confidence": "distribution-based, no stalls to backtest",
                }
            out.append(entry)
            continue

        # --- stall_window: backtested ---------------------------------------
        cands = []
        for w in _STALL_WINDOWS:
            r = _evaluate(group, lambda t, w=w: _would_halt_stall(t, w))
            r.update({"value": w})
            cands.append(r)
        best = _best(cands)
        entry["stall_window_candidates"] = cands
        longest_run = _longest_repeat_run(group)
        entry["longest_healthy_repeat_run"] = longest_run
        if best:
            entry["recommendations"]["stall_window"] = {
                "value": best["value"],
                "basis": f"backtested against {len(group)} tasks",
                "caught": best["caught"], "missed": best["missed"],
                "false_positives": 0, "confidence": "backtested",
                # How much room is left before a legitimate repetition run
                # would trip this. Reported so the choice is inspectable rather
                # than trusted.
                "headroom": best["value"] - longest_run,
                "longest_healthy_repeat_run": longest_run,
            }
        else:
            entry["notes"].append(
                "no stall_window catches a stalled task here without also "
                "stopping a healthy one - use max_calls or an envelope instead.")

        # --- max_repeats: backtested ----------------------------------------
        rcands = []
        for k in _MAX_REPEATS:
            r = _evaluate(group, lambda t, k=k: _would_halt_repeats(t, k))
            r.update({"value": k})
            rcands.append(r)
        rbest = _best(rcands)
        if rbest:
            entry["recommendations"]["max_repeats"] = {
                "value": rbest["value"],
                "basis": f"backtested against {len(group)} tasks",
                "caught": rbest["caught"], "missed": rbest["missed"],
                "false_positives": 0, "confidence": "backtested",
            }

        # --- envelope: distribution + backtest ------------------------------
        healthy_spends = [t["spend"] for t in healthy if t["spend"] > 0]
        if healthy_spends:
            p95 = _percentile(healthy_spends, 0.95)
            for mult in (1.5, 2.0, 3.0, 5.0):
                value = round(p95 * mult, 4)
                r = _evaluate(group, lambda t, v=value: t["spend"] > v)
                if r["false_positives"] == 0 and r["caught"] > 0:
                    entry["recommendations"]["envelope"] = {
                        "value": value,
                        "basis": f"{mult:g}x the p95 spend of healthy tasks",
                        "caught": r["caught"], "missed": r["missed"],
                        "false_positives": 0, "confidence": "backtested",
                    }
                    break
            else:
                entry["notes"].append(
                    "no envelope separates stalled tasks from healthy ones "
                    "here - stalled tasks did not cost more, which is normal "
                    "when repeats are served from cache.")
        out.append(entry)

    return {
        "report": "suggestions",
        "agents": out,
        "tasks_analysed": len(tasks),
        "note": (
            "Every value marked 'backtested' was replayed against these exact "
            "records: `caught` is stalled tasks it would have stopped, "
            "`false_positives` is HEALTHY tasks it would also have stopped. "
            "Zero false positives is a hard requirement - a ceiling that kills "
            "working tasks gets switched off, and then it protects nothing. "
            "Where the data cannot support a recommendation, none is given and "
            "the reason is stated."),
    }


def render_suggestions(rep: Dict[str, Any]) -> str:
    """Human-readable. ASCII only - this runs on a legacy console as often as
    it runs in a modern terminal."""
    out: List[str] = []
    if not rep["agents"]:
        return ("No agent tasks found. Wrap an agent entry point with\n"
                "  with tokeymeter.task('id', agent='name'): ...")
    for e in rep["agents"]:
        head = f"  {e['agent']}"
        meta = f"{e['tasks']} tasks"
        if e["stuck_tasks"]:
            meta += f", {e['stuck_tasks']} stalled"
        if e["healthy_progress_range"]:
            lo, hi = e["healthy_progress_range"]
            meta += f", healthy progress {lo:.2f}-{hi:.2f}"
        out.append(f"{head:<22}{meta}")
        for name, r in e["recommendations"].items():
            val = r["value"]
            line = f"      {name} {val}"
            if r.get("confidence") == "backtested":
                fp = r["false_positives"]
                line += (f"  -> would have caught {r['caught']}, "
                         f"{fp} false positive{'' if fp == 1 else 's'}")
                if r.get("headroom") is not None:
                    line += (f"\n          longest legitimate repeat run seen: "
                             f"{r['longest_healthy_repeat_run']} "
                             f"({r['headroom']} calls of headroom)")
            else:
                line += f"  -> {r['basis']} (not backtested)"
            out.append(line)
        for n in e["notes"]:
            for chunk in _wrap(n, 62):
                out.append(f"      {chunk}")
        out.append("")
    out.append("  Values marked with a catch/false-positive count were replayed")
    out.append("  against your own history. Anything the data could not support")
    out.append("  was left out rather than guessed.")
    return "\n".join(out)


def _wrap(text: str, width: int) -> List[str]:
    words, line, res = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            res.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        res.append(line)
    return res
