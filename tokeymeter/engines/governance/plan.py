"""`tokeymeter plan` — what a policy would do, before it does anything.

WHY A COMMAND AND NOT A FUNCTION
---------------------------------
S4-2 shipped `simulate_rules()`, which is the engine. This is the thing a
platform owner actually uses. The difference matters: a Python function is a
capability, a command is a tool. Nobody issues an estate-wide mandate by
writing a script first.

The ritual is Terraform's, deliberately, because the audience already performs
it every week:

    tokeymeter plan --policy policy.yaml --since 30d
    tokeymeter apply --policy policy.yaml

WHAT IS OURS INSIDE THE BORROWED RITUAL
----------------------------------------
Terraform's plan diffs DESIRED state against ACTUAL infrastructure state. This
replays a proposed ruleset against real per-request cost history, grouped by
task. Same word, different mechanism — and only possible because the ledger
exists. A gateway does not retain per-request cost at that granularity, an
observability platform has traces but no policy engine to replay, a FinOps tool
has spend but no task boundary.

WHAT IT REPORTS
---------------
  * the DIFF: rules added, changed, and removed against what is active now
  * the EFFECT: tasks that would halt, spend that would be avoided
  * the COLLATERAL: halted tasks by agent, so a rule that would have stopped
    `checkout` is visible BEFORE it stops checkout
  * the COVERAGE: how much spend is task-attributed at all, because a plan
    over 40% of the estate is a plan with a blind spot and must say so

Exit codes are for CI: 0 when the plan is clean, 2 when it would halt tasks
belonging to an agent the caller declared protected, 1 on a policy error. A
plan that cannot fail a pipeline is decoration.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["plan_report", "coverage_report", "render_plan", "parse_window"]


_WINDOW_RE = re.compile(r"^(\d+)\s*([hdw])$", re.I)
_UNIT_SECONDS = {"h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_window(spec: str, *, now: Optional[float] = None) -> Tuple[float, float]:
    """Turn `30d` / `24h` / `2w` into a half-open [start, end) epoch window.

    Half-open matches the chargeback and capacity reports exactly, so a plan
    and a close packet built for "the same month" cover byte-identically the
    same records.
    """
    if not isinstance(spec, str):
        raise ValueError(f"window must be a string like '30d', got {spec!r}")
    m = _WINDOW_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"window must be <number><h|d|w>, e.g. '24h', '30d', '2w'; "
            f"got {spec!r}")
    n = int(m.group(1))
    if n <= 0:
        raise ValueError(f"window must be a positive number of units, got {spec!r}")
    end = time.time() if now is None else float(now)
    return end - n * _UNIT_SECONDS[m.group(2).lower()], end


def _in_window(rec: dict, start: Optional[float], end: Optional[float]) -> bool:
    if start is None and end is None:
        return True
    ts = rec.get("timestamp")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return False
    if ts != ts:                       # NaN
        return False
    if start is not None and ts < start:
        return False
    if end is not None and ts >= end:   # half-open
        return False
    return True


def coverage_report(records: Iterable[dict]) -> Dict[str, Any]:
    """How much of the estate a plan can actually see.

    A ceiling can only be simulated for spend that belongs to a task. If half
    the estate is untagged, a plan showing "$40 avoided" is describing half a
    company — so the blind spot is reported as a first-class number rather than
    left for someone to discover.
    """
    total = 0.0
    attributed = 0.0
    services: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        try:
            cost = float(rec.get("estimated_cost") or 0.0)
        except (TypeError, ValueError):
            continue
        if cost != cost or cost in (float("inf"), float("-inf")) or cost < 0:
            continue
        if rec.get("hit"):
            continue
        total += cost
        tag = rec.get("tag") or "(untagged)"
        svc = services.setdefault(str(tag), {"spend_usd": 0.0,
                                             "task_attributed_usd": 0.0})
        svc["spend_usd"] += cost
        if rec.get("task_id"):
            attributed += cost
            svc["task_attributed_usd"] += cost
    for svc in services.values():
        svc["spend_usd"] = round(svc["spend_usd"], 6)
        svc["task_attributed_usd"] = round(svc["task_attributed_usd"], 6)
        svc["governed"] = (svc["task_attributed_usd"] >= svc["spend_usd"] - 1e-9
                           and svc["spend_usd"] > 0)
    pct = (attributed / total * 100.0) if total > 0 else 0.0
    return {
        "report": "coverage",
        "executed_spend_usd": round(total, 6),
        "task_attributed_usd": round(attributed, 6),
        "coverage_pct": round(pct, 2),
        "ungoverned_usd": round(total - attributed, 6),
        "by_service": dict(sorted(services.items())),
        "note": ("Only task-attributed spend can be governed by a task ceiling. "
                 "Spend outside any task() boundary is invisible to a plan and "
                 "is reported here rather than silently excluded."),
    }


def _rule_index(ruleset) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i, r in enumerate(getattr(ruleset, "rules", ())):
        out[r.name or f"<unnamed #{i}>"] = r.as_dict()
    return out


def _diff_rules(active, proposed) -> Dict[str, List[dict]]:
    """Which rules are added, changed, or removed — matched by NAME.

    Names are unique by construction (the loader rejects duplicates), which is
    exactly so a diff and an audit trail can identify a rule across versions.
    """
    a, p = _rule_index(active), _rule_index(proposed)
    added = [p[k] for k in p if k not in a]
    removed = [a[k] for k in a if k not in p]
    changed = []
    for k in p:
        if k in a and a[k] != p[k]:
            changed.append({"name": k, "from": a[k], "to": p[k]})
    return {"added": added, "changed": changed, "removed": removed}


def plan_report(proposed, *, active=None, records: Optional[Iterable[dict]] = None,
                since: Optional[str] = None, env: Optional[str] = None,
                protected_agents: Iterable[str] = (),
                now: Optional[float] = None) -> Dict[str, Any]:
    """Build the full plan: diff, simulated effect, collateral, coverage.

    `protected_agents` are the agents a caller declares must never be halted —
    a revenue path, a compliance-critical workflow. A plan that would stop one
    of them is reported as a VIOLATION and drives a non-zero exit code, so the
    check can gate a pipeline rather than merely inform one.
    """
    from .rules import EMPTY_RULESET, get_rules
    from .simulate import simulate_rules

    if active is None:
        active = get_rules()
    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = list(_sv._tracker._iter_records())
    else:
        records = list(records)

    window: Optional[Tuple[float, float]] = None
    if since:
        window = parse_window(since, now=now)
        records = [r for r in records if _in_window(r, window[0], window[1])]

    sim = simulate_rules(proposed, records, env=env)
    cov = coverage_report(records)

    # A CHANGE is judged by its DELTA, not its absolute effect. If the active
    # policy already halts 20 tasks, a proposal that halts 22 is a change of
    # two — reporting 22 would make every tightening look like a first install
    # and would overstate what the reviewer is actually approving.
    baseline = None
    delta = None
    newly_halted: List[Dict[str, Any]] = []
    if getattr(active, "rules", ()):
        base_sim = simulate_rules(active, records, env=env)
        already = {h["task_id"] for h in base_sim["halted_tasks"]}
        newly_halted = [h for h in sim["halted_tasks"]
                        if h["task_id"] not in already]
        baseline = {
            "tasks_would_halt": base_sim["tasks_would_halt"],
            "avoided_usd": base_sim["avoided_usd"],
        }
        delta = {
            "tasks_would_halt": sim["tasks_would_halt"] - base_sim["tasks_would_halt"],
            "avoided_usd": round(sim["avoided_usd"] - base_sim["avoided_usd"], 6),
            "newly_halted_tasks": len(newly_halted),
        }

    protected = {str(a) for a in protected_agents}
    violations = [h for h in sim["halted_tasks"]
                  if h.get("agent") and str(h["agent"]) in protected]
    # A protected task that ALREADY halts under the live policy is not this
    # change's doing. Separating them keeps the gate honest: a reviewer is
    # accountable for what their diff introduces.
    new_violations = ([h for h in newly_halted
                       if h.get("agent") and str(h["agent"]) in protected]
                      if baseline is not None else violations)

    return {
        "report": "plan",
        "generated_at": time.time() if now is None else float(now),
        "window": ({"since": since,
                    "start": window[0], "end": window[1]} if window else None),
        "env": env,
        "diff": _diff_rules(active, proposed),
        "active_rules": len(getattr(active, "rules", ()) or ()),
        "proposed_rules": len(getattr(proposed, "rules", ()) or ()),
        "effect": {
            "tasks_seen": sim["tasks_seen"],
            "tasks_would_halt": sim["tasks_would_halt"],
            "tasks_completed": sim["tasks_completed"],
            "spend_in_window_usd": sim["spend_in_window_usd"],
            "avoided_usd": sim["avoided_usd"],
            "tasks_already_stall_shaped": sim.get("tasks_already_stall_shaped", 0),
            "compliance_refusals": sim.get("compliance_refusals", 0),
        },
        "refusals_by_reason": sim.get("refusals_by_reason", {}),
        "refusals_by_rule": sim.get("refusals_by_rule", {}),
        "refused_calls": sim.get("refused_calls", []),
        "baseline": baseline,
        "delta": delta,
        "newly_halted": newly_halted[:10],
        "collateral_by_agent": sim["collateral_by_agent"],
        "top_halts": sim["halted_tasks"][:10],
        "protected_agents": sorted(protected),
        "violations": violations,
        "new_violations": new_violations,
        "coverage": cov,
        "agent_resolution": sim["agent_resolution"],
        "unattributed_records": sim["unattributed_records"],
        "excluded_malformed_records": sim["excluded_malformed_records"],
        "note": (
            "Replay of these exact records under the proposed rules. "
            "avoided_usd is spend that occurred AFTER the point each halt "
            "would have landed — not a projection and not a percentage."),
    }


# ── rendering ───────────────────────────────────────────────────────────

def _fmt_then(then: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(then.items()))


def _fmt_when(when: list) -> str:
    if not when:
        return "always"
    parts = []
    for c in when:
        v = c["value"]
        v = "[" + ", ".join(map(str, v)) + "]" if isinstance(v, list) else str(v)
        parts.append(f"{c['field']} {c['op']} {v}")
    return " and ".join(parts)


def render_plan(plan: Dict[str, Any]) -> str:
    """Human-readable plan. ASCII only: this runs on a Windows console under a
    legacy code page as often as it runs in a Linux CI job."""
    d = plan["diff"]
    e = plan["effect"]
    cov = plan["coverage"]
    out: List[str] = []
    n_add, n_chg, n_rem = len(d["added"]), len(d["changed"]), len(d["removed"])

    if not (n_add or n_chg or n_rem):
        out.append("Plan: no changes. The proposed policy matches what is active.")
    else:
        out.append(f"Plan: {n_add} to add, {n_chg} to change, {n_rem} to remove")
    out.append("")

    for r in d["added"]:
        out.append(f"  + {r['name'] or '<unnamed>'}")
        out.append(f"      when {_fmt_when(r['when'])}")
        out.append(f"      then {_fmt_then(r['then'])}")
    for c in d["changed"]:
        out.append(f"  ~ {c['name']}")
        out.append(f"      from {_fmt_then(c['from']['then'])}")
        out.append(f"      to   {_fmt_then(c['to']['then'])}")
    for r in d["removed"]:
        out.append(f"  - {r['name'] or '<unnamed>'}  ({_fmt_then(r['then'])})")
    if n_add or n_chg or n_rem:
        out.append("")

    win = plan.get("window")
    scope = f"last {win['since']}" if win else "the whole ledger"
    out.append(f"Against {scope}:")
    if e["tasks_seen"] == 0:
        out.append("  no tasks in this window - nothing to simulate")
    else:
        out.append(f"  {e['tasks_seen']} tasks seen, {e['tasks_would_halt']} "
                   f"would halt, {e['tasks_completed']} would complete")
        out.append(f"  ${e['avoided_usd']:.4f} avoided of "
                   f"${e['spend_in_window_usd']:.4f} in window")
        d, b = plan.get("delta"), plan.get("baseline")
        if d and b:
            sign = "+" if d["tasks_would_halt"] >= 0 else ""
            out.append("")
            out.append(f"  Change vs the active policy "
                       f"(which halts {b['tasks_would_halt']}, "
                       f"avoids ${b['avoided_usd']:.4f}):")
            av = d["avoided_usd"]
            money = f"+${av:.4f}" if av >= 0 else f"-${abs(av):.4f}"
            out.append(f"    {sign}{d['tasks_would_halt']} tasks halted, "
                       f"{money} avoided")
            if d["newly_halted_tasks"]:
                names = ", ".join(h["task_id"]
                                  for h in plan.get("newly_halted", [])[:4])
                out.append(f"    {d['newly_halted_tasks']} newly halted"
                           + (f": {names}" if names else ""))

    refused = e.get("compliance_refusals", 0)
    if refused:
        out.append("")
        out.append(f"  !! {refused} call(s) would be REFUSED by a compliance rule:")
        for reason, n in sorted(plan.get("refusals_by_reason", {}).items()):
            out.append(f"       {n:>6}  {reason.replace('_', ' ')}")
        for r in plan.get("refused_calls", [])[:3]:
            scope = " ".join(f"{k}={r[k]}" for k in ("region", "data_class")
                             if r.get(k))
            out.append(f"       {r['task_id']} -> {r['subject']!r}"
                       + (f"  ({scope})" if scope else ""))

    shaped = e.get("tasks_already_stall_shaped", 0)
    if shaped:
        out.append("")
        out.append(f"  note: {shaped} task(s) in this window already show a stall")
        out.append(f"        signature but are not halted by this replay. That is")
        out.append(f"        what already-governed history looks like - a policy in")
        out.append(f"        force truncated them, so this plan UNDER-states the")
        out.append(f"        rule's effect on ungoverned traffic.")

    if plan["collateral_by_agent"]:
        out.append("")
        out.append("  By agent:")
        for a, b in sorted(plan["collateral_by_agent"].items()):
            mark = "  <-- PROTECTED" if a in plan["protected_agents"] else ""
            out.append(f"    {a:<20} {b['halted']:>4} halted   "
                       f"${b['avoided_usd']:.4f} avoided{mark}")

    gate = plan.get("new_violations", plan["violations"])
    if gate:
        out.append("")
        word = "NEWLY halted" if plan.get("baseline") else "would be halted"
        out.append(f"  !! {len(gate)} task(s) belonging to a PROTECTED agent "
                   f"{word}:")
        for v in gate[:5]:
            out.append(f"     {v['task_id']} ({v['agent']}) - {v['limit']} "
                       f"at call {v['halted_at_call']}")

    out.append("")
    out.append(f"Coverage: {cov['coverage_pct']:.1f}% of executed spend is "
               f"task-attributed (${cov['ungoverned_usd']:.4f} ungoverned)")
    if cov["coverage_pct"] < 100.0:
        ungoverned = [s for s, v in cov["by_service"].items()
                      if not v["governed"] and v["spend_usd"] > 0]
        if ungoverned:
            out.append(f"  not fully governed: {', '.join(ungoverned[:6])}")
    if plan["excluded_malformed_records"]:
        out.append(f"  excluded {plan['excluded_malformed_records']} malformed "
                   f"record(s)")
    if "UNDER-reports" in plan.get("agent_resolution", ""):
        out.append(f"  note: {plan['agent_resolution']}")
    return "\n".join(out)


# ── CLI ─────────────────────────────────────────────────────────────────

_USAGE = """tokeymeter plan - preview what an execution policy would do

  tokeymeter plan --policy FILE [--since 30d] [--env ci]
                  [--protect agent[,agent...]] [--json]

Replays the proposed rules against your own ledger history and reports the
diff, the tasks that would halt, the spend avoided, and the collateral by
agent - before anything is applied.

Options:
  --policy FILE     policy to preview (JSON, or YAML when PyYAML is installed)
  --active FILE     the policy currently deployed, to diff against. Without it
                    a plan reads as a first install and reports absolute effect
  --since WINDOW    limit to a window: 24h, 30d, 2w  (default: whole ledger)
  --env NAME        simulate under this environment, e.g. --env ci
  --protect AGENTS  comma-separated agents that must never halt; exit 2 if any would
  --json            emit the full machine-readable report

Exit codes: 0 clean, 2 a protected agent would be halted, 1 policy error."""


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    # A BARE INVOCATION IS AN ERROR, not help. `tokeymeter plan` in a CI script
    # with a forgotten --policy must fail the step, never pass it silently
    # having done nothing — that is the "check that cannot fail" problem, and
    # it is the one failure mode a gate must not have.
    if not argv:
        print("tokeymeter plan: --policy FILE is required\n")
        print(_USAGE)
        return 1

    policy_path = None
    active_path = None
    since = env = None
    protect: List[str] = []
    as_json = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--policy" and i + 1 < len(argv):
            policy_path = argv[i + 1]; i += 2
        elif a == "--active" and i + 1 < len(argv):
            active_path = argv[i + 1]; i += 2
        elif a == "--since" and i + 1 < len(argv):
            since = argv[i + 1]; i += 2
        elif a == "--env" and i + 1 < len(argv):
            env = argv[i + 1]; i += 2
        elif a == "--protect" and i + 1 < len(argv):
            protect = [x.strip() for x in argv[i + 1].split(",") if x.strip()]
            i += 2
        elif a == "--json":
            as_json = True; i += 1
        else:
            print(f"tokeymeter plan: unrecognized argument {a!r}\n")
            print(_USAGE)
            return 1

    if not policy_path:
        print("tokeymeter plan: --policy FILE is required\n")
        print(_USAGE)
        return 1
    if not os.path.exists(policy_path):
        print(f"tokeymeter plan: no such policy file: {policy_path}")
        return 1

    from .rules import load_rules_file, RulePolicyError
    try:
        proposed = load_rules_file(policy_path)
    except RulePolicyError as e:
        print(f"tokeymeter plan: {e}")
        return 1

    active = None
    if active_path:
        if not os.path.exists(active_path):
            print(f"tokeymeter plan: no such active policy file: {active_path}")
            return 1
        try:
            active = load_rules_file(active_path)
        except RulePolicyError as e:
            print(f"tokeymeter plan: active policy: {e}")
            return 1

    try:
        plan = plan_report(proposed, active=active, since=since, env=env,
                           protected_agents=protect)
    except ValueError as e:
        print(f"tokeymeter plan: {e}")
        return 1

    print(json.dumps(plan, indent=2, default=str) if as_json
          else render_plan(plan))
    return 2 if plan.get("new_violations", plan["violations"]) else 0
