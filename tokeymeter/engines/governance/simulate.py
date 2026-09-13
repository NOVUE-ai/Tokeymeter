"""Rule simulation — what a policy WOULD have done, before it does anything.

THE PROBLEM THIS SOLVES
-----------------------
A platform owner will not push a ceiling across twelve services on hope. The
question they actually have is not "is this rule well-formed?" — it is "what
would this have done to us last month, and whose pager would it have fired?"

Terraform won its category on `plan` before `apply`. This is that, for AI
execution policy.

WHY NOBODY ELSE CAN OFFER IT
-----------------------------
Simulating a cost ceiling requires replaying it against real per-request cost
history, grouped by task. A gateway does not retain per-request cost at that
granularity. An observability platform has traces but no policy engine to
replay. A FinOps tool has spend but no request-level detail and no task
boundary. We already hold all of it in one ledger, so `plan` is a query rather
than new infrastructure — and it is what makes a no-code guardrail SAFER than
a hand-written one, because a rule typed into a repository gets reviewed by
someone who cannot predict its blast radius either.

WHAT IT REPORTS, AND WHAT IT REFUSES TO
----------------------------------------
For every task in the window it replays the calls in order, applies the
ceilings the ruleset resolves for that task, and reports where the halt would
have landed.

  * `would_halt` — tasks stopped, by which limit, and at which call
  * `avoided_usd` — spend after the halt point, which is what the rule saves
  * `collateral` — halted tasks grouped by agent, so a rule that would have
    stopped `checkout` is visible BEFORE it stops checkout

It reports no projection and no percentage: only what these exact records
would have done under these exact rules. Records missing a task_id are counted
in `unattributed_records` rather than being folded in, because a task ceiling
cannot be simulated for a call that belonged to no task.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .rules import RuleSet, resolve_limits

__all__ = ["simulate_rules"]


def _numbers(rec: dict):
    """Cost and hit-state for a record, or None if the record is unusable.
    Mirrors the ledger sanitizer's posture: a corrupt record is excluded whole
    and counted, never partially folded into a total."""
    try:
        cost = float(rec.get("estimated_cost") or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return cost, bool(rec.get("hit")), bool(rec.get("shadow"))


def _compliance_verdict(ruleset, call, env) -> Optional[Dict[str, Any]]:
    """Would a compliance rule refuse this call?

    ENFORCEMENT AND SIMULATION MUST MOVE TOGETHER. A compliance rule is the one
    a platform owner must never push blind, and a plan that reports "0 would
    halt" for a rule that would refuse half their EU traffic is worse than no
    plan at all. Mirrors compliance.check_endpoint / check_model, including the
    asymmetry that an UNDECLARED endpoint is refused under an endpoint
    allowlist while an undeclared model is not.
    """
    try:
        from tokeymeter.engines.governance import compliance as _c
        from tokeymeter.engines.governance.rules import resolve_limits
    except Exception:
        return None
    ctx = {"data_class": call.get("data_class"), "region": call.get("region"),
           "agent": call.get("agent"), "env": env, "task_id": None,
           "principal": None, "endpoint": call.get("endpoint")}
    only = deny = only_eps = deny_eps = None
    matched = []
    try:
        for rule in ruleset.rules:
            then = rule.then
            if not any(k in then for k in ("only", "deny", "only_endpoints",
                                           "deny_endpoints")):
                continue
            if not rule.matches(ctx):
                continue
            matched.append(rule.name or "<unnamed>")
            if "only" in then:
                new = frozenset(then["only"])
                only = new if only is None else (only & new)
            if "deny" in then:
                deny = frozenset(then["deny"]) | (deny or frozenset())
            if "only_endpoints" in then:
                new = frozenset(then["only_endpoints"])
                only_eps = new if only_eps is None else (only_eps & new)
            if "deny_endpoints" in then:
                deny_eps = frozenset(then["deny_endpoints"]) | (deny_eps or frozenset())
    except Exception:
        return None
    if not matched:
        return None

    ep, model = call.get("endpoint"), call.get("model")
    if only_eps is not None or deny_eps:
        if deny_eps and ep in deny_eps:
            return {"reason": "endpoint_denied", "subject": ep, "rules": matched}
        if only_eps is not None and ep not in only_eps:
            return {"reason": ("endpoint_undeclared" if ep is None
                               else "endpoint_not_permitted"),
                    "subject": ep, "rules": matched}
    if deny and model in deny:
        return {"reason": "model_denied", "subject": model, "rules": matched}
    if only is not None and model is not None and model not in only:
        return {"reason": "model_not_permitted", "subject": model,
                "rules": matched}
    return None


def _stalled(window, stall_window: int, min_novelty) -> bool:
    """Mirror of _TaskState._stall_check_locked, deliberately line-for-line.

    A plan that models enforcement APPROXIMATELY is a plan that lies, and the
    whole value of `plan` rests on it not lying. If the enforcement rule ever
    changes, this must change with it — the test
    `test_simulation_agrees_with_live_enforcement` fails loudly if they drift.
    """
    maxlen = max(2, stall_window or 8)
    if len(window) < maxlen:
        return False
    w = window[-maxlen:]
    fps = [r[0] for r in w if r[0] is not None]
    if len(fps) < len(w):
        return False                          # incomplete sample; never guess
    novelty = len(set(fps)) / float(len(fps))
    threshold = min_novelty if min_novelty is not None else 0.25
    if novelty > threshold:
        return False
    half = len(w) // 2
    first = [r[1] for r in w[:half] if r[1] is not None]
    second = [r[1] for r in w[half:] if r[1] is not None]
    if not first or not second:
        return False
    return sum(second) / len(second) > sum(first) / len(first)


# Whole-task stall SHAPE, using the same thresholds as the agent report: low
# response novelty together with growing input. This is retrospective ("this
# task looks stalled"), distinct from the rolling-window rule that decides
# where enforcement would halt.
_SHAPE_LOW_PROGRESS = 0.25
_SHAPE_GROWTH = 0.25


def _looks_stalled(calls) -> bool:
    fps = [c["rfp"] for c in calls if not c["hit"] and c["rfp"]]
    if len(fps) < 4:
        return False
    if len(set(fps)) / float(len(fps)) > _SHAPE_LOW_PROGRESS:
        return False
    toks = [c["tokens"] for c in calls if not c["hit"] and c["tokens"] is not None]
    if len(toks) < 4:
        return False
    half = len(toks) // 2
    first = sum(toks[:half]) / half
    second = sum(toks[half:]) / (len(toks) - half)
    return first > 0 and (second - first) / first >= _SHAPE_GROWTH


def _agent_resolution(tasks, agent_by_task, agent_of) -> str:
    """State plainly how many tasks could be attributed to an agent. A
    simulation that silently matched no agent-conditioned rule would under-
    report a rule's blast radius, which is the one direction a `plan` must
    never err in."""
    if not tasks:
        return "no tasks in window"
    missing = sum(1 for t in tasks if t not in agent_by_task)
    if missing == 0:
        return "from the ledger"
    if agent_of is not None:
        return f"from the ledger; {missing} task(s) fell back to the supplied map"
    return (f"{missing} of {len(tasks)} task(s) have no agent recorded — "
            f"agent-conditioned rules cannot match for those, so this plan "
            f"UNDER-reports their effect. Records written before v0.16 lack "
            f"the field; pass agent_of= to supply it.")


def simulate_rules(ruleset: RuleSet, records: Optional[Iterable[dict]] = None,
                   *, env: Optional[str] = None,
                   agent_of=None) -> Dict[str, Any]:
    """Replay `ruleset` over historical records and report what it would do.

    Args:
      ruleset: the policy under consideration — not installed, not applied.
      records: ledger records. Defaults to the live ledger.
      env: the environment to simulate under, so a team can ask "what would
        this do in CI?" without deploying to CI.
      agent_of: optional callable mapping a task_id to its agent name. The
        ledger stores task_id, not agent, so without this an agent-conditioned
        rule cannot match — the result says so in `agent_resolution` rather
        than silently matching nothing.

    Never raises on bad data: unusable records are excluded and counted.
    """
    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = list(_sv._tracker._iter_records())

    tasks: Dict[str, List[dict]] = {}
    agent_by_task: Dict[str, str] = {}
    unattributed = 0
    excluded = 0
    for rec in records:
        parsed = _numbers(rec)
        if parsed is None:
            excluded += 1
            continue
        cost, hit, shadow = parsed
        if shadow and hit:
            continue                      # a projection, never real execution
        tid = rec.get("task_id")
        if not tid:
            unattributed += 1
            continue
        try:
            tok = rec.get("input_tokens")
            tok = int(tok) if tok is not None else None
        except (TypeError, ValueError):
            tok = None
        tasks.setdefault(str(tid), []).append({
            "model": rec.get("model"),
            "endpoint": rec.get("endpoint_identity"),
            "data_class": rec.get("data_class"),
            "region": rec.get("region"),
            "cost": cost, "hit": hit,
            "fp": rec.get("prompt_fingerprint"),
            "rfp": rec.get("response_fingerprint"),
            "tokens": tok,
            "ts": rec.get("timestamp") or 0.0,
        })
        a = rec.get("agent")
        if a and str(tid) not in agent_by_task:
            agent_by_task[str(tid)] = str(a)

    halted: List[Dict[str, Any]] = []
    refusals: List[Dict[str, Any]] = []
    refusal_reasons: Dict[str, int] = {}
    refusal_rules: Dict[str, int] = {}
    already_shaped = 0
    completed = 0
    avoided_total = 0.0
    spend_total = 0.0
    by_agent: Dict[str, Dict[str, Any]] = {}

    for tid, calls in tasks.items():
        calls.sort(key=lambda c: c["ts"])
        # The record carries `agent` since v0.16, so an agent-conditioned rule
        # simulates correctly against any ledger written by a current node. The
        # `agent_of` override exists for OLDER ledgers recorded before that
        # field existed, where the mapping has to come from the caller.
        agent = agent_by_task.get(tid)
        if agent is None and agent_of is not None:
            try:
                agent = agent_of(tid)
            except Exception:
                agent = None
        limits, matched = resolve_limits(
            {"agent": agent, "env": env, "task_id": tid, "principal": None},
            {}, ruleset)

        envelope = limits.get("envelope")
        reserve = limits.get("reserve")
        max_calls = limits.get("max_calls")
        max_repeats = limits.get("max_repeats")
        stall_window = limits.get("stall_window")
        min_novelty = limits.get("min_novelty")

        spend = 0.0
        seen: Dict[str, int] = {}
        max_seen_cost = 0.0
        halt_at: Optional[int] = None
        halt_limit: Optional[str] = None
        progress: List[tuple] = []
        stalled = False

        for i, c in enumerate(calls):
            c["agent"] = agent
            verdict = _compliance_verdict(ruleset, c, env)
            if verdict is not None:
                refusal_reasons[verdict["reason"]] = \
                    refusal_reasons.get(verdict["reason"], 0) + 1
                for rn in verdict["rules"]:
                    refusal_rules[rn] = refusal_rules.get(rn, 0) + 1
                if len(refusals) < 20:
                    refusals.append({
                        "task_id": tid, "agent": agent, "call": i,
                        "reason": verdict["reason"],
                        "subject": verdict["subject"],
                        "region": c.get("region"),
                        "data_class": c.get("data_class"),
                        "rules": verdict["rules"]})
            # Same pre-flight order the node uses: envelope (hit-aware), then
            # call count, then repeat count, then stall — and the stall flag is
            # set AFTER a call completes, so it halts on the NEXT pre-flight,
            # exactly as enforcement does.
            if stalled:
                halt_at, halt_limit = i, "stalled"
                break
            if envelope is not None and not c["hit"]:
                hold = reserve if reserve is not None else max_seen_cost
                if spend + hold > envelope:
                    halt_at, halt_limit = i, "envelope"
                    break
            if max_calls is not None and i >= max_calls:
                halt_at, halt_limit = i, "max_calls"
                break
            fp = c["fp"]
            if (max_repeats is not None and fp is not None
                    and seen.get(fp, 0) >= max_repeats):
                halt_at, halt_limit = i, "max_repeats"
                break
            if fp is not None:
                seen[fp] = seen.get(fp, 0) + 1
            if not c["hit"]:
                spend += c["cost"]
                if c["cost"] > max_seen_cost:
                    max_seen_cost = c["cost"]
                if stall_window is not None:
                    progress.append((c["rfp"], c["tokens"]))
                    if _stalled(progress, int(stall_window), min_novelty):
                        stalled = True

        task_total = sum(c["cost"] for c in calls if not c["hit"])
        spend_total += task_total
        if halt_at is None:
            completed += 1
            # A task that already LOOKS stalled but that this replay does not
            # halt is the signature of history a policy has ALREADY truncated:
            # the call where the halt fires was never recorded. Counting it
            # matters because "0 would halt" would otherwise read as "this rule
            # is useless" — and someone might remove a policy that is working.
            if stall_window is not None and _looks_stalled(calls):
                already_shaped += 1
        else:
            avoided = sum(c["cost"] for c in calls[halt_at:] if not c["hit"])
            avoided_total += avoided
            halted.append({
                "task_id": tid, "agent": agent, "limit": halt_limit,
                "halted_at_call": halt_at, "total_calls": len(calls),
                "spend_before_halt_usd": round(spend, 6),
                "avoided_usd": round(avoided, 6),
                "matched_rules": matched,
            })
            b = by_agent.setdefault(agent or "(unknown)",
                                    {"halted": 0, "avoided_usd": 0.0})
            b["halted"] += 1
            b["avoided_usd"] = round(b["avoided_usd"] + avoided, 6)

    halted.sort(key=lambda h: h["avoided_usd"], reverse=True)
    return {
        "report": "rule_simulation",
        "ruleset_version": ruleset.version,
        "ruleset_source": ruleset.source,
        "rules_evaluated": len(ruleset),
        "env": env,
        "tasks_seen": len(tasks),
        "tasks_completed": completed,
        "tasks_would_halt": len(halted),
        "tasks_already_stall_shaped": already_shaped,
        "compliance_refusals": sum(refusal_reasons.values()),
        "refusals_by_reason": dict(sorted(refusal_reasons.items())),
        "refusals_by_rule": dict(sorted(refusal_rules.items())),
        "refused_calls": refusals,
        "spend_in_window_usd": round(spend_total, 6),
        "avoided_usd": round(avoided_total, 6),
        "halted_tasks": halted,
        "collateral_by_agent": by_agent,
        "unattributed_records": unattributed,
        "excluded_malformed_records": excluded,
        "agent_resolution": _agent_resolution(tasks, agent_by_task, agent_of),
        "tasks_without_agent": sum(1 for t in tasks if t not in agent_by_task),
        "note": (
            "Replay of these exact records under these exact rules. No "
            "projection and no percentage: avoided_usd is the spend that "
            "occurred AFTER the point each halt would have landed. Records "
            "with no task_id are counted separately, never folded in, because "
            "a task ceiling cannot be simulated for a call that belonged to "
            "no task."),
    }
