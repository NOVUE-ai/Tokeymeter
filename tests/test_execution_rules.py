"""Execution rules (S4-2) — the company's AI execution policy, as data.

Pinned here, because each was a deliberate decision:

  VALIDATION IS LOUD, EVALUATION IS SILENT. A typo'd field or unknown action is
    rejected at LOAD, naming the rule. Nobody discovers a misspelled
    `enevelope` at 3am because it silently matched nothing.
  LAST MATCH WINS, PER ACTION. A broad default first, a specific override
    after — CSS/firewall semantics. A later rule overrides only the actions it
    actually sets.
  MOST RESTRICTIVE WINS between policy and code. Neither side can loosen what
    the other set, so neither has to trust the other. `reserve` combines by MAX
    because holding more per call halts EARLIER.
  FAIL-OPEN MEANS "POLICY CONTRIBUTES NOTHING", never "the ceiling
    disappears" — limits declared in code still apply when a policy is broken.
  RESOLVED ONCE PER TASK, never per call, so policy costs nothing on the hot
    path.
  SIMULATION MUST PREDICT ENFORCEMENT EXACTLY. A plan that lies is worse than
    no plan.
"""
import json
import os
import tempfile
import time

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance import rules as R
from tokeymeter.engines.governance.simulate import simulate_rules
import tokeymeter.engines.economics.savings as sv


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    R.clear_rules()
    R.set_env(None)
    yield
    R.clear_rules()
    R.set_env(None)
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _step(model="m"):
    @tokeymeter.cache(model=model)
    def step(p):
        set_reported_usage(3000, 800)      # $0.0155 per executed call
        return "r"
    return step


def _records():
    return list(sv._tracker._iter_records())


# ── validation: loud, at load time ──────────────────────────────────────

def test_unknown_field_rejected_with_the_rule_named():
    with pytest.raises(R.RulePolicyError) as ei:
        R.load_rules([{"name": "bad", "when": {"nonexistent": "x"},
                       "then": {"envelope": 1.0}}])
    assert "nonexistent" in str(ei.value)


def test_unknown_action_rejected_with_the_rule_named():
    with pytest.raises(R.RulePolicyError) as ei:
        R.load_rules([{"name": "typo", "then": {"enevelope": 1.0}}])
    assert "typo" in str(ei.value) and "enevelope" in str(ei.value)


def test_unknown_operator_rejected():
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"when": [{"field": "agent", "op": "~=", "value": "x"}],
                       "then": {"envelope": 1.0}}])


def test_rule_with_no_actions_rejected():
    """A rule that sets nothing would silently do nothing."""
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"name": "empty", "when": {"agent": "x"}, "then": {}}])


def test_duplicate_rule_names_rejected():
    """Names must be unique so an audit trail can identify which rule applied."""
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"name": "a", "then": {"envelope": 1.0}},
                      {"name": "a", "then": {"envelope": 2.0}}])


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "abc"])
def test_bad_limit_values_rejected(bad):
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"then": {"envelope": bad}}])


def test_non_integer_call_limit_rejected():
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"then": {"max_calls": 2.5}}])


def test_enforce_must_be_boolean():
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"then": {"enforce": "yes"}}])


def test_in_operator_requires_a_list():
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"when": [{"field": "env", "op": "in", "value": "ci"}],
                       "then": {"envelope": 1.0}}])


def test_malformed_json_rejected():
    with pytest.raises(R.RulePolicyError):
        R.load_rules("{not json")


# ── resolution semantics ────────────────────────────────────────────────

def _rs():
    return R.load_rules({"version": 1, "rules": [
        {"name": "default", "then": {"envelope": 1.00, "max_repeats": 4}},
        {"name": "research", "when": {"agent": "research"},
         "then": {"envelope": 10.00}},
        {"name": "ci", "when": {"env": ["dev", "ci"]},
         "then": {"envelope": 0.25, "enforce": True}},
    ]})


def test_empty_when_matches_everything():
    limits, matched = R.resolve_limits({"agent": "anything"}, {}, _rs())
    assert limits["envelope"] == 1.00
    assert "default" in matched


def test_last_match_wins_per_action():
    """A later rule overrides only the actions it sets — the default's
    max_repeats survives an envelope override."""
    limits, matched = R.resolve_limits(
        {"agent": "research", "env": "ci"}, {}, _rs())
    assert limits["envelope"] == 0.25        # ci overrode research overrode default
    assert limits["max_repeats"] == 4        # still from default
    assert limits["enforce"] is True
    assert matched == ["default", "research", "ci"]


def test_absent_field_matches_nothing_except_negation():
    """An unbound field must never satisfy a rule by accident."""
    c_eq = R.Condition("agent", "==", "support")
    c_ne = R.Condition("agent", "!=", "support")
    assert c_eq.matches({}) is False
    assert c_ne.matches({}) is True


def test_code_can_tighten_policy():
    limits, _ = R.resolve_limits({"agent": "support"}, {"envelope": 0.50}, _rs())
    assert limits["envelope"] == 0.50


def test_policy_can_tighten_code():
    limits, _ = R.resolve_limits({"agent": "research"}, {"envelope": 50.0}, _rs())
    assert limits["envelope"] == 10.00


def test_reserve_combines_by_max_because_holding_more_halts_earlier():
    rs = R.load_rules([{"then": {"reserve": 0.05}}])
    limits, _ = R.resolve_limits({}, {"reserve": 0.01}, rs)
    assert limits["reserve"] == 0.05


def test_enforce_combines_by_or():
    rs = R.load_rules([{"then": {"enforce": True}}])
    limits, _ = R.resolve_limits({}, {"enforce": False}, rs)
    assert limits["enforce"] is True


def test_startswith_and_not_in():
    rs = R.load_rules([
        {"name": "tickets", "when": [{"field": "task_id", "op": "startswith",
                                      "value": "ticket-"}],
         "then": {"envelope": 2.0}},
        {"name": "not-prod", "when": [{"field": "env", "op": "not_in",
                                       "value": ["prod"]}],
         "then": {"max_calls": 5}},
    ])
    limits, _ = R.resolve_limits({"task_id": "ticket-9", "env": "ci"}, {}, rs)
    assert limits["envelope"] == 2.0 and limits["max_calls"] == 5
    limits2, _ = R.resolve_limits({"task_id": "job-9", "env": "prod"}, {}, rs)
    assert limits2 == {}


# ── file loading ────────────────────────────────────────────────────────

def test_json_file_round_trip():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "policy.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"version": 7, "rules": [{"name": "x",
                                            "then": {"envelope": 1.0}}]}, f)
    rs = R.load_rules_file(p)
    assert len(rs) == 1 and rs.version == 7 and rs.source == p


def test_yaml_file_when_available():
    yaml = pytest.importorskip("yaml")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "policy.yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write("version: 2\nrules:\n  - name: d\n    then: {envelope: 1.5}\n")
    rs = R.load_rules_file(p)
    assert rs.version == 2 and rs.rules[0].then["envelope"] == 1.5


def test_explicit_version_beats_file_mtime():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "policy.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"version": 42, "rules": [{"then": {"envelope": 1.0}}]}, f)
    assert R.load_rules_file(p).version == 42


def test_mtime_used_when_no_explicit_version():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "policy.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"then": {"envelope": 1.0}}]}, f)
    assert R.load_rules_file(p).version > 0


# ── integration with task() ─────────────────────────────────────────────

def test_policy_supplies_limits_when_code_declares_none():
    R.set_rules(R.load_rules([{"name": "d",
                               "then": {"envelope": 0.05, "reserve": 0.02,
                                        "enforce": True}}]))
    step = _step()
    with tokeymeter.task("t-1", agent="support") as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(100):
                step(f"d-{i}")
    snap = st.snapshot()
    assert snap["envelope_usd"] == 0.05
    assert snap["spend_usd"] <= 0.05
    assert snap["matched_rules"] == ["d"]


def test_agent_conditioned_rules_apply():
    R.set_rules(R.load_rules([
        {"name": "default", "then": {"envelope": 0.05, "enforce": True}},
        {"name": "research", "when": {"agent": "research"},
         "then": {"envelope": 100.0}},
    ]))
    step = _step()
    with tokeymeter.task("r-1", agent="research") as st:
        for i in range(20):
            step(f"d-{i}")                 # must NOT halt
    assert st.snapshot()["envelope_usd"] == 100.0


def test_env_conditioned_rules_apply():
    R.set_rules(R.load_rules([
        {"name": "ci", "when": {"env": ["dev", "ci"]},
         "then": {"envelope": 0.02, "enforce": True}}]))
    R.set_env("ci")
    step = _step()
    with tokeymeter.task("t-1", agent="support") as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"d-{i}")
    assert st.snapshot()["envelope_usd"] == 0.02


def test_env_from_environment_variable():
    R.set_env(None)
    os.environ["TOKEYMETER_ENV"] = "staging"
    try:
        assert R.current_env() == "staging"
    finally:
        del os.environ["TOKEYMETER_ENV"]


def test_explicit_env_overrides_the_variable():
    os.environ["TOKEYMETER_ENV"] = "prod"
    try:
        R.set_env("ci")
        assert R.current_env() == "ci"
    finally:
        del os.environ["TOKEYMETER_ENV"]
        R.set_env(None)


def test_malformed_env_variable_ignored_not_guessed():
    R.set_env(None)
    os.environ["TOKEYMETER_ENV"] = "not a valid env!!"
    try:
        assert R.current_env() is None
    finally:
        del os.environ["TOKEYMETER_ENV"]


def test_code_declared_limits_still_apply_with_no_policy():
    """Backward compatible: S4-1 behaviour is unchanged when no rules exist."""
    R.clear_rules()
    step = _step()
    with tokeymeter.task("t", envelope=0.02, reserve=0.02, enforce=True) as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"d-{i}")
    assert st.snapshot()["spend_usd"] <= 0.02


def test_policy_and_code_both_apply_most_restrictive():
    R.set_rules(R.load_rules([{"then": {"envelope": 1.00, "enforce": True}}]))
    step = _step()
    with tokeymeter.task("t", envelope=0.02, reserve=0.02) as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"d-{i}")
    snap = st.snapshot()
    assert snap["envelope_usd"] == 0.02       # code was tighter
    assert snap["enforce"] is True            # policy turned enforcement on


# ── fail-open ───────────────────────────────────────────────────────────

def test_broken_policy_never_removes_a_code_declared_ceiling():
    """Fail-open means policy contributes nothing — not that the ceiling
    disappears."""
    class Exploding(R.RuleSet):
        def resolve(self, context):
            raise RuntimeError("policy engine broke")

    R.set_rules(Exploding(R.load_rules([{"then": {"envelope": 5.0}}]).rules))
    step = _step()
    with tokeymeter.task("t", envelope=0.02, reserve=0.02, enforce=True) as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"d-{i}")
    assert st.snapshot()["spend_usd"] <= 0.02


def test_set_rules_rejects_a_non_ruleset():
    with pytest.raises(TypeError):
        R.set_rules({"rules": []})


def test_rules_swap_is_atomic_and_returns_the_previous():
    a = R.load_rules([{"name": "a", "then": {"envelope": 1.0}}])
    b = R.load_rules([{"name": "b", "then": {"envelope": 2.0}}])
    R.set_rules(a)
    prev = R.set_rules(b)
    assert prev is a and R.get_rules() is b


def test_rule_resolution_costs_nothing_on_the_hot_path():
    """Rules resolve ONCE per task. A 200-call task must not pay 200 times."""
    R.set_rules(R.load_rules([{"then": {"envelope": 1000.0}} for _ in range(1)]))
    step = _step()
    t0 = time.perf_counter()
    with tokeymeter.task("t", agent="a"):
        for i in range(200):
            step(f"d-{i}")
    with_rules = time.perf_counter() - t0

    R.clear_rules()
    tokeymeter.reset_savings()
    step2 = _step()
    t1 = time.perf_counter()
    with tokeymeter.task("t2", agent="a"):
        for i in range(200):
            step2(f"e-{i}")
    without = time.perf_counter() - t1
    # generous bound: resolution is once, so the per-call cost must not grow
    assert with_rules < without * 3 + 0.05


# ── simulation: the wedge ───────────────────────────────────────────────

def _history(n_tasks=20, loop_every=5):
    """Ungoverned history with deliberate loops."""
    step = _step()
    agents = {}
    for i in range(n_tasks):
        agent = "checkout" if i % 10 == 0 else "support"
        tid = f"task-{i}"
        agents[tid] = agent
        with tokeymeter.task(tid, agent=agent):
            if i % loop_every == 0:
                for _ in range(20):
                    step("the same stuck call")
            else:
                for j in range(3):
                    step(f"{tid}-{j}")
    return _records(), agents


def test_simulation_reports_what_a_rule_would_have_done():
    recs, agents = _history()
    proposed = R.load_rules([{"name": "cap",
                              "then": {"envelope": 0.05, "reserve": 0.02}}])
    plan = simulate_rules(proposed, recs, agent_of=agents.get)
    assert plan["tasks_seen"] > 0
    assert plan["tasks_would_halt"] > 0
    assert plan["avoided_usd"] > 0
    assert plan["tasks_completed"] + plan["tasks_would_halt"] == plan["tasks_seen"]


def test_simulation_surfaces_collateral_by_agent():
    """A rule that would halt the revenue path must be visible BEFORE apply."""
    recs, agents = _history()
    # A loop is served from cache after its first call, so it costs almost
    # nothing — an envelope alone never catches one. max_repeats does.
    proposed = R.load_rules([{"name": "cap", "then": {"envelope": 0.02,
                                                      "reserve": 0.02,
                                                      "max_repeats": 4}}])
    plan = simulate_rules(proposed, recs, agent_of=agents.get)
    assert "checkout" in plan["collateral_by_agent"]


def test_simulation_predicts_enforcement_exactly():
    """A plan that lies is worse than no plan."""
    rules = R.load_rules([{"name": "r", "then": {"envelope": 0.05,
                                                 "reserve": 0.02,
                                                 "max_repeats": 4,
                                                 "enforce": True}}])
    # ungoverned run
    R.clear_rules()
    recs, agents = _history()
    ungoverned = sum(r["estimated_cost"] for r in recs if not r["hit"])
    plan = simulate_rules(rules, recs, agent_of=agents.get)

    # same workload, enforcement live — from a COLD cache, or every prompt is
    # a hit left over from the ungoverned run and the comparison is meaningless
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset_savings()
    R.set_rules(rules)
    step = _step()
    actual_halts = set()
    for i in range(20):
        agent = "checkout" if i % 10 == 0 else "support"
        tid = f"task-{i}"
        try:
            with tokeymeter.task(tid, agent=agent):
                if i % 5 == 0:
                    for _ in range(20):
                        step("the same stuck call")
                else:
                    for j in range(3):
                        step(f"{tid}-{j}")
        except tokeymeter.TaskLimitExceeded:
            actual_halts.add(tid)
    governed = sum(r["estimated_cost"] for r in _records() if not r["hit"])

    predicted = {h["task_id"] for h in plan["halted_tasks"]}
    assert predicted == actual_halts
    assert abs(plan["avoided_usd"] - (ungoverned - governed)) < 1e-9


def test_simulation_excludes_corrupt_records_and_counts_them():
    recs, agents = _history(n_tasks=4)
    recs = list(recs) + [
        {"task_id": "task-0", "estimated_cost": float("nan"), "hit": False},
        {"task_id": "task-0", "estimated_cost": float("inf"), "hit": False},
        {"task_id": "task-0", "estimated_cost": -5.0, "hit": False},
    ]
    plan = simulate_rules(R.load_rules([{"then": {"envelope": 1.0}}]), recs,
                          agent_of=agents.get)
    assert plan["excluded_malformed_records"] == 3


def test_simulation_counts_records_with_no_task_separately():
    """A task ceiling cannot be simulated for a call that belonged to no task."""
    step = _step()
    step("outside any task")
    plan = simulate_rules(R.load_rules([{"then": {"envelope": 1.0}}]),
                          _records())
    assert plan["unattributed_records"] >= 1
    assert plan["tasks_seen"] == 0


def test_agent_now_resolves_from_the_ledger():
    """S4-3 put `agent` on the record, so agent-conditioned rules — the most
    common kind — can finally be simulated against real history."""
    recs, _ = _history(n_tasks=3)
    plan = simulate_rules(R.load_rules([{"then": {"envelope": 1.0}}]), recs)
    assert plan["agent_resolution"] == "from the ledger"
    assert plan["tasks_without_agent"] == 0


def test_simulation_says_loudly_when_agent_cannot_be_resolved():
    """A ledger written before v0.16 has no `agent` field. The report must SAY
    it is under-reporting rather than silently matching no agent-conditioned
    rule and showing a smaller blast radius than the truth."""
    recs, _ = _history(n_tasks=3)
    old_ledger = [{k: v for k, v in r.items() if k != "agent"} for r in recs]
    plan = simulate_rules(R.load_rules([{"then": {"envelope": 1.0}}]),
                          old_ledger)
    assert "UNDER-reports" in plan["agent_resolution"]
    assert plan["tasks_without_agent"] == plan["tasks_seen"]


def test_supplied_agent_map_covers_an_old_ledger():
    recs, agents = _history(n_tasks=3)
    old_ledger = [{k: v for k, v in r.items() if k != "agent"} for r in recs]
    plan = simulate_rules(R.load_rules([{"then": {"envelope": 1.0}}]),
                          old_ledger, agent_of=agents.get)
    assert "fell back to the supplied map" in plan["agent_resolution"]


def test_simulation_changes_nothing():
    """`plan` must be a pure query — it applies no policy and writes no record."""
    recs, agents = _history(n_tasks=4)
    before = len(_records())
    active_before = R.get_rules()
    simulate_rules(R.load_rules([{"then": {"envelope": 0.01}}]), recs,
                   agent_of=agents.get)
    assert len(_records()) == before
    assert R.get_rules() is active_before


def test_simulation_is_deterministic():
    recs, agents = _history(n_tasks=8)
    proposed = R.load_rules([{"then": {"envelope": 0.05, "reserve": 0.02}}])
    a = simulate_rules(proposed, recs, agent_of=agents.get)
    b = simulate_rules(proposed, recs, agent_of=agents.get)
    assert a["avoided_usd"] == b["avoided_usd"]
    assert [h["task_id"] for h in a["halted_tasks"]] == \
           [h["task_id"] for h in b["halted_tasks"]]


def test_envelope_alone_cannot_catch_a_cached_loop():
    """The property that makes max_repeats necessary: after the first call a
    loop is served from cache, so it spends almost nothing. A policy with only
    an envelope has a hole an agent can spin in forever."""
    R.set_rules(R.load_rules([{"name": "money-only",
                               "then": {"envelope": 1.00, "reserve": 0.02,
                                        "enforce": True}}]))
    step = _step()
    with tokeymeter.task("loop", agent="support") as st:
        for _ in range(200):
            step("the same stuck call")     # no halt: it costs one call
    snap = st.snapshot()
    assert snap["calls"] == 200
    assert snap["spend_usd"] < 0.02          # 199 of 200 were free


def test_max_repeats_catches_what_the_envelope_cannot():
    R.set_rules(R.load_rules([{"name": "both",
                               "then": {"envelope": 1.00, "reserve": 0.02,
                                        "max_repeats": 4, "enforce": True}}]))
    step = _step()
    with tokeymeter.task("loop", agent="support"):
        with pytest.raises(tokeymeter.TaskLoopDetected):
            for _ in range(200):
                step("the same stuck call")
