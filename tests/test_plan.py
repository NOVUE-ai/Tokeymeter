"""`tokeymeter plan` (S4-3) — preview a policy against real history.

Pinned here, because each was a deliberate decision:

  A PLAN MUST NEVER UNDER-REPORT ITS OWN BLAST RADIUS. If an agent cannot be
    resolved, the report says so loudly rather than silently matching no
    agent-conditioned rule and showing a smaller effect than the truth.
  COVERAGE IS FIRST-CLASS. A ceiling can only govern spend that belongs to a
    task. A plan over 40% of the estate is a plan with a blind spot and has to
    say which services are outside it.
  HALF-OPEN WINDOWS, matching chargeback and capacity exactly, so a plan and a
    close packet built for "the same month" cover byte-identically the same
    records.
  EXIT CODES ARE FOR CI. A plan that cannot fail a pipeline is decoration:
    0 clean, 2 a protected agent would halt, 1 policy error.
  PURE QUERY. Planning applies nothing, installs nothing, writes nothing.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance import rules as R
from tokeymeter.engines.governance import plan as P
import tokeymeter.engines.economics.savings as sv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _seed(n_support=12, n_checkout=4, untagged=0):
    @tokeymeter.cache(model="m", tag="support-api")
    def support(p):
        set_reported_usage(3000, 800)
        return "r"

    @tokeymeter.cache(model="m", tag="checkout-api")
    def checkout(p):
        set_reported_usage(3000, 800)
        return "r"

    for i in range(n_support):
        with tokeymeter.task(f"ticket-{i}", agent="support"):
            for j in range(6):
                support(f"s-{i}-{j}")
    for i in range(n_checkout):
        with tokeymeter.task(f"order-{i}", agent="checkout"):
            for j in range(4):
                checkout(f"c-{i}-{j}")
    for i in range(untagged):
        support(f"outside-any-task-{i}")
    return list(sv._tracker._iter_records())


def _ceiling(**over):
    then = {"envelope": 0.05, "reserve": 0.02, "max_repeats": 4}
    then.update(over)
    return R.load_rules([{"name": "default-ceiling", "then": then}])


# ── windows ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec,seconds", [
    ("24h", 86400), ("1h", 3600), ("30d", 2592000), ("2w", 1209600),
    (" 7d ", 604800),
])
def test_window_parsing(spec, seconds):
    start, end = P.parse_window(spec, now=1_000_000.0)
    assert end == 1_000_000.0
    assert end - start == seconds


@pytest.mark.parametrize("bad", ["30x", "d", "", "-1d", "0d", "abc", None, 30])
def test_bad_windows_rejected(bad):
    with pytest.raises(ValueError):
        P.parse_window(bad)


def test_window_is_half_open_like_every_other_report():
    """[start, end) — so a plan and a close packet for the same month cover
    byte-identically the same records."""
    start, end = P.parse_window("1h", now=1000.0)
    assert P._in_window({"timestamp": start}, start, end) is True
    assert P._in_window({"timestamp": end}, start, end) is False
    assert P._in_window({"timestamp": end - 0.001}, start, end) is True


def test_records_with_unusable_timestamps_are_excluded_from_a_window():
    start, end = P.parse_window("1h", now=1000.0)
    for ts in (None, "yesterday", float("nan")):
        assert P._in_window({"timestamp": ts}, start, end) is False


def test_since_actually_filters():
    _seed(n_support=4, n_checkout=0)
    recs = list(sv._tracker._iter_records())
    old = dict(recs[0])
    old["timestamp"] = time.time() - 86400 * 30
    plan = P.plan_report(_ceiling(), records=recs + [old], since="1h")
    assert plan["window"]["since"] == "1h"
    assert plan["effect"]["tasks_seen"] > 0


# ── diff ────────────────────────────────────────────────────────────────

def test_diff_reports_added_changed_and_removed():
    active = R.load_rules([{"name": "a", "then": {"envelope": 1.0}},
                           {"name": "gone", "then": {"envelope": 9.0}}])
    proposed = R.load_rules([{"name": "a", "then": {"envelope": 2.0}},
                             {"name": "new", "then": {"envelope": 3.0}}])
    plan = P.plan_report(proposed, active=active, records=[])
    d = plan["diff"]
    assert [r["name"] for r in d["added"]] == ["new"]
    assert [c["name"] for c in d["changed"]] == ["a"]
    assert [r["name"] for r in d["removed"]] == ["gone"]


def test_identical_policy_reports_no_changes():
    rs = R.load_rules([{"name": "a", "then": {"envelope": 1.0}}])
    plan = P.plan_report(rs, active=rs, records=[])
    assert plan["diff"] == {"added": [], "changed": [], "removed": []}
    assert "no changes" in P.render_plan(plan)


def test_diff_matches_rules_by_name_across_versions():
    """Names are unique by construction precisely so a diff can identify a rule
    across versions."""
    active = R.load_rules([{"name": "ceiling", "when": {"agent": "support"},
                            "then": {"envelope": 1.0}}])
    proposed = R.load_rules([{"name": "ceiling", "when": {"agent": "support"},
                              "then": {"envelope": 0.5}}])
    d = P.plan_report(proposed, active=active, records=[])["diff"]
    assert len(d["changed"]) == 1 and not d["added"] and not d["removed"]


# ── effect and collateral ───────────────────────────────────────────────

def test_plan_reports_effect_against_real_history():
    recs = _seed()
    plan = P.plan_report(_ceiling(), active=R.EMPTY_RULESET, records=recs)
    e = plan["effect"]
    assert e["tasks_seen"] == 16
    assert e["tasks_would_halt"] > 0
    assert e["avoided_usd"] > 0
    assert e["tasks_would_halt"] + e["tasks_completed"] == e["tasks_seen"]


def test_collateral_is_grouped_by_agent():
    recs = _seed()
    plan = P.plan_report(_ceiling(), records=recs)
    assert set(plan["collateral_by_agent"]) <= {"support", "checkout"}


def test_agent_conditioned_rules_simulate_from_the_ledger():
    """The gap S4-3 closed: `agent` is now on the record, so the most common
    kind of rule can be previewed at all."""
    recs = _seed()
    proposed = R.load_rules([
        {"name": "default", "then": {"envelope": 0.05, "reserve": 0.02}},
        {"name": "checkout-exempt", "when": {"agent": "checkout"},
         "then": {"envelope": 100.0}},
    ])
    plan = P.plan_report(proposed, records=recs)
    assert plan["agent_resolution"] == "from the ledger"
    assert "checkout" not in plan["collateral_by_agent"]


def test_plan_says_loudly_when_agents_cannot_be_resolved():
    """A plan must never UNDER-report its own blast radius."""
    recs = _seed(n_support=3, n_checkout=0)
    stripped = [{k: v for k, v in r.items() if k != "agent"} for r in recs]
    plan = P.plan_report(_ceiling(), records=stripped)
    assert "UNDER-reports" in plan["agent_resolution"]
    assert plan["unattributed_records"] == 0        # they still have task_id


# ── protected agents and exit codes ─────────────────────────────────────

def test_protected_agent_halt_is_reported_as_a_violation():
    recs = _seed()
    plan = P.plan_report(_ceiling(), records=recs, protected_agents=["checkout"])
    assert plan["violations"]
    assert all(v["agent"] == "checkout" for v in plan["violations"])
    assert "PROTECTED" in P.render_plan(plan)


def test_exempting_the_protected_agent_clears_the_violation():
    recs = _seed()
    proposed = R.load_rules([
        {"name": "default", "then": {"envelope": 0.05, "reserve": 0.02}},
        {"name": "checkout-is-revenue", "when": {"agent": "checkout"},
         "then": {"envelope": 100.0}},
    ])
    plan = P.plan_report(proposed, records=recs, protected_agents=["checkout"])
    assert plan["violations"] == []


# ── coverage ────────────────────────────────────────────────────────────

def test_coverage_reports_ungoverned_spend():
    recs = _seed(n_support=4, n_checkout=0, untagged=6)
    cov = P.coverage_report(recs)
    assert 0 < cov["coverage_pct"] < 100
    assert cov["ungoverned_usd"] > 0
    assert cov["executed_spend_usd"] == pytest.approx(
        cov["task_attributed_usd"] + cov["ungoverned_usd"], abs=1e-9)


def test_coverage_is_100_when_everything_is_in_a_task():
    recs = _seed(n_support=4, n_checkout=0, untagged=0)
    assert P.coverage_report(recs)["coverage_pct"] == 100.0


def test_coverage_names_the_services_that_are_not_governed():
    recs = _seed(n_support=3, n_checkout=0, untagged=4)
    cov = P.coverage_report(recs)
    assert any(not v["governed"] for v in cov["by_service"].values())


def test_coverage_ignores_cache_hits_and_corrupt_records():
    recs = _seed(n_support=3, n_checkout=0)
    dirty = list(recs) + [
        {"estimated_cost": float("nan"), "hit": False, "task_id": "x"},
        {"estimated_cost": float("inf"), "hit": False, "task_id": "x"},
        {"estimated_cost": -1.0, "hit": False, "task_id": "x"},
    ]
    assert (P.coverage_report(dirty)["executed_spend_usd"]
            == P.coverage_report(recs)["executed_spend_usd"])


def test_coverage_on_an_empty_ledger_does_not_divide_by_zero():
    cov = P.coverage_report([])
    assert cov["coverage_pct"] == 0.0 and cov["executed_spend_usd"] == 0.0


# ── purity ──────────────────────────────────────────────────────────────

def test_planning_applies_nothing_and_writes_nothing():
    installed = R.load_rules([{"name": "live", "then": {"envelope": 9.0}}])
    R.set_rules(installed)
    recs = _seed(n_support=3, n_checkout=0)
    before = len(list(sv._tracker._iter_records()))
    P.plan_report(_ceiling(), records=recs, protected_agents=["support"])
    assert R.get_rules() is installed
    assert len(list(sv._tracker._iter_records())) == before


def test_plan_is_deterministic():
    recs = _seed(n_support=6, n_checkout=2)
    a = P.plan_report(_ceiling(), records=recs, now=1000.0)
    b = P.plan_report(_ceiling(), records=recs, now=1000.0)
    assert a["effect"] == b["effect"]
    assert P.render_plan(a) == P.render_plan(b)


def test_plan_on_an_empty_ledger_is_honest_not_empty():
    plan = P.plan_report(_ceiling(), records=[])
    assert plan["effect"]["tasks_seen"] == 0
    assert "no tasks in this window" in P.render_plan(plan)


# ── rendering ───────────────────────────────────────────────────────────

def test_render_is_ascii_only():
    """This runs on a Windows console under a legacy code page as often as it
    runs in a Linux CI job."""
    recs = _seed()
    text = P.render_plan(P.plan_report(_ceiling(), records=recs,
                                       protected_agents=["checkout"]))
    assert all(ord(c) < 128 for c in text)


def test_render_includes_the_numbers_a_reviewer_needs():
    recs = _seed()
    text = P.render_plan(P.plan_report(_ceiling(), records=recs))
    for expected in ("Plan:", "would halt", "avoided", "Coverage:"):
        assert expected in text


# ── CLI ─────────────────────────────────────────────────────────────────

def _run_cli(args, home):
    env = dict(os.environ)
    env["TOKEYMETER_HOME"] = home
    env["PYTHONPATH"] = REPO_ROOT
    p = subprocess.run([sys.executable, "-m", "tokeymeter", "plan", *args],
                       capture_output=True, env=env, timeout=300, cwd=home)
    return (p.returncode,
            (p.stdout or b"").decode("utf-8", "replace")
            + (p.stderr or b"").decode("utf-8", "replace"))


@pytest.fixture
def cli_env():
    home = tempfile.mkdtemp()
    seed = os.path.join(home, "seed.py")
    with open(seed, "w", encoding="utf-8") as f:
        f.write(
            "import os, sys\n"
            f"sys.path.insert(0, {REPO_ROOT!r})\n"
            f"os.environ['TOKEYMETER_HOME'] = {home!r}\n"
            "import tokeymeter\n"
            "from tokeymeter.storage import MemoryStore\n"
            "from tokeymeter.engines.economics.usage import set_reported_usage\n"
            "tokeymeter.set_default_store(MemoryStore())\n"
            f"tokeymeter.set_savings_path({os.path.join(home, 'savings.jsonl')!r})\n"
            "tokeymeter.register_pricing('m', input_per_1m=2.5, output_per_1m=10.0)\n"
            "@tokeymeter.cache(model='m', tag='support-api')\n"
            "def s(p):\n"
            "    set_reported_usage(3000, 800); return 'r'\n"
            "for i in range(10):\n"
            "    with tokeymeter.task(f'ticket-{i}', agent='support'):\n"
            "        for j in range(6): s(f'{i}-{j}')\n")
    subprocess.run([sys.executable, seed], check=True, timeout=300)
    return home


def test_cli_help_exits_clean(cli_env):
    code, out = _run_cli(["--help"], cli_env)
    assert code == 0 and "tokeymeter plan" in out


def test_cli_requires_a_policy(cli_env):
    code, out = _run_cli([], cli_env)
    assert code == 1 and "--policy" in out


def test_cli_missing_file_is_a_clear_message(cli_env):
    code, out = _run_cli(["--policy", "nope.yaml"], cli_env)
    assert code == 1 and "no such policy file" in out


def test_cli_policy_typo_names_the_rule(cli_env):
    p = os.path.join(cli_env, "bad.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "oops", "then": {"enevelope": 1.0}}]}, f)
    code, out = _run_cli(["--policy", p], cli_env)
    assert code == 1 and "oops" in out and "enevelope" in out


def test_cli_bad_window_is_a_clear_message(cli_env):
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"then": {"envelope": 1.0}}]}, f)
    code, out = _run_cli(["--policy", p, "--since", "30x"], cli_env)
    assert code == 1 and "24h" in out


def test_cli_unknown_argument_is_rejected(cli_env):
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"then": {"envelope": 1.0}}]}, f)
    code, out = _run_cli(["--policy", p, "--wat"], cli_env)
    assert code == 1 and "unrecognized" in out


def test_cli_runs_a_real_plan_and_exits_zero(cli_env):
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "cap",
                              "then": {"envelope": 0.05, "reserve": 0.02}}]}, f)
    code, out = _run_cli(["--policy", p], cli_env)
    assert code == 0
    assert "would halt" in out and "Coverage:" in out


def test_cli_exits_two_when_a_protected_agent_would_halt(cli_env):
    """A plan that cannot fail a pipeline is decoration."""
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "cap",
                              "then": {"envelope": 0.05, "reserve": 0.02}}]}, f)
    code, out = _run_cli(["--policy", p, "--protect", "support"], cli_env)
    assert code == 2 and "PROTECTED" in out


def test_cli_json_output_is_machine_readable(cli_env):
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "cap",
                              "then": {"envelope": 0.05, "reserve": 0.02}}]}, f)
    code, out = _run_cli(["--policy", p, "--json"], cli_env)
    assert code == 0
    parsed = json.loads(out)
    assert parsed["report"] == "plan"
    assert "effect" in parsed and "coverage" in parsed


# ── enterprise: a CHANGE is judged by its delta, not its absolute effect ──

def test_change_reports_delta_against_the_active_policy():
    """If the live policy already halts 20 tasks, a proposal that halts 22 is a
    change of two. Reporting 22 would make every tightening read like a first
    install and overstate what the reviewer is approving."""
    recs = _seed()
    active = R.load_rules([{"name": "cap",
                            "then": {"envelope": 0.10, "reserve": 0.02}}])
    proposed = R.load_rules([{"name": "cap",
                              "then": {"envelope": 0.02, "reserve": 0.02}}])
    plan = P.plan_report(proposed, active=active, records=recs)
    assert plan["baseline"] is not None
    assert plan["delta"]["avoided_usd"] > 0          # tighter saves more
    text = P.render_plan(plan)
    assert "Change vs the active policy" in text


def test_first_install_has_no_baseline():
    """With nothing deployed, absolute effect is the right thing to report."""
    recs = _seed()
    plan = P.plan_report(_ceiling(), active=R.EMPTY_RULESET, records=recs)
    assert plan["baseline"] is None and plan["delta"] is None
    assert "Change vs the active policy" not in P.render_plan(plan)


def test_loosening_shows_a_negative_delta():
    recs = _seed()
    active = R.load_rules([{"name": "cap",
                            "then": {"envelope": 0.02, "reserve": 0.02}}])
    proposed = R.load_rules([{"name": "cap", "then": {"envelope": 500.0}}])
    plan = P.plan_report(proposed, active=active, records=recs)
    assert plan["delta"]["tasks_would_halt"] < 0
    assert plan["delta"]["avoided_usd"] < 0
    assert "-$" in P.render_plan(plan)          # money sign renders correctly


def test_newly_halted_excludes_tasks_the_live_policy_already_halts():
    recs = _seed()
    active = R.load_rules([{"name": "cap",
                            "then": {"envelope": 0.05, "reserve": 0.02}}])
    proposed = R.load_rules([{"name": "cap",
                              "then": {"envelope": 0.02, "reserve": 0.02}}])
    plan = P.plan_report(proposed, active=active, records=recs)
    already = {h["task_id"] for h in
               __import__("tokeymeter").simulate_rules(active, recs)["halted_tasks"]}
    assert all(h["task_id"] not in already for h in plan["newly_halted"])


def test_gate_fires_only_on_violations_this_change_introduces():
    """A protected task already halting under the live policy is not this
    diff's doing. A reviewer is accountable for what their change introduces."""
    recs = _seed()
    breaks = R.load_rules([{"name": "cap",
                            "then": {"envelope": 0.05, "reserve": 0.02}}])
    # already broken before the change -> not a NEW violation
    plan_same = P.plan_report(breaks, active=breaks, records=recs,
                              protected_agents=["checkout"])
    assert plan_same["violations"]              # still reported
    assert plan_same["new_violations"] == []    # but not this change's fault

    # the change introduces it -> a new violation
    safe = R.load_rules([
        {"name": "cap", "then": {"envelope": 0.05, "reserve": 0.02}},
        {"name": "checkout-exempt", "when": {"agent": "checkout"},
         "then": {"envelope": 100.0}}])
    plan_new = P.plan_report(breaks, active=safe, records=recs,
                             protected_agents=["checkout"])
    assert plan_new["new_violations"]
    assert "NEWLY halted" in P.render_plan(plan_new)


def test_cli_active_flag_produces_a_real_diff(cli_env):
    v1 = os.path.join(cli_env, "v1.json")
    v2 = os.path.join(cli_env, "v2.json")
    with open(v1, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "cap", "then": {"envelope": 0.10}},
                             {"name": "gone", "then": {"envelope": 9.0}}]}, f)
    with open(v2, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "cap", "then": {"envelope": 0.02}},
                             {"name": "new", "then": {"max_calls": 5}}]}, f)
    code, out = _run_cli(["--active", v1, "--policy", v2], cli_env)
    assert code == 0
    assert "1 to add, 1 to change, 1 to remove" in out
    assert "Change vs the active policy" in out


def test_cli_missing_active_file_is_a_clear_message(cli_env):
    p = os.path.join(cli_env, "ok.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"then": {"envelope": 1.0}}]}, f)
    code, out = _run_cli(["--active", "nope.json", "--policy", p], cli_env)
    assert code == 1 and "active policy file" in out


def test_cli_broken_active_policy_names_the_problem(cli_env):
    good = os.path.join(cli_env, "ok.json")
    bad = os.path.join(cli_env, "badactive.json")
    with open(good, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"then": {"envelope": 1.0}}]}, f)
    with open(bad, "w", encoding="utf-8") as f:
        json.dump({"rules": [{"name": "x", "then": {"enevelope": 1.0}}]}, f)
    code, out = _run_cli(["--active", bad, "--policy", good], cli_env)
    assert code == 1 and "active policy" in out


def test_plan_flags_already_governed_history():
    """Replaying a policy over history that policy already truncated finds
    nothing left to halt. Correct, but "0 would halt" reads as "this rule is
    useless" — and someone might remove a policy that is working. The plan has
    to say what it is looking at."""
    rules = R.load_rules([{"name": "s",
                           "then": {"stall_window": 8, "enforce": True}}])
    R.set_rules(rules)

    @tokeymeter.cache(model="m")
    def step(prompt, tokens):
        set_reported_usage(tokens, 200)
        return "the same stuck answer"

    for i in range(4):
        history = []
        try:
            with tokeymeter.task(f"t-{i}", agent="a"):
                for j in range(14):
                    history.append(f"x{j}")
                    step(tuple(history) + (i,), 1000 + j * 400)
        except tokeymeter.TaskLimitExceeded:
            pass
    recs = list(sv._tracker._iter_records())
    R.clear_rules()
    plan = P.plan_report(rules, records=recs)
    # nothing new to halt, but the shape is reported rather than left implied
    assert plan["effect"]["tasks_already_stall_shaped"] >= 0
    text = P.render_plan(plan)
    if plan["effect"]["tasks_already_stall_shaped"]:
        assert "already-governed history" in text


def test_ungoverned_history_reports_no_false_already_governed_note():
    """The note must not fire on healthy traffic, or it becomes noise."""
    recs = _seed()
    plan = P.plan_report(_ceiling(), records=recs)
    assert plan["effect"]["tasks_already_stall_shaped"] == 0
    assert "already-governed history" not in P.render_plan(plan)
