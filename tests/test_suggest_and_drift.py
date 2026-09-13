"""Threshold suggestion, drift, and the coding-agent install guide.

Pinned here, because each was a deliberate decision:

  A SUGGESTION IS BACKTESTED OR IT IS NOT MADE. Every recommended value is
    replayed against real history and reported with both counts that decide it:
    stalled tasks caught, and HEALTHY tasks it would also have stopped.
  ZERO FALSE POSITIVES IS A HARD REQUIREMENT, not a preference. A ceiling that
    kills working tasks gets switched off within a day, and then it protects
    nothing.
  REFUSING TO ANSWER IS AN ANSWER. Too little history, no stalls to tune
    against, or batch-shaped work each produce a stated reason instead of a
    fabricated number.
  TIES GO TO THE TIGHTEST SETTING, and that direction was MEASURED. Holdout
    testing on unseen traffic showed the looser choice catching 28 of 40 stalled
    tasks where the tighter caught 40 of 40, with no false positives either way.
    Safety comes from the zero-false-positive filter, not from looseness.
  DRIFT COMPARES ONLY WHAT EXISTS IN BOTH WINDOWS. A new agent has nothing to
    drift from; one that stopped running has not improved.
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
from tokeymeter.engines.governance.suggest import (
    suggest_thresholds, render_suggestions)
from tokeymeter.engines.governance.agents import (
    agent_report, compare_reports, render_comparison)
import tokeymeter.engines.economics.savings as sv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _caller():
    @tokeymeter.cache(model="m")
    def call(prompt, tokens, resp):
        set_reported_usage(tokens, 300)
        return resp
    return call


def _healthy(agent="support", n=20, calls=9, prefix="ok"):
    call = _caller()
    for i in range(n):
        h = []
        with tokeymeter.task(f"{prefix}-{i}", agent=agent):
            for j in range(calls):
                h.append(f"s{j}")
                call(tuple(h) + (prefix, i), 900 + j * 330, f"found {i}-{j}")


def _stalled(agent="support", n=4, prefix="bad"):
    call = _caller()
    for i in range(n):
        h = []
        with tokeymeter.task(f"{prefix}-{i}", agent=agent):
            for j in range(14):
                h.append(f"r{j}")
                call(tuple(h) + (prefix, i, "x"), 1100 + j * 420, "tool error")


def _batch(agent="extract", n=12):
    call = _caller()
    for i in range(n):
        with tokeymeter.task(f"doc-{i}", agent=agent):
            for j in range(8):
                call(f"doc{i}-{j}", 1400, ["APPROVED", "DENIED"][j % 2])


def _records():
    return list(sv._tracker._iter_records())


def _agent(rep, name):
    return next(a for a in rep["agents"] if a["agent"] == name)


# ── the recommendation is backtested ────────────────────────────────────

def test_a_recommendation_carries_both_counts():
    """A suggestion without a false-positive count is a guess wearing a
    number's clothes."""
    _healthy()
    _stalled()
    e = _agent(suggest_thresholds(_records()), "support")
    r = e["recommendations"]["stall_window"]
    assert r["confidence"] == "backtested"
    assert r["caught"] >= 1
    assert r["false_positives"] == 0
    assert "backtested against" in r["basis"]


def test_the_recommended_window_catches_every_stalled_task():
    _healthy()
    _stalled(n=4)
    e = _agent(suggest_thresholds(_records()), "support")
    assert e["stuck_tasks"] == 4
    assert e["recommendations"]["stall_window"]["caught"] == 4
    assert e["recommendations"]["stall_window"]["missed"] == 0


def test_no_candidate_with_a_false_positive_is_ever_chosen():
    """Hard requirement: a ceiling that kills working tasks gets switched off,
    and then it protects nothing."""
    _healthy()
    _stalled()
    e = _agent(suggest_thresholds(_records()), "support")
    for rec in e["recommendations"].values():
        if rec.get("confidence") == "backtested":
            assert rec["false_positives"] == 0


def test_every_candidate_is_reported_not_just_the_winner():
    """A user must be able to see the working, not just the answer."""
    _healthy()
    _stalled()
    e = _agent(suggest_thresholds(_records()), "support")
    assert len(e["stall_window_candidates"]) >= 4
    for c in e["stall_window_candidates"]:
        assert {"value", "caught", "missed", "false_positives"} <= set(c)


# ── refusing to answer ──────────────────────────────────────────────────

def test_too_little_history_is_refused_with_a_reason():
    """A threshold fitted to nine tasks is noise, and shipping it would be
    worse than saying nothing."""
    _healthy(n=5, prefix="few")
    e = _agent(suggest_thresholds(_records()), "support")
    assert "stall_window" not in e["recommendations"]
    assert any("too few" in n for n in e["notes"])


def test_no_stalls_means_nothing_to_tune_against():
    _healthy(n=20)
    e = _agent(suggest_thresholds(_records()), "support")
    assert e["stuck_tasks"] == 0
    assert "stall_window" not in e["recommendations"]
    assert any("nothing to tune against" in n for n in e["notes"])


def test_batch_work_is_recognised_and_stall_detection_declined():
    """Low novelty with FLAT input is a classifier doing its job. Recommending
    stall detection here would fight the workload every time."""
    _batch()
    e = _agent(suggest_thresholds(_records()), "extract")
    assert e["shape"] == "batch"
    assert "stall_window" not in e["recommendations"]
    assert "max_calls" in e["recommendations"]
    assert any("batch work" in n for n in e["notes"])


def test_an_unbacktested_value_is_labelled_as_such():
    _batch()
    e = _agent(suggest_thresholds(_records()), "extract")
    r = e["recommendations"]["max_calls"]
    assert r["confidence"] != "backtested"
    assert r["caught"] is None


def test_an_empty_ledger_guides_rather_than_guessing():
    rep = suggest_thresholds([])
    assert rep["agents"] == []
    assert "wrap an agent entry point" in render_suggestions(rep).lower()


# ── mixed estates ───────────────────────────────────────────────────────

def test_each_agent_is_judged_on_its_own_shape():
    """The whole point: one estate, different workloads, different advice."""
    _healthy(agent="support", n=20)
    _stalled(agent="support", n=4)
    _batch(agent="extract")
    _healthy(agent="research", n=5, prefix="r")
    rep = suggest_thresholds(_records())
    assert "stall_window" in _agent(rep, "support")["recommendations"]
    assert _agent(rep, "extract")["shape"] == "batch"
    assert _agent(rep, "research")["recommendations"] == {}


def test_the_render_is_ascii_safe():
    _healthy()
    _stalled()
    _batch()
    text = render_suggestions(suggest_thresholds(_records()))
    assert all(ord(c) < 128 for c in text)


def test_suggestions_change_nothing():
    _healthy()
    _stalled()
    before = len(_records())
    suggest_thresholds(_records())
    assert len(_records()) == before


def test_corrupt_records_do_not_break_the_analysis():
    _healthy()
    _stalled()
    dirty = _records() + [
        {"task_id": "x", "agent": "support", "estimated_cost": float("nan"),
         "hit": False},
        {"task_id": "x", "agent": "support", "estimated_cost": float("inf"),
         "hit": False},
        {"task_id": "x", "agent": "support", "estimated_cost": -1.0,
         "hit": False},
    ]
    rep = suggest_thresholds(dirty)
    assert _agent(rep, "support")["recommendations"]


# ── drift ───────────────────────────────────────────────────────────────

def _report(calls, stalled, cost=0.05, progress=1.0):
    return {"agents": [{"agent": "extract",
                        "median_calls_per_task": calls,
                        "median_cost_per_task_usd": cost,
                        "p95_cost_per_task_usd": cost * 1.2,
                        "median_progress": progress,
                        "stalled_tasks": stalled}]}


def test_a_regression_is_flagged():
    cmp = compare_reports(_report(11.2, 2, 0.081, 0.88),
                          _report(6.4, 0, 0.046, 1.00))
    assert cmp["regressed"] == 1
    row = cmp["agents"][0]
    assert row["metrics"]["calls/task"]["worse"] is True
    assert row["stalled"]["worse"] is True
    assert "regressed" in render_comparison(cmp)


def test_an_improvement_is_not_flagged():
    cmp = compare_reports(_report(6.4, 0, 0.046, 1.00),
                          _report(11.2, 2, 0.081, 0.88))
    assert cmp["regressed"] == 0
    assert "regressed" not in render_comparison(cmp).replace(
        "0 agents regressed", "")


def test_falling_progress_counts_as_worse_even_though_the_number_dropped():
    """Direction matters per metric: more calls is worse, less progress is
    worse."""
    cmp = compare_reports(_report(6.4, 0, 0.046, 0.5),
                          _report(6.4, 0, 0.046, 1.0))
    assert cmp["agents"][0]["metrics"]["progress"]["worse"] is True


def test_a_new_agent_has_nothing_to_drift_from():
    cur = {"agents": [{"agent": "brand-new", "median_calls_per_task": 5,
                       "stalled_tasks": 0}]}
    cmp = compare_reports(cur, {"agents": []})
    assert cmp["agents"][0]["new"] is True
    assert cmp["regressed"] == 0
    assert "NEW in this window" in render_comparison(cmp)


def test_an_agent_that_stopped_running_has_not_improved():
    prev = {"agents": [{"agent": "retired", "median_calls_per_task": 5,
                        "stalled_tasks": 0}]}
    cmp = compare_reports({"agents": []}, prev)
    assert cmp["agents"][0]["gone"] is True
    assert cmp["regressed"] == 0


def test_a_change_from_zero_is_undefined_not_infinite():
    cmp = compare_reports(_report(5, 0, 0.05), _report(0, 0, 0.05))
    assert cmp["agents"][0]["metrics"]["calls/task"]["change_pct"] is None


def test_comparison_render_is_ascii_safe():
    cmp = compare_reports(_report(11.2, 2, 0.081, 0.88),
                          _report(6.4, 0, 0.046, 1.00))
    assert all(ord(c) < 128 for c in render_comparison(cmp))


# ── the CLI ─────────────────────────────────────────────────────────────

def _cli(args, home, timeout=120):
    env = dict(os.environ, TOKEYMETER_HOME=home, PYTHONPATH=REPO_ROOT)
    p = subprocess.run([sys.executable, "-m", "tokeymeter", "agents", *args],
                       capture_output=True, env=env, timeout=timeout)
    return (p.returncode,
            (p.stdout or b"").decode("utf-8", "replace")
            + (p.stderr or b"").decode("utf-8", "replace"))


@pytest.fixture
def two_weeks():
    """A ledger spanning two windows, with a regression in the newer one."""
    home = tempfile.mkdtemp()
    led = os.path.join(home, "savings.jsonl")
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.set_savings_path(led)
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    call = _caller()

    def week(prefix, per_task, stuck):
        for i in range(12):
            h = []
            with tokeymeter.task(f"{prefix}-ok-{i}", agent="extract"):
                for j in range(per_task):
                    h.append(f"s{j}")
                    call(tuple(h) + (prefix, i), 900 + j * 300, f"{prefix} {i}-{j}")
        for i in range(stuck):
            h = []
            with tokeymeter.task(f"{prefix}-bad-{i}", agent="extract"):
                for j in range(14):
                    h.append(f"r{j}")
                    call(tuple(h) + (prefix, i, "x"), 1100 + j * 420, "tool error")

    week("w1", 6, 0)
    week("w2", 11, 2)
    now = time.time()
    recs = [json.loads(l) for l in open(led, encoding="utf-8") if l.strip()]
    with open(led, "w", encoding="utf-8") as f:
        for r in recs:
            old = str(r.get("task_id", "")).startswith("w1")
            r["timestamp"] = now - (10 * 86400 if old else 3 * 86400)
            f.write(json.dumps(r) + "\n")
    yield home
    tokeymeter.set_in_memory_savings(True)


def test_cli_suggest_runs_and_shows_the_counts(two_weeks):
    code, out = _cli(["--suggest"], two_weeks)
    assert code == 0
    assert "would have caught" in out and "false positive" in out


def test_cli_compare_detects_the_regression(two_weeks):
    code, out = _cli(["--since", "7d", "--compare"], two_weeks)
    assert code == 0
    assert "regressed" in out and "calls/task" in out


def test_cli_compare_requires_a_window(two_weeks):
    code, out = _cli(["--compare"], two_weeks)
    assert code == 1 and "--since" in out


def test_cli_rejects_an_unknown_flag(two_weeks):
    code, out = _cli(["--suggest", "--wat"], two_weeks)
    assert code == 1 and "unrecognized" in out


def test_cli_rejects_a_bad_window(two_weeks):
    code, out = _cli(["--since", "30x"], two_weeks)
    assert code == 1


def test_cli_suggest_json_is_machine_readable(two_weeks):
    code, out = _cli(["--suggest", "--json"], two_weeks)
    assert code == 0
    parsed = json.loads(out)
    assert parsed["report"] == "suggestions"
    assert parsed["agents"]


def test_cli_survives_a_legacy_console(two_weeks):
    env = dict(os.environ, TOKEYMETER_HOME=two_weeks, PYTHONPATH=REPO_ROOT,
               PYTHONIOENCODING="cp1252")
    for args in (["--suggest"], ["--since", "7d", "--compare"]):
        p = subprocess.run([sys.executable, "-m", "tokeymeter", "agents", *args],
                           capture_output=True, env=env, timeout=120)
        assert p.returncode == 0


# ── the coding-agent guide ──────────────────────────────────────────────

def test_agents_md_ships_and_states_the_one_rule_that_matters():
    """A coding agent following this must not wrap individual model calls —
    that is the mistake that makes the whole signal meaningless."""
    path = os.path.join(REPO_ROOT, "AGENTS.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "AGENT ENTRY POINT" in text
    assert "Never wrap individual model calls" in text


def test_agents_md_forbids_enforcing_on_a_guess():
    text = open(os.path.join(REPO_ROOT, "AGENTS.md"), encoding="utf-8").read()
    assert "Do not add `enforce=True` in the first change" in text
    assert "--suggest" in text


def test_agents_md_states_the_anti_segment():
    """Honesty filters out people who would install, get nothing, and say so."""
    text = open(os.path.join(REPO_ROOT, "AGENTS.md"), encoding="utf-8").read()
    assert "single-shot" in text
    assert "Claude Code" in text


def test_agents_md_commands_all_exist():
    text = open(os.path.join(REPO_ROOT, "AGENTS.md"), encoding="utf-8").read()
    import re
    cmds = set(re.findall(r"python -m tokeymeter (\w+)", text))
    assert cmds
    for c in cmds:
        env = dict(os.environ, PYTHONPATH=REPO_ROOT,
                   TOKEYMETER_HOME=tempfile.mkdtemp())
        p = subprocess.run([sys.executable, "-m", "tokeymeter", c, "--help"],
                           capture_output=True, env=env, timeout=120)
        assert p.returncode == 0, f"AGENTS.md references `{c}` which fails"


# ── the claim must survive live enforcement ─────────────────────────────

def test_the_recommended_window_behaves_as_predicted_when_enforced():
    """The whole promise: "would have caught 4, 0 false positives" has to be
    true when the value is actually applied. A backtest that does not predict
    enforcement is a guess with extra steps."""
    _healthy(agent="d", n=20, calls=9)
    _stalled(agent="d", n=4)
    rec = _agent(suggest_thresholds(_records()), "d")["recommendations"]["stall_window"]
    window = rec["value"]

    # replay the same workload with the recommended value enforced
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset_savings()
    call = _caller()
    healthy_halted = stuck_halted = 0
    for i in range(20):
        h = []
        try:
            with tokeymeter.task(f"ok-{i}", agent="d",
                                 stall_window=window, enforce=True):
                for j in range(9):
                    h.append(f"s{j}")
                    call(tuple(h) + ("ok", i), 900 + j * 330, f"found {i}-{j}")
        except tokeymeter.TaskStalled:
            healthy_halted += 1
    for i in range(4):
        h = []
        try:
            with tokeymeter.task(f"bad-{i}", agent="d",
                                 stall_window=window, enforce=True):
                for j in range(14):
                    h.append(f"r{j}")
                    call(tuple(h) + ("bad", i, "x"), 1100 + j * 420, "tool error")
        except tokeymeter.TaskStalled:
            stuck_halted += 1

    assert stuck_halted == rec["caught"]
    assert healthy_halted == rec["false_positives"] == 0


def test_an_estate_with_no_healthy_baseline_still_recommends():
    """Every task stalled. There is no healthy work to protect, so a
    recommendation is safe — and withholding one would be unhelpful."""
    _stalled(agent="a", n=15)
    e = _agent(suggest_thresholds(_records()), "a")
    assert e["stuck_tasks"] == 15
    assert e["recommendations"]["stall_window"]["caught"] == 15


def test_long_healthy_tasks_are_not_mistaken_for_stalls():
    """A 30-call task that keeps producing new answers is working, however long
    it runs. Length alone must never trigger advice to halt it."""
    call = _caller()
    for i in range(20):
        h = []
        with tokeymeter.task(f"long-{i}", agent="b"):
            for j in range(30):
                h.append(f"s{j}")
                call(tuple(h) + (i,), 900 + j * 300, f"new {i}-{j}")
    e = _agent(suggest_thresholds(_records()), "b")
    assert e["stuck_tasks"] == 0
    assert "stall_window" not in e["recommendations"]


def test_unmeasurable_responses_produce_no_stall_advice():
    """If responses could not be scored there is no progress signal, and
    recommending a stall window would be advice with nothing behind it."""
    @tokeymeter.cache(model="m")
    def opaque(prompt, tokens):
        set_reported_usage(tokens, 300)

        class Reply:                      # no text form: not scoreable
            pass
        return Reply()

    for i in range(15):
        h = []
        with tokeymeter.task(f"o-{i}", agent="c"):
            for j in range(10):
                h.append(f"s{j}")
                opaque(tuple(h) + (i,), 900 + j * 300)
    e = _agent(suggest_thresholds(_records()), "c")
    assert e["tasks_scored"] == 0
    assert "stall_window" not in e["recommendations"]


# ── the tie-break, decided by holdout rather than by intuition ───────────

def test_the_tie_break_goes_to_the_tightest_survivor():
    """Measured, not assumed. Holdout testing on unseen traffic showed the
    LOOSER choice catching 28 of 40 stalled tasks where the tighter caught
    40 of 40, with no false positives either way. Safety is already provided by
    the zero-false-positive filter, not by looseness."""
    _healthy(agent="d", n=25)
    _stalled(agent="d", n=5)
    e = _agent(suggest_thresholds(_records()), "d")
    chosen = e["recommendations"]["stall_window"]["value"]
    clean = [c for c in e["stall_window_candidates"]
             if c["false_positives"] == 0 and c["caught"] > 0]
    best = max(c["caught"] for c in clean)
    tied = [c["value"] for c in clean if c["caught"] == best]
    assert chosen == min(tied)


def test_legitimate_repetition_excludes_the_tight_windows():
    """An agent that retries an upstream several times and then recovers is not
    stalled. Windows at or below that run must fail the backtest, which is what
    makes the tightest survivor safe to choose."""
    call = _caller()
    for i in range(25):
        h = []
        with tokeymeter.task(f"ok-{i}", agent="r"):
            for j in range(18):
                h.append(f"s{j}")
                resp = "retrying upstream" if 3 <= j < 10 else f"new {i}-{j}"
                call(tuple(h) + (i,), 800 + j * 350, resp)
    _stalled(agent="r", n=5)
    e = _agent(suggest_thresholds(_records()), "r")
    survivors = [c["value"] for c in e["stall_window_candidates"]
                 if c["false_positives"] == 0 and c["caught"] > 0]
    assert 4 not in survivors and 6 not in survivors
    assert e["recommendations"]["stall_window"]["value"] >= 12


def test_headroom_is_reported_so_the_choice_is_inspectable():
    """"How much room before a legitimate repeat trips this" must be a number,
    not a feeling."""
    _healthy(agent="d", n=25)
    _stalled(agent="d", n=5)
    r = _agent(suggest_thresholds(_records()),
               "d")["recommendations"]["stall_window"]
    assert r["headroom"] == r["value"] - r["longest_healthy_repeat_run"]
    assert r["headroom"] > 0


def test_a_fitted_threshold_generalises_to_unseen_traffic():
    """The property that makes a backtest worth anything: fit on one workload,
    verify on a differently-seeded one."""
    import random

    def workload(seed, window=None, enforce=False):
        rng = random.Random(seed)
        tokeymeter.set_default_store(MemoryStore())
        tokeymeter.reset_savings()
        call = _caller()
        fp = tp = 0
        kw = ({"stall_window": window, "enforce": True} if enforce else {})
        for i in range(20):
            h = []
            try:
                with tokeymeter.task(f"ok-{i}", agent="g", **kw):
                    for j in range(rng.randint(6, 16)):
                        h.append(f"s{j}")
                        call(tuple(h) + (seed, i), 800 + j * rng.randint(200, 450),
                             f"new {i}-{j}")
            except tokeymeter.TaskStalled:
                fp += 1
        for i in range(5):
            h = []
            try:
                with tokeymeter.task(f"bad-{i}", agent="g", **kw):
                    for j in range(rng.randint(9, 20)):
                        h.append(f"r{j}")
                        call(tuple(h) + (seed, i, "x"),
                             1000 + j * rng.randint(300, 500), "tool error")
            except tokeymeter.TaskStalled:
                tp += 1
        return tp, fp

    workload(11)                                   # fit
    rec = _agent(suggest_thresholds(_records()),
                 "g")["recommendations"]["stall_window"]
    tp, fp = workload(9911, window=rec["value"], enforce=True)   # unseen
    assert fp == 0                                 # generalises safely
    assert tp >= 4                                 # and still catches


# ── one sequence, three components ──────────────────────────────────────

def _sequence_fixture():
    """A stuck task and healthy ones, as raw records with timestamps."""
    recs = []
    for j in range(14):
        recs.append({"task_id": "stuck", "agent": "a", "hit": False,
                     "estimated_cost": 0.01, "input_tokens": 1000 + j * 400,
                     "response_fingerprint": "same",
                     "prompt_fingerprint": f"p{j}", "timestamp": 1000.0 + j})
    for i in range(15):
        for j in range(10):
            recs.append({"task_id": f"ok{i}", "agent": "a", "hit": False,
                         "estimated_cost": 0.01, "input_tokens": 900 + j * 300,
                         "response_fingerprint": f"n{i}-{j}",
                         "prompt_fingerprint": f"q{i}-{j}",
                         "timestamp": 1000.0 + j})
    return recs


def _verdicts(recs):
    from tokeymeter.engines.governance.simulate import simulate_rules
    s = suggest_thresholds(recs)["agents"][0]["recommendations"] \
        .get("stall_window", {}).get("value")
    a = agent_report(recs)["agents"][0]["stalled_tasks"]
    p = simulate_rules(R_load([{"name": "x", "then": {"stall_window": 4}}]),
                       recs)["tasks_would_halt"]
    return s, a, p


def R_load(rules):
    from tokeymeter.engines.governance import rules as R
    return R.load_rules(rules)


def test_advice_does_not_depend_on_record_order():
    """The progress signal reads the FIRST half of a task against the SECOND,
    so a task whose records arrived interleaved — several processes appending
    to one ledger, or two ledgers concatenated — would be judged on a sequence
    that never happened. Verified broken before this test existed: shuffling
    turned a valid recommendation into no recommendation at all."""
    import random
    recs = _sequence_fixture()
    ordered = _verdicts(recs)
    shuffled = list(recs)
    random.Random(1).shuffle(shuffled)
    assert _verdicts(shuffled) == ordered
    assert _verdicts(list(reversed(recs))) == ordered


def test_the_report_card_is_also_order_independent():
    import random
    recs = _sequence_fixture()
    before = agent_report(recs)["agents"][0]["stalled_tasks"]
    shuffled = list(recs)
    random.Random(9).shuffle(shuffled)
    assert agent_report(shuffled)["agents"][0]["stalled_tasks"] == before


def test_suggest_report_and_simulate_answer_from_one_sequence():
    """If they could drift, the number a user reads would not be the number
    that stops their agent."""
    recs = _sequence_fixture()
    s, a, p = _verdicts(recs)
    assert s is not None
    assert a == p == 1
