"""Progress signal (S4-4) — stall detection and the agent report card.

ONE PRIMITIVE, TWO SURFACES. The stall that halts a task at 3am and the number
a platform owner reads on Monday are computed the same way, so the report can
never disagree with the enforcement.

Pinned here, because each was forced by a real failure:

  THE PROMPT CANNOT TELL YOU AN AGENT IS STUCK. A real agent carries its
    conversation history, so every prompt hash is unique even on the thirtieth
    identical failure. Our shipped max_repeats missed that workload entirely.
  THE RESPONSE CAN. Progress is novelty in the OUTPUT.
  NOVELTY ALONE IS A FALSE-POSITIVE MACHINE. A classifier returning "APPROVED"
    for 500 documents scores near zero and is working perfectly. What separates
    it from a stall is that its input does not GROW.
  CACHE HITS ARE EXCLUDED. A hit returns a byte-identical response by
    definition; counting them would make every well-cached workload look
    stalled.
"""
import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution import task as T
from tokeymeter.engines.governance import rules as R
from tokeymeter.engines.governance.agents import agent_report, render_agents
import tokeymeter.engines.economics.savings as sv


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    R.clear_rules()
    yield
    R.clear_rules()
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _caller():
    """A model call whose input size and response are both controllable —
    the two axes the whole signal rests on."""
    @tokeymeter.cache(model="m")
    def call(prompt, tokens=1200, resp="ok"):
        set_reported_usage(tokens, 200)
        return resp
    return call


def _records():
    return list(sv._tracker._iter_records())


# ── the response fingerprint ────────────────────────────────────────────

def test_same_response_different_prompts_shares_a_fingerprint():
    """The whole basis of the signal."""
    call = _caller()
    with tokeymeter.task("t", agent="a"):
        call("alpha", resp="identical")
        call("beta", resp="identical")
    recs = _records()
    assert len({r["prompt_fingerprint"] for r in recs}) == 2
    assert len({r["response_fingerprint"] for r in recs}) == 1


def test_response_fingerprint_is_a_non_reversible_digest():
    fp = T.fingerprint_response("a response body")
    assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 12
    assert "response" not in fp


@pytest.mark.parametrize("value", [None, "", b""])
def test_empty_responses_yield_no_fingerprint(value):
    assert T.fingerprint_response(value) is None


def test_unhashable_response_never_breaks_a_call():
    class Exploding:
        def __str__(self):
            raise RuntimeError("boom")
    assert T.fingerprint_response(Exploding()) is None


def test_huge_response_is_hashed_from_a_bounded_prefix():
    """An unbounded hash would put an unbounded cost on the hot path."""
    fp = T.fingerprint_response("x" * 5_000_000)
    assert fp is not None


def test_bytes_and_chunk_lists_are_scoreable():
    """S4-5 narrowed this deliberately: only shapes we KNOW are content get
    scored. A dict is refused because it may carry a per-call request id, which
    would fabricate novelty for a stuck agent. See test_progress_scoreability."""
    assert T.fingerprint_response(b"raw bytes") is not None
    assert T.fingerprint_response(["chunk one", "chunk two"]) is not None
    assert T.fingerprint_response({"structured": "response"}) is None


# ── stall detection: the three workload shapes ──────────────────────────

def test_stuck_conversational_agent_is_halted():
    """The workload our shipped max_repeats missed completely: history grows,
    so every prompt hash is unique, but the answer never changes."""
    call = _caller()
    n = {"c": 0}
    history = []
    with pytest.raises(tokeymeter.TaskStalled) as ei:
        with tokeymeter.task("stuck", stall_window=8, enforce=True):
            for i in range(40):
                history.append(f"retry {i}")
                n["c"] += 1
                call(tuple(history), tokens=1000 + n["c"] * 400,
                     resp="tool error: cannot parse PDF")
    assert n["c"] < 40                     # stopped well before the end
    assert ei.value.limit == "stalled"


def test_max_repeats_alone_would_have_missed_it():
    """Proof the new signal was necessary, not additive."""
    call = _caller()
    history = []
    with tokeymeter.task("stuck", max_repeats=4, enforce=True) as st:
        for i in range(30):
            history.append(f"retry {i}")
            call(tuple(history), tokens=1000 + i * 400, resp="same error")
    assert st.snapshot()["halted_reason"] is None      # never fired
    assert len({r["prompt_fingerprint"] for r in _records()}) == 30


def test_batch_classifier_is_not_a_stall():
    """500 documents that all return APPROVED score near zero progress and are
    working perfectly. Flat input is what makes it legitimate."""
    call = _caller()
    with tokeymeter.task("batch", stall_window=8, enforce=True) as st:
        for i in range(60):
            call(f"document-{i}", tokens=1500, resp="APPROVED")
    snap = st.snapshot()
    assert snap["stalled"] is False
    assert snap["progress_novelty"] is not None and snap["progress_novelty"] < 0.3


def test_working_conversational_agent_is_not_a_stall():
    call = _caller()
    history = []
    with tokeymeter.task("working", stall_window=8, enforce=True) as st:
        for i in range(40):
            history.append(f"step {i}")
            call(tuple(history), tokens=1000 + i * 400, resp=f"extracted {i}")
    assert st.snapshot()["stalled"] is False
    assert st.snapshot()["progress_novelty"] == 1.0


def test_growth_alone_is_not_a_stall():
    """Every healthy conversational agent grows its context. Growth without a
    novelty collapse must never halt anything."""
    call = _caller()
    history = []
    with tokeymeter.task("t", stall_window=8, enforce=True) as st:
        for i in range(30):
            history.append(f"s{i}")
            call(tuple(history), tokens=500 + i * 900, resp=f"unique {i}")
    assert st.snapshot()["stalled"] is False


def test_observe_mode_records_a_stall_without_raising():
    call = _caller()
    history = []
    with tokeymeter.task("t", stall_window=8) as st:      # enforce defaults False
        for i in range(30):
            history.append(f"r{i}")
            call(tuple(history), tokens=1000 + i * 400, resp="same error")
    assert st.snapshot()["stalled"] is True
    assert st.snapshot()["halted_reason"] == "stalled"


def test_no_stall_window_means_no_detection():
    """Opt-in: a task that did not ask for stall detection never gets it."""
    call = _caller()
    history = []
    with tokeymeter.task("t", enforce=True) as st:
        for i in range(30):
            history.append(f"r{i}")
            call(tuple(history), tokens=1000 + i * 400, resp="same error")
    assert st.snapshot()["stalled"] is False


def test_a_short_task_is_never_judged():
    """Two calls that happen to agree are noise, not evidence."""
    call = _caller()
    with tokeymeter.task("t", stall_window=8, enforce=True) as st:
        for i in range(3):
            call(f"p{i}", tokens=1000 + i * 500, resp="same")
    assert st.snapshot()["stalled"] is False


def test_cache_hits_do_not_count_toward_progress():
    """A hit returns a byte-identical response by definition. Counting hits
    would make every well-cached workload look stalled."""
    call = _caller()
    with tokeymeter.task("t", stall_window=4, enforce=True) as st:
        for _ in range(30):
            call("identical prompt", tokens=1200, resp="identical")   # 1 miss, 29 hits
    assert st.snapshot()["stalled"] is False


def test_stall_can_be_expressed_as_policy():
    """The platform owner sets it centrally, not the developer in code."""
    R.set_rules(R.load_rules([{"name": "stall-guard",
                               "then": {"stall_window": 8, "enforce": True}}]))
    call = _caller()
    history = []
    with pytest.raises(tokeymeter.TaskStalled):
        with tokeymeter.task("t", agent="support"):
            for i in range(40):
                history.append(f"r{i}")
                call(tuple(history), tokens=1000 + i * 400, resp="same error")


def test_stalled_subclasses_the_common_base():
    assert issubclass(tokeymeter.TaskStalled, tokeymeter.TaskLimitExceeded)


# ── the report card ─────────────────────────────────────────────────────

def _estate():
    call = _caller()
    for i in range(9):                       # support: 1 in 9 gets stuck
        h = []
        with tokeymeter.task(f"ticket-{i}", agent="support"):
            stuck = i == 0
            for j in range(10):
                h.append(f"turn {j}")
                call(tuple(h) + (i,), tokens=900 + j * 350,
                     resp="tool error" if stuck else f"found {i}-{j}")
    for i in range(6):                       # extract: legitimate batch work
        with tokeymeter.task(f"doc-{i}", agent="extract"):
            for j in range(8):
                call(f"classify {i}-{j}", tokens=1500, resp="APPROVED")
    return _records()


def test_report_separates_a_stall_from_legitimate_repetition():
    """The single most important property: extract scores low progress and must
    NOT be flagged; support scores high and must still surface its one stall."""
    rep = agent_report(_estate())
    by = {r["agent"]: r for r in rep["agents"]}
    assert by["extract"]["median_progress"] < 0.3
    assert by["extract"]["stalled_tasks"] == 0        # flat input: fine
    assert by["support"]["median_progress"] == 1.0
    assert by["support"]["stalled_tasks"] == 1        # growing input: stall


def test_report_names_the_stalled_task():
    rep = agent_report(_estate())
    support = next(r for r in rep["agents"] if r["agent"] == "support")
    assert support["worst_stalls"][0]["task_id"] == "ticket-0"
    assert support["worst_stalls"][0]["input_growth"] > 0


def test_report_gives_cost_per_task_and_the_tail():
    rep = agent_report(_estate())
    for r in rep["agents"]:
        assert r["median_cost_per_task_usd"] > 0
        assert r["p95_cost_per_task_usd"] >= r["median_cost_per_task_usd"]
        assert r["median_calls_per_task"] > 0


def test_report_surfaces_untasked_spend():
    call = _caller()
    _estate()
    for i in range(5):
        call(f"outside-any-task-{i}")
    rep = agent_report(_records())
    assert rep["untasked_spend_usd"] > 0
    assert "belongs to no task" in render_agents(rep)


def test_report_excludes_corrupt_records_and_counts_them():
    recs = list(_estate()) + [
        {"task_id": "x", "agent": "a", "estimated_cost": float("nan"), "hit": False},
        {"task_id": "x", "agent": "a", "estimated_cost": float("inf"), "hit": False},
        {"task_id": "x", "agent": "a", "estimated_cost": -1.0, "hit": False},
    ]
    assert agent_report(recs)["excluded_malformed_records"] == 3


def test_report_on_an_empty_ledger_tells_you_what_to_do():
    """It must name BOTH steps. The old message said only "wrap an agent entry
    point" — which is what a reader had just done, so they were sent in a
    circle with an empty ledger and no idea why. Wrapping the CLIENT is what
    puts calls on the ledger at all; the task boundary only groups them."""
    rep = agent_report([])
    assert rep["agents"] == []
    text = render_agents(rep).lower()
    assert "wrap your client" in text          # the step that was missing
    assert "tokeymeter.task" in text           # and the one that was there
    assert "firstrun" in text                  # and a way out if stuck


def test_report_handles_tasks_with_no_agent():
    call = _caller()
    with tokeymeter.task("no-agent-here"):
        for i in range(5):
            call(f"p{i}")
    rep = agent_report(_records())
    assert any(r["agent"] == "(unnamed)" for r in rep["agents"])


def test_report_render_is_ascii_only():
    text = render_agents(agent_report(_estate()))
    assert all(ord(c) < 128 for c in text)


def test_report_is_deterministic():
    recs = _estate()
    a, b = agent_report(recs), agent_report(recs)
    assert a["agents"] == b["agents"]
    assert render_agents(a) == render_agents(b)


def test_report_changes_nothing():
    recs = _estate()
    before = len(_records())
    agent_report(recs)
    assert len(_records()) == before


def test_report_and_enforcement_agree():
    """One primitive, two surfaces: a task the detector halts in flight must be
    the task the report flags afterwards. If these ever disagree, the number a
    platform owner reads is not the number that stopped their agent."""
    call = _caller()
    history = []
    with tokeymeter.task("agreed", agent="support", stall_window=8) as st:
        for i in range(30):
            history.append(f"r{i}")
            call(tuple(history), tokens=1000 + i * 400, resp="same error")
    assert st.snapshot()["stalled"] is True
    rep = agent_report(_records())
    support = next(r for r in rep["agents"] if r["agent"] == "support")
    assert "agreed" in {s["task_id"] for s in support["worst_stalls"]}


# ── the two defects the highest-order audit found ───────────────────────

def test_simulation_models_stall_detection():
    """`plan` was written before stall detection existed and never learned it.
    A platform owner would have run a stall policy through plan, seen "0 tasks
    would halt", applied it, and watched tasks halt in production. A plan that
    lies is worse than no plan."""
    from tokeymeter.engines.governance.simulate import simulate_rules
    call = _caller()
    history = []
    for i in range(6):
        with tokeymeter.task(f"t-{i}", agent="a"):
            for j in range(12):
                history.append(f"x{j}")
                call(tuple(history) + (i,), tokens=1000 + j * 400,
                     resp="same error")
    recs = _records()
    rules = R.load_rules([{"name": "s", "then": {"stall_window": 8}}])
    sim = simulate_rules(rules, recs)
    rep = agent_report(recs)
    assert sim["tasks_would_halt"] == rep["agents"][0]["stalled_tasks"] == 6


def test_simulation_agrees_with_live_enforcement_on_stalls():
    """Guards the mirror in simulate._stalled against drifting from the
    enforcement rule in _TaskState._stall_check_locked. If either changes
    without the other, this fails."""
    from tokeymeter.engines.governance.simulate import simulate_rules
    rules = R.load_rules([{"name": "s",
                           "then": {"stall_window": 8, "enforce": True}}])

    def run(live):
        tokeymeter.set_default_store(MemoryStore())
        tokeymeter.reset_savings()
        R.set_rules(live)
        call = _caller()
        halted = set()
        for i in range(12):
            history = []
            try:
                with tokeymeter.task(f"t-{i}", agent="a"):
                    for j in range(12):
                        history.append(f"x{j}")
                        if i % 3 == 0:          # stuck
                            call(tuple(history) + (i,), tokens=1000 + j * 400,
                                 resp="stuck")
                        elif i % 3 == 1:        # batch, flat input
                            call(f"doc-{i}-{j}", tokens=1400, resp="APPROVED")
                        else:                   # working
                            call(tuple(history) + (i, "w"),
                                 tokens=1000 + j * 400, resp=f"new {i}-{j}")
            except tokeymeter.TaskLimitExceeded:
                halted.add(f"t-{i}")
        return _records(), halted

    ungoverned, _ = run(None)
    predicted = {h["task_id"]
                 for h in simulate_rules(rules, ungoverned)["halted_tasks"]}
    _, actual = run(rules)
    R.clear_rules()
    assert predicted == actual


def test_progress_rolls_up_to_a_parent_task():
    """Spend rolls up to ancestors; progress must too, or a sub-agent can spin
    forever under a parent that has stall detection enabled — the ceiling would
    be only as deep as the innermost `with`."""
    call = _caller()
    calls = {"n": 0}
    with tokeymeter.task("outer", stall_window=8, enforce=True) as outer:
        with pytest.raises(tokeymeter.TaskLimitExceeded) as ei:
            with tokeymeter.task("inner", enforce=True):
                for i in range(30):
                    calls["n"] += 1
                    call(f"n{i}", tokens=1000 + i * 400, resp="stuck")
    assert ei.value.task_id == "outer"      # the OUTER stall stopped it
    assert calls["n"] < 30
    assert outer.snapshot()["stalled"] is True


def test_a_working_sub_agent_is_not_halted_by_the_parent():
    call = _caller()
    with tokeymeter.task("outer", stall_window=8, enforce=True) as outer:
        with tokeymeter.task("inner", enforce=True):
            for i in range(30):
                call(f"m{i}", tokens=1000 + i * 400, resp=f"new {i}")
    assert outer.snapshot()["stalled"] is False


def test_progress_sampling_is_bounded_in_a_long_task():
    call = _caller()
    with tokeymeter.task("long", stall_window=8) as st:
        for i in range(5_000):
            call(f"u{i}", tokens=1000, resp=f"r{i}")
    assert len(st._recent) == 8


def test_a_task_boundary_alone_records_nothing_and_the_guidance_says_so():
    """The dead end this guidance exists to prevent: a reader who wraps only
    the entry point, with an unwrapped client, records ZERO calls — and the
    empty report must not send them back to the step they already did."""
    def their_agent(ticket):          # a plain, unwrapped client
        return "did some work"

    with tokeymeter.task("ticket-1", agent="support"):
        their_agent("t1")

    assert _records() == []
    text = render_agents(agent_report()).lower()
    assert "wrap your client" in text


def test_the_empty_report_names_the_ledger_it_looked_at():
    """"Nothing was recorded" and "you are pointing at the wrong ledger" produce
    an identical empty report, and the second is far more common — a mistyped
    TOKEYMETER_HOME, or cmd.exe `set VAR=...` pasted into PowerShell where it
    silently does nothing. Naming the path turns a dead end into a diagnosis."""
    text = render_agents(agent_report([]))
    assert "looked in:" in text
    # A PATH, not one particular filename — the ledger location is
    # configurable and the gate runs with a different one.
    line = next(l for l in text.splitlines() if "looked in:" in l)
    assert len(line.split("looked in:")[1].strip()) > 3
