"""Tier 1 embedding surfaces — halt routing, the shareable report, adapters.

Pinned here, because each was a deliberate decision:

  THE HOOK FIRES WHERE THE HALT IS DECIDED, not where it is raised. Ordinary
    agent retry code catches RuntimeError and swallows the halt — measured, 22
    swallowed on one stuck task — so a hook riding on the exception would stay
    silent in exactly the case an operator needs to hear about.
  EXACTLY ONCE PER TASK. A halted task keeps being checked on every later call;
    notifying per check turns one stuck ticket into forty identical alerts, and
    a channel that cries wolf gets muted.
  RECORD-ONLY MODE STILL NOTIFIES, and the event says `enforced=False`. A team
    that has not turned enforcement on still wants to know their agent went
    nowhere — and an alert that does not distinguish "stopped" from "would have
    been stopped" is misleading.
  A HANDLER CANNOT BREAK A REQUEST. Handlers run in the caller's request path.
    One that raises is swallowed and counted; one that keeps raising is retired
    rather than left to burn latency on every halt forever.
  THE HTML IS A FILE, NOT A DASHBOARD. No server, no port, no network request
    when opened, and every interpolated value escaped — an operator who names an
    agent `<script>` gets a page that SAYS `<script>`.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance.agents import agent_report
from tokeymeter.engines.governance.agents_html import (
    render_agents_html, write_agents_html)
import tokeymeter.engines.economics.savings as sv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    tokeymeter.clear_halt_handlers()
    yield
    tokeymeter.clear_halt_handlers()
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _caller(resp="tool error: cannot parse"):
    @tokeymeter.cache(model="m")
    def call(prompt, tokens):
        set_reported_usage(tokens, 300)
        return resp
    return call


def _run_stuck(task_id="ticket-88214", agent="support", swallow=False, **kw):
    call = _caller()
    history = []
    opts = {"stall_window": 8, "enforce": True}
    opts.update(kw)
    try:
        with tokeymeter.task(task_id, agent=agent, **opts) as t:
            for i in range(30):
                history.append(f"retry {i}")
                if swallow:
                    try:
                        call(tuple(history), 1000 + i * 400)
                    except Exception:          # what real retry code does
                        pass
                else:
                    call(tuple(history), 1000 + i * 400)
    except tokeymeter.TaskLimitExceeded:
        pass
    return t


# ── the hook fires when it matters ──────────────────────────────────────

def test_a_swallowed_halt_still_notifies():
    """The requirement that shapes the whole design. TaskLimitExceeded
    subclasses RuntimeError, so `except Exception: continue` swallows it and
    nothing upstream would ever learn the agent went nowhere."""
    seen = []
    tokeymeter.on_halt(seen.append)
    t = _run_stuck(swallow=True)
    assert len(seen) == 1
    assert t.snapshot()["calls"] == 8        # bounded, despite the swallowing
    assert seen[0].task_id == "ticket-88214"
    assert seen[0].reason == "stalled"


def test_it_fires_exactly_once_however_long_the_task_runs():
    """A halted task keeps being checked. Forty alerts for one stuck ticket
    would get the channel muted."""
    seen = []
    tokeymeter.on_halt(seen.append)
    _run_stuck(swallow=True)
    assert len(seen) == 1


@pytest.mark.parametrize("kwargs,expected", [
    ({"stall_window": 8}, "stalled"),
    ({"max_repeats": 4}, "max_repeats"),
    ({"max_calls": 5}, "max_calls"),
    ({"envelope": 0.02, "reserve": 0.01}, "envelope"),
])
def test_every_halt_type_notifies(kwargs, expected):
    seen = []
    tokeymeter.on_halt(seen.append)
    call = _caller()
    history = []
    try:
        with tokeymeter.task("t", agent="a", enforce=True, **kwargs):
            for i in range(60):
                history.append(f"r{i}")
                if expected == "stalled":
                    # growing context: every prompt unique, same answer back
                    call(tuple(history), 1000 + i * 400)
                elif expected == "max_repeats":
                    # BYTE-IDENTICAL arguments, or the fingerprint differs and
                    # there is no repeat to count. The token count is part of
                    # the call, so it has to be constant too.
                    call("identical", 1000)
                else:
                    # distinct prompts, so nothing is cache-served — a repeated
                    # prompt is free and would never breach an envelope
                    call(f"u{i}", 1000 + i * 400)
    except tokeymeter.TaskLimitExceeded:
        pass
    assert len(seen) == 1
    assert seen[0].reason == expected


def test_record_only_mode_notifies_and_says_it_did_not_stop():
    """A team that has not enabled enforcement still wants to know. An alert
    that does not distinguish the two is misleading."""
    seen = []
    tokeymeter.on_halt(seen.append)
    t = _run_stuck(enforce=False)
    assert len(seen) == 1
    assert seen[0].enforced is False
    assert t.snapshot()["calls"] == 30       # genuinely not stopped
    assert "would have been stopped" in seen[0].summary()


def test_a_healthy_task_never_notifies():
    seen = []
    tokeymeter.on_halt(seen.append)
    call = _caller()
    history = []
    with tokeymeter.task("ok", agent="a", stall_window=8, enforce=True):
        for i in range(20):
            history.append(f"s{i}")

            @tokeymeter.cache(model="m")
            def novel(prompt, tokens, n):
                set_reported_usage(tokens, 300)
                return f"finding {n}"
            novel(tuple(history), 900 + i * 300, i)
    assert seen == []


# ── the event is safe to send anywhere ──────────────────────────────────

def test_the_event_carries_no_prompt_or_response():
    """It is designed to be serialised straight into a Slack channel."""
    seen = []
    tokeymeter.on_halt(seen.append)
    _run_stuck(swallow=True)
    blob = json.dumps(seen[0].as_dict(), default=str)
    assert "tool error" not in blob
    assert "retry 1" not in blob


def test_the_summary_is_one_actionable_line():
    seen = []
    tokeymeter.on_halt(seen.append)
    _run_stuck(swallow=True)
    line = seen[0].summary()
    assert "\n" not in line
    assert "ticket-88214" in line and "support" in line and "stopped" in line
    assert "progress" in line


def test_the_event_is_json_serialisable():
    seen = []
    tokeymeter.on_halt(seen.append)
    _run_stuck(swallow=True)
    json.loads(json.dumps(seen[0].as_dict(), default=str))


# ── a handler can never break a request ─────────────────────────────────

def test_a_raising_handler_does_not_break_the_caller():
    """Handlers run in the caller's request path. A broken alert must not
    become a broken agent."""
    def boom(_event):
        raise RuntimeError("webhook down")

    tokeymeter.on_halt(boom)
    t = _run_stuck(swallow=True)             # must not raise anything unexpected
    assert t.snapshot()["calls"] == 8


def test_a_persistently_failing_handler_is_retired():
    """Left alone it would burn latency on every halt forever, and nobody would
    learn why nothing arrives."""
    calls = {"n": 0}

    def boom(_event):
        calls["n"] += 1
        raise RuntimeError("down")

    tokeymeter.on_halt(boom)
    for i in range(8):
        tokeymeter.reset_savings()
        tokeymeter.set_default_store(MemoryStore())
        _run_stuck(task_id=f"t-{i}", swallow=True)
    assert calls["n"] <= 5                   # retired after the failure budget
    assert tokeymeter.halt_handler_count() == 0


def test_one_broken_handler_does_not_starve_the_others():
    good = []

    def boom(_event):
        raise RuntimeError("down")

    tokeymeter.on_halt(boom)
    tokeymeter.on_halt(good.append)
    _run_stuck(swallow=True)
    assert len(good) == 1


def test_a_slow_handler_is_the_callers_latency_not_a_crash():
    def slow(_event):
        import time
        time.sleep(0.05)

    tokeymeter.on_halt(slow)
    t = _run_stuck(swallow=True)
    assert t.snapshot()["calls"] == 8


# ── registration surface ────────────────────────────────────────────────

def test_a_per_task_handler_runs_alongside_the_global_one():
    globals_seen, task_seen = [], []
    tokeymeter.on_halt(globals_seen.append)
    call = _caller()
    history = []
    try:
        with tokeymeter.task("t", agent="a", stall_window=8, enforce=True,
                             on_halt=task_seen.append):
            for i in range(30):
                history.append(f"r{i}")
                try:
                    call(tuple(history), 1000 + i * 400)
                except Exception:
                    pass
    except tokeymeter.TaskLimitExceeded:
        pass
    assert len(task_seen) == 1 and len(globals_seen) == 1


def test_a_non_callable_handler_fails_at_the_with_not_at_halt_time():
    """A typo must surface immediately, not silently at 3am."""
    with pytest.raises(TypeError):
        with tokeymeter.task("t", agent="a", on_halt="notify-me"):
            pass
    with pytest.raises(TypeError):
        tokeymeter.on_halt("not callable")


def test_handlers_can_be_removed_and_counted():
    def h(_e):
        pass
    tokeymeter.on_halt(h)
    assert tokeymeter.halt_handler_count() == 1
    assert tokeymeter.remove_halt_handler(h) is True
    assert tokeymeter.remove_halt_handler(h) is False
    assert tokeymeter.halt_handler_count() == 0


def test_registering_the_same_handler_twice_does_not_double_deliver():
    seen = []
    tokeymeter.on_halt(seen.append)
    tokeymeter.on_halt(seen.append)
    _run_stuck(swallow=True)
    assert len(seen) == 1


def test_on_halt_works_as_a_decorator():
    seen = []

    @tokeymeter.on_halt
    def handler(event):
        seen.append(event)

    _run_stuck(swallow=True)
    assert len(seen) == 1


def test_concurrent_tasks_each_notify_once():
    import concurrent.futures as cf
    seen = []
    lock = threading.Lock()

    def record(e):
        with lock:
            seen.append(e.task_id)

    tokeymeter.on_halt(record)
    call = _caller()

    def work(i):
        history = []
        try:
            with tokeymeter.task(f"t-{i}", agent="a", stall_window=8,
                                 enforce=True):
                for j in range(20):
                    history.append(f"r{j}")
                    try:
                        call(tuple(history) + (i,), 1000 + j * 400)
                    except Exception:
                        pass
        except tokeymeter.TaskLimitExceeded:
            pass

    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(work, range(40)))
    assert len(seen) == 40
    assert len(set(seen)) == 40              # one per task, none duplicated


# ── the shareable report ────────────────────────────────────────────────

def _estate():
    call = _caller()
    for i in range(15):
        history = []
        with tokeymeter.task(f"tk-{i}", agent="support"):
            for j in range(9):
                history.append(f"s{j}")

                @tokeymeter.cache(model="m")
                def novel(prompt, tokens, n):
                    set_reported_usage(tokens, 300)
                    return f"found {n}"
                novel(tuple(history) + (i,), 900 + j * 330, f"{i}-{j}")
    for i in range(3):
        history = []
        with tokeymeter.task(f"stuck-{i}", agent="support"):
            for j in range(14):
                history.append(f"r{j}")
                call(tuple(history) + (i, "x"), 1100 + j * 420)
    return agent_report()


def test_the_html_is_self_contained():
    """No server, no port, and no network request when it opens — it will be
    read on a laptop that has never had Python on it."""
    doc = render_agents_html(_estate())
    body = doc.replace("http-equiv", "")
    assert "<script" not in body.lower()
    assert "src=" not in body and "://" not in body


def test_the_html_contains_no_prompt_or_response_text():
    doc = render_agents_html(_estate())
    assert "tool error" not in doc and "found 0-0" not in doc


def test_the_html_shows_the_stalled_tasks_by_name():
    doc = render_agents_html(_estate())
    assert "support" in doc
    assert "stuck-0" in doc


def test_operator_supplied_names_are_escaped():
    """An operator who names an agent `<script>` gets a page that SAYS
    `<script>`, not one that runs it."""
    report = {"agents": [{"agent": "<script>alert(1)</script>", "tasks": 1,
                          "spend_usd": 0.1, "median_progress": 1.0,
                          "stalled_tasks": 0}]}
    doc = render_agents_html(report)
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_the_html_renders_an_empty_estate_without_failing():
    doc = render_agents_html({"agents": []})
    assert "No agent tasks recorded yet" in doc


@pytest.mark.parametrize("report", [
    {"agents": [{"agent": "a"}]},
    {"agents": [{"agent": "a", "median_progress": None,
                 "median_cost_per_task_usd": None, "tasks": None}]},
    {},
])
def test_the_html_survives_missing_fields(report):
    """A report that fails to render is worth less than one with a gap."""
    doc = render_agents_html(report)
    assert "<html" in doc


def test_the_html_states_what_is_and_is_not_in_it():
    """It gets forwarded to people who did not install it and have every right
    to ask what is inside."""
    doc = render_agents_html(_estate())
    assert "hashed, never read" in doc
    assert "snapshot" in doc.lower()


def test_writing_the_file_returns_an_absolute_path():
    home = tempfile.mkdtemp()
    path = write_agents_html(_estate(), os.path.join(home, "sub", "r.html"))
    assert os.path.isabs(path) and os.path.exists(path)
    assert open(path, encoding="utf-8").read().startswith("<!doctype html>")


def test_the_cli_writes_the_file():
    home = tempfile.mkdtemp()
    env = dict(os.environ, TOKEYMETER_HOME=home, PYTHONPATH=REPO_ROOT)
    out_path = os.path.join(home, "report.html")
    p = subprocess.run([sys.executable, "-m", "tokeymeter", "agents",
                        "--html", out_path],
                       capture_output=True, env=env, timeout=120)
    assert p.returncode == 0
    assert os.path.exists(out_path)


# ── framework adapters ──────────────────────────────────────────────────

def test_a_langchain_model_is_governed_end_to_end():
    """Most of the target users are on LangChain. This was already built and
    unexported — invisible is the same as missing."""
    class FakeChatModel:
        def __init__(self):
            self.n = 0

        def invoke(self, prompt):
            self.n += 1

            class Msg:
                content = "tool error: cannot parse"
            return Msg()

    llm = FakeChatModel()
    rt = tokeymeter.wrap_langchain_llm(llm, model="m")
    history = []
    with pytest.raises(tokeymeter.TaskStalled):
        with tokeymeter.task("lc-1", agent="langchain-agent",
                             stall_window=8, enforce=True):
            for i in range(30):
                history.append(f"turn {i}")
                rt.execute(" ".join(history))
    assert llm.n < 30                        # bounded
    recs = list(sv._tracker._iter_records())
    assert recs and recs[0]["task_id"] == "lc-1"
    assert recs[0]["agent"] == "langchain-agent"


def test_a_langchain_halt_reaches_a_registered_handler():
    seen = []
    tokeymeter.on_halt(seen.append)

    class FakeChatModel:
        def invoke(self, prompt):
            class Msg:
                content = "same"
            return Msg()

    rt = tokeymeter.wrap_langchain_llm(FakeChatModel(), model="m")
    history = []
    try:
        with tokeymeter.task("lc-2", agent="lc", stall_window=8, enforce=True):
            for i in range(30):
                history.append(f"t{i}")
                try:
                    rt.execute(" ".join(history))
                except Exception:
                    pass
    except tokeymeter.TaskLimitExceeded:
        pass
    assert len(seen) == 1 and seen[0].task_id == "lc-2"


def test_a_non_langchain_object_is_rejected_clearly():
    with pytest.raises(TypeError) as ei:
        tokeymeter.wrap_langchain_llm(object(), model="m")
    assert "invoke" in str(ei.value)


def test_importing_the_adapters_needs_no_framework_installed():
    """The adapters duck-type whatever they are handed, so the package still
    declares zero dependencies."""
    import importlib
    mod = importlib.import_module("tokeymeter.runtime.frameworks")
    assert hasattr(mod, "wrap_langchain_llm")
    assert "langchain" not in sys.modules


def test_the_spectrum_labels_never_clip_the_panel():
    """An agent scoring exactly 1.00 is the common case, and a label centred at
    100% hangs off the edge. The usable span is inset for that reason."""
    import re
    report = {"agents": [
        {"agent": "perfect", "tasks": 5, "median_progress": 1.0,
         "stalled_tasks": 0, "spend_usd": 1.0},
        {"agent": "stuck", "tasks": 5, "median_progress": 0.0,
         "stalled_tasks": 5, "spend_usd": 1.0}]}
    doc = render_agents_html(report)
    lefts = [float(m) for m in re.findall(r'left:([\d.]+)%', doc)]
    assert lefts, "nothing was placed on the axis"
    assert all(5.0 <= v <= 95.0 for v in lefts), f"off-panel: {lefts}"


def test_a_clustered_estate_falls_back_to_a_legend():
    """A horizontal axis cannot label five agents that all score 1.00 — and an
    estate where everything is healthy is normal, not an edge case. The dots
    keep their true position; the names move below."""
    report = {"agents": [
        {"agent": f"svc-{i}", "tasks": 5, "median_progress": 1.0 - i * 0.01,
         "stalled_tasks": 0, "spend_usd": 0.1} for i in range(5)]}
    doc = render_agents_html(report)
    assert 'class="legend"' in doc
    assert doc.count('class="dot') == 5          # every agent still plotted
    for i in range(5):
        assert f"svc-{i}" in doc                 # and still named


def test_a_spread_estate_labels_on_the_axis():
    report = {"agents": [
        {"agent": "alpha", "tasks": 5, "median_progress": 0.1,
         "stalled_tasks": 3, "spend_usd": 1.0},
        {"agent": "omega", "tasks": 5, "median_progress": 0.95,
         "stalled_tasks": 0, "spend_usd": 1.0}]}
    doc = render_agents_html(report)
    assert 'class="legend"' not in doc
    assert doc.count('class="node"') == 2


def test_the_report_carries_the_novue_mark():
    doc = render_agents_html({"agents": []})
    assert "<svg" in doc and "NOVUE" in doc
    # inline SVG must not carry xmlns: its value is a URL, and this file makes
    # no network request of any kind
    assert "xmlns" not in doc


def test_stalled_tasks_cluster_into_one_counted_mark():
    """Six tasks stuck the same way all score the same progress, and six
    hairlines drawn on top of each other look like one. A median of 1.00 beside
    six stalls is not a contradiction — it is two populations — and the axis
    exists to make that visible."""
    report = {"agents": [{
        "agent": "svc", "tasks": 18, "median_progress": 1.0,
        "stalled_tasks": 6, "spend_usd": 0.0037,
        "worst_stalls": [{"task_id": f"b{i}", "progress": 0.07,
                          "input_growth": 1.34, "calls": 14,
                          "spend_usd": 0.0007} for i in range(6)]}]}
    doc = render_agents_html(report)
    assert doc.count('class="cluster"') == 1     # one mark, not six hairlines
    assert "6 stalled" in doc                    # and it says how many


def test_small_spend_is_not_rounded_away():
    """A real gpt-4o-mini run spent $0.0038. Two decimals renders that as
    $0.00, which reads as "this cost nothing"."""
    doc = render_agents_html({"agents": [
        {"agent": "a", "tasks": 5, "median_progress": 1.0,
         "stalled_tasks": 0, "spend_usd": 0.0038}]})
    assert "$0.0038" in doc and "$0.00 spent" not in doc


def test_large_spend_uses_ordinary_money_formatting():
    doc = render_agents_html({"agents": [
        {"agent": "a", "tasks": 5, "median_progress": 1.0,
         "stalled_tasks": 0, "spend_usd": 1432.5}]})
    assert "$1,432.50" in doc


def test_the_axis_count_matches_the_table_count():
    """`worst_stalls` is a truncated sample, so counting its entries printed
    "5 stalled" on the axis beside a "6" in the table. Two numbers on one page
    disagreeing is worse than no number at all."""
    import re
    report = {"agents": [{
        "agent": "svc", "tasks": 18, "median_progress": 0.16,
        "stalled_tasks": 6, "spend_usd": 0.0037,
        "worst_stalls": [{"task_id": f"b{i}", "progress": 0.07,
                          "input_growth": 1.34, "calls": 14,
                          "spend_usd": 0.0007} for i in range(5)]}]}
    doc = render_agents_html(report)
    axis = re.findall(r"<em>(\d+) stalled</em>", doc)
    pill = re.findall(r'class="pill">(\d+)<', doc)
    assert axis == pill == ["6"], f"axis {axis} vs table {pill}"


def test_a_realistic_agent_name_is_not_truncated():
    """"service-diagnostic" is 18 characters and a perfectly ordinary name."""
    doc = render_agents_html({"agents": [
        {"agent": "service-diagnostic", "tasks": 18, "median_progress": 0.16,
         "stalled_tasks": 0, "spend_usd": 1.0},
        {"agent": "warranty-triage", "tasks": 10, "median_progress": 0.9,
         "stalled_tasks": 0, "spend_usd": 1.0}]})
    assert "service-diagnostic" in doc
    width = int(doc.split(".node{")[1].split("width:")[1].split("px")[0])
    assert width >= 140, "a label this narrow ellipsises ordinary agent names"


def test_table_columns_do_not_collide():
    """A right-aligned column with no left padding rendered its header as
    "CallsProgress" and ran the progress bar into the calls figure."""
    doc = render_agents_html({"agents": [
        {"agent": "a", "tasks": 5, "median_progress": 0.5,
         "stalled_tasks": 0, "spend_usd": 1.0}]})
    css = doc.split("<style>")[1].split("</style>")[0]
    assert "padding:0 14px 9px 14px" in css      # th has left padding
    assert "th:first-child,td:first-child{padding-left:0}" in css


def _full_report():
    return {"total_tasks": 9, "total_spend_usd": 0.0181,
            "untasked_spend_usd": 0.000099, "excluded_malformed_records": 2,
            "thresholds": {"low_progress": 0.25, "input_growth": 0.25,
                           "min_calls_scored": 4},
            "agents": [{"agent": "svc", "tasks": 9, "median_progress": 0.16,
                        "stalled_tasks": 3, "spend_usd": 0.0181,
                        "stalled_spend_usd": 0.0148, "responses_scored": 60,
                        "responses_executed": 72,
                        "median_cost_per_task_usd": 0.0007,
                        "p95_cost_per_task_usd": 0.0014,
                        "median_calls_per_task": 5.0,
                        "worst_stalls": [{"task_id": f"bad-{i}",
                                          "progress": 0.07,
                                          "input_growth": 1.34, "calls": 14,
                                          "spend_usd": 0.0049}
                                         for i in range(3)]}]}


def test_the_report_leads_with_what_the_stalls_cost():
    """Not what you spent — what you spent on tasks that produced nothing. It
    is the number a platform lead acts on, and it was missing entirely."""
    doc = render_agents_html(_full_report())
    assert "$0.0148" in doc          # the wasted figure
    assert "82%" in doc              # and its share of everything


def test_the_report_states_how_a_stall_was_decided():
    """A reader cannot challenge, or trust, a flag whose rule is invisible."""
    doc = render_agents_html(_full_report())
    assert "How a stall was decided" in doc
    assert "0.25" in doc and "25%" in doc and "4 scored" in doc


def test_the_report_states_what_it_could_and_could_not_see():
    """A report that quietly ignores a sixth of the calls is worse than one
    that says so."""
    doc = render_agents_html(_full_report())
    assert "60 of 72 executed responses could be scored" in doc
    assert "2 malformed records were skipped" in doc
    assert "Not everything is wrapped" in doc      # untasked spend named


def test_a_clean_window_says_so_without_alarm():
    doc = render_agents_html({
        "total_tasks": 5, "total_spend_usd": 1.0, "untasked_spend_usd": 0.0,
        "agents": [{"agent": "a", "tasks": 5, "median_progress": 1.0,
                    "stalled_tasks": 0, "spend_usd": 1.0,
                    "responses_scored": 20, "responses_executed": 20}]})
    assert "Nothing stopped making progress" in doc
    assert "Not everything is wrapped" not in doc


def test_one_colour_system_across_the_axis_and_the_table():
    """Colouring the axis dot by "has stalls" while the table bar used the
    progress band put the SAME agent in cyan on one and amber on the other.
    Both rules were individually right and the page still read as a
    contradiction. Colour now means one thing: how much progress."""
    import re
    doc = render_agents_html({"agents": [
        {"agent": "healthy", "tasks": 5, "median_progress": 0.95,
         "stalled_tasks": 0, "spend_usd": 1.0},
        {"agent": "middling", "tasks": 5, "median_progress": 0.50,
         "stalled_tasks": 0, "spend_usd": 1.0},
        {"agent": "stuck", "tasks": 5, "median_progress": 0.10,
         "stalled_tasks": 4, "spend_usd": 1.0}]})
    axis = dict(re.findall(r"<b>([^<]+)</b>.*?background:(var\(--\w+\))", doc,
                           re.S))
    table = dict(re.findall(r'class="name">([^<]+)<.*?width:\d+%;'
                            r"background:(var\(--\w+\))", doc))
    assert axis and table
    for name, tone in table.items():
        assert axis[name] == tone, f"{name}: axis {axis[name]} vs table {tone}"
