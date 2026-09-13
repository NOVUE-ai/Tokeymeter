"""`tokeymeter watch` — the live view.

Pinned here, because each was a deliberate decision:

  IT IS A READER, NOT A PARTICIPANT. It tails the ledger, opens no port, keeps
    no state agents depend on, and killing it must not affect a running agent.
    A monitor that can take production down is not a monitor.
  IT JUDGES A STALL EXACTLY AS ENFORCEMENT DOES — low response novelty AND
    rising input. If the two ever diverge, the screen would contradict the halt.
  IT SHOWS ONLY WHAT THE LEDGER HOLDS. `avoided` is the single derived figure
    and is always marked with a tilde; anything that cannot be derived honestly
    (which tool an agent was stuck on, which this node never sees) is absent
    rather than invented.
  A TORN FINAL LINE IS NORMAL, not corruption — another process may be
    mid-write — so it is skipped and re-read whole on the next tick.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

import tokeymeter
from tokeymeter import watch as W
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def ledger():
    """A real ledger written by real calls — never hand-built records, so the
    view is tested against what the node actually emits."""
    home = tempfile.mkdtemp()
    path = os.path.join(home, "savings.jsonl")
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.set_savings_path(path)
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)

    @tokeymeter.cache(model="gpt-4o")
    def turn(msgs, tokens, resp):
        set_reported_usage(tokens, 300)
        return resp

    for i in range(2):                       # healthy: novel answers
        h = []
        with tokeymeter.task(f"order-{i}", agent="checkout"):
            for j in range(5):
                h.append(f"s{j}")
                turn(tuple(h) + (i,), 900 + j * 120, f"step {i}-{j}")

    h = []                                    # stalled: same answer, growing input
    with tokeymeter.task("ticket-88214", agent="support"):
        for j in range(12):
            h.append(f"retry {j}")
            turn(tuple(h), 1100 + j * 420, "tool error: cannot parse")

    with tokeymeter.task("doc-1", agent="extract"):   # cache-heavy
        for _ in range(6):
            turn(("same doc",), 1400, "extracted")

    yield home, path
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _run(args, home, timeout=60):
    env = dict(os.environ, TOKEYMETER_HOME=home, PYTHONPATH=REPO_ROOT)
    p = subprocess.run([sys.executable, "-m", "tokeymeter", "watch", *args],
                       capture_output=True, env=env, timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out + err


# ── the signal ──────────────────────────────────────────────────────────

def test_a_stalled_task_is_flagged(ledger):
    home, _ = ledger
    code, out = _run(["--replay", "--speed", "999"], home)
    assert code == 0
    assert "ticket-88214" in out and "STALLED" in out


def test_healthy_tasks_are_not_flagged(ledger):
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    stalled_lines = [l for l in out.splitlines() if "STALLED" in l]
    assert all("order-" not in l for l in stalled_lines)
    assert "1.00" in out                      # a healthy progress score is shown


def test_the_view_agrees_with_enforcement(ledger):
    """If the screen and the halt ever disagree, the number a user reads is not
    the number that stopped their agent."""
    home, path = ledger
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    t = W._Task("ticket-88214")
    for r in recs:
        if r.get("task_id") == "ticket-88214":
            t.add(r, 0.0)
    assert t.stalling is True
    assert t.progress is not None and t.progress <= W._LOW_PROGRESS
    assert t.growth is not None and t.growth >= W._GROWTH

    ok = W._Task("order-0")
    for r in recs:
        if r.get("task_id") == "order-0":
            ok.add(r, 0.0)
    assert ok.stalling is False


def test_a_batch_workload_with_flat_input_is_not_a_stall():
    """The false positive that would sink the screen: identical answers with
    FLAT input is a classifier doing its job."""
    t = W._Task("batch")
    for i in range(10):
        t.add({"task_id": "batch", "hit": False, "estimated_cost": 0.01,
               "response_fingerprint": "sha256:same", "input_tokens": 1400}, 0.0)
    assert t.progress is not None and t.progress < 0.25   # low novelty
    assert t.stalling is False                            # but not a stall


def test_too_few_samples_are_never_scored(ledger):
    t = W._Task("new")
    t.add({"task_id": "new", "hit": False, "estimated_cost": 0.01,
           "response_fingerprint": "sha256:a", "input_tokens": 100}, 0.0)
    assert t.progress is None                 # one sample says nothing


# ── what it shows ───────────────────────────────────────────────────────

def test_cache_hits_are_counted_and_cost_nothing(ledger):
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    assert "cached" in out


def test_the_footer_totals_the_session(ledger):
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    assert "tasks -" in out and "spent" in out and "stalled" in out
    # `avoided` appears only when there is a real figure to show — see
    # test_no_zero_avoided_figure_is_printed.


def test_avoided_is_always_marked_as_an_estimate(ledger):
    """It is the one derived figure. Presenting it as measured would be a lie
    about the only number we cannot observe."""
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    for line in out.splitlines():
        if "avoided" in line:
            assert "~$" in line


def test_no_tool_name_is_invented(ledger):
    """We never see which tool an agent called. Showing one would be fabricated."""
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    assert "stuck on" not in out


# ── it is a reader ──────────────────────────────────────────────────────

def test_watch_writes_nothing(ledger):
    home, path = ledger
    before = os.path.getsize(path)
    before_dir = sorted(os.listdir(home))
    _run(["--replay", "--speed", "999"], home)
    assert os.path.getsize(path) == before
    assert sorted(os.listdir(home)) == before_dir


def test_a_torn_final_line_is_skipped_not_fatal(ledger):
    """Another process may be mid-write. That is normal, not corruption."""
    home, path = ledger
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"timestamp": 1.0, "hit": fal')
    code, out = _run(["--replay", "--speed", "999"], home)
    assert code == 0 and "STALLED" in out


def test_a_corrupt_record_does_not_poison_totals(ledger):
    home, path = ledger
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": "x", "hit": False,
                            "estimated_cost": float("inf"),
                            "input_tokens": 1}) + "\n")
    code, out = _run(["--replay", "--speed", "999"], home)
    assert code == 0 and "inf" not in out.lower()


def test_no_ledger_guides_instead_of_crashing():
    home = tempfile.mkdtemp()
    code, out = _run([], home)
    assert code == 0
    assert "No ledger yet" in out and "tokeymeter.task" in out


# ── the surfaces ────────────────────────────────────────────────────────

def test_json_mode_emits_one_parseable_event_per_line(ledger):
    home, _ = ledger
    code, out = _run(["--replay", "--json"], home)
    assert code == 0
    events = [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]
    assert events and events[0]["event"] == "stalled"
    assert events[0]["task_id"] == "ticket-88214"
    for key in ("progress", "input_growth", "avoided_usd_estimate", "calls"):
        assert key in events[0]


def test_output_is_ascii_safe_on_a_legacy_console(ledger):
    """This runs in cmd.exe as often as it runs in a modern terminal."""
    home, _ = ledger
    env = dict(os.environ, TOKEYMETER_HOME=home, PYTHONPATH=REPO_ROOT,
               PYTHONIOENCODING="cp1252")
    p = subprocess.run([sys.executable, "-m", "tokeymeter", "watch",
                        "--replay", "--speed", "999"],
                       capture_output=True, env=env, timeout=60)
    assert p.returncode == 0
    assert b"STALLED" in (p.stdout or b"")


def test_help_exits_clean():
    code, out = _run(["--help"], tempfile.mkdtemp())
    assert code == 0 and "tokeymeter watch" in out


def test_an_unknown_flag_is_rejected(ledger):
    home, _ = ledger
    code, out = _run(["--replay", "--wat"], home)
    assert code == 1 and "unrecognized" in out


def test_a_bad_speed_is_rejected(ledger):
    home, _ = ledger
    code, out = _run(["--replay", "--speed", "fast"], home)
    assert code == 1


def test_avoided_is_derived_from_the_agents_own_task_length(ledger):
    """The first version assumed 32 remaining calls, which reported "avoided
    ~$0.2344" for tasks that cost $0.1383 in TOTAL — claiming to have saved
    more than the task ever spent. The honest basis is how long this agent's
    healthy tasks actually run."""
    home, path = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    spend_by_task = {}
    for r in recs:
        if r.get("task_id") and not r.get("hit"):
            spend_by_task[r["task_id"]] = spend_by_task.get(r["task_id"], 0.0) \
                + float(r.get("estimated_cost") or 0.0)
    biggest = max(spend_by_task.values())
    for line in out.splitlines():
        if "avoided ~$" in line:
            amount = float(line.split("avoided ~$")[1].split()[0])
            assert amount <= biggest, (
                f"claimed to avoid ${amount:.4f}, more than the most expensive "
                f"task in this ledger ever cost (${biggest:.4f})")


def test_no_avoided_figure_is_printed_without_a_basis():
    """Returning 0.0 and printing nothing beats substituting a guess."""
    t = W._Task("solo")
    t.agent = "a"
    for i in range(6):
        t.add({"task_id": "solo", "hit": False, "estimated_cost": 0.01,
               "response_fingerprint": "same", "input_tokens": 100 + i * 200},
              0.0)
    assert t.avoided_estimate(None) == 0.0        # no history to judge against
    assert t.avoided_estimate(3) == 0.0           # already past typical length


def test_no_zero_avoided_figure_is_printed(ledger):
    """"avoided ~$0.0000" reads as "this saved you nothing", which is worse
    than saying nothing at all. Seen on a real run, on every stall line."""
    home, _ = ledger
    _, out = _run(["--replay", "--speed", "999"], home)
    assert "$0.0000 avoided" not in out
    assert "avoided ~$0.0000" not in out


def test_prior_stalls_are_the_baseline_when_there_are_enough():
    """A stuck agent does not stop where a healthy one does — it retries until
    its own loop limit. Measured on a real run: healthy tasks ran 5 calls while
    stalled ones ran 14, so judging a halt at call 4 against the HEALTHY length
    understated the saving to nothing."""
    t = W._Task("t")
    t.agent = "a"
    for i in range(8):
        t.add({"task_id": "t", "hit": False, "estimated_cost": 0.01,
               "response_fingerprint": "same", "input_tokens": 100 + i * 300},
              0.0)
    # halted at 8 calls; peers that also stalled ran 20
    assert t.avoided_estimate(20) > t.avoided_estimate(9)
    assert t.avoided_estimate(5) == 0.0        # already past that length


def test_the_bar_is_a_fill_not_a_height_ramp():
    """The old ramp drew a stalled agent as U+2581, a hairline — the single
    most important state on the screen, and invisible in a screen recording.
    Fill is legible at any size."""
    class _P:
        def __init__(self, p): self._p = p
        @property
        def progress(self): return self._p

    assert W._spark(_P(1.00), False) == "\u2588" * 4
    assert W._spark(_P(0.50), False).count("\u2588") == 2
    assert W._spark(_P(0.12), False).count("\u2588") == 1
    # a scored-but-tiny value must still show SOMETHING, or a stalled agent
    # renders identically to one that was never measured
    assert W._spark(_P(0.01), False).count("\u2588") >= 1


def test_unscored_renders_empty_not_faint():
    """"Nothing to measure" must not look like "nearly stalled"."""
    class _P:
        progress = None
    assert W._spark(_P(), False).strip() == ""
    assert W._spark(_P(), True).strip() == ""


def test_ansi_is_requested_and_the_fallback_is_readable(capsys):
    """Windows consoles ignore escape sequences unless a process asks, with no
    error — the clear-screen code is swallowed and every frame prints BELOW the
    last, so a recorded demo becomes the same table stacked forty times."""
    assert isinstance(W._enable_ansi(), bool)
    W._clear(False)
    out = capsys.readouterr().out
    assert "-" * 20 in out            # a rule separates frames instead
    W._clear(True)
    assert "\033[J" in capsys.readouterr().out


def test_a_frame_always_fits_the_window():
    """A frame taller than the terminal scrolls no matter how thoroughly the
    screen is cleared, and the live view degrades into a log. Measured on a
    real Windows recording: the same table stacked over and over, which is the
    one thing a recorded demo must not be."""
    import os
    import shutil as _sh
    tasks = {}
    for i in range(30):                       # far more tasks than fit
        t = W._Task(f"t-{i}")
        t.agent = "svc"
        for j in range(6):
            t.add({"task_id": f"t-{i}", "hit": False, "estimated_cost": 0.001,
                   "response_fingerprint": f"n{i}-{j}",
                   "input_tokens": 200 + j * 300}, float(j))
        tasks[f"t-{i}"] = t
    events = [f"t-{i} STALLED - 1 distinct answer in 4 calls" for i in range(4)]
    totals = {"tasks": 30, "spend": 0.02, "halted": 6, "avoided": 0.004}

    original = _sh.get_terminal_size
    try:
        for height in (18, 20, 24, 30, 40):
            _sh.get_terminal_size = (
                lambda fallback=(80, 24), _h=height: os.terminal_size((110, _h)))
            lines = len(W._render(tasks, events, totals, False).splitlines())
            assert lines <= height - 2, (
                f"a {lines}-line frame in a {height}-line window will scroll")
    finally:
        _sh.get_terminal_size = original


def test_the_clear_drops_scrollback_too(capsys):
    """Plain ESC[2J leaves the scrollback intact, so a frame that once
    overflowed keeps its history and the view stacks."""
    W._clear(True)
    out = capsys.readouterr().out
    assert "\033[H" in out           # home BEFORE clearing
    assert "\033[3J" in out          # and drop the scrollback
