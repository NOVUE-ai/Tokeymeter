"""`tokeymeter watch` — see tasks as they run, and the moment one stops working.

WHY THIS EXISTS
---------------
`tokeymeter agents` reports after the fact. This shows the thing happening:
progress collapsing while spend keeps rising. That relationship is the one
number no other tool computes, and watching it fall is far more convincing than
reading that it fell.

It is deliberately a READER. It tails the ledger every service already writes,
opens no port, starts no server, keeps no state of its own, and can be killed
at any moment without affecting a single running agent. A monitor that can take
production down is not a monitor.

WHAT IS MEASURED AND WHAT IS ESTIMATED
--------------------------------------
Everything in the table is read from the ledger: calls, spend, cache hits,
distinct responses, input growth.

`avoided` is the ONE derived figure, and it is labelled with a tilde wherever it
appears. When a task halts we know what its calls cost so far; we do not know
what the calls that never happened would have cost. The estimate assumes the
remaining calls would have resembled the observed ones, which is the most
defensible assumption available and still an assumption. Anything we cannot
derive honestly — such as which tool an agent was stuck on, which this node
never sees — is simply not shown.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

__all__ = ["main"]

# A task is considered finished when nothing new has arrived for this long, so
# a completed task eventually leaves the live table instead of accumulating.
_IDLE_SECONDS = 20.0
# Progress is judged over the most recent calls, matching the enforcement rule.
_WINDOW = 8
_LOW_PROGRESS = 0.25
_GROWTH = 0.25
# A FILL bar, not a height ramp. The old ramp drew a stalled agent as
# U+2581 (lower one-eighth block) — a hairline, and the single most important
# state on the screen. Fill is legible at any size and survives a screen
# recording, which a one-pixel underline does not.
_CELLS = 4


def _enable_ansi() -> bool:
    """Turn on ANSI escape handling, and say whether it worked.

    Windows consoles do not process escape sequences unless a process asks,
    and there is no error when they do not: the clear-screen code is simply
    swallowed and every frame prints BELOW the last. The result looks like a
    log, not a live view — the same table stacked forty times, which is
    exactly what a recorded demo must not be.

    conhost supports this from Windows 10 1511 via ENABLE_VIRTUAL_TERMINAL_
    PROCESSING (0x4); Windows Terminal and every POSIX terminal already do.
    Returns False when the screen cannot be cleared, so the caller can fall
    back to something readable instead of something broken.
    """
    if os.name != "nt":
        return sys.stdout.isatty()
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x4))
    except Exception:
        return False


def _clear(enabled: bool) -> None:
    """Blank the screen for the next frame, or separate frames legibly when the
    console cannot be cleared. Printing forty stacked copies of the same table
    is worse than printing a rule between them."""
    if enabled:
        # Home first, then clear to end, then drop the scrollback buffer.
        # Plain \033[2J leaves the scrollback intact, so a frame taller than
        # the window still scrolls and the view stacks instead of replacing.
        print("\033[H\033[J\033[3J", end="")
    else:
        print("\n" + "-" * 66 + "\n")


def _ascii_only() -> bool:
    """Legacy consoles cannot render block characters. Ask, do not assume."""
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return not ("utf" in enc or "u8" in enc)


class _Task:
    __slots__ = ("task_id", "agent", "calls", "executed", "hits", "spend",
                 "fps", "tokens", "last_seen", "halted", "rules")

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.agent: Optional[str] = None
        self.calls = 0
        self.executed = 0
        self.hits = 0
        self.spend = 0.0
        self.fps: List[Optional[str]] = []
        self.tokens: List[Optional[int]] = []
        self.last_seen = 0.0
        self.halted = False
        self.rules: Optional[str] = None

    def add(self, rec: dict, now: float) -> None:
        self.calls += 1
        self.last_seen = now
        if not self.agent and rec.get("agent"):
            self.agent = str(rec["agent"])
        if not self.rules and rec.get("policy_rules"):
            self.rules = str(rec["policy_rules"])
        if rec.get("hit"):
            self.hits += 1
            return
        self.executed += 1
        try:
            c = float(rec.get("estimated_cost") or 0.0)
            if c == c and abs(c) != float("inf") and c >= 0:
                self.spend += c
        except (TypeError, ValueError):
            pass
        self.fps.append(rec.get("response_fingerprint"))
        try:
            t = rec.get("input_tokens")
            self.tokens.append(int(t) if t is not None else None)
        except (TypeError, ValueError):
            self.tokens.append(None)

    @property
    def progress(self) -> Optional[float]:
        """Distinct responses over the recent window, or None when there is not
        enough to say. Two samples that happen to agree are noise."""
        fps = [f for f in self.fps[-_WINDOW:] if f]
        if len(fps) < 2:
            return None
        return len(set(fps)) / float(len(fps))

    @property
    def growth(self) -> Optional[float]:
        toks = [t for t in self.tokens[-_WINDOW:] if t is not None]
        if len(toks) < 4:
            return None
        half = len(toks) // 2
        first = sum(toks[:half]) / half
        second = sum(toks[half:]) / (len(toks) - half)
        return (second - first) / first if first > 0 else None

    @property
    def stalling(self) -> bool:
        """Both signals, exactly as enforcement judges it: the answers stopped
        being new AND the input kept growing."""
        p, g = self.progress, self.growth
        return (p is not None and p <= _LOW_PROGRESS
                and g is not None and g >= _GROWTH)

    def avoided_estimate(self, typical_calls: Optional[float] = None) -> float:
        """Cost of the calls this halt prevented.

        HOW LONG WOULD IT HAVE RUN? That is the whole question, and a fixed
        guess gets it badly wrong. The first version assumed 32 more calls,
        which reported "avoided ~$0.2344" for tasks that cost $0.1383 in
        total — claiming to have saved more than the task ever spent, by
        roughly 2x.

        The honest source is the agent's OWN history: if its tasks typically
        run 12 calls and this one was stopped at 4, then 8 calls were
        prevented. Both the length and the per-call cost come from observed
        traffic, so the figure can be over-estimated only if this particular
        task would have been unusually long — and it is still marked with a
        tilde everywhere it appears.

        Returns 0.0 when there is no basis to estimate from, rather than
        substituting a guess.
        """
        if not self.executed or not typical_calls:
            return 0.0
        remaining = max(0.0, typical_calls - self.executed)
        if remaining <= 0:
            return 0.0
        return (self.spend / self.executed) * remaining


def _spark(task: "_Task", ascii_mode: bool) -> str:
    """A proportional fill bar. Not scored renders as empty cells rather than
    a faint one, so "nothing to measure" never looks like "nearly stalled"."""
    p = task.progress
    fill, empty = ("#", ".") if ascii_mode else ("\u2588", "\u00b7")
    if p is None:
        return " " * _CELLS
    n = max(1, round(max(0.0, min(1.0, p)) * _CELLS)) if p > 0 else 0
    return fill * n + empty * (_CELLS - n)


def _read_new(path: str, offset: int) -> "tuple[List[dict], int]":
    """Read whatever has been appended since the last poll.

    A torn final line is normal — another process may be mid-write — so it is
    skipped and re-read next tick rather than treated as corruption. The file
    shrinking means it was trimmed or rotated, and we start over.
    """
    out: List[dict] = []
    try:
        size = os.path.getsize(path)
    except OSError:
        return out, offset
    if size < offset:
        offset = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            data = f.read()
            offset = f.tell()
    except OSError:
        return out, offset
    if not data:
        return out, offset
    lines = data.split("\n")
    if not data.endswith("\n"):
        # incomplete final line; rewind so it is read whole next time
        offset -= len(lines[-1].encode("utf-8", "replace"))
        lines = lines[:-1]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, offset


def _render(tasks: Dict[str, _Task], events: List[str], totals: Dict[str, Any],
            ascii_mode: bool) -> str:
    """Fit the WINDOW, not the data.

    A frame taller than the terminal scrolls no matter how thoroughly the
    screen is cleared, and the live view degrades into a log — which is what a
    recorded demo must never look like. So the number of task rows is derived
    from the actual window height, leaving room for the header, the event
    lines and the footer.
    """
    try:
        rows = shutil.get_terminal_size(fallback=(80, 24)).lines
    except Exception:
        rows = 24
    budget = max(3, rows - 11)          # header, blank lines, 4 events, footer
    live = sorted((t for t in tasks.values()),
                  key=lambda t: -t.last_seen)[:budget]
    lines = []
    for t in live:
        p = t.progress
        ptxt = "  . " if p is None else f"{p:.2f}"
        flag = "!" if t.stalling and not t.halted else " "
        cached = f"{t.hits} cached" if t.hits else ""
        rule = f"  {t.rules}" if t.rules else ""
        lines.append(
            f"  {t.task_id[:16]:16} {(t.agent or '-')[:10]:10} "
            f"call {t.calls:>3}  ${t.spend:7.4f}  {_spark(t, ascii_mode)} "
            f"{ptxt}  {flag} {cached}{rule}".rstrip())
    body = "\n".join(lines) if lines else "  (waiting for tasks)"
    ev = "\n".join(f"  {e}" for e in events[-4:])
    n = totals["tasks"]
    foot = (f"  {n} task{'' if n == 1 else 's'} - ${totals['spend']:.4f} spent - "
            f"{totals['halted']} stalled")
    if round(totals["avoided"], 4) > 0:
        foot += f" - ~${totals['avoided']:.4f} avoided"
    parts = [body]
    if ev:
        parts += ["", ev]
    parts += ["", foot]
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("tokeymeter watch - see tasks as they run\n\n"
              "  tokeymeter watch                  follow live\n"
              "  tokeymeter watch --replay         replay the existing ledger\n"
              "  tokeymeter watch --replay --speed 4\n"
              "  tokeymeter watch --json           one event per line, for piping\n\n"
              "Reads the ledger. Opens no port, starts no server, and can be\n"
              "killed at any time without affecting a running agent.")
        return 0

    replay = "--replay" in argv
    as_json = "--json" in argv
    speed = 1.0
    if "--speed" in argv:
        try:
            speed = max(0.1, float(argv[argv.index("--speed") + 1]))
        except (IndexError, ValueError):
            print("tokeymeter watch: --speed needs a number")
            return 1
    unknown = [a for a in argv
               if a not in ("--replay", "--json", "--speed")
               and not a.replace(".", "").isdigit()]
    if unknown:
        print(f"tokeymeter watch: unrecognized argument {unknown[0]!r}")
        return 1

    from tokeymeter import paths
    path = paths.savings_path()
    if not os.path.exists(path):
        print(f"No ledger yet at {path}\n\n"
              "Wrap an agent entry point and run it:\n"
              "  with tokeymeter.task('id', agent='name'):\n"
              "      agent.run(...)\n\n"
              "Or try `tokeymeter firstrun` for an offline demo.")
        return 0

    ansi = _enable_ansi()
    tasks: Dict[str, _Task] = {}
    events: List[str] = []
    # Observed calls-per-task, per agent. The basis for "how long would this
    # have run", and it is only ever read from traffic already seen.
    agent_lengths: Dict[str, Dict[str, int]] = {}
    # Lengths of tasks that ALREADY stalled, kept apart from healthy ones. A
    # stuck agent does not stop where a healthy one does — it retries until its
    # own loop limit — so prior stalls are the honest evidence for "how long
    # would this have run".
    stalled_lengths: Dict[str, Dict[str, int]] = {}
    totals = {"tasks": 0, "spend": 0.0, "halted": 0, "avoided": 0.0}
    ascii_mode = _ascii_only()
    offset = 0
    ticks_idle = 0

    def ingest(rec: dict, now: float) -> None:
        tid = rec.get("task_id")
        if not tid:
            return
        tid = str(tid)
        first = tid not in tasks
        t = tasks.setdefault(tid, _Task(tid))
        if first:
            totals["tasks"] += 1
        # Track how long this agent's HEALTHY tasks run. Keyed by task so a
        # task in progress refines its own entry rather than adding a new one;
        # halted tasks are excluded, since their length is what we prevented
        # and using it as the baseline would argue in a circle.
        if t.agent:
            bucket = stalled_lengths if t.halted else agent_lengths
            bucket.setdefault(t.agent, {})[tid] = t.calls
        before = t.stalling
        t.add(rec, now)
        try:
            c = float(rec.get("estimated_cost") or 0.0)
            if c == c and abs(c) != float("inf") and c >= 0 and not rec.get("hit"):
                totals["spend"] += c
        except (TypeError, ValueError):
            pass
        if t.stalling and not before and not t.halted:
            t.halted = True
            totals["halted"] += 1
            # PRIOR STALLS FIRST. Measured on a real run: healthy diagnostic
            # tasks ran 5 calls while stalled ones ran 14, so judging a halt at
            # call 4 against the healthy length said "avoided ~$0.0000" —
            # understating as badly as the fixed-32 guess overstated. A stuck
            # task's peers are other stuck tasks.
            stalled_seen = sorted(stalled_lengths.get(t.agent or "", {}).values())
            healthy_seen = sorted(agent_lengths.get(t.agent or "", {}).values())
            seen = stalled_seen if len(stalled_seen) >= 3 else healthy_seen
            # Three samples or nothing: two is a coincidence, not a length.
            typical = seen[len(seen) // 2] if len(seen) >= 3 else None
            avoided = t.avoided_estimate(typical)
            totals["avoided"] += avoided
            p = t.progress or 0.0
            g = t.growth or 0.0
            scored = [f for f in t.fps[-_WINDOW:] if f]
            n_distinct = len(set(scored))
            msg = (f"{tid} STALLED - "
                   f"{n_distinct} distinct answer{'' if n_distinct == 1 else 's'} "
                   f"in {len(scored)} calls, input +{g:.0%}")
            # Only when it survives rounding. "avoided ~$0.0000" reads as
            # "this saved you nothing", which is worse than saying nothing.
            if round(avoided, 4) > 0:
                msg += f" - avoided ~${avoided:.4f}"
            events.append(msg)
            if as_json:
                print(json.dumps({"event": "stalled", "task_id": tid,
                                  "agent": t.agent, "calls": t.calls,
                                  "spend_usd": round(t.spend, 6),
                                  "progress": round(p, 4),
                                  "input_growth": round(g, 4),
                                  "avoided_usd_estimate": round(avoided, 6)}),
                      flush=True)

    if replay:
        recs, _ = _read_new(path, 0)
        recs.sort(key=lambda r: r.get("timestamp") or 0.0)
        if not recs:
            print("Ledger is empty - nothing to replay.")
            return 0
        for i, rec in enumerate(recs):
            ingest(rec, i)
            if not as_json and i % max(1, int(speed)) == 0:
                _clear(ansi)
                print("  tokeymeter watch (replay)\n")
                print(_render(tasks, events, totals, ascii_mode))
                time.sleep(0.04)
        if not as_json:
            _clear(ansi)
            print("  tokeymeter watch (replay complete)\n")
            print(_render(tasks, events, totals, ascii_mode))
        return 0

    try:
        while True:
            recs, offset = _read_new(path, offset)
            now = time.time()
            for rec in recs:
                ingest(rec, now)
            for tid in [k for k, v in tasks.items()
                        if now - v.last_seen > _IDLE_SECONDS]:
                tasks.pop(tid, None)
            if not as_json:
                _clear(ansi)
                print("  tokeymeter watch   (ctrl-c to stop)\n")
                print(_render(tasks, events, totals, ascii_mode))
                ticks_idle = 0 if recs else ticks_idle + 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        if not as_json:
            print("\n  stopped. Nothing was changed - watch only reads.\n")
        return 0
