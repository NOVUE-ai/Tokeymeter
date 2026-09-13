"""Task boundary — WHICH unit of work every call belongs to.

The third sibling of `identity.principal` and `execution.endpoint`, and the
one that changes what can be governed.

WHY THIS EXISTS
---------------
A single model call is a function; an agent is a program. An agent task is
10-100 calls that belong together, and every tool in the stack meters the
wrong unit:

  * agent frameworks cap ITERATIONS, which is a count, not a cost. Context
    grows every turn, so call 24 can cost 60x call 3 and no counter can tell.
  * observability platforms show the trace AFTER the money is gone.
  * a gateway sees N unrelated requests: task identity is application context
    and never crosses a network boundary, whoever owns the network.

Only something inside the process that spawned the task can hold the boundary.
That is the whole reason this module can exist here and nowhere else.

WHAT IT PROVIDES
----------------
    with tokeymeter.task("ticket-4f2a", envelope=1.00, max_repeats=4):
        agent.run(ticket)                     # every call inside is bound

  * `task_id` on every record  -> cost per task, the unit that maps to the bill
  * `prompt_fingerprint`       -> loop detection, content-blind
  * envelope / call / repeat limits -> a CEILING, checked BEFORE the next call

CONTENT-BLIND BY CONSTRUCTION
-----------------------------
A loop is detected as the same PROMPT FINGERPRINT appearing repeatedly inside
one task — the agent asking the same thing and making no progress. The
fingerprint is a truncated SHA-256 of the cache key, which is itself a digest
of the call arguments. We compare hashes; we never read a prompt. A record
gains two fields and still carries zero content.

THE ENVELOPE: TRIGGER MODE VS HARD MODE
----------------------------------------
A call's cost is not knowable until it returns, so a naive check ("have I spent
too much yet?") always lets the call that crosses the line complete. That is a
TRIGGER, not a cap, and final spend lands at `envelope + one call`.

The fix is the pattern a hotel uses on your card: hold the worst case before
the charge, settle for the actual afterwards.

    task(envelope=1.00, reserve=0.05)   # no single call can cost more than 5c

With `reserve` set, the check becomes "would the WORST CASE of the next call
breach the envelope?" — so the call that would have crossed the line is never
made, and final spend NEVER exceeds the envelope. That is a hard cap.

Without `reserve`, the task adapts: after the first call it holds back the
largest cost it has actually observed in this task, which tightens the bound
automatically for every call but the first. State which you have:

    reserve declared    -> spend <= envelope                    (hard cap)
    no reserve, call 1  -> spend <= envelope + cost_of_call_1   (unbounded first call)
    no reserve, call n  -> spend <= envelope + largest_seen     (adaptive)

If a call ever costs MORE than the declared reserve, the reservation was
under-declared: the overrun is counted in `reserve_breaches` and surfaced in
the snapshot rather than hidden, because a guarantee you cannot audit is not a
guarantee.

ENFORCEMENT DOES NOT DEPEND ON THE CALLER COOPERATING
------------------------------------------------------
`TaskLimitExceeded` subclasses `RuntimeError` (matching the shipped
`KeyBudgetExceeded`), so a broad `except Exception:` in agent code — which is
almost universal in retry loops — will swallow the halt. That is fine, and
deliberately so: the guarantee does not rest on the caller re-raising.

Because the ceiling is evaluated BEFORE each call, a swallowed halt simply
means the next attempt is refused too. Measured: an agent that caught 198
halts and kept looping 200 times made 2 upstream calls and stayed inside its
envelope. The failure mode of swallowing is a busy loop, not a budget breach.

WHAT COUNTS, AND WHEN
---------------------
  attempt  counted PRE-flight  -> a call that RAISES still consumes the call
                                  and repeat budget, so an agent failing every
                                  call upstream is bounded. Counting only
                                  successes would leave that runaway free.
  spend    settled POST-call   -> actual cost only, and only for an executed
                                  call. A cache hit cost nothing upstream, so
                                  no spend is held for one either: refusing a
                                  free call would stop an agent early for no
                                  reason.
  nesting  rolls UP            -> spend accrues to every ancestor and the whole
                                  chain is checked, so an inner task can never
                                  exceed an outer envelope.

THE ENVELOPE AND max_repeats CATCH DIFFERENT FAILURES — SET BOTH
----------------------------------------------------------------
They are orthogonal, and a policy with only one of them has a hole.

An envelope catches an EXPENSIVE task: many distinct calls, growing context,
real money. A stuck task is often the opposite — it repeats the SAME call, so
after the first execution every repeat is served from cache. Measured: a
20-call loop cost $0.0155, because 19 of the 20 were hits. An envelope alone
would never have stopped it, and the agent would spin forever inside its
budget.

    envelope     bounds the money
    max_repeats  bounds the spinning
    max_calls    bounds both, without depending on cost at all

Cache hits are counted toward repeats precisely so the second ceiling still
works when caching has made the loop free.

PRE-FLIGHT, NOT POST-MORTEM
---------------------------
Limits are evaluated BEFORE the wrapped function runs, against state
accumulated from earlier calls in the same task. Checking afterwards would
mean the spend has already happened — which is precisely the failure mode of
every alert-based tool. A ceiling that reports is not a ceiling.

FAIL-OPEN, EXCEPT WHERE ASKED
-----------------------------
All bookkeeping is defensive: an error inside task accounting must never break
a caller's request. The ONLY thing that raises is a deliberate, configured
limit breach, and only when `enforce=True`. The default is observe-and-record,
so adopting the boundary can never take down an agent that was working.

CONCURRENCY — READ THIS BEFORE FANNING OUT
------------------------------------------
The contextvar holds a mutable state object, and all mutation is lock-guarded,
so many callers may safely share one task.

Propagation follows standard Python contextvar rules, which are NOT uniform:

  * asyncio tasks INHERIT the binding automatically (a new Task copies the
    current context). An async agent fanning out with `asyncio.gather` needs
    nothing extra — every child call counts toward the same envelope.
  * plain threads DO NOT. A worker started by `ThreadPoolExecutor` begins with
    a fresh context and would see no task at all.

That second case is a correctness trap: enforcement would silently not apply
while the caller believed it did, which is worse than having no ceiling. Two
supported ways to carry the binding into threads:

    # explicit, per worker
    with tokeymeter.task("job-1", envelope=5.00, enforce=True) as t:
        def work(i):
            with tokeymeter.bind_task(t):
                return step(i)
        with ThreadPoolExecutor() as ex:
            list(ex.map(work, range(10)))

    # or copy the whole context, the stdlib way
    ctx = contextvars.copy_context()
    ex.submit(ctx.run, step, i)

`task_snapshot()` reports `calls`, so a caller can assert the count it expects
and catch a propagation mistake in a test rather than in production.
"""
from __future__ import annotations

import collections
import contextlib
import contextvars
import hashlib
import math
import time
import re
import threading
from typing import Any, Dict, Iterator, Optional

__all__ = [
    "task", "bind_task", "set_task", "get_task", "current_task_id",
    "current_agent",
    "task_snapshot",
    "TaskLimitExceeded", "TaskEnvelopeExceeded", "TaskLoopDetected",
    "TaskStalled",
    "TaskCallLimitExceeded",
]


# ── exceptions ──────────────────────────────────────────────────────────

class TaskLimitExceeded(RuntimeError):
    """A declared task limit was reached. Base class so a caller can catch one
    thing and treat any ceiling breach uniformly (mark the work for human
    review, retry with a larger envelope, or fail the task cleanly)."""

    def __init__(self, message: str, *, task_id: str, limit: str,
                 observed, allowed) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.limit = limit
        self.observed = observed
        self.allowed = allowed


class TaskEnvelopeExceeded(TaskLimitExceeded):
    """Accumulated spend for this task reached its declared envelope."""


class TaskLoopDetected(TaskLimitExceeded):
    """The same prompt fingerprint repeated inside one task beyond the declared
    limit: the agent is asking the same thing and making no progress. Detected
    from hash repetition — no prompt is ever read."""


class TaskStalled(TaskLimitExceeded):
    """The agent stopped making progress: its recent responses stopped being
    novel while its input kept growing. It is paying more and more to learn
    nothing new. Detected from response-hash novelty — no output is read."""


class TaskCallLimitExceeded(TaskLimitExceeded):
    """This task made more calls than its declared maximum."""


# ── identifier grammar (same posture as endpoint/principal) ─────────────

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")


def _validate_id(tid: str) -> str:
    """A task id is an OPERATOR-SUPPLIED IDENTIFIER — a ticket number, a job
    id, a run id. Validated here so a malformed or content-smuggling value can
    never enter the record stream."""
    if not isinstance(tid, str) or not _ID_RE.match(tid):
        raise ValueError(
            "task id must be 1-128 chars of [A-Za-z0-9._:-] starting "
            "alphanumeric (an identifier such as a ticket or run id, never "
            f"content); got {tid!r}")
    return tid


def _validate_limit(name: str, value, *, integer: bool = False):
    """Limits must be finite and positive. A NaN slips past `> 0` (every
    comparison with NaN is False), which would silently disable the ceiling —
    exactly the class of bug the ledger hardening passes kept finding."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got a bool")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or None, got {value!r}")
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    if integer:
        if v != int(v):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(v)
    return v


def _current_principal_safe() -> Optional[str]:
    """The bound principal, or None. Rule resolution must never fail because
    identity is unavailable."""
    try:
        from tokeymeter.engines.governance.identity import get_principal
        return get_principal()
    except Exception:
        return None


# ── task state ──────────────────────────────────────────────────────────

def _novelty_or_none(recent):
    """Distinct scoreable responses over scoreable samples, or None when there
    are too few to mean anything. Two responses that happen to agree are noise,
    not evidence."""
    fps = [r[0] for r in recent if r[0]]
    if len(fps) < 2:
        return None
    return round(len(set(fps)) / float(len(fps)), 4)


class _TaskState:
    """Live accounting for one in-flight task. Mutable and shared across any
    threads or async tasks spawned inside the block, so every mutation is
    lock-guarded."""

    __slots__ = ("task_id", "envelope_usd", "reserve_usd", "max_calls",
                 "max_repeats", "enforce", "_lock", "spend_usd", "calls",
                 "_fingerprints", "halted_reason", "max_observed_call_usd",
                 "on_halt",
                 "reserve_breaches", "parent", "fingerprints_evicted",
                 "agent", "matched_rules", "stall_window", "min_novelty",
                 "_recent", "stalled")

    # A loop is the SAME fingerprint repeating, so entries seen exactly once
    # are never loop evidence. Cap the map and evict those first: a long task
    # with tens of thousands of DISTINCT calls must not grow without bound.
    _FINGERPRINT_CAP = 10_000

    def __init__(self, task_id: str, envelope_usd: Optional[float],
                 reserve_usd: Optional[float], max_calls: Optional[int],
                 max_repeats: Optional[int], enforce: bool,
                 stall_window: Optional[int] = None,
                 min_novelty: Optional[float] = None,
                 parent: "Optional[_TaskState]" = None,
                 agent: Optional[str] = None,
                 matched_rules: tuple = ()) -> None:
        self.parent = parent
        self.agent = agent
        self.matched_rules = matched_rules
        self.task_id = task_id
        self.envelope_usd = envelope_usd
        self.reserve_usd = reserve_usd
        self.max_calls = max_calls
        self.max_repeats = max_repeats
        self.enforce = bool(enforce)
        self._lock = threading.Lock()
        self.spend_usd = 0.0
        self.calls = 0
        self._fingerprints: Dict[str, int] = {}
        self.halted_reason: Optional[str] = None
        self.on_halt = None
        # Largest single-call cost actually seen in this task. Used as an
        # implicit hold when no reserve was declared, so the bound tightens
        # after the first call instead of staying one-call-wide forever.
        self.max_observed_call_usd: float = 0.0
        # Calls whose ACTUAL cost exceeded the declared reserve — i.e. the
        # caller's upper bound was wrong. Surfaced, never swallowed: a hard
        # guarantee you cannot audit is not a guarantee.
        self.reserve_breaches: int = 0
        # Distinct fingerprints dropped at the cap. Surfaced, because loop
        # detection is weaker for prompts we stopped tracking.
        self.fingerprints_evicted: int = 0
        self.stall_window = stall_window
        self.min_novelty = min_novelty
        # Rolling window of (response_fingerprint, input_tokens) for EXECUTED
        # calls only. Cache hits are excluded deliberately: a hit returns a
        # byte-identical response by definition, so counting them would make
        # every well-cached workload look stalled.
        self._recent: "collections.deque" = collections.deque(
            maxlen=max(2, stall_window or 8))
        self.stalled: bool = False

    # -- pre-flight -----------------------------------------------------

    def check_before_call(self, fingerprint: Optional[str],
                          is_hit: bool = False) -> None:
        """Evaluate every declared limit against state from EARLIER calls in
        this task, then count this attempt.

        `is_hit` matters: a cache hit costs nothing upstream, so no spend is
        held for it. Holding the reserve for a free call would refuse work that
        cannot breach the envelope — a false positive that stops an agent early
        for no reason. Hits still count toward calls and repeats, because a
        loop served from cache is still a loop.

        The ATTEMPT is counted here, not on completion, so a call that RAISES
        still consumes the call and repeat budget. An agent failing every call
        upstream is the exact runaway this exists to stop, and counting only
        successes would leave it unbounded.
        """
        breach = self._check_locked(fingerprint, is_hit)
        if breach is None and self.parent is not None:
            # An inner task can never exceed an outer one: check the whole
            # chain before the call proceeds.
            try:
                self.parent.check_before_call(fingerprint, is_hit)
            except TaskLimitExceeded:
                self._count_attempt(fingerprint)
                raise
        if breach is None:
            self._count_attempt(fingerprint)
            return
        cls, limit, observed, allowed, message = breach
        with self._lock:
            first = self.halted_reason is None
            if first:
                self.halted_reason = limit
        # Notify from the one-time transition, not from the raise. Ordinary
        # agent retry code catches RuntimeError and swallows the halt —
        # measured, 22 swallowed on one stuck task — so a hook riding on the
        # exception would stay silent in exactly the case an operator needs.
        # Firing here also means record-only mode still reports.
        if first:
            self._notify_halt(limit, observed, allowed, message)
        if not self.enforce:
            self._count_attempt(fingerprint)
            return
        raise cls(message, task_id=self.task_id, limit=limit,
                  observed=observed, allowed=allowed)

    def _notify_halt(self, reason: str, observed, allowed, message) -> None:
        """Build the event and hand it off. NEVER raises and never blocks the
        decision: notification is best-effort, enforcement is not."""
        try:
            from tokeymeter.engines.execution import halts as _halts
            snap = self.snapshot()
            event = _halts.HaltEvent(
                task_id=self.task_id, agent=self.agent, reason=reason,
                calls=snap.get("calls", 0),
                executed_calls=snap.get("executed_calls", snap.get("calls", 0)),
                spend_usd=float(snap.get("spend_usd") or 0.0),
                progress=snap.get("progress_novelty"),
                input_growth=self._input_growth(),
                enforced=bool(self.enforce),
                limit_value=allowed, observed_value=observed,
                message=str(message), timestamp=time.time())
            _halts._emit(event, self.on_halt)
        except Exception:                            # noqa: BLE001
            pass

    def _input_growth(self) -> Optional[float]:
        """Second-half input mean over first-half, for the alert text."""
        try:
            with self._lock:
                toks = [t for _, t in self._recent if t is not None]
            if len(toks) < 4:
                return None
            half = len(toks) // 2
            first = sum(toks[:half]) / half
            second = sum(toks[half:]) / (len(toks) - half)
            return (second - first) / first if first > 0 else None
        except Exception:
            return None

    def _check_locked(self, fingerprint: Optional[str], is_hit: bool):
        with self._lock:
            if self.envelope_usd is not None and not is_hit:
                hold = (self.reserve_usd if self.reserve_usd is not None
                        else self.max_observed_call_usd)
                if self.spend_usd + hold > self.envelope_usd:
                    kind = "would exceed" if hold > 0 else "has reached"
                    return (TaskEnvelopeExceeded, "envelope",
                            round(self.spend_usd, 6), self.envelope_usd,
                            f"task {self.task_id!r} has spent "
                            f"${self.spend_usd:.4f} and the next call {kind} "
                            f"its ${self.envelope_usd:.2f} envelope "
                            f"(holding ${hold:.4f} for it)")
            if self.stalled:
                w = self._recent
                fps = [r[0] for r in w if r[0] is not None]
                novelty = (len(set(fps)) / float(len(fps))) if fps else 0.0
                thr = self.min_novelty if self.min_novelty is not None else 0.25
                return (TaskStalled, "stalled", round(novelty, 4), thr,
                        f"task {self.task_id!r} has stopped making progress: "
                        f"{len(set(fps))} distinct responses in the last "
                        f"{len(fps)} calls while its input kept growing - it "
                        f"is paying more to learn nothing new")
            if self.max_calls is not None and self.calls >= self.max_calls:
                return (TaskCallLimitExceeded, "max_calls",
                        self.calls, self.max_calls,
                        f"task {self.task_id!r} reached its limit of "
                        f"{self.max_calls} calls")
            if (self.max_repeats is not None and fingerprint is not None
                    and self._fingerprints.get(fingerprint, 0) >= self.max_repeats):
                seen = self._fingerprints.get(fingerprint, 0)
                return (TaskLoopDetected, "max_repeats", seen, self.max_repeats,
                        f"task {self.task_id!r} repeated the same request "
                        f"{seen} times with no progress (fingerprint "
                        f"{fingerprint}) — this is a loop, not work")
            return None

    def _count_attempt(self, fingerprint: Optional[str]) -> None:
        """Count the attempt on this task only — ancestors count their own via
        the recursive check, so a nested call is never double-counted."""
        with self._lock:
            self.calls += 1
            if fingerprint is None:
                return
            fps = self._fingerprints
            if fingerprint in fps:
                fps[fingerprint] += 1
                return
            if len(fps) >= self._FINGERPRINT_CAP:
                # Evict singletons first — they are not loop evidence.
                singles = [k for k, v in fps.items() if v == 1]
                for k in singles:
                    del fps[k]
                self.fingerprints_evicted += len(singles)
                if len(fps) >= self._FINGERPRINT_CAP:
                    self.fingerprints_evicted += 1
                    return
            fps[fingerprint] = 1

    # -- accounting -----------------------------------------------------

    def record_progress(self, response_fp: Optional[str],
                        input_tokens, executed: bool) -> None:
        """Sample one EXECUTED call for stall detection, then evaluate.

        Returns the breach tuple when the task has stalled and enforcement is
        on, so the caller can raise on the NEXT pre-flight rather than
        mid-record — enforcement always happens before a call, never after.
        """
        if not executed:
            return
        try:
            tokens = int(input_tokens) if input_tokens is not None else None
        except (TypeError, ValueError):
            tokens = None
        # Progress rolls UP the whole chain, exactly as spend does. Sampling
        # only the innermost task would let a sub-agent spin forever under a
        # parent that has stall detection enabled — the same bypass that was
        # fixed for the envelope, and it has to be fixed here too or the
        # ceiling is only as deep as the innermost `with`.
        node = self
        while node is not None:
            if node.stall_window is not None:
                with node._lock:
                    node._recent.append((response_fp, tokens))
                    if node._stall_check_locked() is not None:
                        node.stalled = True
                        first_stall = node.halted_reason is None
                        if first_stall:
                            node.halted_reason = "stalled"
                    else:
                        first_stall = False
                if first_stall:
                    node._notify_halt("stalled", None, node.stall_window,
                                      f"task {node.task_id!r} has stopped "
                                      f"making progress")
            node = node.parent

    def record_call(self, fingerprint: Optional[str], cost_usd: float,
                    executed: bool) -> None:
        """Settle a completed call: add its ACTUAL cost.

        The call and its fingerprint were already counted pre-flight, so this
        only folds in spend — and only for an executed call, because a cache
        hit cost nothing upstream. Spend rolls up the whole parent chain so an
        inner task consumes its ancestors' envelopes too.
        """
        if not executed:
            return
        try:
            c = float(cost_usd)
        except (TypeError, ValueError):
            return
        if not math.isfinite(c) or c <= 0:
            return
        node = self
        while node is not None:
            with node._lock:
                node.spend_usd += c
                if c > node.max_observed_call_usd:
                    node.max_observed_call_usd = c
                if node.reserve_usd is not None and c > node.reserve_usd:
                    node.reserve_breaches += 1
            node = node.parent

    def _stall_check_locked(self):
        """Is the agent paying more and more to learn nothing new?

        TWO signals, and BOTH are required, because either alone is a false
        positive machine:

          novelty   distinct responses in the recent window. A stuck agent gets
                    the same answer back every time.
          growth    input tokens rising across that window. A conversational
                    agent accumulates history, so a stuck one costs more each
                    turn while learning nothing.

        Novelty alone would condemn a batch classifier: 500 documents that all
        return "APPROVED" have novelty 0.002 and are working perfectly. Its
        input does NOT grow — each document is independent — which is exactly
        what separates legitimate repetition from a stall.

        Growth alone would condemn every healthy conversational agent, whose
        context grows by design.
        """
        w = self._recent
        if self.stall_window is None or len(w) < w.maxlen:
            return None
        fps = [r[0] for r in w if r[0] is not None]
        if len(fps) < len(w):
            return None                       # incomplete sample; never guess
        novelty = len(set(fps)) / float(len(fps))
        threshold = self.min_novelty if self.min_novelty is not None else 0.25
        if novelty > threshold:
            return None
        half = len(w) // 2
        first = [r[1] for r in list(w)[:half] if r[1] is not None]
        second = [r[1] for r in list(w)[half:] if r[1] is not None]
        if not first or not second:
            return None
        if sum(second) / len(second) <= sum(first) / len(first):
            return None                       # input flat: legitimate repetition
        return (novelty, threshold, len(set(fps)), len(fps))

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            top = max(self._fingerprints.values()) if self._fingerprints else 0
            return {
                "task_id": self.task_id,
                "agent": self.agent,
                "matched_rules": list(self.matched_rules),
                "calls": self.calls,
                "spend_usd": round(self.spend_usd, 6),
                "envelope_usd": self.envelope_usd,
                "reserve_usd": self.reserve_usd,
                "max_observed_call_usd": round(self.max_observed_call_usd, 6),
                "reserve_breaches": self.reserve_breaches,
                "max_calls": self.max_calls,
                "max_repeats": self.max_repeats,
                "enforce": self.enforce,
                "distinct_fingerprints": len(self._fingerprints),
                "max_fingerprint_repeats": top,
                "fingerprints_evicted": self.fingerprints_evicted,
                "parent_task_id": self.parent.task_id if self.parent else None,
                "halted_reason": self.halted_reason,
                "stalled": self.stalled,
                # Novelty is computed over SCOREABLE responses only. Dividing
                # by every sample would report 0.0 — indistinguishable from a
                # total stall — for a task whose responses simply could not be
                # measured. None means "not scored", which is the truth.
                "progress_novelty": _novelty_or_none(self._recent),
                "responses_scored": sum(1 for r in self._recent if r[0]),
                "responses_sampled": len(self._recent),
            }


_TASK: contextvars.ContextVar[Optional[_TaskState]] = contextvars.ContextVar(
    "tokeymeter_task", default=None)


# ── public surface ──────────────────────────────────────────────────────

def set_task(state: Optional[_TaskState]) -> Optional[_TaskState]:
    """Bind a task state directly (None to clear); returns the previous value.
    Prefer the `task()` context manager — this exists for frameworks that
    cannot use a `with` block around their entry point."""
    prev = _TASK.get()
    _TASK.set(state)
    return prev


def get_task() -> Optional[_TaskState]:
    """The task state bound to the current context, or None."""
    return _TASK.get()


def current_agent() -> Optional[str]:
    """The agent bound to the current context, or None. Stamped on every record
    so an agent-conditioned rule can be simulated against real history."""
    st = _TASK.get()
    return st.agent if st is not None else None


def current_task_id() -> Optional[str]:
    """The bound task id, or None. This is what the record path stamps."""
    st = _TASK.get()
    return st.task_id if st is not None else None


def task_snapshot() -> Optional[Dict[str, object]]:
    """A copy of the current task's live accounting, or None outside a task.
    Useful in an exception handler to report what the task had consumed."""
    st = _TASK.get()
    return st.snapshot() if st is not None else None


@contextlib.contextmanager
def task(task_id: str, *, agent: Optional[str] = None,
         on_halt=None,
         env: Optional[str] = None, envelope: Optional[float] = None,
         reserve: Optional[float] = None, max_calls: Optional[int] = None,
         max_repeats: Optional[int] = None,
         stall_window: Optional[int] = None,
         min_novelty: Optional[float] = None,
         enforce: Optional[bool] = None) -> Iterator[_TaskState]:
    """Bind a task boundary for a scope, with optional ceilings.

    Args:
      task_id: an operator-supplied identifier — a ticket, job, or run id.
        Stamped on every record made inside the block.
      agent: the KIND of task ("support", "extract"). task_id identifies the
        instance; `agent` is what execution rules key on, so one policy line
        governs every ticket the support agent ever handles.
      env: deployment environment ("dev", "ci", "prod"). Defaults to
        TOKEYMETER_ENV. Lets a rule give CI a different ceiling from
        production without touching application code.
      envelope: maximum USD this task may spend.
      reserve: an upper bound on what ANY SINGLE call in this task can cost.
        Supplying it turns the envelope from a trigger into a HARD CAP: the
        worst case is held before each call, so the call that would cross the
        line is never made and final spend never exceeds `envelope`. Omit it
        and the task holds back the largest cost it has actually observed
        instead — unbounded on the first call, tightening after. If a real call
        ever costs more than `reserve`, the overrun is counted in
        `reserve_breaches` and shown in the snapshot rather than hidden.
      max_calls: maximum calls this task may make.
      max_repeats: maximum times one prompt fingerprint may repeat before the
        task is judged to be looping.
      enforce: when True, a breach raises. When False (the default) the breach
        is recorded and reported but the call proceeds — so a team can measure
        what a ceiling WOULD have stopped before it stops anything. Adopting
        the boundary is therefore risk-free; turning on the ceiling is a
        separate, deliberate decision.

    Always restores the prior binding, including on exception, so nested tasks
    and concurrent agents compose correctly.
    """
    tid = _validate_id(task_id)
    agent_id = _validate_id(agent) if agent is not None else None

    # Execution rules (S4-2) supply the limits this task did not declare, and
    # tighten the ones it did. Resolved ONCE here, never per call, so policy
    # costs nothing on the hot path. Fail-open: if anything goes wrong the
    # code-declared limits still apply — policy failing must never remove a
    # ceiling the application already set.
    matched_rules: tuple = ()
    resolved: Dict[str, Any] = {
        "envelope": envelope, "reserve": reserve, "max_calls": max_calls,
        "max_repeats": max_repeats, "stall_window": stall_window,
        "min_novelty": min_novelty,
    }
    if enforce is not None:
        resolved["enforce"] = enforce
    try:
        from tokeymeter.engines.governance import rules as _rules
        ctx = {
            "agent": agent_id,
            "env": env if env is not None else _rules.current_env(),
            "task_id": tid,
            "principal": _current_principal_safe(),
        }
        resolved, names = _rules.resolve_limits(ctx, resolved)
        matched_rules = tuple(names)
    except Exception:
        resolved = {k: v for k, v in resolved.items() if v is not None}

    state = _TaskState(
        task_id=tid,
        agent=agent_id,
        matched_rules=matched_rules,
        envelope_usd=_validate_limit("envelope", resolved.get("envelope")),
        reserve_usd=_validate_limit("reserve", resolved.get("reserve")),
        max_calls=_validate_limit("max_calls", resolved.get("max_calls"),
                                  integer=True),
        max_repeats=_validate_limit("max_repeats", resolved.get("max_repeats"),
                                    integer=True),
        enforce=bool(resolved.get("enforce", False)),
        stall_window=_validate_limit("stall_window",
                                     resolved.get("stall_window"), integer=True),
        min_novelty=resolved.get("min_novelty"),
        parent=_TASK.get(),          # nested tasks nest their limits too
    )
    # A per-task handler runs alongside any registered globally. Validated here
    # so a typo surfaces at the `with`, not silently at halt time when it is
    # far too late to be useful.
    if on_halt is not None:
        if not callable(on_halt):
            raise TypeError(f"on_halt must be callable, got {on_halt!r}")
        state.on_halt = on_halt
    token = _TASK.set(state)
    try:
        yield state
    finally:
        _TASK.reset(token)


@contextlib.contextmanager
def bind_task(state) -> Iterator[None]:
    """Attach an EXISTING task to the current context — the supported way to
    carry a task boundary into a worker thread.

    A plain thread starts with a fresh context and would otherwise see no task,
    so its calls would escape the ceiling silently. Hand the state object from
    `with task(...) as t` to the worker and bind it there. Restores the prior
    binding on exit, including on exception.

    Passing None is legal and binds nothing, so a worker written this way is
    safe to call from outside a task too.
    """
    if state is not None and not isinstance(state, _TaskState):
        raise TypeError(
            "bind_task expects the state object yielded by task(...), "
            f"got {type(state).__name__}")
    token = _TASK.set(state)
    try:
        yield
    finally:
        _TASK.reset(token)


# ── fingerprint ─────────────────────────────────────────────────────────

_FINGERPRINT_PREFIX = "sha256:"


# Responses are hashed with a bounded prefix. A model response is normally a
# few KB, but a pathological one can be megabytes, and hashing the whole thing
# on every call would put an unbounded cost on the hot path. 64KB is far beyond
# the point where two genuinely different answers still agree.
_RESPONSE_HASH_LIMIT = 65536


# WHAT WE ARE WILLING TO CALL CONTENT.
#
# Scoring the wrong thing is the failure mode that matters here. Two examples,
# both found by attacking this function:
#
#   a generator     -> "<generator object g at 0x7f9968...>" — the address
#                      changes every call
#   an SDK response -> "ChatCompletion(id='chatcmpl-a3f9...', ...)" — the
#                      request id changes every call
#
# Either would give two IDENTICAL responses two DIFFERENT fingerprints, so a
# completely stuck agent would score a perfect 1.00 and nothing would warn. A
# metric that lies is worse than a missing one.
#
# So only shapes we KNOW are content get scored: text, bytes, and a sequence of
# those (what `cache_stream` assembles). Anything else is scored only through
# the caller's own `extract_text`, which is this codebase's existing idiom for
# "pull the content out of a provider object" and is already threaded to the
# record path. With no extractor, the answer is "not scored" — refusing costs a
# column entry, guessing costs the whole signal's credibility.
_MAX_CHUNKS_SCANNED = 4096


def _content_bytes(result) -> Optional[bytes]:
    """The bytes to hash, or None when this shape is not safely content."""
    if isinstance(result, (bytes, bytearray)):
        return bytes(result[:_RESPONSE_HASH_LIMIT])
    if isinstance(result, str):
        return result[:_RESPONSE_HASH_LIMIT].encode("utf-8", "replace")
    if isinstance(result, (list, tuple)):
        # Assembled stream chunks. Every element must itself be content, or the
        # container inherits whatever instability the element carries.
        if len(result) > _MAX_CHUNKS_SCANNED:
            return None
        parts = []
        total = 0
        for chunk in result:
            if isinstance(chunk, str):
                b = chunk.encode("utf-8", "replace")
            elif isinstance(chunk, (bytes, bytearray)):
                b = bytes(chunk)
            else:
                return None
            parts.append(b)
            total += len(b)
            if total >= _RESPONSE_HASH_LIMIT:
                break
        joined = b"".join(parts)[:_RESPONSE_HASH_LIMIT]
        return joined or None
    return None


def fingerprint_response(result, extract_response_text=None) -> Optional[str]:
    """Non-reversible digest of a response's CONTENT, for progress detection.

    Returns None — meaning "not scored" — whenever the response cannot be
    reduced to stable content. Scoreable: text, bytes, and sequences of those,
    which covers the assembled chunks `cache_stream` produces. A provider
    object is scoreable only via `extract_text`, because its text form
    typically carries a per-call request id that would fabricate novelty.

    A live iterator is never touched: reading it would consume the caller's
    stream.

    Never raises and never blocks a call — an exploding object yields None and
    the task loses one sample rather than the request.
    """
    try:
        if result is None:
            return None
        raw = _content_bytes(result)
        if raw is None:
            # A live iterator must not be consumed to be measured.
            if hasattr(result, "__next__") or hasattr(result, "__anext__"):
                return None
            if extract_response_text is not None:
                try:
                    text = extract_response_text(result)
                except Exception:
                    return None
                if isinstance(text, str) and text:
                    raw = text[:_RESPONSE_HASH_LIMIT].encode("utf-8", "replace")
        if not raw:
            return None
        return _FINGERPRINT_PREFIX + hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return None


def fingerprint_of(cache_key: Optional[str]) -> Optional[str]:
    """Derive the record's prompt fingerprint from a cache key.

    The cache key is already a SHA-256 digest of the call arguments (plus any
    tenant/namespace/lineage prefixes), so this is a hash of a hash: no content
    can survive it, and identical calls inside one task collapse to the same
    value — which is exactly what loop detection needs.

    Truncated to the `sha256:<12hex>` form the codebase already uses for
    non-reversible correlation tokens, keeping records small while leaving
    collisions vanishingly unlikely within a single task.
    """
    if not cache_key or not isinstance(cache_key, str):
        return None
    digest = hashlib.sha256(cache_key.encode("utf-8", "replace")).hexdigest()[:12]
    return _FINGERPRINT_PREFIX + digest


# ── record-path helpers (never raise except on deliberate enforcement) ──

def before_call(cache_key: Optional[str],
                is_hit: bool = False) -> Optional[str]:
    """Called on the request path before the wrapped function runs.

    Returns the fingerprint for this call (or None outside a task / without a
    key). Raises TaskLimitExceeded only when a declared limit is breached and
    enforcement is on — every other failure is swallowed, because task
    accounting must never be the reason a request fails.
    """
    st = _TASK.get()
    if st is None:
        return None
    try:
        fp = fingerprint_of(cache_key)
    except Exception:
        return None
    try:
        st.check_before_call(fp, is_hit)
    except TaskLimitExceeded:
        raise
    except Exception:
        return fp
    return fp


def after_call(fingerprint: Optional[str], cost_usd: float,
               executed: bool, response_fp: Optional[str] = None,
               input_tokens=None) -> None:
    """Called once a call has completed and its cost is known. Never raises."""
    st = _TASK.get()
    if st is None:
        return
    try:
        st.record_call(fingerprint, cost_usd, executed)
        st.record_progress(response_fp, input_tokens, executed)
    except Exception:
        pass
