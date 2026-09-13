"""Halt notification — routing a halt to wherever a team already looks.

WHY A HOOK AND NOT AN INTEGRATION
---------------------------------
The obvious ask is "send halts to Slack" — and the obvious build is a Slack
client. That would be a mistake. It means an HTTP dependency, an auth scheme
and a token store in a package whose `dependencies` list is empty, and empty is
exactly why a security team can approve this in an afternoon. Multiply by Jira,
Linear, Teams and PagerDuty and the library now talks to the internet in four
ways, each of which breaks when somebody's API changes.

So this ships the primitive instead. A handler is a function. Every integration
is a few lines written by the person who already has those credentials
configured, and none of them are ours to maintain:

    def notify(halt):
        urlopen(SLACK_WEBHOOK, json.dumps({"text": halt.summary()}).encode())

    tokeymeter.on_halt(notify)

IT MUST FIRE EVEN WHEN THE CALLER SWALLOWS THE EXCEPTION
--------------------------------------------------------
This is the requirement that shapes everything here. TaskLimitExceeded
subclasses RuntimeError, so ordinary agent retry code — `except Exception:
continue` — swallows it. Measured on a stuck agent: 22 halts swallowed, the
task correctly bounded at 8 calls, and nothing upstream any the wiser.

A hook that rode on the exception would therefore report nothing in precisely
the case an operator most needs to hear about. So notification fires where the
halt is DECIDED, not where it is raised, and fires whether or not enforcement
is even on: a team running in record-only mode still wants to know their agent
went nowhere.

EXACTLY ONCE PER TASK
---------------------
A halted task keeps being checked on every subsequent call. Notifying per check
would turn one stuck ticket into forty identical alerts, and an alert channel
that cries wolf gets muted — which costs the operator the signal entirely. The
`halted_reason` flag is set once under a lock, so notification is emitted from
that same one-time transition.

A HANDLER CANNOT BREAK A REQUEST
--------------------------------
Handlers run inside the caller's request path, so a slow webhook is the
caller's latency and a raising handler would be the caller's outage. Every
handler is wrapped: exceptions are swallowed and counted, and a handler that
fails repeatedly is retired rather than left to fail forever. Notification is
best-effort; execution is not.

CONTENT-BLIND, LIKE EVERYTHING ELSE
-----------------------------------
A HaltEvent carries identifiers, counts and money. No prompt, no response, no
fingerprints — the same 26-field discipline the ledger has. Anything sent to
Slack by a handler is therefore safe to put in a channel by construction.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

__all__ = ["HaltEvent", "on_halt", "remove_halt_handler",
           "clear_halt_handlers", "halt_handler_count"]

log = logging.getLogger("tokeymeter.halts")

# A handler that keeps raising is retired. Left alone it would burn latency on
# every halt forever, and the operator would never learn why nothing arrived.
_MAX_HANDLER_FAILURES = 5


class HaltEvent:
    """What stopped, why, and what it had cost by then.

    Deliberately a small, flat, content-blind object: it is designed to be
    serialised straight into an alert without anyone auditing what might be
    inside it.
    """

    __slots__ = ("task_id", "agent", "reason", "calls", "executed_calls",
                 "spend_usd", "progress", "input_growth", "enforced",
                 "limit_value", "observed_value", "message", "timestamp")

    def __init__(self, *, task_id: str, agent: Optional[str], reason: str,
                 calls: int, executed_calls: int, spend_usd: float,
                 progress: Optional[float], input_growth: Optional[float],
                 enforced: bool, limit_value: Any, observed_value: Any,
                 message: str, timestamp: float) -> None:
        self.task_id = task_id
        self.agent = agent
        self.reason = reason
        self.calls = calls
        self.executed_calls = executed_calls
        self.spend_usd = spend_usd
        self.progress = progress
        self.input_growth = input_growth
        # False means the ceiling was reached in record-only mode: the task
        # was NOT stopped. An alert that does not say which is misleading.
        self.enforced = enforced
        self.limit_value = limit_value
        self.observed_value = observed_value
        self.message = message
        self.timestamp = timestamp

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def summary(self) -> str:
        """One line, ready to send. States plainly whether it was stopped."""
        who = f"{self.agent}/" if self.agent else ""
        verb = "stopped" if self.enforced else "would have been stopped"
        line = (f"{who}{self.task_id} {verb} after {self.calls} calls "
                f"(${self.spend_usd:.4f}) - {self.reason}")
        if self.reason == "stalled" and self.progress is not None:
            line += f", progress {self.progress:.2f}"
            if self.input_growth is not None:
                line += f", input +{self.input_growth:.0%}"
        return line

    def __repr__(self) -> str:                      # pragma: no cover
        return f"<HaltEvent {self.summary()}>"


_LOCK = threading.RLock()
_HANDLERS: List[Callable[[HaltEvent], None]] = []
_FAILURES: Dict[int, int] = {}


def on_halt(handler: Callable[[HaltEvent], None]
            ) -> Callable[[HaltEvent], None]:
    """Register a handler called once per halted task, process-wide.

    Registered once by a platform team rather than per call site — that is the
    whole point, since editing every agent to add an alert is the reason alerts
    do not get added. Returns the handler, so it also works as a decorator.
    """
    if not callable(handler):
        raise TypeError(f"halt handler must be callable, got {handler!r}")
    with _LOCK:
        if handler not in _HANDLERS:
            _HANDLERS.append(handler)
            _FAILURES.pop(id(handler), None)
    return handler


def remove_halt_handler(handler: Callable[[HaltEvent], None]) -> bool:
    with _LOCK:
        try:
            _HANDLERS.remove(handler)
            _FAILURES.pop(id(handler), None)
            return True
        except ValueError:
            return False


def clear_halt_handlers() -> None:
    with _LOCK:
        _HANDLERS.clear()
        _FAILURES.clear()


def halt_handler_count() -> int:
    with _LOCK:
        return len(_HANDLERS)


def _emit(event: HaltEvent, task_handler=None) -> None:
    """Deliver to the task's own handler and every global one. Never raises.

    Order is deliberate: the task's own handler runs first, because it is the
    most specific and the most likely to be doing something the caller cares
    about right now.
    """
    handlers: List[Callable[[HaltEvent], None]] = []
    if task_handler is not None:
        handlers.append(task_handler)
    with _LOCK:
        handlers.extend(_HANDLERS)
    for h in handlers:
        try:
            h(event)
        except Exception as exc:                    # noqa: BLE001
            key = id(h)
            with _LOCK:
                n = _FAILURES.get(key, 0) + 1
                _FAILURES[key] = n
                retire = n >= _MAX_HANDLER_FAILURES and h in _HANDLERS
                if retire:
                    _HANDLERS.remove(h)
                    _FAILURES.pop(key, None)
            log.warning("tokeymeter: halt handler %r failed (%s)",
                        getattr(h, "__name__", h), exc)
            if retire:
                log.error("tokeymeter: retired halt handler %r after %d "
                          "consecutive failures - halts will no longer be "
                          "delivered to it", getattr(h, "__name__", h),
                          _MAX_HANDLER_FAILURES)
        else:
            with _LOCK:
                _FAILURES.pop(id(h), None)
