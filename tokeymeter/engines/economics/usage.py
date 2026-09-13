"""Reported-usage channel (T1.1) — provider truth over estimation.

The provider's OWN token counts (`response.usage`) are the ground truth for
what a miss actually cost. The SDK wrappers set them here, in-context, right
after the real call returns; the decorator consumes them when it builds the
record. Where no reported usage exists (bare callables, hits — where tokens
quantify an AVOIDED cost), the chars÷4 estimate remains, and every record
says which it carries via `token_source`: "reported" | "estimated".
Consume-once semantics: a stale value can never bleed into the next call.
"""
from __future__ import annotations
import contextvars
from typing import Optional, Tuple

_REPORTED: contextvars.ContextVar[Optional[Tuple[int, int]]] = \
    contextvars.ContextVar("tokeymeter_reported_usage", default=None)


def set_reported_usage(input_tokens, output_tokens) -> None:
    """Called by SDK wrappers with the provider's reported counts. Fail-safe:
    malformed values are ignored (the estimate remains)."""
    try:
        i, o = int(input_tokens), int(output_tokens)
        if i >= 0 and o >= 0:
            _REPORTED.set((i, o))
    except (TypeError, ValueError):
        pass


def consume_reported_usage() -> Optional[Tuple[int, int]]:
    v = _REPORTED.get()
    if v is not None:
        _REPORTED.set(None)
    return v


def peek_reported_usage() -> Optional[Tuple[int, int]]:
    """Read the reported usage WITHOUT consuming it. Used by the envelope
    meta-stamping path (S1.1), which runs before the record path and must not
    disturb the record's consume-once semantics: peek here, consume there,
    same tuple both times."""
    return _REPORTED.get()


# ── queue wait (T-selfhost): time a miss spent WAITING before execution ──
# Self-hosted serving layers (vLLM, TGI) expose per-request queue time — the
# interval between admission and the first compute step. It is the signal
# behind Queue Economics and SLO cost: waiting is capacity you paid for and
# did not use. The SDK wrappers read it from the serving response/metrics and
# set it here, in-context, right after the real call; the decorator consumes
# it onto the miss record. Consume-once, same as reported usage: a stale value
# can never bleed into the next record. Absent (None) => the record's
# queue_wait_ms stays None (unknown), never a fabricated zero.
_QUEUE_WAIT_MS: contextvars.ContextVar[Optional[float]] = \
    contextvars.ContextVar("tokeymeter_queue_wait_ms", default=None)


def set_queue_wait_ms(ms) -> None:
    """Called by SDK wrappers with the serving layer's reported queue wait
    (milliseconds). Fail-safe: malformed or negative values are ignored (the
    record simply carries queue_wait_ms=None)."""
    try:
        v = float(ms)
        if v >= 0.0:
            _QUEUE_WAIT_MS.set(v)
    except (TypeError, ValueError):
        pass


def consume_queue_wait_ms() -> Optional[float]:
    v = _QUEUE_WAIT_MS.get()
    if v is not None:
        _QUEUE_WAIT_MS.set(None)
    return v


# Known locations where self-hosted serving layers surface per-request queue
# time. Checked in order; the FIRST that yields a non-negative number wins.
# Deliberately conservative: we read only fields a serving layer explicitly
# populates. We do NOT derive queue wait from client-side wall-clock (that
# measures network + our own overhead, not the serving queue) — an unknown
# queue wait must read as None, never as a fabricated number. Extend this list
# as stacks expose the signal; never loosen it to guessing.
#   - vLLM (recent builds) can attach scheduler timings under
#     response.usage or an extension block; keys vary by version, so we probe
#     a small set of documented names rather than assume one schema.
_QUEUE_WAIT_KEYS = (
    "queue_time_ms", "queue_wait_ms", "time_in_queue_ms",
    "queue_time", "time_in_queue",          # some builds report SECONDS here
)
_QUEUE_WAIT_SECOND_KEYS = frozenset({"queue_time", "time_in_queue"})


def extract_queue_wait_ms(resp) -> Optional[float]:
    """Best-effort read of serving-layer queue time from a response object,
    normalized to milliseconds. Returns None when the stack did not surface it
    (the honest default — an absent measurement, not zero). Never raises.

    Tolerant of shape: checks the response's ``usage`` object and any
    ``extra``/``metrics`` mapping for a small set of documented field names.
    Second-valued fields are converted to ms; ms-valued fields pass through.
    """
    try:
        candidates = []
        u = getattr(resp, "usage", None)
        if u is not None:
            candidates.append(u)
        for attr in ("extra", "metrics", "timings"):
            m = getattr(resp, attr, None)
            if m is not None:
                candidates.append(m)
        for holder in candidates:
            for key in _QUEUE_WAIT_KEYS:
                val = None
                if isinstance(holder, dict):
                    val = holder.get(key)
                else:
                    val = getattr(holder, key, None)
                if val is None:
                    continue
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if num < 0:
                    continue
                if key in _QUEUE_WAIT_SECOND_KEYS:
                    num *= 1000.0
                return num
        return None
    except Exception:
        return None
