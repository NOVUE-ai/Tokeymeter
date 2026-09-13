"""
Cache observability events.

Tokeymeter emits a CacheEvent on every lookup decision, every store, and every
error. Subscribers register callbacks; events are dispatched synchronously
in order of subscription.

Design principles:
  - **Fail-open**: subscriber errors NEVER crash the cache. Always caught
    and logged. Observability code has bugs; cache code should keep
    working when it does.
  - **No backpressure**: subscribers are expected to be fast. Slow work
    (HTTP, DB writes) should be queued by the subscriber itself.
  - **Snapshot iteration**: we iterate a list snapshot so a subscriber
    can subscribe/unsubscribe other handlers mid-emit without breaking.

Wiring examples:

    import tokeymeter

    # Datadog counter
    @tokeymeter.events.on_event
    def to_datadog(event):
        tag = f"hit_type:{event.hit_type or 'miss'}"
        statsd.increment("tokeymeter.lookup", tags=[tag, f"model:{event.model}"])

    # Custom log line for misses only
    @tokeymeter.events.on_event
    def log_misses(event):
        if not event.hit:
            log.info("cache miss key=%s prompt=%r",
                     event.cache_key[:8] if event.cache_key else None,
                     event.prompt_preview)

    # Debug helper: look at the most recent event
    result = await ask("hello")
    print(tokeymeter.events.last_event())
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger("tokeymeter.events")


@dataclass
class CacheEvent:
    """A single cache lookup or store event.

    All fields are populated on a best-effort basis; fields not relevant
    to a particular event type may be None or zero.
    """
    timestamp: float
    event_type: str           # "lookup_hit" | "lookup_miss" | "store" | "error"
    hit: bool
    hit_type: Optional[str]   # "exact" | "semantic" | "single_flight"
                              # | "shadow_exact" | "shadow_semantic"
                              # | "shadow_single_flight" | None
    model: str
    cache_key: Optional[str]
    prompt_preview: Optional[str]  # first ~200 chars, REDACTED if redactor used
    latency_ms: float
    estimated_cost_usd: float      # what this call cost (miss)
                                    # or would have cost (hit / shadow hit)
    input_tokens: int
    output_tokens: int
    shadow: bool = False
    tag: Optional[str] = None
    function_name: Optional[str] = None
    principal: Optional[str] = None   # v0.14 identity binding
    token_source: Optional[str] = None  # v0.14 T1.1 reported|estimated
    endpoint_identity: Optional[str] = None  # v0.14 self-host: declared endpoint id
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


_subscribers_lock = threading.Lock()
_subscribers: List[Callable[[CacheEvent], None]] = []
_last_event: Optional[CacheEvent] = None
_last_event_lock = threading.Lock()


def subscribe(callback: Callable[[CacheEvent], None]) -> Callable[[CacheEvent], None]:
    """Register a function to be called on every cache event.

    Idempotent: registering the same callback twice has no effect beyond
    the first registration. Returns the callback (so it can be used as
    a decorator: `@tokeymeter.events.subscribe`).
    """
    with _subscribers_lock:
        if callback not in _subscribers:
            _subscribers.append(callback)
    return callback


# Decorator alias
on_event = subscribe


def unsubscribe(callback: Callable[[CacheEvent], None]) -> None:
    """Remove a previously-registered subscriber. No-op if not subscribed."""
    with _subscribers_lock:
        try:
            _subscribers.remove(callback)
        except ValueError:
            pass


def clear_subscribers() -> None:
    """Remove all subscribers. Useful for tests."""
    global _last_event
    with _subscribers_lock:
        _subscribers.clear()
    with _last_event_lock:
        _last_event = None


def subscriber_count() -> int:
    with _subscribers_lock:
        return len(_subscribers)


def emit(event: CacheEvent) -> None:
    """Dispatch an event to all subscribers. Subscriber errors are caught.

    This function NEVER raises. If something goes wrong with the event
    machinery itself, it's logged and returns silently.
    """
    global _last_event
    try:
        with _last_event_lock:
            _last_event = event
        # Snapshot under lock; iterate without holding the lock so a
        # subscriber can re-subscribe / unsubscribe without deadlock.
        with _subscribers_lock:
            snapshot = list(_subscribers)
        for sub in snapshot:
            try:
                sub(event)
            except Exception as e:
                log.debug("tokeymeter.events: subscriber %r raised: %s", sub, e)
    except Exception as e:  # pragma: no cover
        log.debug("tokeymeter.events: emit machinery error: %s", e)


def last_event() -> Optional[CacheEvent]:
    """Return the most recently emitted event, or None if none yet.

    Useful for debugging a specific call: invoke your function, then
    `tokeymeter.events.last_event()` to see exactly what the cache did.
    """
    with _last_event_lock:
        return _last_event
