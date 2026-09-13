# Tokeymeter Exception Policy

Tokeymeter sits between an application and its LLM provider, so its first duty is
**never to take down the caller's app**. But its product promise is *trustworthy,
content-blind, no silent failures*. Those two duties only conflict if exceptions
are handled carelessly. This policy reconciles them.

## The three rules

**1. I/O / backend / untrusted-input boundaries fail OPEN — and are visible.**
A cache backend, cipher, Redis connection, redactor, compressor, or user-supplied
callback can fail for reasons outside our control. There, a broad `except` is
correct: the wrapped call still succeeds with degraded behavior. But the failure
must not be *silent* — it emits a **source-specific degraded event**
(`store.get`, `store.set`, `sqlite_write`, `redis_write`, `redis_unhealthy`,
`redis_serialize`, `redis_encrypt`, `redactor`, `compression_fallback`,
`semantic_vec_disabled`, …) so operators see exactly what degraded and how often
via `tokeymeter.degraded.degraded_counts()`.

**2. Pure-logic paths fail SAFE, and surface unexpected errors.**
Internal logic (cache-key composition, token counting, routing, reduction math)
should not hide programmer bugs. The pattern:

```python
try:
    ...                      # the logic
except (TypeError, ValueError, AttributeError):
    return SAFE_FALLBACK     # EXPECTED conditions — handle quietly
except Exception as e:
    _internal_error("where", e)   # UNEXPECTED — surface the bug…
    return SAFE_FALLBACK          # …but still fail safe
```

- *Expected* exceptions (exotic argument types, malformed inputs) are caught
  narrowly and handled quietly — they are not bugs.
- *Unexpected* exceptions are a Tokeymeter bug. They are surfaced as an
  `internal_error:<where>` degraded event so they can be told apart from a
  backend/ops failure (which is expected and recoverable) and reported — while
  the call still succeeds.

**3. "Fail safe" in the cache means MISS, never a wrong hit.**
The one outcome Tokeymeter must never produce is a *silent wrong answer* — e.g.
serving one tenant's cached response to another. So whenever key composition,
lookup, or isolation logic is uncertain, it returns a **miss** (recompute the
real answer) rather than a possibly-wrong hit. Correctness dominates the
optimization: a missed cache hit costs money; a wrong hit costs trust.

## Why the distinction matters to operators

A solo developer and a bank both need to tell two situations apart:

- `redis_unhealthy` / `store.get` — *"my backend is having a moment."* Expected,
  recoverable, an ops signal. Tokeymeter degraded gracefully; the app kept working.
- `internal_error:*` — *"Tokeymeter itself hit a bug it didn't anticipate."*
  Should be ~never. If it appears, it is worth a bug report. The app still kept
  working (fail-safe), but the behavior should be investigated.

Collapsing both into a silent `except Exception: pass` would have erased that
distinction — and a silently-disabled cache or a skewed savings number is exactly
the kind of thing that erodes enterprise trust without ever raising an alarm.

## Checklist for new `except` blocks

- Is this an I/O / backend / user-callback boundary? → broad catch OK; emit a
  source-specific degraded event.
- Is this internal logic? → catch expected types narrowly; route unexpected ones
  through `_internal_error(...)`; never swallow at debug level only.
- Could the fallback ever be *wrong* (not just degraded)? → it must be a cache
  **miss** / recompute instead.
- Does the handler emit or log enough that an operator could see it happened? If
  not, it is too silent.
