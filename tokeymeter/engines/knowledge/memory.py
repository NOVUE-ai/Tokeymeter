"""
Conversation memory layer (v0.7).

The wedge: in a chat app, every turn re-sends the entire prior conversation
as context. Total context tokens grow QUADRATICALLY with turn count. A
20-turn conversation pays for the prior history 20 times.

Tokeymeter memory turns that into LINEAR growth: oldest turns get summarized
once into a compact representation; recent turns stay full-fidelity. The
summary is itself cached — re-summarization only happens when the buffer
overflows again.

Pieces:
  - Turn:             a single (user, assistant) exchange
  - MemoryStore:      where turns live (InMemory or SQLite)
  - Summarizer:       turns -> compact str. Pluggable.
  - ConversationMemory: orchestrates buffer + summary + fidelity
  - with_memory:      decorator that wires it into your call

Design contract:
  - Fail-open everywhere. If the store breaks or the summarizer raises,
    the wrapped call still works (just without memory benefit that call).
  - Per-session async lock prevents corrupted concurrent writes.
  - Summary is cached by (session_id, buffer_state_hash). Only invalidated
    when the buffer state changes.
  - fidelity_rate audit: sample R% of calls, run with FULL history AND
    summarized history, log output similarity. Same pattern as v0.6's
    verify_rate.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union

log = logging.getLogger("tokeymeter.memory")

# Set once we've warned that the sync wrapper was used from a running event loop,
# so the advisory fires at most once per process rather than on every call.
_WARNED_SYNC_IN_LOOP = False


def _emit_degraded(source: str, error: BaseException) -> None:
    """Surface a memory-store boundary failure as a degraded event so a fail-open
    swallow is VISIBLE in metrics rather than silent at debug level. These paths
    stay fail-open (the wrapped call still succeeds with a safe fallback) per
    docs/EXCEPTION_POLICY — this only adds observability. Never raises."""
    try:
        from tokeymeter.engines.reliability.degraded import emit_degraded
        emit_degraded(source, error)
    except Exception:
        pass


# ============================================================
#                       Data model
# ============================================================

@dataclass
class Turn:
    """A single user/assistant exchange."""
    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def token_estimate(self) -> int:
        # Same heuristic as the rest of Tokeymeter (chars/4)
        return max(1, (len(self.user) + len(self.assistant)) // 4)


# ============================================================
#                       MemoryStore
# ============================================================

class MemoryStore(Protocol):
    """Where conversation turns and summaries live."""
    def append_turn(self, session_id: str, turn: Turn) -> None: ...
    def get_turns(self, session_id: str) -> List[Turn]: ...
    def get_summary(self, session_id: str) -> Optional[str]: ...
    def set_summary(self, session_id: str, summary: str, buffer_hash: str) -> None: ...
    def get_summary_buffer_hash(self, session_id: str) -> Optional[str]: ...
    def clear_session(self, session_id: str) -> None: ...
    def list_sessions(self) -> List[str]: ...
    def trim_turns(self, session_id: str, keep_last_n: int) -> None: ...


class InMemoryMemoryStore:
    """Thread-safe in-memory store. Lost on process exit. Best for tests."""

    def __init__(self):
        self._turns: Dict[str, List[Turn]] = {}
        self._summaries: Dict[str, tuple] = {}  # session_id -> (summary, buffer_hash)
        self._lock = threading.Lock()

    def append_turn(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            self._turns.setdefault(session_id, []).append(turn)

    def get_turns(self, session_id: str) -> List[Turn]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def get_summary(self, session_id: str) -> Optional[str]:
        with self._lock:
            entry = self._summaries.get(session_id)
            return entry[0] if entry else None

    def set_summary(self, session_id: str, summary: str, buffer_hash: str) -> None:
        with self._lock:
            self._summaries[session_id] = (summary, buffer_hash)

    def get_summary_buffer_hash(self, session_id: str) -> Optional[str]:
        with self._lock:
            entry = self._summaries.get(session_id)
            return entry[1] if entry else None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)
            self._summaries.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        with self._lock:
            keys = set(self._turns.keys()) | set(self._summaries.keys())
            return sorted(keys)

    def trim_turns(self, session_id: str, keep_last_n: int) -> None:
        """Drop all but the most recent keep_last_n turns for a session."""
        if keep_last_n < 0:
            return
        with self._lock:
            turns = self._turns.get(session_id)
            if turns is not None and len(turns) > keep_last_n:
                self._turns[session_id] = turns[-keep_last_n:] if keep_last_n else []


class SQLiteMemoryStore:
    """Durable file-backed memory store.

    Schema:
        memory_turns:    session_id, position, user_text, assistant_text,
                         timestamp, metadata_json
        memory_summary:  session_id PK, summary, buffer_hash, updated_at
    """

    def __init__(self, path: Optional[str] = None):
        from tokeymeter import paths as _paths
        self._path = os.path.expanduser(
            path if path is not None else _paths.state_path("memory.db"))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        # check_same_thread=False so async/worker threads can use it
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS memory_turns (
                    session_id TEXT NOT NULL,
                    position   INTEGER NOT NULL,
                    user_text  TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    timestamp  REAL NOT NULL,
                    metadata_json TEXT,
                    PRIMARY KEY (session_id, position)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS memory_summary (
                    session_id TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    buffer_hash TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

    def append_turn(self, session_id: str, turn: Turn) -> None:
        try:
            with self._lock, self._conn() as c:
                cur = c.execute(
                    "SELECT COALESCE(MAX(position), -1) FROM memory_turns WHERE session_id = ?",
                    (session_id,),
                )
                pos = cur.fetchone()[0] + 1
                c.execute(
                    "INSERT INTO memory_turns (session_id, position, user_text, "
                    "assistant_text, timestamp, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, pos, turn.user, turn.assistant,
                     turn.timestamp, json.dumps(turn.metadata) if turn.metadata else None),
                )
        except sqlite3.Error as e:
            log.debug("memory: append_turn sqlite error: %s", e)

    def get_turns(self, session_id: str) -> List[Turn]:
        try:
            with self._lock, self._conn() as c:
                rows = c.execute(
                    "SELECT user_text, assistant_text, timestamp, metadata_json "
                    "FROM memory_turns WHERE session_id = ? ORDER BY position ASC",
                    (session_id,),
                ).fetchall()
        except sqlite3.Error as e:
            log.debug("memory: get_turns sqlite error: %s", e)
            return []

        result: List[Turn] = []
        for u, a, ts, m in rows:
            try:
                meta = json.loads(m) if m else {}
            except json.JSONDecodeError:
                meta = {}
            result.append(Turn(user=u, assistant=a, timestamp=ts, metadata=meta))
        return result

    def get_summary(self, session_id: str) -> Optional[str]:
        try:
            with self._lock, self._conn() as c:
                row = c.execute(
                    "SELECT summary FROM memory_summary WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row[0] if row else None
        except sqlite3.Error:
            return None

    def set_summary(self, session_id: str, summary: str, buffer_hash: str) -> None:
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT INTO memory_summary (session_id, summary, buffer_hash, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "    summary = excluded.summary, "
                    "    buffer_hash = excluded.buffer_hash, "
                    "    updated_at = excluded.updated_at",
                    (session_id, summary, buffer_hash, time.time()),
                )
        except sqlite3.Error as e:
            log.debug("memory: set_summary sqlite error: %s", e)

    def get_summary_buffer_hash(self, session_id: str) -> Optional[str]:
        try:
            with self._lock, self._conn() as c:
                row = c.execute(
                    "SELECT buffer_hash FROM memory_summary WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row[0] if row else None
        except sqlite3.Error:
            return None

    def clear_session(self, session_id: str) -> None:
        try:
            with self._lock, self._conn() as c:
                c.execute("DELETE FROM memory_turns WHERE session_id = ?", (session_id,))
                c.execute("DELETE FROM memory_summary WHERE session_id = ?", (session_id,))
        except sqlite3.Error as e:
            log.debug("memory: clear_session sqlite error: %s", e)

    def list_sessions(self) -> List[str]:
        try:
            with self._lock, self._conn() as c:
                rows = c.execute(
                    "SELECT DISTINCT session_id FROM memory_turns "
                    "UNION SELECT session_id FROM memory_summary "
                    "ORDER BY session_id"
                ).fetchall()
                return [r[0] for r in rows]
        except sqlite3.Error:
            return []

    def trim_turns(self, session_id: str, keep_last_n: int) -> None:
        """Delete all but the most recent keep_last_n turns for a session.

        Positions are monotonic, so we keep the highest keep_last_n positions
        and delete the rest. Positions are NOT renumbered (append uses MAX+1),
        so ordering remains correct.
        """
        if keep_last_n < 0:
            return
        try:
            with self._lock, self._conn() as c:
                row = c.execute(
                    "SELECT MAX(position) FROM memory_turns WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if not row or row[0] is None:
                    return
                threshold = row[0] - keep_last_n + 1  # delete positions strictly below
                c.execute(
                    "DELETE FROM memory_turns WHERE session_id = ? AND position < ?",
                    (session_id, threshold),
                )
        except sqlite3.Error as e:
            log.debug("memory: trim_turns sqlite error: %s", e)


# ============================================================
#                       Summarizer
# ============================================================

class Summarizer(Protocol):
    """Reduces a list of turns to a single compact string.

    Implementations MAY be async. ConversationMemory awaits them either way.
    """
    def summarize(self, turns: List[Turn]) -> Union[str, Awaitable[str]]: ...


class TruncationSummarizer:
    """Zero-dependency summarizer. Deterministic. No LLM calls.

    Strategy: produce a header with the first turn (which usually contains
    the user's goal / system context), then a one-line summary per turn
    truncated to keep_chars characters. The result reads as a compact
    transcript.

    Not as good as an LLM summary, but free and predictable. A reasonable
    default for "I want memory to work without paying for summarization."
    """

    def __init__(self, keep_chars_per_turn: int = 80, include_first_turn: bool = True):
        self._chars = keep_chars_per_turn
        self._include_first = include_first_turn

    def _truncate(self, text: str) -> str:
        t = text.strip().replace("\n", " ")
        if len(t) > self._chars:
            return t[: self._chars - 1].rstrip() + "…"
        return t

    def summarize(self, turns: List[Turn]) -> str:
        if not turns:
            return ""
        lines: List[str] = []
        if self._include_first and turns:
            lines.append(
                f"[Conversation began with]\n"
                f"  User: {self._truncate(turns[0].user)}\n"
                f"  Assistant: {self._truncate(turns[0].assistant)}"
            )
            rest = turns[1:]
        else:
            rest = turns
        if rest:
            lines.append(f"[Earlier turns ({len(rest)})]")
            for i, t in enumerate(rest):
                lines.append(
                    f"  {i+1}. U: {self._truncate(t.user)} | A: {self._truncate(t.assistant)}"
                )
        return "\n".join(lines)


class CallableSummarizer:
    """Wraps any callable (turns) -> str. Used to plug in an LLM summarizer.

    Example:
        async def my_llm_summarize(turns):
            text = "\\n".join(f"U:{t.user} A:{t.assistant}" for t in turns)
            return await my_llm_call(f"Summarize this conversation:\\n{text}")

        summarizer = CallableSummarizer(my_llm_summarize)
    """

    def __init__(self, fn: Callable[[List[Turn]], Union[str, Awaitable[str]]]):
        self._fn = fn

    def summarize(self, turns: List[Turn]) -> Union[str, Awaitable[str]]:
        return self._fn(turns)


# ============================================================
#                  ConversationMemory
# ============================================================

@dataclass
class _ContextResult:
    """Internal: what get_context computed."""
    text: str
    messages: List[Dict[str, str]]
    summary_text: Optional[str]
    recent_turns: List[Turn]
    total_turns: int
    used_summary: bool
    tokens_full_history: int       # what we'd have paid without memory
    tokens_with_memory: int        # what we'd pay with memory
    tokens_saved: int


class ConversationMemory:
    """Tiered-fidelity conversation memory with summary caching.

    Args:
        recent_window: number of most-recent turns kept full-fidelity.
        summary_threshold: when total turns exceed this, older turns
            get summarized. Must be >= recent_window.
        store: where turns live. Defaults to SQLite in ~/.tokeymeter/memory.db.
        summarizer: how to reduce old turns. Defaults to TruncationSummarizer.
        max_turns: hard cap per session (oldest dropped after this).

    Concurrency:
        One asyncio.Lock per session ensures correctness when concurrent
        async calls touch the same session. The store itself is thread-safe.
    """

    def __init__(
        self,
        *,
        recent_window: int = 5,
        summary_threshold: int = 10,
        store: Optional[MemoryStore] = None,
        summarizer: Optional[Summarizer] = None,
        max_turns: int = 1000,
    ):
        if recent_window < 0:
            raise ValueError("recent_window must be >= 0")
        if summary_threshold < recent_window:
            raise ValueError("summary_threshold must be >= recent_window")
        if max_turns < summary_threshold:
            raise ValueError("max_turns must be >= summary_threshold")

        self._recent_window = recent_window
        self._summary_threshold = summary_threshold
        self._max_turns = max_turns
        self._store = store if store is not None else InMemoryMemoryStore()
        self._summarizer = summarizer if summarizer is not None else TruncationSummarizer()

        # Per-session locks, created lazily and LRU-bounded so a long-running
        # service that sees unboundedly many session IDs cannot leak memory here.
        # Evicting an *unheld* lock is safe: the next access just makes a new one.
        self._session_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
        self._max_session_locks = 10_000
        self._locks_guard = threading.Lock()

        # Stats counters (process-local; the persistent stats live in savings.jsonl)
        self._stats = {
            "turns_added": 0,
            "context_built": 0,
            "summaries_generated": 0,
            "summary_cache_hits": 0,
            "tokens_saved_total": 0,
        }
        self._stats_lock = threading.Lock()

    # ---- Lock acquisition ----

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
                self._evict_idle_locks_locked()
            else:
                self._session_locks.move_to_end(session_id)  # mark MRU
            return lock

    def _evict_idle_locks_locked(self) -> None:
        """Bound the lock map: drop least-recently-used locks that are not held.
        Caller must hold _locks_guard."""
        while len(self._session_locks) > self._max_session_locks:
            evicted = False
            for sid, lk in list(self._session_locks.items()):  # oldest-first
                if not lk.locked():
                    del self._session_locks[sid]
                    evicted = True
                    break
            if not evicted:
                break  # everything currently held; don't force-evict

    # ---- Buffer hashing (cache invalidation key) ----

    @staticmethod
    def _buffer_hash(turns_to_summarize: List[Turn]) -> str:
        """Stable hash of the turns about to be summarized.

        If two calls produce the same hash, the cached summary is reused.
        """
        h = hashlib.sha256()
        for t in turns_to_summarize:
            h.update(t.user.encode("utf-8", "replace"))
            h.update(b"\x00")
            h.update(t.assistant.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()

    # ---- Public API ----

    async def add_turn(
        self,
        session_id: str,
        user: str,
        assistant: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a user/assistant exchange. Never raises."""
        if not isinstance(session_id, str) or not session_id:
            return
        try:
            lock = self._get_lock(session_id)
            async with lock:
                turn = Turn(user=user, assistant=assistant, metadata=metadata or {})
                await asyncio.to_thread(self._store.append_turn, session_id, turn)
                # Enforce the hard per-session cap: prune oldest turns so the
                # session can never grow without bound (the docstring contract).
                all_turns = await asyncio.to_thread(self._store.get_turns, session_id)
                if len(all_turns) > self._max_turns:
                    trim = getattr(self._store, "trim_turns", None)
                    if callable(trim):
                        await asyncio.to_thread(trim, session_id, self._max_turns)
                    else:
                        log.debug("memory: store %s lacks trim_turns; cannot "
                                  "enforce max_turns", type(self._store).__name__)
                with self._stats_lock:
                    self._stats["turns_added"] += 1
        except Exception as e:
            log.debug("memory: add_turn failed: %s", e)
            _emit_degraded("memory.add_turn", e)

    async def get_context(
        self,
        session_id: str,
        *,
        return_format: str = "messages",
        force_full: bool = False,
    ) -> _ContextResult:
        """Build the context prefix to inject into the next LLM call.

        Returns a _ContextResult with both .text and .messages forms so
        callers can pick the shape that matches their LLM API.

        Args:
            force_full: if True, no summarization happens — all turns are
                included verbatim. Used by the fidelity audit to compare
                summarized vs full-history outputs.
        """
        try:
            all_turns = await asyncio.to_thread(self._store.get_turns, session_id)
        except Exception as e:
            log.debug("memory: get_turns failed: %s", e)
            _emit_degraded("memory.get_context", e)
            all_turns = []

        total = len(all_turns)
        tokens_full = sum(t.token_estimate() for t in all_turns)

        if total == 0:
            return _ContextResult(
                text="", messages=[], summary_text=None,
                recent_turns=[], total_turns=0, used_summary=False,
                tokens_full_history=0, tokens_with_memory=0, tokens_saved=0,
            )

        # Decide: do we need to summarize?
        if force_full or total <= self._summary_threshold:
            # All turns fit in the buffer; no summarization needed
            recent = all_turns
            summary_text = None
            used_summary = False
        else:
            # Split: oldest get summarized, newest stay full
            split_at = total - self._recent_window
            to_summarize = all_turns[:split_at]
            recent = all_turns[split_at:]
            summary_text = await self._get_or_build_summary(session_id, to_summarize)
            used_summary = True

            # ---- Industrial-grade safeguard ----
            # If the summary is LONGER than what it replaces (rare, happens with
            # very short turns), fall back to the full history. The memory layer
            # must NEVER make a user's call more expensive than the no-memory baseline.
            summary_token_estimate = len(summary_text) // 4 if summary_text else 0
            replaced_token_estimate = sum(t.token_estimate() for t in to_summarize)
            if summary_token_estimate >= replaced_token_estimate:
                recent = all_turns
                summary_text = None
                used_summary = False

        # Build the two output forms
        messages = self._build_messages(summary_text, recent)
        text = self._build_text(summary_text, recent)

        tokens_with_memory = (
            (len(summary_text) // 4 if summary_text else 0)
            + sum(t.token_estimate() for t in recent)
        )
        tokens_saved = max(0, tokens_full - tokens_with_memory) if not force_full else 0

        with self._stats_lock:
            self._stats["context_built"] += 1
            self._stats["tokens_saved_total"] += tokens_saved

        return _ContextResult(
            text=text,
            messages=messages,
            summary_text=summary_text,
            recent_turns=recent,
            total_turns=total,
            used_summary=used_summary,
            tokens_full_history=tokens_full,
            tokens_with_memory=tokens_with_memory,
            tokens_saved=tokens_saved,
        )

    async def _get_or_build_summary(
        self, session_id: str, turns_to_summarize: List[Turn]
    ) -> str:
        """Get cached summary if buffer state matches; otherwise compute fresh."""
        try:
            new_hash = self._buffer_hash(turns_to_summarize)
            cached_hash = await asyncio.to_thread(
                self._store.get_summary_buffer_hash, session_id
            )
            if cached_hash == new_hash:
                cached = await asyncio.to_thread(self._store.get_summary, session_id)
                if cached is not None:
                    with self._stats_lock:
                        self._stats["summary_cache_hits"] += 1
                    return cached

            # Rebuild
            result = self._summarizer.summarize(turns_to_summarize)
            if asyncio.iscoroutine(result):
                summary = await result
            else:
                summary = result
            if not isinstance(summary, str):
                summary = ""

            await asyncio.to_thread(
                self._store.set_summary, session_id, summary, new_hash
            )
            with self._stats_lock:
                self._stats["summaries_generated"] += 1
            return summary
        except Exception as e:
            log.debug("memory: summary build failed: %s", e)
            _emit_degraded("memory.summary", e)
            # Fall back to "no summary" — caller gets only recent turns
            return ""

    @staticmethod
    def _build_messages(
        summary: Optional[str], recent: List[Turn]
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if summary:
            msgs.append({"role": "system", "content": f"Earlier context:\n{summary}"})
        for t in recent:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    @staticmethod
    def _build_text(summary: Optional[str], recent: List[Turn]) -> str:
        lines: List[str] = []
        if summary:
            lines.append(f"Earlier context:\n{summary}\n")
        for t in recent:
            lines.append(f"User: {t.user}")
            lines.append(f"Assistant: {t.assistant}")
        return "\n".join(lines).strip()

    # ---- Admin ----

    async def clear_session(self, session_id: str) -> None:
        try:
            await asyncio.to_thread(self._store.clear_session, session_id)
        except Exception as e:
            log.debug("memory: clear_session failed: %s", e)
            _emit_degraded("memory.clear_session", e)
        finally:
            # Remove the per-session lock so cleared sessions don't leak it.
            # Skip if currently held (an in-flight op will let a later clear
            # reap it); the LRU bound is the backstop for that case.
            with self._locks_guard:
                lk = self._session_locks.get(session_id)
                if lk is not None and not lk.locked():
                    self._session_locks.pop(session_id, None)

    async def inspect_session(self, session_id: str) -> dict:
        """Return diagnostic info for a session. Never raises."""
        try:
            turns = await asyncio.to_thread(self._store.get_turns, session_id)
            summary = await asyncio.to_thread(self._store.get_summary, session_id)
            tokens_total = sum(t.token_estimate() for t in turns)
            return {
                "session_id": session_id,
                "total_turns": len(turns),
                "tokens_total": tokens_total,
                "has_summary": summary is not None,
                "summary_preview": (summary[:200] + "...") if summary and len(summary) > 200 else summary,
                "first_turn_ts": turns[0].timestamp if turns else None,
                "last_turn_ts": turns[-1].timestamp if turns else None,
            }
        except Exception as e:
            log.debug("memory: inspect_session failed: %s", e)
            _emit_degraded("memory.inspect_session", e)
            return {"session_id": session_id, "error": str(e)}

    def stats(self) -> dict:
        """Process-local counters. The durable stats are in savings.jsonl."""
        with self._stats_lock:
            return dict(self._stats)

    async def list_sessions(self) -> List[str]:
        try:
            return await asyncio.to_thread(self._store.list_sessions)
        except Exception as e:
            _emit_degraded("memory.list_sessions", e)
            return []

    async def get_full_context(self, session_id: str) -> _ContextResult:
        """Build context WITHOUT summarization. Used by fidelity audits.

        Returns the full conversation history as messages + text, with
        tokens_saved=0 (no savings because no summarization).
        """
        try:
            all_turns = await asyncio.to_thread(self._store.get_turns, session_id)
        except Exception as e:
            _emit_degraded("memory.get_full_context", e)
            all_turns = []

        if not all_turns:
            return _ContextResult(
                text="", messages=[], summary_text=None,
                recent_turns=[], total_turns=0, used_summary=False,
                tokens_full_history=0, tokens_with_memory=0, tokens_saved=0,
            )

        messages = self._build_messages(None, all_turns)
        text = self._build_text(None, all_turns)
        tokens = sum(t.token_estimate() for t in all_turns)
        return _ContextResult(
            text=text, messages=messages, summary_text=None,
            recent_turns=all_turns, total_turns=len(all_turns),
            used_summary=False,
            tokens_full_history=tokens, tokens_with_memory=tokens, tokens_saved=0,
        )


# (v0.12.1 review) A stale duplicate of the with_memory section that previously
# lived here was removed: every name it defined was rebound by the section
# below, so the whole block was provably dead at import time.

# ============================================================
#                  with_memory decorator
# ============================================================

# Module-level fidelity audit log. Bounded; oldest entries dropped past cap.
_memory_fidelity_log: List[dict] = []
_memory_fidelity_lock = threading.Lock()


def _record_memory_fidelity(rec: dict) -> None:
    """Append a memory-fidelity record. Never raises."""
    try:
        with _memory_fidelity_lock:
            _memory_fidelity_log.append(rec)
            if len(_memory_fidelity_log) > 10_000:
                del _memory_fidelity_log[: len(_memory_fidelity_log) - 10_000]
    except Exception:
        pass


def memory_fidelity_log() -> List[dict]:
    """Return a copy of the in-memory fidelity audit log."""
    with _memory_fidelity_lock:
        return list(_memory_fidelity_log)


def _memory_jaccard_similarity(a: Any, b: Any) -> float:
    """Token-set Jaccard similarity on string representations."""
    sa = str(a) if not isinstance(a, str) else a
    sb = str(b) if not isinstance(b, str) else b
    if sa == sb:
        return 1.0
    ta = set(sa.lower().split())
    tb = set(sb.lower().split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _default_text_injector(original_prompt: str, ctx: _ContextResult) -> str:
    """Default: prepend conversation context as a text prefix.

    For OpenAI-style `messages=` users, supply a custom injector that
    builds `[summary_system, *recent_msgs, user_msg]` instead.
    """
    if not ctx.text:
        return original_prompt
    return f"{ctx.text}\n\nUser: {original_prompt}"


def with_memory(
    fn: Optional[Callable] = None,
    *,
    memory: ConversationMemory,
    session_arg: str = "session_id",
    prompt_arg: str = "prompt",
    context_injector: Optional[Callable[[str, _ContextResult], str]] = None,
    fidelity_rate: float = 0.0,
    fidelity_similarity_fn: Optional[Callable[[Any, Any], float]] = None,
    record_turns: bool = True,
):
    """Decorate a function with rolling conversation memory.

    On each call:
      1. Pulls session_id from kwargs[session_arg]. If absent, skips
         memory entirely (fail-open).
      2. Builds context via memory.get_context(session_id).
      3. Injects context into kwargs[prompt_arg] using context_injector
         (default: prepend conversation as text).
      4. Calls the wrapped function with the expanded prompt.
      5. Appends the (ORIGINAL prompt, response) as a new turn.

    Composes naturally with @tokeymeter.cache — place @with_memory ABOVE @cache:

        @tokeymeter.with_memory(memory=mem, session_arg="session_id")
        @tokeymeter.cache(...)
        async def ask(prompt: str, session_id: str): ...

    Works on both sync and async functions. Detection is automatic.

    Args:
        memory: a ConversationMemory instance.
        session_arg: kwarg name carrying the session id. Caller MUST pass
            this as a keyword argument (positional won't work cleanly).
        prompt_arg: kwarg name holding the prompt text. Mutated in-place
            by the injector. Caller must pass this as a keyword arg too.
        context_injector: callable (original_prompt, context_result) -> str.
            Default prepends `ctx.text` to the prompt.
        fidelity_rate: in [0, 1]. Fraction of calls that get audited by
            running the function with BOTH summarized and full context,
            then comparing outputs via fidelity_similarity_fn.
        fidelity_similarity_fn: (summarized_result, full_result) -> float
            in [0, 1]. Defaults to token-set Jaccard.
        record_turns: if False, don't add the (prompt, response) pair to
            memory. Useful for read-only audit modes.
    """
    if memory is None or not isinstance(memory, ConversationMemory):
        raise TypeError("with_memory requires a ConversationMemory instance")

    injector = context_injector or _default_text_injector

    def decorator(func: Callable) -> Callable:
        is_coro = inspect.iscoroutinefunction(func)

        if is_coro:
            async def async_wrapper(*args, **kwargs):
                session_id = kwargs.get(session_arg)
                original_prompt = kwargs.get(prompt_arg)

                # Fail-open: missing session or prompt → bypass memory
                if not isinstance(session_id, str) or not session_id:
                    return await func(*args, **kwargs)
                if not isinstance(original_prompt, str):
                    return await func(*args, **kwargs)

                # Build context (summarized by default)
                try:
                    ctx = await memory.get_context(session_id)
                except Exception as e:
                    log.debug("memory: get_context failed: %s", e)
                    return await func(*args, **kwargs)

                # Inject context. If the injector raises, fall back to original.
                try:
                    expanded = injector(original_prompt, ctx)
                    if not isinstance(expanded, str):
                        expanded = original_prompt
                except Exception as e:
                    log.debug("memory: injector failed: %s", e)
                    expanded = original_prompt

                new_kwargs = dict(kwargs)
                new_kwargs[prompt_arg] = expanded

                result = await func(*args, **new_kwargs)

                # ----- Fidelity audit (sampled) -----
                if (fidelity_rate > 0
                        and ctx.used_summary
                        and _should_fidelity_sample(fidelity_rate)):
                    try:
                        full_ctx = await memory.get_context(session_id, force_full=True)
                        full_expanded = injector(original_prompt, full_ctx)
                        full_kwargs = dict(kwargs)
                        full_kwargs[prompt_arg] = full_expanded
                        full_result = await func(*args, **full_kwargs)
                        sim_fn = fidelity_similarity_fn or _memory_jaccard_similarity
                        sim = float(sim_fn(result, full_result))
                        _record_memory_fidelity({
                            "timestamp": time.time(),
                            "session_id": session_id,
                            "function_name": func.__name__,
                            "similarity": sim,
                            "tokens_summarized": ctx.tokens_with_memory,
                            "tokens_full": ctx.tokens_full_history,
                            "tokens_saved": ctx.tokens_saved,
                            "total_turns": ctx.total_turns,
                        })
                    except Exception as e:
                        log.debug("memory: fidelity audit failed: %s", e)

                # Record the new turn (original prompt — NOT expanded)
                if record_turns:
                    try:
                        await memory.add_turn(session_id, original_prompt, str(result))
                    except Exception as e:
                        log.debug("memory: add_turn failed: %s", e)

                return result

            async_wrapper.__wrapped__ = func  # type: ignore[attr-defined]
            async_wrapper.__name__ = func.__name__
            async_wrapper.__doc__ = func.__doc__
            return async_wrapper

        # ---- Sync path: drive the async memory ops via asyncio.run/get_loop ----
        def sync_wrapper(*args, **kwargs):
            session_id = kwargs.get(session_arg)
            original_prompt = kwargs.get(prompt_arg)

            if not isinstance(session_id, str) or not session_id:
                return func(*args, **kwargs)
            if not isinstance(original_prompt, str):
                return func(*args, **kwargs)

            try:
                ctx = _run_sync(memory.get_context(session_id))
            except Exception as e:
                log.debug("memory: get_context (sync) failed: %s", e)
                return func(*args, **kwargs)

            try:
                expanded = injector(original_prompt, ctx)
                if not isinstance(expanded, str):
                    expanded = original_prompt
            except Exception as e:
                log.debug("memory: injector (sync) failed: %s", e)
                expanded = original_prompt

            new_kwargs = dict(kwargs)
            new_kwargs[prompt_arg] = expanded

            result = func(*args, **new_kwargs)

            if (fidelity_rate > 0
                    and ctx.used_summary
                    and _should_fidelity_sample(fidelity_rate)):
                try:
                    full_ctx = _run_sync(
                        memory.get_context(session_id, force_full=True)
                    )
                    full_expanded = injector(original_prompt, full_ctx)
                    full_kwargs = dict(kwargs)
                    full_kwargs[prompt_arg] = full_expanded
                    full_result = func(*args, **full_kwargs)
                    sim_fn = fidelity_similarity_fn or _memory_jaccard_similarity
                    sim = float(sim_fn(result, full_result))
                    _record_memory_fidelity({
                        "timestamp": time.time(),
                        "session_id": session_id,
                        "function_name": func.__name__,
                        "similarity": sim,
                        "tokens_summarized": ctx.tokens_with_memory,
                        "tokens_full": ctx.tokens_full_history,
                        "tokens_saved": ctx.tokens_saved,
                        "total_turns": ctx.total_turns,
                    })
                except Exception as e:
                    log.debug("memory: sync fidelity audit failed: %s", e)

            if record_turns:
                try:
                    _run_sync(memory.add_turn(session_id, original_prompt, str(result)))
                except Exception as e:
                    log.debug("memory: sync add_turn failed: %s", e)

            return result

        sync_wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        sync_wrapper.__name__ = func.__name__
        sync_wrapper.__doc__ = func.__doc__
        return sync_wrapper

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


def _should_fidelity_sample(rate: float) -> bool:
    """Decide whether this call should fire a memory-fidelity audit."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    import random as _r
    return _r.random() < rate


def _run_sync(coro):
    """Run an async coroutine from synchronous code, handling already-running loops.

    If no loop is running on this thread, uses ``asyncio.run``. If a loop IS
    running on this thread, the coroutine is run on a one-shot worker thread.

    IMPORTANT: that worker-thread path avoids the
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    crash and avoids nesting loops — but the synchronous ``.result()`` wait still
    BLOCKS the calling thread. Since this branch is only reached when a loop is
    running on the calling thread, the calling thread is the loop thread, so the
    event loop is blocked for the duration of the coroutine. In other words: the
    sync ``@with_memory`` wrapper is safe to call from an event loop, but it is
    NOT non-blocking there. In async code, use the async wrapper (decorate an
    ``async def``) so memory I/O is awaited instead of blocking the loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)

    # We're on a thread with a running loop. Run the coroutine on a side thread to
    # avoid the "asyncio.run() from a running loop" error — but be honest that the
    # .result() wait below blocks THIS (the loop) thread until it completes.
    global _WARNED_SYNC_IN_LOOP
    if not _WARNED_SYNC_IN_LOOP:
        _WARNED_SYNC_IN_LOOP = True
        warnings.warn(
            "tokeymeter: the synchronous @with_memory wrapper was called from a "
            "running event loop. It will work, but it BLOCKS the loop thread while "
            "memory I/O runs. In async code, apply @with_memory to an 'async def' "
            "so the work is awaited instead of blocking the loop.",
            RuntimeWarning, stacklevel=3,
        )
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
