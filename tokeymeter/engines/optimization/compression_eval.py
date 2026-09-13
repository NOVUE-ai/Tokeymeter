"""
Compression eval loop — continuous monitoring of compression's failure mode.

WHY THIS EXISTS
The catastrophic failure of prompt compression is dropping the ANSWER-BEARING
chunk: the pruned prompt is silently missing the fact the model needed, so the
model answers wrong with full confidence and a 200 OK. Exactly the silent-wrong-
answer risk the semantic cache had — and the only way to KNOW compression is safe
in production (not just on clean test cases) is to continuously estimate the
HARMFUL-DROP RATE: how often a dropped chunk actually contained what the question
needed. This is that monitor.

WHAT IT DOES (ports the semantic eval loop pattern)
1. RECORD every compression event content-blind (hashes of query + kept + dropped
   text, the ratio, tokens saved). Raw text is NEVER persisted — only used
   transiently for grading.
2. SAMPLE a fraction of events for automated grading.
3. GRADE with TWO signals:
   - LLM judge (automated): "can the question still be fully answered from the
     KEPT context alone?" NO => a harmful drop (we lost the answer). Deferred +
     batched so it never adds latency.
   - User feedback (authoritative): the app reports a wrong answer on a compressed
     prompt; overrides the judge for that event.
4. TRACK the harmful-drop rate over a rolling window, plus the distribution of
   compression ratios (aggressive drops are where harm concentrates).
5. ALERT on breach (harmful-drop rate above tolerance). Backing off compression
   aggressiveness on breach is an OPT-IN toggle (monitor-first by default).

FAIL-SAFE: a judge infrastructure error counts the event as SAFE (it never
inflates the harmful-drop rate on its own failures); monitoring never breaks a call.

Pure-stdlib core, no mandatory deps, content-blind persistence.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

log = logging.getLogger("tokeymeter.compression.eval")


def _hash(text: str, secret: str = "") -> str:
    h = hashlib.sha256()
    if secret:
        h.update(secret.encode("utf-8", "ignore")); h.update(b"|")
    h.update((text or "").encode("utf-8", "ignore"))
    return h.hexdigest()[:24]


class _CompRecord:
    """One compression observation. PERSISTENT fields are content-blind."""
    __slots__ = ("cid", "query_hash", "kept_hash", "dropped_hash", "ratio",
                 "tokens_saved", "ts", "harmful", "grade_source")

    def __init__(self, cid, query_hash, kept_hash, dropped_hash, ratio,
                 tokens_saved, ts):
        self.cid = cid
        self.query_hash = query_hash
        self.kept_hash = kept_hash
        self.dropped_hash = dropped_hash
        self.ratio = ratio
        self.tokens_saved = tokens_saved
        self.ts = ts
        self.harmful: Optional[bool] = None       # True = needed chunk was dropped
        self.grade_source: Optional[str] = None   # "judge" | "feedback"


class CompressionEvalLoop:
    """Continuous harmful-drop monitor for prompt compression.

    Typical wiring (handled by wrap(compression_eval=...)):
        loop = CompressionEvalLoop(sample_rate=0.05, harmful_tolerance=0.02,
                                   judge_fn=my_judge, alert_fn=my_alert)
        # on each compression the engine calls:
        loop.record_compression(query, kept_text, dropped_text, ratio, tokens_saved)
        # periodically:
        loop.grade_pending()
        # the app reports a wrong answer on a compressed prompt (authoritative):
        loop.report_feedback(query, answer_was_correct=False)
        # read the provable number:
        loop.report()  # -> {harmful_drop_rate, ratio distribution, ...}
    """

    def __init__(
        self,
        sample_rate: float = 0.05,
        harmful_tolerance: float = 0.02,
        window: int = 1000,
        judge_fn: Optional[Callable[[str, str, str], bool]] = None,
        alert_fn: Optional[Callable[[dict], None]] = None,
        auto_backoff: bool = False,
        backoff_step: float = 0.1,
        compressor_ref: Optional[object] = None,
        hash_secret: str = "",
        min_graded_for_alert: int = 20,
    ):
        """
        Args:
            sample_rate: fraction of compressions to buffer for LLM-judge grading.
            harmful_tolerance: harmful-drop rate above which the alert fires.
            window: rolling window size (most-recent N events for stats).
            judge_fn: callable(query, kept_text, dropped_text) -> bool, where
                True = SAFE drop (the kept context can still answer the question),
                False = HARMFUL drop (a needed chunk was dropped). Injected so the
                core has no SDK dependency.
            alert_fn: callable(report_dict) on breach. Logs a warning if absent.
            auto_backoff: OPT-IN. On breach, make compression LESS aggressive by
                raising the compressor's context_target_ratio by backoff_step
                (keep more). Default False = monitor-only.
            backoff_step: ratio increment applied on breach when auto_backoff.
            compressor_ref: the QueryAwareCompressor to back off (needs a settable
                context_target_ratio).
            hash_secret: per-install secret for content-blind hashing.
            min_graded_for_alert: minimum graded events before alerting.
        """
        self._sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self._tol = float(harmful_tolerance)
        self._judge_fn = judge_fn
        self._alert_fn = alert_fn
        self._auto_backoff = bool(auto_backoff)
        self._backoff_step = float(backoff_step)
        self._compressor = compressor_ref
        self._secret = hash_secret
        self._min_graded = int(min_graded_for_alert)

        self._records: Deque[_CompRecord] = deque(maxlen=window)
        self._by_query: Dict[str, _CompRecord] = {}
        self._pending: Deque[tuple] = deque(maxlen=2000)   # (cid, query, kept, dropped)
        self._counter = 0
        self._alerts_fired = 0
        self._last_alert: Optional[dict] = None
        self._lock = threading.Lock()

    # ── recording ──
    def record_compression(self, query: str, kept_text: str, dropped_text: str,
                           ratio: float, tokens_saved: int = 0) -> str:
        """Record a compression event content-blind; if sampled and there was an
        actual drop, buffer its text for harmful-drop grading. Non-blocking."""
        with self._lock:
            self._counter += 1
            cid = f"c{self._counter}"
            rec = _CompRecord(
                cid=cid,
                query_hash=_hash(query, self._secret),
                kept_hash=_hash(kept_text, self._secret),
                dropped_hash=_hash(dropped_text, self._secret),
                ratio=float(ratio) if ratio is not None else None,
                tokens_saved=int(tokens_saved or 0),
                ts=time.time(),
            )
            self._records.append(rec)
            self._by_query[rec.query_hash] = rec
            sampled = (self._sample_rate > 0 and
                       (self._counter * self._sample_rate) % 1 < self._sample_rate)
            # only worth grading if something was actually dropped
            if sampled and self._judge_fn is not None and (dropped_text or "").strip():
                self._pending.append((cid, query, kept_text, dropped_text))
            return cid

    # ── grading signal 1: automated LLM judge ──
    def grade_pending(self, judge_fn: Optional[Callable] = None, limit: int = 50) -> int:
        """Grade buffered sampled events. Stores ONLY the boolean (text discarded).
        Returns the number graded. Out-of-band from serving."""
        fn = judge_fn or self._judge_fn
        if fn is None:
            return 0
        graded = 0
        while graded < limit:
            with self._lock:
                if not self._pending:
                    break
                cid, query, kept, dropped = self._pending.popleft()
            try:
                safe = bool(fn(query, kept, dropped))   # True = safe drop
            except Exception as e:
                log.debug("tokeymeter.comp-eval: judge error: %s", e)
                continue
            with self._lock:
                rec = self._find(cid)
                if rec is not None and rec.grade_source != "feedback":
                    rec.harmful = (not safe)
                    rec.grade_source = "judge"
            graded += 1
        if graded:
            self._maybe_alert()
        return graded

    # ── grading signal 2: real user feedback (authoritative) ──
    def report_feedback(self, query: str, answer_was_correct: bool) -> bool:
        """The app reports whether the answer to a recently-compressed prompt was
        correct. A wrong answer => harmful drop (authoritative; overrides judge)."""
        qh = _hash(query, self._secret)
        with self._lock:
            rec = self._by_query.get(qh)
            if rec is None:
                return False
            rec.harmful = (not bool(answer_was_correct))
            rec.grade_source = "feedback"
        self._maybe_alert()
        return True

    # ── metrics ──
    def harmful_drop_rate(self) -> dict:
        """The provable number: harmful-drop rate over graded events, by source."""
        with self._lock:
            recs = list(self._records)
        graded = [r for r in recs if r.harmful is not None]
        fb = [r for r in graded if r.grade_source == "feedback"]
        jd = [r for r in graded if r.grade_source == "judge"]

        def rate(rs):
            return (sum(1 for r in rs if r.harmful) / len(rs)) if rs else None

        return {
            "overall": rate(graded),
            "by_feedback": rate(fb),
            "by_judge": rate(jd),
            "graded": len(graded),
            "graded_feedback": len(fb),
            "graded_judge": len(jd),
            "window_events": len(recs),
        }

    def report(self) -> dict:
        with self._lock:
            recs = list(self._records)
            ratios = [r.ratio for r in recs if r.ratio is not None]
            saved = sum(r.tokens_saved for r in recs)
            alerts = self._alerts_fired
            last = self._last_alert
            cur_ratio = self._current_ratio()
        hdr = self.harmful_drop_rate()
        dist = {}
        if ratios:
            aggressive = sum(1 for r in ratios if r <= 0.4)  # kept <=40% = aggressive
            dist = {
                "min": round(min(ratios), 3),
                "mean": round(sum(ratios) / len(ratios), 3),
                "max": round(max(ratios), 3),
                "aggressive_events": aggressive,
                "aggressive_pct": round(100.0 * aggressive / len(ratios), 1),
            }
        return {
            "total_compressions": self._counter,
            "window_events": len(recs),
            "harmful_drop_rate": hdr["overall"],
            "harmful_by_feedback": hdr["by_feedback"],
            "harmful_by_judge": hdr["by_judge"],
            "graded": hdr["graded"],
            "pending_grading": len(self._pending),
            "total_tokens_saved": saved,
            "ratio_distribution": dist,
            "current_target_ratio": cur_ratio,
            "harmful_tolerance": self._tol,
            "alerts_fired": alerts,
            "last_alert": last,
            "auto_backoff": self._auto_backoff,
        }

    # ── alerting + opt-in backoff ──
    def _maybe_alert(self) -> None:
        hdr = self.harmful_drop_rate()
        rate = hdr["overall"]
        if rate is None or hdr["graded"] < self._min_graded:
            return
        if rate <= self._tol:
            return
        payload = {
            "harmful_drop_rate": round(rate, 4),
            "tolerance": self._tol,
            "graded": hdr["graded"],
            "current_target_ratio": self._current_ratio(),
            "message": (f"compression harmful-drop rate {rate:.1%} exceeds tolerance "
                        f"{self._tol:.1%} — compression is dropping answer-bearing chunks"),
        }
        with self._lock:
            self._alerts_fired += 1
            self._last_alert = payload
        if self._auto_backoff:
            self._backoff(payload)
        if self._alert_fn is not None:
            try:
                self._alert_fn(payload)
            except Exception as e:
                log.debug("tokeymeter.comp-eval: alert_fn error: %s", e)
        else:
            log.warning("tokeymeter.comp-eval: %s", payload["message"])

    def _backoff(self, payload: dict) -> None:
        cur = self._current_ratio()
        if cur is None:
            return
        new = min(1.0, cur + self._backoff_step)   # keep MORE (less aggressive)
        if self._set_ratio(new):
            payload["auto_backed_off_to"] = round(new, 3)
            log.warning("tokeymeter.comp-eval: backed off compression %.2f -> %.2f",
                        cur, new)

    def _current_ratio(self) -> Optional[float]:
        c = self._compressor
        if c is None:
            return None
        for attr in ("context_target_ratio", "_context_target_ratio"):
            if hasattr(c, attr):
                try:
                    return float(getattr(c, attr))
                except Exception:
                    return None
        return None

    def _set_ratio(self, value: float) -> bool:
        c = self._compressor
        if c is None or not hasattr(c, "context_target_ratio"):
            return False
        try:
            c.context_target_ratio = float(value)
            return True
        except Exception:
            return False

    def _find(self, cid: str) -> Optional[_CompRecord]:
        for r in reversed(self._records):
            if r.cid == cid:
                return r
        return None


def make_openai_compression_judge(client: object, model: str = "gpt-4o-mini") -> Callable:
    """Build an LLM-judge for grade_pending using an OpenAI-style client.
    Returns judge(query, kept_text, dropped_text) -> bool, where True = SAFE drop
    (the question can still be answered from the kept context). Kept out of the
    core so the eval loop has no SDK dependency.

    Fail-safe: on judge error, returns True (do NOT inflate the harmful-drop rate
    on infrastructure errors)."""
    def judge(query: str, kept_text: str, dropped_text: str) -> bool:
        prompt = (
            "Prompt compression removed some context before sending to an LLM, to "
            "save tokens. Decide if that was SAFE.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT THAT WAS KEPT:\n{str(kept_text)[:1500]}\n\n"
            f"CONTEXT THAT WAS REMOVED:\n{str(dropped_text)[:1000]}\n\n"
            "Can the QUESTION still be fully and correctly answered using ONLY the "
            "KEPT context? Reply ONLY 'YES' (safe — removed content was not needed) "
            "or 'NO' (harmful — the removed content was needed to answer).")
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=4, temperature=0)
            ans = (r.choices[0].message.content or "").strip().upper()
            return not ans.startswith("N")   # NO = harmful -> return False
        except Exception:
            return True   # fail-safe
    return judge
