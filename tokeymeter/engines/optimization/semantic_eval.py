"""
Semantic eval loop — continuous false-positive monitoring for the semantic cache.

WHY THIS EXISTS
A semantic cache can silently serve a wrong answer: a bad hit returns an
incorrect response with full confidence and a 200 OK. The only way to KNOW your
cache is correct in production (not just on a test set) is to continuously
estimate the false-positive rate — by sampling served hits and grading them, and
by recording real user corrections. The documented discipline: "a semantic cache
without an eval loop is a footgun." This is that eval loop, built clean-room from
the public production pattern (sample, grade, track FP-rate, alert on breach).

WHAT IT DOES
1. RECORD every semantic hit content-blind (hashes of query+matched prompt, the
   similarity and verification scores, timestamp, tenant) into a rolling window.
   The raw text is NEVER persisted — only used transiently for grading.
2. SAMPLE a configurable fraction of hits for automated grading.
3. GRADE with TWO signals, integrated:
   - LLM judge (automated): "did the cached answer actually answer this query?"
     Deferred + batched so it never adds latency to the hit.
   - User feedback (authoritative): the app reports real corrections; this is the
     strongest signal and overrides the judge for the same hit.
4. TRACK the false-positive rate over the window, plus the similarity-score
   distribution — flagging near-threshold matches (the documented danger zone
   where similarity barely clears the cutoff but intent differs).
5. ALERT on breach (FP rate above tolerance) via a callback. Auto-tighten the
   threshold on breach is an OPT-IN toggle (monitor-first by default — never
   silently change behavior unless the operator asks).

MOAT ALIGNMENT
The accumulated grading data (which matches were good/bad, per-tenant, over time)
is exactly the compounding governance state. And it produces the clean, PROVABLE
false-positive number an auditor needs — turning "trust me" into "here is the
measured, monitored rate."

DESIGN
- Pure-stdlib core (counters, deque window, hashing). No mandatory deps.
- Content-blind persistence (only hashes + scores + grade stored).
- Non-blocking: record_hit just appends; grading is deferred/sampled.
- The LLM judge is an injected callable (no hard dependency on any SDK).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

log = logging.getLogger("tokeymeter.semantic.eval")


def _hash(text: str, secret: str = "") -> str:
    """Content-blind hash of a prompt (with optional per-install secret)."""
    h = hashlib.sha256()
    if secret:
        h.update(secret.encode("utf-8", "ignore"))
        h.update(b"|")
    h.update((text or "").encode("utf-8", "ignore"))
    return h.hexdigest()[:24]


class _HitRecord:
    """One semantic-hit observation. PERSISTENT fields are content-blind: only
    hashes and scores. Raw text lives only in the transient grading buffer."""
    __slots__ = ("hit_id", "query_hash", "prompt_hash", "similarity",
                 "verify_score", "tenant", "ts", "grade", "grade_source")

    def __init__(self, hit_id, query_hash, prompt_hash, similarity,
                 verify_score, tenant, ts):
        self.hit_id = hit_id
        self.query_hash = query_hash
        self.prompt_hash = prompt_hash
        self.similarity = similarity
        self.verify_score = verify_score
        self.tenant = tenant
        self.ts = ts
        self.grade: Optional[bool] = None        # True=good hit, False=false positive
        self.grade_source: Optional[str] = None  # "judge" | "feedback"


class SemanticEvalLoop:
    """Continuous false-positive monitor for a semantic cache.

    Typical wiring (handled for you by wrap(eval=True)):
        loop = SemanticEvalLoop(sample_rate=0.05, fp_tolerance=0.02,
                                judge_fn=my_judge, alert_fn=my_alert)
        # on each served semantic hit, the cache calls:
        loop.record_hit(query, matched_prompt, response, similarity, verify_score)
        # periodically (or on a timer) grade the sampled buffer:
        loop.grade_pending()
        # the app reports real corrections (authoritative):
        loop.report_feedback(query, was_correct=False)
        # read the provable number:
        loop.report()  # -> {fp_rate, score distribution, near-threshold, ...}
    """

    def __init__(
        self,
        sample_rate: float = 0.05,
        fp_tolerance: float = 0.02,
        window: int = 1000,
        judge_fn: Optional[Callable[[str, str, object], bool]] = None,
        alert_fn: Optional[Callable[[dict], None]] = None,
        auto_tighten: bool = False,
        tighten_step: float = 0.01,
        cache: Optional[object] = None,
        hash_secret: str = "",
        near_threshold_band: float = 0.03,
        min_graded_for_alert: int = 20,
    ):
        """
        Args:
            sample_rate: fraction of hits to buffer for automated LLM-judge
                grading (0..1). The research norm is ~0.01-0.05.
            fp_tolerance: false-positive rate above which the alert fires.
                Research norms: ~0.02 general, ~0.005 regulated.
            window: rolling window size (most-recent N hits kept for stats).
            judge_fn: callable(query, matched_prompt, cached_response) -> bool
                (True = the cached answer correctly answers the query). Injected
                so the core has no SDK dependency. Optional — feedback alone works.
            alert_fn: callable(report_dict) invoked when the FP rate breaches
                tolerance. Optional (logs a warning if absent).
            auto_tighten: OPT-IN. When True, a breach nudges the cache threshold
                up by `tighten_step`. Default False = monitor-only (never silently
                change behavior).
            tighten_step: threshold increment applied on breach when auto_tighten.
            cache: the SemanticCache to auto-tighten (needs a settable threshold).
            hash_secret: per-install secret for content-blind hashing.
            near_threshold_band: similarity within this of the cache threshold is
                counted as a "near-threshold" (danger-zone) hit.
            min_graded_for_alert: minimum graded samples before alerting (avoids
                firing on tiny, noisy samples).
        """
        self._sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self._fp_tolerance = float(fp_tolerance)
        self._judge_fn = judge_fn
        self._alert_fn = alert_fn
        self._auto_tighten = bool(auto_tighten)
        self._tighten_step = float(tighten_step)
        self._cache = cache
        self._secret = hash_secret
        self._near_band = float(near_threshold_band)
        self._min_graded_for_alert = int(min_graded_for_alert)

        self._records: Deque[_HitRecord] = deque(maxlen=window)
        self._by_query: Dict[str, _HitRecord] = {}   # query_hash -> most recent record
        # transient grading buffer: (hit_id, query_text, prompt_text, response).
        # in-memory, bounded, CLEARED after grading — raw text never persisted.
        self._pending: Deque[tuple] = deque(maxlen=2000)
        self._counter = 0
        self._alerts_fired = 0
        self._last_alert: Optional[dict] = None
        self._lock = threading.Lock()

    # ── recording ─────────────────────────────────────────────────────────────

    def record_hit(self, query: str, matched_prompt: str, response: object,
                   similarity: float, verify_score: Optional[float] = None,
                   tenant: Optional[str] = None) -> str:
        """Record a served semantic hit (content-blind) and, if sampled, buffer
        its text for automated grading. Returns a hit_id. Fast + non-blocking."""
        with self._lock:
            self._counter += 1
            hit_id = f"h{self._counter}"
            rec = _HitRecord(
                hit_id=hit_id,
                query_hash=_hash(query, self._secret),
                prompt_hash=_hash(matched_prompt, self._secret),
                similarity=float(similarity) if similarity is not None else None,
                verify_score=(float(verify_score) if verify_score is not None else None),
                tenant=tenant,
                ts=time.time(),
            )
            self._records.append(rec)
            self._by_query[rec.query_hash] = rec
            # sample for automated grading
            sampled = (self._sample_rate > 0 and
                       (self._counter * self._sample_rate) % 1 < self._sample_rate)
            if sampled and self._judge_fn is not None:
                self._pending.append((hit_id, query, matched_prompt, response))
            return hit_id

    # ── grading signal 1: automated LLM judge (deferred, batched) ──────────────

    def grade_pending(self, judge_fn: Optional[Callable] = None,
                      limit: int = 50) -> int:
        """Grade buffered sampled hits with the LLM judge. Stores ONLY the
        boolean grade (text is discarded). Returns the number graded. Call this
        periodically (timer/cron) — it is out-of-band from serving."""
        fn = judge_fn or self._judge_fn
        if fn is None:
            return 0
        graded = 0
        while graded < limit:
            with self._lock:
                if not self._pending:
                    break
                hit_id, query, prompt, response = self._pending.popleft()
            try:
                ok = bool(fn(query, prompt, response))
            except Exception as e:
                log.debug("tokeymeter.eval: judge error: %s", e)
                continue
            with self._lock:
                rec = self._find(hit_id)
                # never let the judge override an authoritative user-feedback grade
                if rec is not None and rec.grade_source != "feedback":
                    rec.grade = ok
                    rec.grade_source = "judge"
            graded += 1
        if graded:
            self._maybe_alert()
        return graded

    # ── grading signal 2: real user feedback (authoritative) ───────────────────

    def report_feedback(self, query: str, was_correct: bool,
                        tenant: Optional[str] = None) -> bool:
        """The app reports a real correction for a recently-served hit (e.g. the
        user rejected/regenerated the answer). This is the STRONGEST signal and
        overrides any LLM-judge grade for that hit. Returns True if a matching
        recent hit was found and graded."""
        qh = _hash(query, self._secret)
        with self._lock:
            rec = self._by_query.get(qh)
            if rec is None:
                return False
            rec.grade = bool(was_correct)
            rec.grade_source = "feedback"
        self._maybe_alert()
        return True

    # ── metrics ────────────────────────────────────────────────────────────────

    def false_positive_rate(self) -> dict:
        """The provable number. FP rate over graded hits in the window, split by
        signal source. {overall, by_judge, by_feedback, graded, window_hits}."""
        with self._lock:
            recs = list(self._records)
        graded = [r for r in recs if r.grade is not None]
        fb = [r for r in graded if r.grade_source == "feedback"]
        jd = [r for r in graded if r.grade_source == "judge"]

        def rate(rs):
            return (sum(1 for r in rs if r.grade is False) / len(rs)) if rs else None

        return {
            "overall": rate(graded),
            "by_feedback": rate(fb),
            "by_judge": rate(jd),
            "graded": len(graded),
            "graded_feedback": len(fb),
            "graded_judge": len(jd),
            "window_hits": len(recs),
        }

    def report(self) -> dict:
        """Full monitoring report: FP rate, score distribution, near-threshold
        (danger-zone) count, current threshold, alerts fired."""
        with self._lock:
            recs = list(self._records)
            thr = self._current_threshold()
            alerts = self._alerts_fired
            last = self._last_alert
        sims = [r.similarity for r in recs if r.similarity is not None]
        near = 0
        if thr is not None:
            near = sum(1 for s in sims if 0 <= (s - thr) <= self._near_band)
        fpr = self.false_positive_rate()
        dist = {}
        if sims:
            dist = {
                "min": round(min(sims), 4),
                "max": round(max(sims), 4),
                "mean": round(sum(sims) / len(sims), 4),
                "near_threshold": near,
                "near_threshold_pct": round(100.0 * near / len(sims), 1),
            }
        return {
            "total_hits_observed": self._counter,
            "window_hits": len(recs),
            "false_positive_rate": fpr["overall"],
            "fp_by_feedback": fpr["by_feedback"],
            "fp_by_judge": fpr["by_judge"],
            "graded": fpr["graded"],
            "pending_grading": len(self._pending),
            "score_distribution": dist,
            "current_threshold": thr,
            "fp_tolerance": self._fp_tolerance,
            "alerts_fired": alerts,
            "last_alert": last,
            "auto_tighten": self._auto_tighten,
        }

    # ── alerting + opt-in auto-tighten ─────────────────────────────────────────

    def _maybe_alert(self) -> None:
        fpr = self.false_positive_rate()
        rate = fpr["overall"]
        if rate is None or fpr["graded"] < self._min_graded_for_alert:
            return
        if rate <= self._fp_tolerance:
            return
        payload = {
            "false_positive_rate": round(rate, 4),
            "tolerance": self._fp_tolerance,
            "graded": fpr["graded"],
            "current_threshold": self._current_threshold(),
            "message": (f"semantic cache false-positive rate {rate:.1%} exceeds "
                        f"tolerance {self._fp_tolerance:.1%}"),
        }
        with self._lock:
            self._alerts_fired += 1
            self._last_alert = payload
        # opt-in self-correction
        if self._auto_tighten:
            self._tighten(payload)
        # notify
        if self._alert_fn is not None:
            try:
                self._alert_fn(payload)
            except Exception as e:
                log.debug("tokeymeter.eval: alert_fn error: %s", e)
        else:
            log.warning("tokeymeter.eval: %s", payload["message"])

    def _tighten(self, payload: dict) -> None:
        thr = self._current_threshold()
        if thr is None:
            return
        new = min(1.0, thr + self._tighten_step)
        ok = self._set_threshold(new)
        if ok:
            payload["auto_tightened_to"] = round(new, 4)
            log.warning("tokeymeter.eval: auto-tightened threshold %.3f -> %.3f",
                        thr, new)

    # ── cache threshold accessors (defensive about the cache's internals) ──────

    def _current_threshold(self) -> Optional[float]:
        c = self._cache
        if c is None:
            return None
        for attr in ("_threshold", "threshold"):
            if hasattr(c, attr):
                try:
                    return float(getattr(c, attr))
                except Exception:
                    return None
        return None

    def _set_threshold(self, value: float) -> bool:
        c = self._cache
        if c is None:
            return False
        if hasattr(c, "_threshold"):
            try:
                c._threshold = float(value)
                return True
            except Exception:
                return False
        return False

    def _find(self, hit_id: str) -> Optional[_HitRecord]:
        for r in reversed(self._records):
            if r.hit_id == hit_id:
                return r
        return None


def make_openai_judge(client: object, model: str = "gpt-4o-mini") -> Callable:
    """Build an LLM-judge callable for grade_pending using an OpenAI-style client.
    Returns judge(query, matched_prompt, cached_response) -> bool (True = the
    cached answer correctly answers the new query). Kept out of the core so the
    eval loop has no SDK dependency."""
    def judge(query: str, matched_prompt: str, cached_response: object) -> bool:
        # extract response text best-effort
        text = ""
        try:
            text = cached_response.choices[0].message.content  # OpenAI object
        except Exception:
            text = str(cached_response)
        prompt = (
            "A semantic cache served a stored answer for a NEW question because it "
            "judged them equivalent. Decide if that was CORRECT.\n\n"
            f"NEW QUESTION:\n{query}\n\n"
            f"CACHED QUESTION (the stored answer was written for this):\n{matched_prompt}\n\n"
            f"CACHED ANSWER SERVED:\n{str(text)[:1200]}\n\n"
            "Is the cached answer a CORRECT answer to the NEW question? "
            "Respond ONLY 'YES' or 'NO'.")
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5, temperature=0)
            ans = (r.choices[0].message.content or "").strip().upper()
            return ans.startswith("Y")
        except Exception:
            # fail-safe: if the judge call fails, treat as correct (do NOT inflate
            # the false-positive rate on infrastructure errors)
            return True
    return judge
