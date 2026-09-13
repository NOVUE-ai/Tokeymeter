"""
Router — zero-dependency model routing for cost reduction (Camp A).

The idea (FrugalGPT / RouteLLM, grounded): not every prompt needs the biggest
model. Easy prompts go to a cheap model; hard prompts go to a capable one.
Done well this is one of the largest cost levers — but the routing decision
itself must be cheap and trustworthy, or it becomes its own problem.

This router is MODEL-FREE by default, in keeping with Tokeymeter's local-first
thesis: it scores prompt complexity from cheap, explainable signals (length,
reasoning cues, structure, task type) rather than a learned classifier. A
learned router (DistilBERT-class, like the research) can be plugged in later
as an opt-in upgrade — same graded pattern as compression.

Safety stance (mirrors compression):
  - Conservative by default: when complexity is uncertain, route UP (to the
    capable model). Saving money must never silently degrade a hard answer.
  - Escalation hook (cascade): an optional confidence check on the cheap
    model's output can trigger a retry on the capable model.
  - Honest accounting: every routing decision records which tier was chosen,
    the estimated cost, and the estimated saving vs always-capable.

Routing is a *suggestion engine*: it tells you which model tier to use. The
caller wires the actual model calls; the router never makes network calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from tokeymeter.engines.economics.pricing import estimate_cost, estimate_tokens

_WORD = re.compile(r"[A-Za-z0-9_]+")

# Signals that a prompt is HARD (needs the capable model). Each is a cheap,
# explainable heuristic — not a model. Grounded in what routing research finds
# correlates with difficulty: reasoning, code, multi-step, long context.
_HARD_CUES = re.compile(
    r"(?i)\b(?:reason|prove|derive|analyze|analyse|explain why|step by step|"
    r"chain of thought|debug|refactor|optimize|algorithm|complexity|"
    r"trade-?off|architect|design a|implement|theorem|integral|differential|"
    r"compare and contrast|critique|synthesize|edge case|why does|how would)\b"
)
_CODE_CUE = re.compile(r"```|def |class |function |SELECT |import |#include")
_EASY_CUES = re.compile(
    r"(?i)\b(?:what is|who is|when (?:is|was)|where is|define|translate|"
    r"capital of|spell|how many|list the|name the|yes or no|true or false)\b"
)


@dataclass
class RouteDecision:
    tier: str                 # "cheap" | "capable"
    model: str                # concrete model chosen for that tier
    complexity: float         # 0..1 (0 = trivial, 1 = hard) — see win_rate below
    reason: str               # human-readable why
    est_cost_usd: float       # estimated cost at the chosen model
    est_saved_usd: float      # vs always using the capable model
    confident: bool           # False -> caller may want a cascade check
    win_rate: float = 0.0     # P(capable model meaningfully needed) in [0,1].
    #                           The win-rate framing (does the cheap model
    #                           suffice?) is what we route on; `complexity` is
    #                           kept as an alias for backward compatibility and
    #                           equals win_rate for the heuristic scorer.


# ============================================================================
# Win-rate scorer interface — the single-method abstraction.
#
# A scorer answers ONE question: given a prompt, what is the probability that
# the capable (expensive) model is *meaningfully needed* — i.e. that the cheap
# model would NOT do as well? This is the right thing to optimize (not raw
# "complexity"): a short prompt can need the strong model ("prove this lemma")
# and a long one may not ("summarize this text"). Routing on predicted
# quality-win is what prevents the failure mode where an easy-LOOKING-but-hard
# task gets cheaped out and the user gets a bad answer.
#
# Every scorer implements only `score(prompt) -> float in [0,1]`. The heuristic
# below is the zero-dependency default. A trained scorer (our own, from public
# matrix-factorization math, fit on real traffic) can be supplied later behind
# this same interface with no change to the router or the wrapper — "swap the
# brain, keep the wiring."
# ============================================================================


class WinRateScorer:
    """Abstract single-method scorer. Implementations return P(capable needed)."""

    def score(self, prompt: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def batch_score(self, prompts: List[str]) -> List[float]:
        """Vectorizable hook (used for threshold calibration). Default loops."""
        return [self.score(p) for p in prompts]


class HeuristicWinRateScorer(WinRateScorer):
    """Zero-dependency win-rate scorer from cheap, explainable signals.

    Returns P(capable model meaningfully needed) in [0,1], with a reason. This is
    the existing complexity heuristic reframed as a win-rate estimate: the same
    signals (length, hard/easy cues, code, multi-step) now read as evidence that
    the cheap model would or would not suffice.
    """

    def score_with_reason(self, prompt: str) -> Tuple[float, str, bool]:
        if not isinstance(prompt, str) or not prompt.strip():
            return 1.0, "empty/invalid -> route up", False

        words = _WORD.findall(prompt)
        n = len(words)
        score = 0.0
        reasons: List[str] = []

        # 1) length: longer prompts skew harder (more context to reason over)
        if n > 400:
            score += 0.45; reasons.append("very_long")
        elif n > 150:
            score += 0.25; reasons.append("long")
        elif n < 16:
            score -= 0.15; reasons.append("short")

        # 2) explicit hard cues (reasoning/analysis verbs)
        hard_hits = len(_HARD_CUES.findall(prompt))
        if hard_hits:
            score += min(0.20 * hard_hits, 0.45); reasons.append(f"hard_cues:{hard_hits}")

        # 3) code present -> usually needs the capable model
        if _CODE_CUE.search(prompt):
            score += 0.30; reasons.append("code")

        # 4) easy cues (factual/lookup/translate) pull down
        easy_hits = len(_EASY_CUES.findall(prompt))
        if easy_hits:
            score -= min(0.15 * easy_hits, 0.35); reasons.append(f"easy_cues:{easy_hits}")

        # 5) multiple questions / multi-step -> harder
        q = prompt.count("?")
        if q >= 3:
            score += 0.15; reasons.append("multi_question")

        win = max(0.0, min(1.0, 0.4 + score))  # center ~0.4, clamp 0..1

        # confidence: clear when the score is near 0 or near 1; mid-range is
        # uncertain -> the router routes up on low confidence.
        confident = win <= 0.30 or win >= 0.70
        reason = ",".join(reasons) if reasons else "neutral"
        return win, reason, confident

    def score(self, prompt: str) -> float:
        return self.score_with_reason(prompt)[0]


@dataclass
class Router:
    """Win-rate model router (zero-dep heuristic brain by default).

    Routes on predicted *quality-win*: the probability that the capable model is
    meaningfully needed. Above `threshold` -> capable; below (and confident) ->
    cheap. Conservative by default: uncertain -> route up, so cost saving never
    silently degrades a hard answer.

    Args:
        cheap_model:    model id for easy prompts (e.g. "gpt-4o-mini").
        capable_model:  model id for hard prompts (e.g. "gpt-4o").
        threshold:      win-rate above which we route to the capable model.
                        Lower = more cautious (route up sooner) = safer/pricier.
                        Tune with `calibrate()` to hit a target capable-model %.
        est_output_tokens: assumed output size for cost estimation.
        scorer:         a WinRateScorer (the swap seam for a trained brain).
        complexity_fn:  legacy override scorer(prompt)->0..1. If given, it is
                        wrapped as the scorer (kept for backward compatibility).
    """
    cheap_model: str = "gpt-4o-mini"
    capable_model: str = "gpt-4o"
    threshold: float = 0.5
    est_output_tokens: int = 256
    scorer: Optional[WinRateScorer] = None
    complexity_fn: Optional[Callable[[str], float]] = None

    def __post_init__(self):
        if self.scorer is None:
            if self.complexity_fn is not None:
                self.scorer = _CallableScorer(self.complexity_fn)
            else:
                self.scorer = HeuristicWinRateScorer()

    def score(self, prompt: str) -> float:
        """P(capable model meaningfully needed) in [0,1] — the win-rate."""
        try:
            return max(0.0, min(1.0, float(self.scorer.score(prompt))))
        except Exception:
            return 1.0  # fail safe: uncertain -> treat as needing capable

    def route(self, prompt: str) -> RouteDecision:
        win, reason, confident = self._score_with_reason(prompt)
        in_toks = estimate_tokens(prompt)

        cheap_cost = estimate_cost(self.cheap_model, in_toks, self.est_output_tokens)
        capable_cost = estimate_cost(self.capable_model, in_toks, self.est_output_tokens)

        # Conservative rule: route UP when uncertain. Only take the cheap model
        # when win-rate is clearly below threshold AND we're confident.
        if win < self.threshold and confident:
            return RouteDecision(
                tier="cheap", model=self.cheap_model, complexity=win, win_rate=win,
                reason=reason, est_cost_usd=cheap_cost,
                est_saved_usd=max(capable_cost - cheap_cost, 0.0),
                confident=confident,
            )
        return RouteDecision(
            tier="capable", model=self.capable_model, complexity=win, win_rate=win,
            reason=reason if win >= self.threshold else f"{reason} (uncertain -> route up)",
            est_cost_usd=capable_cost, est_saved_usd=0.0, confident=confident,
        )

    def calibrate(self, prompts: List[str], target_capable_pct: float) -> float:
        """Set `threshold` so ~target_capable_pct of `prompts` route to capable.

        Clean-room reimplementation of RouteLLM's threshold calibration: score a
        representative sample, then pick the win-rate percentile that sends the
        target fraction to the capable model. Decouples the spend policy ("route
        ~15% to the capable model") from the scoring mechanism. Returns and sets
        the calibrated threshold.
        """
        target = max(0.0, min(1.0, float(target_capable_pct)))
        scores = sorted(self.scorer.batch_score([p for p in prompts if isinstance(p, str)]))
        if not scores:
            return self.threshold
        # the top `target` fraction (highest win-rates) should go to capable, so
        # the threshold sits at the (1 - target) quantile of the score list.
        idx = int((1.0 - target) * len(scores))
        idx = max(0, min(len(scores) - 1, idx))
        self.threshold = float(scores[idx])
        return self.threshold

    # ---- internal: get score + reason + confidence in one pass ----
    def _score_with_reason(self, prompt: str) -> Tuple[float, str, bool]:
        s = self.scorer
        # Legacy complexity_fn path: preserve original semantics exactly —
        # report "custom_scorer", and on any failure fall back to the heuristic.
        if isinstance(s, _CallableScorer):
            try:
                win = max(0.0, min(1.0, float(s._fn(prompt))))
                return win, "custom_scorer", True
            except Exception:
                return HeuristicWinRateScorer().score_with_reason(prompt)
        if isinstance(s, HeuristicWinRateScorer):
            return s.score_with_reason(prompt)
        # generic scorer: derive confidence from how decisive the score is
        try:
            win = max(0.0, min(1.0, float(s.score(prompt))))
        except Exception:
            return 1.0, "scorer_error -> route up", False
        confident = win <= 0.30 or win >= 0.70
        return win, "scored", confident


class _CallableScorer(WinRateScorer):
    """Adapts a plain callable(prompt)->float into a WinRateScorer (back-compat
    for the legacy `complexity_fn` argument)."""

    def __init__(self, fn: Callable[[str], float]):
        self._fn = fn

    def score(self, prompt: str) -> float:
        return max(0.0, min(1.0, float(self._fn(prompt))))


# ============================================================================
# Cascade — try cheap first, escalate to capable only if the cheap answer
# fails a confidence check. Grounded in FrugalGPT (cascade of models with a
# scoring function) and cascade-routing (Dekoninck et al., 2025).
# ============================================================================

# Signals that a CHEAP MODEL'S RESPONSE is low-confidence / likely inadequate.
# Zero-dep, explainable heuristics — the opt-in deep version uses a scorer model.
_UNSURE_CUES = re.compile(
    r"(?i)\b(?:i'?m not sure|i am not sure|i don'?t know|i cannot|i can'?t (?:help|answer|determine)|"
    r"unclear|unable to|insufficient (?:information|context|data)|"
    r"as an ai|i don'?t have (?:enough|access)|it depends|cannot determine|"
    r"would need more|please provide more|no information)\b"
)


@dataclass
class CascadeResult:
    answer: str
    tier: str                 # "cheap" (accepted) | "capable" (escalated)
    escalated: bool
    confidence: float         # 0..1 confidence in the cheap answer
    reason: str               # why accepted or escalated
    est_cost_usd: float
    est_saved_usd: float      # vs always-capable


@dataclass
class Cascade:
    """Try the cheap model first; escalate to capable only on low confidence.

    Unlike routing (which decides BEFORE calling, from the prompt), a cascade
    decides AFTER the cheap call, from the response — so it can catch cases the
    prompt-based router would misjudge. The cost: a cheap call is "wasted" on
    escalated prompts. Net savings depend on the escalation rate; this is
    measured, not assumed.

    Args:
        cheap_fn:        callable(prompt)->str for the cheap model.
        capable_fn:      callable(prompt)->str for the capable model.
        cheap_model/capable_model: ids for cost accounting.
        confidence_fn:   optional callable(prompt, response)->0..1 to override the
                         zero-dep heuristic (the opt-in learned scorer seam).
        min_confidence:  escalate if confidence < this.
        est_output_tokens: assumed output size for cost estimation.
    """
    cheap_fn: Callable[[str], str]
    capable_fn: Callable[[str], str]
    cheap_model: str = "gpt-4o-mini"
    capable_model: str = "gpt-4o"
    confidence_fn: Optional[Callable[[str, str], float]] = None
    min_confidence: float = 0.55
    est_output_tokens: int = 256

    def run(self, prompt: str, *args, **kwargs) -> CascadeResult:
        in_toks = estimate_tokens(prompt)
        cheap_cost = estimate_cost(self.cheap_model, in_toks, self.est_output_tokens)
        capable_cost = estimate_cost(self.capable_model, in_toks, self.est_output_tokens)

        # 1) always try cheap first
        cheap_answer = self.cheap_fn(prompt, *args, **kwargs)

        # 2) score confidence in the cheap answer
        conf, reason = self._confidence(prompt, cheap_answer)

        # 3) accept or escalate
        if conf >= self.min_confidence:
            return CascadeResult(
                answer=cheap_answer, tier="cheap", escalated=False,
                confidence=conf, reason=reason,
                est_cost_usd=cheap_cost,
                est_saved_usd=max(capable_cost - cheap_cost, 0.0),
            )
        capable_answer = self.capable_fn(prompt, *args, **kwargs)
        return CascadeResult(
            answer=capable_answer, tier="capable", escalated=True,
            confidence=conf, reason=f"{reason} -> escalated",
            # escalated path pays BOTH calls; saving is negative vs always-capable
            est_cost_usd=cheap_cost + capable_cost,
            est_saved_usd=-(cheap_cost),
        )

    def _confidence(self, prompt: str, response: str) -> Tuple[float, str]:
        if self.confidence_fn is not None:
            try:
                c = max(0.0, min(1.0, float(self.confidence_fn(prompt, response))))
                return c, "custom_scorer"
            except Exception:
                pass  # fall through to heuristic

        if not isinstance(response, str) or not response.strip():
            return 0.0, "empty_response"

        conf = 0.8  # start optimistic
        reasons: List[str] = []

        # explicit unsure/refusal phrasing -> low confidence
        if _UNSURE_CUES.search(response):
            conf -= 0.5; reasons.append("unsure_phrasing")

        # extremely short answer to a non-trivial prompt -> suspicious
        if len(response.split()) < 3 and len(prompt.split()) > 8:
            conf -= 0.3; reasons.append("too_short")

        # answer echoes the question with little added -> low value
        if response.strip().lower() == prompt.strip().lower():
            conf -= 0.6; reasons.append("echoed_prompt")

        conf = max(0.0, min(1.0, conf))
        return conf, (",".join(reasons) if reasons else "ok")
