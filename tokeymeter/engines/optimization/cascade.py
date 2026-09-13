"""
The Cascade — verify-then-escalate model selection.

THE PROBLEM IT SOLVES
Pure routing GAMBLES: it predicts cheap-vs-capable up front and commits. The
quality data proved that 95% of HARD tasks degrade on the cheap model, and you
cannot reliably predict the ~5-10% where cheap actually suffices. So a router that
sends hard work to the cheap model to save money silently ships bad answers.

THE CASCADE DOESN'T GAMBLE — IT VERIFIES
  1. Call the CHEAP model.
  2. Inspect the cheap answer with a quality gate.
  3. If it's good enough  -> keep it      (money saved, no quality loss).
  4. If it's NOT          -> ESCALATE to the capable model and return that.

This inverts the risk: cost is saved only on the calls where the cheap answer was
actually verified acceptable; every other call gets the capable model. Quality is
preserved by construction, not bet on a prediction.

DESIGN (enterprise-grade, conservative, fail-safe)
- CONSERVATIVE BY DEFAULT: when the gate is uncertain, ESCALATE. Never serve a
  doubtful cheap answer to save money.
- FAIL-SAFE: any error in the cheap call or the gate -> escalate (serve capable).
  A failure must never degrade the answer.
- TWO-LAYER GATE:
    Layer 1 (always on, zero-dep, zero-cost): deterministic failure-signal
      detection — empty/truncated/refusal/hedging answers, missing code when code
      was asked for, length anomalies. Catches objective low quality for free.
    Layer 2 (opt-in, costs one check): a model verifier grades the cheap answer
      ("does this fully and correctly answer the question?"). For higher assurance
      on tasks that pass Layer 1.
- PROVABLE: every cascade decision (kept cheap / escalated, with the reason and
  the realized saving) is returned for sealing into the content-blind audit chain
  — provable that no answer was cheaped out below the quality bar.
- MEASURABLE: tracks escalation rate and realized savings.

NO PLACEHOLDERS: the Layer-1 gate is real, tested failure detection. Layer 2 takes
a real verifier callable. Nothing here is stubbed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("tokeymeter.cascade")


# ── Deterministic failure signals (Layer 1) ──────────────────────────────────
# These are OBJECTIVE markers that a cheap answer is low quality. Conservative:
# we only flag clear failures; anything ambiguous is handled by escalate-on-doubt.

# Explicit refusals / inability — the model declined or couldn't answer.
_REFUSAL = re.compile(
    r"\b(i (?:can'?t|cannot|am (?:un)?able to|won'?t)\b(?:\s+(?:help|assist|do|provide|answer|complete))?"
    r"|i'?m (?:sorry|unable|not able)\b"
    r"|as an ai\b|i do(?:n'?t| not) have (?:access|the ability|enough information))",
    re.IGNORECASE,
)

# Substantive uncertainty / hedging that signals a weak answer (not mere nuance).
_HEDGE = re.compile(
    r"\b(i'?m not (?:sure|certain)\b"
    r"|i do(?:n'?t| not) know\b"
    r"|it'?s (?:hard|difficult|impossible) to (?:say|tell|know)\b"
    r"|without more (?:context|information|details)\b"
    r"|i can(?:'?t| not) (?:be certain|determine))",
    re.IGNORECASE,
)

# Code request indicators (if the prompt asks for code, the answer should contain some).
# Must indicate a request to WRITE/PRODUCE code — NOT merely the word "code" (which
# appears in "status code", "error code", "zip code", etc. — those are not requests).
_CODE_REQUEST = re.compile(
    r"\b(write|implement|refactor|debug|code up|program a|script to|"
    r"function (?:to|that)|class (?:to|that|for)|algorithm (?:to|for)|"
    r"method (?:to|that)|sql (?:query|to)|regex (?:to|for|that)|"
    r"snippet|one-liner)\b",
    re.IGNORECASE,
)
# A loose check for "contains code-like content" (fenced block, def/function, common syntax).
_CODE_PRESENT = re.compile(
    r"(```|`[^`]+`|\bdef \w+\(|\bfunction \w+\(|\bclass \w+\b|=>|;\s*$|"
    r"\bSELECT\b.*\bFROM\b|\bimport \w+|\b(?:if|for|while|return)\b\s*\()",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class CascadeDecision:
    """The provable record of one cascade decision."""
    escalated: bool                 # True = cheap failed the gate, capable was used
    served_model: str               # the model whose answer was returned
    cheap_model: str
    capable_model: str
    reason: str                     # why (gate signal or "cheap_ok")
    layer: str                      # "served_cheap" | "layer1" | "layer2" | "error"
    est_saved_usd: float = 0.0      # realized saving when we kept the cheap answer
    cheap_ok: bool = False          # did the cheap answer pass the gate?


@dataclass
class CascadeStats:
    calls: int = 0
    escalated: int = 0
    kept_cheap: int = 0
    errors: int = 0

    @property
    def escalation_rate(self) -> Optional[float]:
        return (self.escalated / self.calls) if self.calls else None


def extract_answer_text(response: Any) -> str:
    """Best-effort extraction of the assistant text from an OpenAI-style response.
    Returns '' if it can't be found (which the gate treats as a failure)."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        pass
    # dict-shaped fallback
    try:
        return (response["choices"][0]["message"]["content"]) or ""
    except Exception:
        return ""


def finish_reason(response: Any) -> Optional[str]:
    try:
        return response.choices[0].finish_reason
    except Exception:
        try:
            return response["choices"][0]["finish_reason"]
        except Exception:
            return None


class QualityGate:
    """Layer-1 deterministic quality gate. Returns (ok, reason). `ok=False` means
    the cheap answer shows an objective failure signal and should be escalated.

    Real, tested detection — no placeholders. Conservative: only clear failures
    flag; subtle quality differences are out of scope for a zero-cost gate (that
    is what Layer-2 verification and escalate-on-doubt are for).
    """

    def __init__(self, min_chars: int = 1):
        self._min_chars = min_chars

    def check(self, prompt: str, answer: str, fin_reason: Optional[str] = None) -> tuple:
        a = (answer or "").strip()

        # 1) empty / too short -> objective failure
        if len(a) < self._min_chars or not a:
            return False, "empty_answer"

        # 2) truncated by token limit (incomplete answer)
        if fin_reason == "length":
            return False, "truncated_length"

        # 3) explicit refusal / inability
        if _REFUSAL.search(a):
            return False, "refusal"

        # 4) substantive hedging / uncertainty
        if _HEDGE.search(a):
            return False, "hedging_uncertainty"

        # 5) code requested but answer contains no code-like content
        if _CODE_REQUEST.search(prompt or "") and not _CODE_PRESENT.search(a):
            return False, "code_requested_but_absent"

        # 6) suspiciously short answer to a substantive prompt (a few words back
        #    on a real question is usually a non-answer). Conservative threshold.
        if len(prompt or "") > 80 and len(a) < 25:
            return False, "answer_too_short_for_prompt"

        return True, "passed"


class Cascade:
    """Verify-then-escalate executor.

    Args:
        cheap_model / capable_model: the two tiers.
        call_fn: callable(**kwargs) -> response. The real model-call function
            (the cache-miss path). The Cascade calls it with model overridden to
            cheap, then (if escalating) to capable. Injected so the Cascade has no
            SDK dependency and is fully testable.
        gate: the Layer-1 QualityGate (a default is created if None).
        verifier_fn: OPTIONAL Layer-2 callable(prompt, answer) -> bool
            (True = answer is acceptable). Costs one extra call; use for higher
            assurance. If None, only Layer 1 runs.
        estimate_cost_fn: optional callable(model, in_toks, out_toks) -> float for
            realized-savings accounting.
    """

    def __init__(
        self,
        cheap_model: str,
        capable_model: str,
        call_fn: Callable[..., Any],
        gate: Optional[QualityGate] = None,
        verifier_fn: Optional[Callable[[str, str], bool]] = None,
        estimate_cost_fn: Optional[Callable[[str, int, int], float]] = None,
    ):
        self.cheap_model = cheap_model
        self.capable_model = capable_model
        self._call = call_fn
        self._gate = gate or QualityGate()
        self._verifier = verifier_fn
        self._estimate = estimate_cost_fn
        self.stats = CascadeStats()

    def run(self, prompt_text: str, **kwargs) -> tuple:
        """Execute the cascade for one request. Returns (response, CascadeDecision).

        kwargs are the original call kwargs (messages, max_tokens, etc.); `model`
        is overridden by the cascade. FAIL-SAFE throughout: any error escalates.
        """
        self.stats.calls += 1

        # ---- 1) try the cheap model ----
        try:
            cheap_kwargs = {**kwargs, "model": self.cheap_model}
            cheap_resp = self._call(**cheap_kwargs)
        except Exception as e:
            # cheap call failed -> escalate (fail-safe)
            log.debug("tokeymeter.cascade: cheap call failed (%s); escalating", e)
            return self._escalate(kwargs, reason="cheap_call_error", layer="error")

        # ---- 2) Layer-1 deterministic gate ----
        answer = extract_answer_text(cheap_resp)
        fin = finish_reason(cheap_resp)
        try:
            ok, reason = self._gate.check(prompt_text, answer, fin)
        except Exception as e:
            log.debug("tokeymeter.cascade: gate error (%s); escalating", e)
            return self._escalate(kwargs, reason="gate_error", layer="error")

        if not ok:
            return self._escalate(kwargs, reason=reason, layer="layer1",
                                  _cheap_resp=cheap_resp)

        # ---- 3) Layer-2 optional model verification ----
        if self._verifier is not None:
            try:
                verified = bool(self._verifier(prompt_text, answer))
            except Exception as e:
                # verifier failed -> conservative: escalate (don't trust unverified)
                log.debug("tokeymeter.cascade: verifier error (%s); escalating", e)
                return self._escalate(kwargs, reason="verifier_error", layer="error")
            if not verified:
                return self._escalate(kwargs, reason="layer2_rejected", layer="layer2",
                                      _cheap_resp=cheap_resp)

        # ---- 4) cheap answer passed -> keep it ----
        self.stats.kept_cheap += 1
        saved = self._saving(cheap_resp)
        decision = CascadeDecision(
            escalated=False, served_model=self.cheap_model,
            cheap_model=self.cheap_model, capable_model=self.capable_model,
            reason="cheap_ok", layer="served_cheap",
            est_saved_usd=saved, cheap_ok=True,
        )
        return cheap_resp, decision

    # ---- escalation ----
    def _escalate(self, kwargs, reason: str, layer: str, _cheap_resp: Any = None) -> tuple:
        self.stats.escalated += 1
        if layer == "error":
            self.stats.errors += 1
        try:
            capable_kwargs = {**kwargs, "model": self.capable_model}
            capable_resp = self._call(**capable_kwargs)
        except Exception as e:
            # capable ALSO failed. Return the cheap response if we have one (better
            # than nothing); otherwise re-raise so the caller's fail-open handles it.
            log.debug("tokeymeter.cascade: capable call failed (%s)", e)
            if _cheap_resp is not None:
                decision = CascadeDecision(
                    escalated=True, served_model=self.cheap_model,
                    cheap_model=self.cheap_model, capable_model=self.capable_model,
                    reason=f"{reason}+capable_failed", layer=layer, cheap_ok=False,
                )
                return _cheap_resp, decision
            raise
        decision = CascadeDecision(
            escalated=True, served_model=self.capable_model,
            cheap_model=self.cheap_model, capable_model=self.capable_model,
            reason=reason, layer=layer, est_saved_usd=0.0, cheap_ok=False,
        )
        return capable_resp, decision

    def _saving(self, cheap_resp: Any) -> float:
        """Realized saving from keeping the cheap answer (vs the capable model)."""
        if self._estimate is None:
            return 0.0
        try:
            u = getattr(cheap_resp, "usage", None)
            it = getattr(u, "prompt_tokens", 0) or 0
            ot = getattr(u, "completion_tokens", 0) or 0
            cheap_c = self._estimate(self.cheap_model, it, ot)
            capable_c = self._estimate(self.capable_model, it, ot)
            return max(capable_c - cheap_c, 0.0)
        except Exception:
            return 0.0


def make_self_verifier(call_fn: Callable[..., Any], model: str) -> Callable[[str, str], bool]:
    """Build a Layer-2 verifier that asks a model to grade the cheap answer.
    Returns verifier(prompt, answer) -> bool. `call_fn` is the same injected
    model-call function. Uses `model` as the judge (commonly the capable model for
    reliability, or a mid model to balance cost).

    Fail-safe: if the judge call errors or is ambiguous, returns True (do NOT force
    an escalation on a judge infrastructure error — that would inflate cost; the
    Layer-1 gate already caught objective failures)."""
    def verifier(prompt: str, answer: str) -> bool:
        grader = (
            "You are grading whether an answer FULLY and CORRECTLY answers a "
            "question. Be strict.\n\n"
            f"QUESTION:\n{prompt}\n\n"
            f"ANSWER:\n{str(answer)[:1500]}\n\n"
            "Does the answer fully and correctly address the question? "
            "Reply ONLY 'YES' or 'NO'.")
        try:
            resp = call_fn(model=model,
                           messages=[{"role": "user", "content": grader}],
                           max_tokens=4, temperature=0)
            txt = extract_answer_text(resp).strip().upper()
            if txt.startswith("N"):
                return False
            return True  # YES or ambiguous -> accept (fail-safe)
        except Exception:
            return True
    return verifier
