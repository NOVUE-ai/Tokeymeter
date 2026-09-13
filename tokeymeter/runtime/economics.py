"""Economics engine in the kernel path (W4: ECON-1 cost trace, ECON-2 strict
budgets, ECON-3 approval bridge).

HONESTY RULES (pinned):
- Reported usage BEATS estimates: if the shipped wrapper consumed the
  provider's counts, those are the truth (T1.1 precedence, unchanged).
- A model without a price yields cost_usd=None + cost_source describing why
  — NEVER zero, NEVER invented.
- Stale pricing surfaces: `pricing_age_days` is attached when it exceeds
  the configured warning threshold (T1.2).

Budgets (ECON-2) delegate to the shipped keys module — fingerprint-only
storage, month-roll, and the Phase-1 item-2 concurrency discipline all
inherited, not reimplemented. `budget.mode`:
    "enforce"  exceed → BudgetExceeded raised on the strict path
    "soft"     exceed → verdict "warn", request proceeds
    "approval" exceed → GOV-4 CheckpointHook decides (ECON-3)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .engine import Engine
from .enforcement import (CheckpointHook, record_verdict, run_checkpoint)

from tokeymeter.engines.economics import keys as _keys
from tokeymeter.engines.economics import pricing as _pricing
from tokeymeter.engines.economics import usage as _usage


class BudgetExceeded(PermissionError):
    retryable = False
    policy = "budget"

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"BudgetExceeded({detail})")
        self.detail = detail


class EconomicsEngine(Engine):
    name = "economics"

    def __init__(self, *, budget_key: Optional[str] = None,
                 budget_mode: str = "enforce",
                 checkpoint: Optional[CheckpointHook] = None,
                 pricing_age_warn_days: int = 45) -> None:
        if budget_mode not in ("enforce", "soft", "approval"):
            raise ValueError("budget_mode must be enforce|soft|approval")
        if budget_mode == "approval" and checkpoint is None:
            raise ValueError("budget_mode='approval' requires a checkpoint")
        self._budget_key = budget_key
        self._mode = budget_mode
        self._checkpoint = checkpoint
        self._age_warn = pricing_age_warn_days

    # ------------------------------------------------ before: budget ------
    def before_request(self, ctx: Dict[str, Any]) -> None:
        model = ctx["meta"].get("model", ctx["request"].model)
        est_in = _pricing.estimate_tokens(ctx["request"].payload)
        ctx["meta"]["tokens_in_estimated"] = est_in
        est_cost, _src = _pricing.estimate_cost_with_source(model, est_in, 0)
        ctx["kernel"].hooks.emit_strict("budget_check", ctx)
        if self._budget_key is None:
            return
        try:
            with _keys.key(self._budget_key):
                _keys.check_current(estimated_cost=est_cost or 0.0)
            record_verdict(ctx, "budget", "allow")
        except _keys.KeyBudgetExceeded as exc:
            if self._mode == "soft":
                record_verdict(ctx, "budget", "warn",
                               detail=f"key:{self._budget_key}")
                return
            if self._mode == "approval" and self._checkpoint is not None:
                run_checkpoint(ctx, self._checkpoint,
                               reason=f"budget_exceeded:{self._budget_key}")
                record_verdict(ctx, "budget", "approved_over_budget",
                               detail=f"key:{self._budget_key}")
                return
            record_verdict(ctx, "budget", "block",
                           detail=f"key:{self._budget_key}")
            raise BudgetExceeded(f"key:{self._budget_key}") from exc

    # ------------------------------------------------ after: cost truth ---
    def after_response(self, ctx: Dict[str, Any]) -> None:
        model = ctx["meta"].get("model", ctx["request"].model)
        reported = _usage.consume_reported_usage()
        if reported is not None:
            tokens_in, tokens_out = reported
            usage_source = "reported"
        else:
            payload = ctx.get("response_payload")
            usage = getattr(payload, "usage", None)
            p_in = getattr(usage, "prompt_tokens",
                           getattr(usage, "input_tokens", None))
            p_out = getattr(usage, "completion_tokens",
                            getattr(usage, "output_tokens", None))
            if isinstance(p_in, int) and isinstance(p_out, int):
                tokens_in, tokens_out = p_in, p_out
                usage_source = "response_usage"
            else:
                tokens_in = ctx["meta"].get("tokens_in_estimated", 0)
                out_text = payload if isinstance(payload, str) else ""
                tokens_out = _pricing.estimate_tokens(out_text) \
                    if out_text else 0
                usage_source = "estimated"
        ctx["meta"]["tokens_in"] = tokens_in
        ctx["meta"]["tokens_out"] = tokens_out
        ctx["meta"]["usage_source"] = usage_source

        cost, source = _pricing.estimate_cost_with_source(
            model, tokens_in, tokens_out)
        # A KNOWN cost requires a real price provenance. The pricing
        # module's generic 'default' fallback is a guess — downstream budgets
        # and chargeback must not treat a guess as truth, so cost_usd is None
        # and the guess (with its 'default' source) is kept separately for
        # inspect. Only 'registered'/'list' provenance yields a real number.
        if cost is None or source not in ("registered", "list"):
            ctx["meta"]["cost_usd"] = None            # NEVER zero, NEVER guessed
            ctx["meta"]["cost_source"] = source or "unknown"
            if cost is not None:
                ctx["meta"]["cost_estimated_usd"] = float(cost)
        else:
            ctx["meta"]["cost_usd"] = float(cost)
            ctx["meta"]["cost_source"] = source
            age = _pricing.pricing_age_days()
            if age > self._age_warn:
                ctx["meta"]["pricing_age_days"] = age
            if self._budget_key is not None:
                try:
                    _keys.on_spend(self._budget_key, float(cost), hit=False, shadow=False)
                except Exception:
                    pass  # spend recording is fail-open; enforcement is not
        ctx["kernel"].hooks.emit("cost_recorded", ctx)
