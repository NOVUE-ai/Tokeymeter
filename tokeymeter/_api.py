"""
Tokeymeter — the token-economics layer.  A NOVUE product.

Not a gateway. Not a proxy. An in-process, local-first SDK that *lowers*
your LLM bill — deeper than any gateway bothers to — and proves how much,
per workload, with no inflated claims.

Everyone else charts your LLM costs. Tokeymeter lowers them: in-process,
automatically, and provably — then shows you the bill that's already smaller.

Quick start
-----------
    from tokeymeter import meter

    # Wrap ANY callable — any provider, any API on earth, even a local model.
    # Tokeymeter only assumes "text in, text out".
    @meter(model="gpt-4o-mini", tag="support")
    def ask(prompt: str) -> str:
        return my_llm_call(prompt)        # openai / anthropic / gemini / local / custom

    ask("hello")                          # cached, deduped, compressed, metered
    print(report())                       # measured savings — real tokens x real price

Design promises
---------------
- In-process, not a proxy: no hop, no server to run, microsecond overhead.
- Local-first: content stays in your process; we learn from patterns, never secrets.
- Provider-agnostic: wraps any callable; works with APIs that don't exist yet.
- Honest: every saving is measured, never asserted. Lossless where promised
  (exact cache, single-flight); bounded with a quality budget + fallback where not
  (compression). Never silent quality loss.
- Fail-open: if any optimizer errors, your original call still succeeds.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Union

# The proven engine. Tokeymeter is a marketable, focused surface over it.
import tokeymeter as _engine
# Engine versions of names this facade also defines — bound directly to avoid
# the facade shadowing them in the package namespace (which would recurse).
from .cache_optimize import compress_tools as _eng_compress_tools
from .decorator import no_optimize as _eng_no_optimize

__all__ = [
    "meter",
    "report",
    "reset",
    "compressor",
    "router",
    "metered_route",
    "cascade",
    "optimize_cache",
    "check_cache",
    "compress_tools",
    "no_optimize",
]



def meter(
    fn: Optional[Callable] = None,
    *,
    model: str = "_default",
    tag: Optional[str] = None,
    # cost levers (all on by default — we lower the bill, not just chart it)
    cache: bool = True,
    semantic: bool = False,
    single_flight: bool = True,
    compress: Union[bool, Any] = False,
    compress_query: Optional[str] = None,
    # honesty / safety
    verify_rate: float = 0.0,
    high_stakes: bool = False,
    # measurement-only mode: see what you *would* save without changing behavior
    shadow: bool = False,
    # advanced passthrough
    semantic_threshold: float = 0.92,
    ttl: Optional[float] = None,
    redactor: Optional[Callable[[str], str]] = None,
    prompt_arg: Optional[Union[str, int]] = None,
    store: Optional[Any] = None,
):
    """Wrap any text-in/text-out callable to lower and measure its LLM cost.

    The defaults are the product: exact caching + single-flight collapse +
    metering are ON, so simply decorating a function already reduces spend
    and records measured savings. Turn on `semantic=True` for near-duplicate
    reuse and `compress=True` (or pass a compressor) for deep token reduction.

    Parameters
    ----------
    model:
        Model name used for price-accurate cost estimation (see pricing table).
    tag:
        Workload label ("support", "rag", ...) so `report()` can attribute
        savings per workload / team / feature — curated to your firm.
    cache / single_flight:
        Lossless levers. Exact cache returns the stored answer; single-flight
        collapses concurrent identical calls. On by default.
    semantic:
        Bounded lever. Returns a near-equivalent cached answer above
        `semantic_threshold`. High-stakes calls are never semantically cached.
    compress:
        The differentiator. `True` uses the built-in structural compressor;
        pass a compressor instance (see `tokeymeter.compressor(...)`) for deep
        perplexity-based reduction. Always carries a quality budget + fallback.
    verify_rate:
        Fraction of compressed calls to fidelity-audit (original vs compressed
        output similarity), recorded for honest quality reporting.
    high_stakes:
        Mark a call as never-cache, never-semantic — correctness over savings.
    shadow:
        Measure-only. Computes what you *would* save without altering behavior
        — safe to run in production before flipping optimization on.
    """
    resolved_compressor = _resolve_compress(compress, compress_query)

    return _engine.cache(
        fn,
        model=model,
        tag=tag,
        enabled=cache,
        semantic=semantic,
        semantic_threshold=semantic_threshold,
        single_flight=single_flight,
        compressor=resolved_compressor,
        verify_rate=verify_rate,
        high_stakes=high_stakes,
        shadow=shadow,
        ttl=ttl,
        redactor=redactor,
        prompt_arg=prompt_arg,
        store=store,
    )


def _resolve_compress(compress: Union[bool, Any], query: Optional[str] = None) -> Optional[Any]:
    """bool/str -> built-in compressor; instance -> passthrough; False/None -> off.

    compress=True   -> SalienceCompressor (the differentiator: model-free, deep,
                       redundancy- and salience-aware; zero dependencies).
    compress="light"-> StructuralCompressor (conservative; whitespace/fillers).
    compress="deep" -> LLMLinguaCompressor (opt-in, needs the extra installed).

    All compressors are wrapped in a confidence-gated SafeCompressor: the
    compressed prompt ships only if it passes self-verification (ratio band,
    query-term survival); otherwise it falls back to the original. Fallback is
    rare by design, counted internally, and available via report(detail=True).

    query: when set (RAG/Q&A), makes compression query-aware AND enforces that
           the query's key terms survive — the safe mode for context.
    """
    if compress is False or compress is None:
        return None
    if compress is True:
        c = _engine.SalienceCompressor()
    elif isinstance(compress, str):
        c = compressor(compress)
    else:
        c = compress  # already a Compressor instance
    # wrap in the confidence gate (conservative, strong defaults)
    safe = _engine.SafeCompressor(inner=c)
    if query:
        safe = safe.for_query(query)
    return safe


def compressor(kind: str = "salience", **kwargs) -> Any:
    """Build a compressor for `meter(compress=...)`.

    kind="salience"   -> model-free salience pruning (DEFAULT differentiator):
                         self-information + structural salience + redundancy
                         removal. Zero dependencies, local, fast, extractive.
    kind="light"      -> StructuralCompressor: conservative whitespace/filler/
                         substitution cleanup. Zero deps.
    kind="deep"       -> LLMLinguaCompressor: perplexity-based (opt-in extra).
    """
    kind = kind.lower()
    if kind in ("salience", "default"):
        return _engine.SalienceCompressor(**kwargs)
    if kind in ("light", "structural", "heuristic"):
        return _engine.StructuralCompressor(**kwargs)
    if kind in ("deep", "lingua", "llmlingua", "perplexity"):
        return _engine.LLMLinguaCompressor(**kwargs)
    raise ValueError(
        f"unknown compressor kind {kind!r}; use 'salience', 'light', or 'deep'"
    )


def report(detail: bool = False) -> dict:
    """Measured savings to date: tokens saved, cost saved, hit rates, per tag/model.

    Every figure is computed from real token counts x real list prices — what you
    actually saved, not a projection. In shadow mode, includes `would_have_saved`.

    detail=True also includes `compression` stats: how often compression shipped
    vs fell back to the original, with fallback reasons. Quiet by default; this
    lets you *prove* the compressor is capable (low fallback rate) when you look.
    """
    rep = _engine.savings_report()
    if detail:
        rep["compression"] = _engine.compression_stats()
        rep["ledger_health"] = _engine.savings_ledger_health()
    return rep


def reset() -> None:
    """Clear the local savings ledger and compression stats (fresh window)."""
    _engine.reset_savings()
    _engine.reset_compression_stats()


def doctor() -> dict:
    """Operational self-check, cheap enough to call often.

    Reports the meter's own cache-hit overhead (p50/p95/p99), the savings ledger's
    health, how many fail-open degraded events have fired and from where, and the
    active write mode. Use it to *prove* overhead is near-invisible and to catch a
    regression before it ships.
    """
    from . import overhead as _ovh
    from . import degraded as _deg
    health = _engine.savings_ledger_health()
    mode = ("in_memory" if health.get("in_memory")
            else "buffered" if health.get("buffered") else "sync")
    return {
        "cache_hit_overhead": _ovh.percentiles(),
        "savings_mode": mode,
        "ledger_health": health,
        "degraded_events": _deg.degraded_event_count(),
        "degraded_by_source": _deg.degraded_counts(),
    }


def no_optimize():
    """Context manager: run a block with all optimization off (force live calls)."""
    return _eng_no_optimize()


def router(cheap_model: str, capable_model: str, threshold: float = 0.5, **kwargs):
    """Build a model router: easy prompts -> cheap model, hard -> capable.

    Zero-dependency and conservative by default (routes UP when uncertain, so
    saving money never silently degrades a hard answer). The router *suggests*
    a tier; you wire the actual calls:

        rt = tk.router(cheap_model="gpt-4o-mini", capable_model="gpt-4o")
        d = rt.route(prompt)              # -> RouteDecision
        answer = (cheap_fn if d.tier == "cheap" else capable_fn)(prompt)

    Pass `complexity_fn=` to plug in a learned router (the opt-in deep seam).
    """
    return _engine.Router(
        cheap_model=cheap_model,
        capable_model=capable_model,
        threshold=threshold,
        **kwargs,
    )


def metered_route(
    cheap_fn: Callable,
    capable_fn: Callable,
    *,
    cheap_model: str,
    capable_model: str,
    threshold: float = 0.5,
    tag: Optional[str] = None,
    cache: bool = True,
    semantic: bool = False,
    single_flight: bool = True,
    compress: Union[bool, Any] = False,
    complexity_fn: Optional[Callable[[str], float]] = None,
    prompt_arg: Optional[Union[str, int]] = None,
):
    """Unified routing + optimization: route easy->cheap / hard->capable, then
    run the chosen model through the SAME cache/compress/meter pipeline.

    This is the architecture made whole: routing is a real pipeline stage, not a
    bolt-on. The cheap and capable callables are each metered (and cached and
    compressed) under their own model id, so report() attributes savings
    correctly across both tiers AND across the routing decision itself.

        ask = tk.metered_route(
            cheap_fn=lambda p: gpt4o_mini(p),
            capable_fn=lambda p: gpt4o(p),
            cheap_model="gpt-4o-mini", capable_model="gpt-4o",
            tag="support", compress=True,
        )
        ask("What is your refund policy?")   # routed, cached, compressed, metered

    Routing is conservative (uncertain -> capable). The returned callable exposes
    .route(prompt) for inspection and .router for the underlying Router.
    """
    rt = _engine.Router(
        cheap_model=cheap_model, capable_model=capable_model,
        threshold=threshold, complexity_fn=complexity_fn,
    )
    cheap_metered = meter(
        cheap_fn, model=cheap_model, tag=tag, cache=cache, semantic=semantic,
        single_flight=single_flight, compress=compress, prompt_arg=prompt_arg,
    )
    capable_metered = meter(
        capable_fn, model=capable_model, tag=tag, cache=cache, semantic=semantic,
        single_flight=single_flight, compress=compress, prompt_arg=prompt_arg,
    )

    def _dispatch(prompt, *args, **kwargs):
        decision = rt.route(prompt)
        if decision.tier == "cheap":
            return cheap_metered(prompt, *args, **kwargs)
        return capable_metered(prompt, *args, **kwargs)

    _dispatch.route = rt.route
    _dispatch.router = rt
    return _dispatch


def cascade(
    cheap_fn: Callable,
    capable_fn: Callable,
    *,
    cheap_model: str,
    capable_model: str,
    min_confidence: float = 0.55,
    confidence_fn: Optional[Callable[[str, str], float]] = None,
    tag: Optional[str] = None,
    cache: bool = True,
    compress: Union[bool, Any] = False,
):
    """Cascade: try the cheap model first, escalate to capable only if the cheap
    answer fails a confidence check. Each model path runs through the full
    cache/compress/meter pipeline.

    Unlike routing (decides before calling, from the prompt), a cascade decides
    after the cheap call, from the response — catching cases prompt-based routing
    misjudges. Trade-off: escalated prompts pay for both calls, so net savings
    depend on the escalation rate (measured in report(), never assumed).

        ask = tk.cascade(
            cheap_fn=lambda p: gpt4o_mini(p),
            capable_fn=lambda p: gpt4o(p),
            cheap_model="gpt-4o-mini", capable_model="gpt-4o", tag="support",
        )
        result = ask("What is your refund policy?")   # -> CascadeResult
        print(result.answer, result.tier, result.escalated)

    Pass `confidence_fn=` to plug in a learned answer-quality scorer (opt-in deep seam).
    """
    cheap_metered = meter(cheap_fn, model=cheap_model, tag=tag, cache=cache, compress=compress)
    capable_metered = meter(capable_fn, model=capable_model, tag=tag, cache=cache, compress=compress)
    casc = _engine.Cascade(
        cheap_fn=cheap_metered, capable_fn=capable_metered,
        cheap_model=cheap_model, capable_model=capable_model,
        min_confidence=min_confidence, confidence_fn=confidence_fn,
    )

    def _run(prompt, *args, **kwargs):
        return casc.run(prompt, *args, **kwargs)

    _run.cascade = casc
    return _run


def optimize_cache(messages, min_prefix_tokens: int = 1024, place_breakpoint: bool = True):
    """Reorder/annotate a structured chat prompt to maximize the provider's
    cacheable prefix (OpenAI auto 50% / Anthropic explicit 90% caching).

    Model-free, in-process, content-blind. Returns (optimized_messages, report).
    Semantics preserved — conversation turn order is never changed; only the
    leading static prefix is measured and (optionally) marked with cache_control.

        msgs, rep = tk.optimize_cache(messages)
        if rep.meets_min_prefix:
            ...  # provider caching will engage on the prefix
    """
    return _engine.CacheOptimizer(
        min_prefix_tokens=min_prefix_tokens, place_breakpoint=place_breakpoint
    ).optimize(messages)


def compress_tools(tools):
    """Model-free structural compression of tool/function descriptions (trims
    verbose filler, preserves names + parameter schemas exactly).

        tools, stats = tk.compress_tools(my_tools)
    """
    return _eng_compress_tools(tools)


def check_cache(messages, min_prefix_tokens: int = 1024):
    """Audit a structured prompt's cache health WITHOUT modifying it (Tier A).

    Detects volatile content (timestamps, UUIDs, request IDs, tokens, per-user
    identity) sitting in the cacheable prefix that would silently break provider
    caching — and recommends moving it. Changes nothing; the developer decides.
    Content-blind: findings mask the offending value (never reproduced verbatim).

        rep = tk.check_cache(messages)
        print(rep.estimated_cache_health)   # 'good' | 'broken' | 'too_small' | 'no_prefix'
        for f in rep.volatile_findings:
            print(f.kind, f.message_index, f.recommendation)

    Returns the CacheReport (the messages are not altered).
    """
    _, report = _engine.CacheOptimizer(
        min_prefix_tokens=min_prefix_tokens, place_breakpoint=False, detect_volatile=True
    ).optimize(messages)
    return report
