"""
Prompt compression — Method 2 from the cost-reduction roadmap.

Compresses input prompts before they reach the cache and the API.
Three compressors ship by default, all composable:

  - StructuralCompressor: rule-based, deterministic, zero deps.
    Whitespace normalization, politeness-filler removal, few-shot
    deduplication. Conservative by default; aggressive options opt-in.

  - LLMLinguaCompressor: wraps Microsoft's LLMLingua-2 (a small BERT-class
    classifier that scores per-token importance). Requires
    pip install tokeymeter[compression]. ~50ms/prompt on CPU.

  - compose(*compressors): stack them. Each runs on the output of the
    previous. Per-stage failure is logged and the stage is skipped —
    composition never makes the output worse than the input.

Pipeline placement:
  prompt → compress → cache lookup → (miss) → cache + LLM call

Two verbose prompts that compress to the same thing share a cache entry.
Compression compounds with caching.

Fidelity contract:
  Every compressor returns a CompressionResult. On any internal error,
  CompressionResult.after == CompressionResult.before and .safe == False.
  The decorator treats safe=False as "compression skipped, use original".

  Compression NEVER makes a call fail. Worst case: it's a no-op.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from tokeymeter.engines.economics.pricing import estimate_tokens

log = logging.getLogger("tokeymeter.compression")


@dataclass
class CompressionResult:
    """The output of a single compress() call.

    Fields:
        before:         original prompt text
        after:          compressed prompt text
        tokens_before:  estimated token count of `before`
        tokens_after:   estimated token count of `after`
        ratio:          tokens_after / tokens_before (lower = better compression)
        duration_ms:    wall time to produce the compression
        method:         human-readable identifier (e.g. "structural",
                        "llmlingua", "compose:structural→llmlingua")
        safe:           True if compression succeeded;
                        False if it failed and `after` equals `before`
    """
    before: str
    after: str
    tokens_before: int
    tokens_after: int
    ratio: float
    duration_ms: float
    method: str
    safe: bool = True
    extra: dict = field(default_factory=dict)


class Compressor(Protocol):
    """Compressor protocol. Anything callable with this shape works.

    Implementations MUST NOT raise. On failure, return a CompressionResult
    with `safe=False` and `after == before`.
    """
    def compress(self, text: str) -> CompressionResult: ...


# ============================================================
#                  StructuralCompressor
# ============================================================

# Conservative defaults: phrases that almost always have no information value.
# We use word boundaries and case-insensitive matching. Multi-word patterns
# come first so they win over their sub-patterns.
_DEFAULT_FILLERS = [
    r"\bcould you please kindly\b",
    r"\bcould you kindly\b",
    r"\bcould you please\b",
    r"\bplease kindly\b",
    r"\bif you wouldn'?t mind,?\b",
    r"\bif it'?s not too much trouble,?\b",
    r"\bas (?:I )?mentioned (?:earlier|before|previously),?\b",
    r"\bI would like (?:you )?to\b",
    r"\bI'?d like (?:you )?to\b",
    r"\bgo ahead and\b",
    r"\bbasically,?\b",
    r"\bessentially,?\b",
]

# Substitutions: pattern → replacement (instead of pure deletion).
_DEFAULT_SUBSTITUTIONS = [
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bat the present time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
]

# Whitespace patterns
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")

# Few-shot example marker (configurable via __init__)
_DEFAULT_EXAMPLE_MARKER = re.compile(
    r"(?im)^\s*(?:example\s*\d*|input\s*\d*\s*:|q\s*\d*\s*:)",
)


class StructuralCompressor:
    """Rule-based compressor. Deterministic, zero dependencies.

    All transformations are reversible-in-spirit: they remove text that
    is unlikely to affect model output. We do NOT do semantic rewriting
    here — that's the LLMLingua compressor's job.

    Default behavior is conservative. To get more aggressive compression,
    enable `remove_fillers` (on by default), `substitutions` (on by default),
    and set `max_examples` to a small integer.
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        remove_fillers: bool = True,
        substitutions: bool = True,
        custom_fillers: Optional[List[str]] = None,
        custom_substitutions: Optional[List[tuple]] = None,
        max_examples: Optional[int] = None,
        skip_inside_quotes: bool = True,
    ):
        """
        Args:
            normalize_whitespace: collapse multiple spaces, trim trailing,
                normalize excessive newlines.
            remove_fillers: strip polite filler phrases ("could you please",
                "if you wouldn't mind", etc).
            substitutions: replace verbose phrases with terse equivalents
                ("in order to" → "to").
            custom_fillers: list of regex strings to add to the default
                filler list. Use raw strings, e.g. r"\\bplease\\b".
            custom_substitutions: list of (regex_str, replacement) tuples.
            max_examples: if set, keeps only the first N detected few-shot
                examples in the prompt. Detection is heuristic; defaults
                are conservative.
            skip_inside_quotes: don't touch text inside "..." or '...'
                or ```...``` — preserves user-quoted content and code blocks.
        """
        self._normalize_ws = normalize_whitespace
        self._remove_fillers = remove_fillers
        self._substitutions = substitutions
        self._skip_quotes = skip_inside_quotes
        self._max_examples = max_examples

        fillers = list(_DEFAULT_FILLERS)
        if custom_fillers:
            fillers.extend(custom_fillers)
        self._filler_patterns = [re.compile(p, re.IGNORECASE) for p in fillers]

        subs = list(_DEFAULT_SUBSTITUTIONS)
        if custom_substitutions:
            subs.extend(custom_substitutions)
        self._sub_patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in subs]

    # ---- Quote-aware transformation ----

    def _apply_outside_quotes(self, text: str, transform: Callable[[str], str]) -> str:
        """Apply `transform` to text segments OUTSIDE quoted regions.

        Recognized quoted regions: "...", '...', ```...```, `...`.
        """
        if not self._skip_quotes:
            return transform(text)

        # Pattern that matches a "preserved" run (code block, string)
        # or any other character. We rebuild the string by applying the
        # transform only to non-preserved regions.
        preserved = re.compile(
            r"```[\s\S]*?```"        # fenced code block
            r"|`[^`\n]*`"            # inline code
            r'|"(?:\\.|[^"\\])*"'    # double-quoted string
            r"|'(?:\\.|[^'\\])*'"    # single-quoted string
        )
        out_parts: List[str] = []
        last_end = 0
        for m in preserved.finditer(text):
            # Plain segment before the match → transform
            plain = text[last_end:m.start()]
            if plain:
                out_parts.append(transform(plain))
            # Preserved segment → pass through unchanged
            out_parts.append(m.group(0))
            last_end = m.end()
        tail = text[last_end:]
        if tail:
            out_parts.append(transform(tail))
        return "".join(out_parts)

    # ---- Individual transformations ----

    def _strip_fillers(self, text: str) -> str:
        for pat in self._filler_patterns:
            text = pat.sub("", text)
        return text

    def _apply_substitutions(self, text: str) -> str:
        for pat, repl in self._sub_patterns:
            text = pat.sub(repl, text)
        return text

    def _normalize_whitespace_text(self, text: str) -> str:
        text = _MULTI_SPACE.sub(" ", text)
        text = _TRAILING_WS.sub("\n", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        # Strip leading/trailing whitespace on the whole text
        return text.strip()

    def _collapse_ws_segment(self, text: str) -> str:
        """Collapse runs of whitespace WITHOUT stripping segment boundaries.

        Used per-segment via _apply_outside_quotes so that whitespace inside
        fenced code blocks / quoted strings is preserved verbatim and spaces
        adjacent to those regions are not eaten. The whole-text strip happens
        once at the end of compress()."""
        text = _MULTI_SPACE.sub(" ", text)
        text = _TRAILING_WS.sub("\n", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text

    def _cap_examples(self, text: str, n: int) -> str:
        """Heuristic: if the text appears to contain >n labelled examples,
        keep the first n and drop the rest. Conservative — only triggers
        when there's a clear pattern.
        """
        matches = list(_DEFAULT_EXAMPLE_MARKER.finditer(text))
        if len(matches) <= n:
            return text
        # Cut at the (n+1)th match
        cut_at = matches[n].start()
        return text[:cut_at].rstrip()

    # ---- Public API ----

    def compress(self, text: str) -> CompressionResult:
        start = time.perf_counter()
        if not isinstance(text, str) or not text:
            tb = estimate_tokens(text or "")
            return CompressionResult(
                before=text or "", after=text or "",
                tokens_before=tb, tokens_after=tb,
                ratio=1.0,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="structural", safe=True,
            )

        try:
            after = text

            if self._substitutions:
                after = self._apply_outside_quotes(after, self._apply_substitutions)
            if self._remove_fillers:
                after = self._apply_outside_quotes(after, self._strip_fillers)
            if self._normalize_ws:
                # Preserve whitespace inside fenced code blocks / quoted strings
                # verbatim by normalizing only outside protected regions, then
                # strip the whole result once.
                after = self._apply_outside_quotes(after, self._collapse_ws_segment)
                after = after.strip()
            if self._max_examples is not None and self._max_examples >= 0:
                after = self._cap_examples(after, self._max_examples)

            tb = estimate_tokens(text)
            ta = estimate_tokens(after)
            ratio = ta / tb if tb else 1.0
            return CompressionResult(
                before=text, after=after,
                tokens_before=tb, tokens_after=ta,
                ratio=ratio,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="structural",
                safe=True,
            )
        except Exception as e:
            log.debug("tokeymeter.compression: structural compressor failed: %s", e)
            tb = estimate_tokens(text)
            return CompressionResult(
                before=text, after=text,
                tokens_before=tb, tokens_after=tb,
                ratio=1.0,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="structural",
                safe=False,
            )

    def compress_ir(self, ir):
        """Phase 1: compress a PromptIR span-aware, returning a new PromptIR.

        Contract:
          * STRING-sourced IRs delegate to compress(reconstruct(ir)) wholesale,
            so the final text is BYTE-IDENTICAL to today's behavior (per-span
            compression of a flat string is NOT equivalent at span boundaries,
            so we deliberately do not do it).
          * MESSAGE-sourced IRs are compressed PER SPAN: spans with
            must_be_verbatim (e.g. SYSTEM_INSTRUCTION, TOOL_DEFINITION, CODE,
            QUOTED) are passed through UNCHANGED; other spans (user/assistant
            content) are conservatively compressed. Messages are independent
            units, so there is no cross-span boundary to corrupt.

        Never raises: on any error it returns the input IR unchanged (fail-open).
        """
        from dataclasses import replace as _replace
        from tokeymeter.engines.optimization.prompt_ir import PromptIR, Span, Origin, SpanKind, permissions_for, reconstruct
        try:
            if ir.source_kind != "messages":
                after_text = self.compress(reconstruct(ir)).after
                span = Span(0, SpanKind.USER_QUERY, after_text, Origin.USER,
                            permissions_for(SpanKind.USER_QUERY))
                return PromptIR((span,), "string", ir.parser_conf,
                                ir.workload_tag, ir.lineage, ir.high_stakes)
            new_spans = []
            for s in ir.spans:
                if s.permissions.must_be_verbatim:
                    new_spans.append(s)
                else:
                    new_spans.append(_replace(s, text=self.compress(s.text).after))
            return PromptIR(tuple(new_spans), "messages", ir.parser_conf,
                            ir.workload_tag, ir.lineage, ir.high_stakes,
                            _messages_meta=ir._messages_meta)
        except Exception as e:
            log.debug("tokeymeter.compression: compress_ir failed, returning input: %s", e)
            return ir


# ============================================================
#                   LLMLinguaCompressor
# ============================================================

class LLMLinguaCompressor:
    """Wraps Microsoft's LLMLingua-2 model-based compressor.

    LLMLingua-2 is a BERT-class classifier (~100M params) trained to score
    each token's importance for downstream LLM output quality. We keep
    high-importance tokens and drop low-importance ones to hit a target
    compression ratio. The compressor runs locally on CPU (~50ms/prompt
    for ~500 tokens; GPU is faster).

    Reference:
      "LLMLingua-2: Data Distillation for Efficient and Faithful
       Task-Agnostic Prompt Compression" — Microsoft Research, 2024.
       https://github.com/microsoft/LLMLingua

    Requires `pip install tokeymeter[compression]`. On missing dep, construction
    raises a clear RuntimeError so the user knows what to install.
    """

    def __init__(
        self,
        target_ratio: float = 0.5,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_llmlingua2: bool = True,
    ):
        """
        Args:
            target_ratio: Fraction of tokens to KEEP (0.5 = compress to half).
                Lower = more aggressive. Range [0.1, 0.99].
            model_name: HuggingFace model id. Default uses the multilingual
                LLMLingua-2 base model.
            device: "cpu" / "cuda" / "mps". Defaults to CPU.
            use_llmlingua2: True for v2 (recommended). False falls back to v1.
        """
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LLMLinguaCompressor requires the [compression] extras. "
                "Install with: pip install tokeymeter[compression]"
            ) from e

        model_name = model_name or (
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        )
        self._target_ratio = max(0.1, min(0.99, float(target_ratio)))
        self._model_name = model_name
        try:
            self._impl = PromptCompressor(
                model_name=model_name,
                device_map=device or "cpu",
                use_llmlingua2=use_llmlingua2,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"LLMLinguaCompressor: failed to load model {model_name!r}: {e}"
            ) from e

    def compress(self, text: str) -> CompressionResult:
        start = time.perf_counter()
        if not isinstance(text, str) or not text:
            tb = estimate_tokens(text or "")
            return CompressionResult(
                before=text or "", after=text or "",
                tokens_before=tb, tokens_after=tb,
                ratio=1.0,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="llmlingua", safe=True,
            )

        try:
            # rate is the fraction to KEEP in LLMLingua's API
            result = self._impl.compress_prompt(text, rate=self._target_ratio)
            compressed = (
                result.get("compressed_prompt", text)
                if isinstance(result, dict)
                else text
            )
            tb = estimate_tokens(text)
            ta = estimate_tokens(compressed)
            ratio = ta / tb if tb else 1.0
            return CompressionResult(
                before=text, after=compressed,
                tokens_before=tb, tokens_after=ta,
                ratio=ratio,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="llmlingua",
                safe=True,
                extra={"model": self._model_name},
            )
        except Exception as e:
            log.debug("tokeymeter.compression: llmlingua failed: %s", e)
            tb = estimate_tokens(text)
            return CompressionResult(
                before=text, after=text,
                tokens_before=tb, tokens_after=tb,
                ratio=1.0,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="llmlingua",
                safe=False,
            )


# ============================================================
#                       compose()
# ============================================================

class _ComposedCompressor:
    """Internal: chains multiple compressors. Each runs on the previous output."""

    def __init__(self, *compressors: Compressor):
        if not compressors:
            raise ValueError("compose() requires at least one compressor")
        self._compressors = compressors

    def compress(self, text: str) -> CompressionResult:
        start = time.perf_counter()
        if not isinstance(text, str) or not text:
            tb = estimate_tokens(text or "")
            return CompressionResult(
                before=text or "", after=text or "",
                tokens_before=tb, tokens_after=tb,
                ratio=1.0,
                duration_ms=(time.perf_counter() - start) * 1000,
                method="compose", safe=True,
            )

        current = text
        methods: List[str] = []
        any_unsafe = False
        for c in self._compressors:
            try:
                r = c.compress(current)
                if r.safe:
                    current = r.after
                    methods.append(r.method)
                else:
                    any_unsafe = True
            except Exception as e:
                log.debug("tokeymeter.compression: stage in compose() failed: %s", e)
                any_unsafe = True

        tb = estimate_tokens(text)
        ta = estimate_tokens(current)
        ratio = ta / tb if tb else 1.0
        return CompressionResult(
            before=text, after=current,
            tokens_before=tb, tokens_after=ta,
            ratio=ratio,
            duration_ms=(time.perf_counter() - start) * 1000,
            method="compose:" + "→".join(methods) if methods else "compose:noop",
            # safe=True even if some stages failed — as long as we returned
            # something at least as good as the original. any_unsafe is in extra.
            safe=True,
            extra={"any_stage_unsafe": any_unsafe, "stage_methods": methods},
        )


def compose(*compressors: Compressor) -> Compressor:
    """Chain multiple compressors. Returns a Compressor.

    Example:
        from tokeymeter.engines.optimization.compression import compose, StructuralCompressor
        pipeline = compose(
            StructuralCompressor(),
            # LLMLinguaCompressor(target_ratio=0.5),  # if installed
        )
        result = pipeline.compress("Could you please tell me about Python...")
    """
    return _ComposedCompressor(*compressors)


# ============================================================
#                  Safe compress entry point
# ============================================================

def safe_compress(
    compressor: Optional[Compressor], text: str
) -> CompressionResult:
    """Module-level helper: run a compressor, never raise, return a result.

    If `compressor` is None or text is empty, returns a no-op result.
    """
    if compressor is None or not isinstance(text, str) or not text:
        tb = estimate_tokens(text or "")
        return CompressionResult(
            before=text or "", after=text or "",
            tokens_before=tb, tokens_after=tb,
            ratio=1.0,
            duration_ms=0.0,
            method="noop",
            safe=True,
        )
    try:
        return compressor.compress(text)
    except Exception as e:
        log.debug("tokeymeter.compression: compressor raised: %s", e)
        tb = estimate_tokens(text)
        return CompressionResult(
            before=text, after=text,
            tokens_before=tb, tokens_after=tb,
            ratio=1.0,
            duration_ms=0.0,
            method="failed",
            safe=False,
        )
