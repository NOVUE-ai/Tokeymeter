"""
SalienceCompressor — model-free, local, segment-level prompt compression.

The differentiator tier for Tokeymeter. It sits between the conservative
StructuralCompressor (whitespace/filler/substitution) and the heavy
LLMLinguaCompressor (needs a model), giving substantially deeper token
reduction with ZERO required dependencies — pure stdlib + math — so it runs
in-process, locally, fast, and offline.

Grounded in the prompt-compression literature:
  - Selective Context (Li et al., 2023): prune low self-information units;
    we compute self-information from the prompt's own token statistics
    instead of a language model, so there is no model dependency.
  - The survey line (Parse-Trees-Guided, 2024) noting model-based scorers
    ignore cheap linguistic structure: we use structure (questions,
    instructions, entities, boilerplate) as salience signal.
  - Redundancy removal: near-duplicate segments are dropped (common in RAG
    context where retrieved chunks overlap).

Design contract (Compressor protocol):
  - compress(text) -> CompressionResult, MUST NOT raise.
  - On any failure, fall open: return after == before, safe=False.
  - Carries a budget: never prune below `min_keep_ratio` of segments, and
    always keep query/instruction segments. No silent destruction of meaning.

This is NOT semantic rewriting (that's LLMLingua's job). It is extractive:
it keeps a faithful subset of the original segments, preserving wording and
auditability — every output token appeared in the input.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from tokeymeter.engines.optimization.compression import CompressionResult
from tokeymeter.engines.economics.pricing import estimate_tokens


# Segment splitter: sentences/lines/bullet items. Keeps code/quotes intact by
# not splitting inside fenced blocks (handled by the caller's quote-awareness
# upstream; here we treat fenced blocks as single atomic segments).
_FENCE = re.compile(r"```[\s\S]*?```")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_WORD = re.compile(r"[A-Za-z0-9_]+")


# Structural salience cues — segments matching these are PROTECTED (never pruned).
_QUESTION = re.compile(r"\?\s*$")
_INSTRUCTION = re.compile(
    r"(?i)\b(?:must|should|do not|don'?t|never|always|require[ds]?|"
    r"return|output|format|answer|step|first|then|finally|"
    r"you are|your task|instructions?|rules?|constraints?)\b"
)
# Boilerplate / low-value cues — segments matching these are DEMOTED (prune first).
_BOILERPLATE = re.compile(
    r"(?i)\b(?:as mentioned|as noted|please note|kindly|"
    r"feel free|by the way|in other words|that is to say|"
    r"it is worth noting|needless to say|for what it'?s worth)\b"
)


@dataclass
class SalienceCompressor:
    """Extractive, model-free segment pruning by self-information + structure.

    Args:
        target_ratio:   desired tokens_after / tokens_before (e.g. 0.6 = keep ~60%).
                        Acts as a budget; actual ratio may be higher if protected
                        segments exceed the budget (we never drop protected ones).
        min_keep_ratio: hard floor — never keep fewer than this fraction of segments
                        (safety against over-pruning short prompts).
        protect_questions:    never prune segments ending in '?'.
        protect_instructions: never prune instruction-like segments.
        drop_redundant:       remove near-duplicate segments (Jaccard >= dup_threshold).
        dup_threshold:        token-set Jaccard above which two segments are "duplicate".
        min_segments:         prompts with fewer segments than this are returned
                              unchanged (nothing meaningful to prune).
        query:                OPTIONAL. The question/task the prompt must answer. When
                              set, segments are scored by relevance to the query and the
                              most-relevant segments are PROTECTED from pruning — this is
                              the safety mechanism for RAG/context, where dropping the
                              answer-bearing chunk would silently corrupt the output.
        query_protect_top:    how many top query-relevant segments to protect (default 3).
        query_weight:         how strongly query-relevance boosts a segment's score.
    """
    target_ratio: float = 0.6
    min_keep_ratio: float = 0.34
    protect_questions: bool = True
    protect_instructions: bool = True
    drop_redundant: bool = True
    dup_threshold: float = 0.82
    min_segments: int = 4
    query: Optional[str] = None
    query_protect_top: int = 3
    query_weight: float = 2.0

    # ---- public API (Compressor protocol) ----
    def compress(self, text: str) -> CompressionResult:
        t0 = time.perf_counter()
        try:
            return self._compress_inner(text, t0)
        except Exception as e:  # never raise — fall open
            return CompressionResult(
                before=text, after=text,
                tokens_before=estimate_tokens(text),
                tokens_after=estimate_tokens(text),
                ratio=1.0,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                method="salience",
                safe=False,
                extra={"error": type(e).__name__},
            )

    def for_query(self, query: str) -> "SalienceCompressor":
        """Return a query-aware copy that protects segments answering `query`.

        Use when the question/task is known (RAG, Q&A): the answer-bearing
        context is protected from pruning, the safest mode for context.
        """
        from dataclasses import replace
        return replace(self, query=query)

    # ---- internals ----
    def _compress_inner(self, text: str, t0: float) -> CompressionResult:
        tokens_before = estimate_tokens(text)
        segments = self._segment(text)

        if len(segments) < self.min_segments:
            # nothing meaningful to prune; fall open cleanly (safe=True, ratio 1.0)
            return self._noop(text, tokens_before, t0)

        # 1) score every segment (query-aware if a query is set)
        info = self._self_information(segments)
        query_tokens = set(_WORD.findall(self.query.lower())) if self.query else set()
        scored: List[_Seg] = []
        for idx, seg in enumerate(segments):
            scored.append(self._score_segment(seg, idx, info, query_tokens))

        # 1b) query-aware protection: protect the top-N segments most relevant to
        #     the query, so the answer-bearing context is never pruned.
        if query_tokens:
            ranked = sorted(
                (s for s in scored if not s.protected),
                key=lambda s: s.query_overlap, reverse=True,
            )
            for s in ranked[: self.query_protect_top]:
                if s.query_overlap > 0:
                    s.protected = True

        # 2) drop near-duplicate, non-protected segments
        if self.drop_redundant:
            scored = self._dedupe(scored)

        # 3) select segments to keep: all protected, then highest-scoring others
        #    until we hit the token budget.
        kept = self._select(scored, tokens_before)

        # 4) reassemble in original order
        kept_sorted = sorted(kept, key=lambda s: s.index)
        after = "\n".join(s.text for s in kept_sorted).strip()

        # safety: if we somehow produced empty / longer output, fall open
        tokens_after = estimate_tokens(after)
        if not after or tokens_after >= tokens_before:
            return self._noop(text, tokens_before, t0)

        return CompressionResult(
            before=text, after=after,
            tokens_before=tokens_before, tokens_after=tokens_after,
            ratio=tokens_after / max(tokens_before, 1),
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            method="salience",
            safe=True,
            extra={
                "segments_before": len(segments),
                "segments_after": len(kept_sorted),
                "protected_kept": sum(1 for s in kept_sorted if s.protected),
            },
        )

    def _noop(self, text: str, tokens_before: int, t0: float) -> CompressionResult:
        return CompressionResult(
            before=text, after=text,
            tokens_before=tokens_before, tokens_after=tokens_before,
            ratio=1.0,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            method="salience", safe=True,
            extra={"noop": True},
        )

    def _segment(self, text: str) -> List[str]:
        """Split into segments, treating fenced code blocks as atomic units."""
        parts: List[str] = []
        last = 0
        for m in _FENCE.finditer(text):
            pre = text[last:m.start()]
            parts.extend(self._split_prose(pre))
            parts.append(m.group(0))  # code block kept whole
            last = m.end()
        parts.extend(self._split_prose(text[last:]))
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _split_prose(prose: str) -> List[str]:
        if not prose.strip():
            return []
        return _SENT_SPLIT.split(prose)

    def _self_information(self, segments: List[str]) -> dict:
        """Corpus-free self-information: -log2(p(token)) from this prompt's own
        token frequencies. Rare tokens (within this prompt) carry more info.
        No model, no external corpus — fully local and deterministic."""
        counts: Counter = Counter()
        total = 0
        for seg in segments:
            for w in _WORD.findall(seg.lower()):
                counts[w] += 1
                total += 1
        total = max(total, 1)
        return {w: -math.log2(c / total) for w, c in counts.items()}

    def _score_segment(self, seg: str, idx: int, info: dict, query_tokens: set) -> "_Seg":
        words = _WORD.findall(seg.lower())
        n = max(len(words), 1)
        # mean self-information per token = how information-dense the segment is
        si = sum(info.get(w, 0.0) for w in words) / n

        protected = False
        if self.protect_questions and _QUESTION.search(seg):
            protected = True
        if self.protect_instructions and _INSTRUCTION.search(seg):
            protected = True
        # fenced code is always protected
        if seg.startswith("```"):
            protected = True

        # demote obvious boilerplate
        if _BOILERPLATE.search(seg):
            si *= 0.4

        # query relevance: overlap between this segment and the question/task.
        # Boosts the score so relevant context survives budget-based pruning even
        # if it isn't in the protected top-N.
        query_overlap = 0.0
        if query_tokens:
            seg_set = set(words)
            if seg_set:
                hits = len(seg_set & query_tokens)
                query_overlap = hits / max(len(query_tokens), 1)
                si += self.query_weight * query_overlap

        return _Seg(index=idx, text=seg, words=words, score=si,
                    protected=protected, query_overlap=query_overlap)

    def _dedupe(self, scored: List["_Seg"]) -> List["_Seg"]:
        kept: List[_Seg] = []
        seen_sets: List[set] = []
        for s in scored:
            if s.protected:
                kept.append(s); seen_sets.append(set(s.words)); continue
            ws = set(s.words)
            dup = False
            for prev in seen_sets:
                if not ws or not prev:
                    continue
                j = len(ws & prev) / len(ws | prev)
                if j >= self.dup_threshold:
                    dup = True
                    break
            if not dup:
                kept.append(s); seen_sets.append(ws)
        return kept

    def _select(self, scored: List["_Seg"], tokens_before: int) -> List["_Seg"]:
        budget = int(tokens_before * self.target_ratio)
        floor = max(1, int(math.ceil(len(scored) * self.min_keep_ratio)))

        protected = [s for s in scored if s.protected]
        optional = sorted(
            (s for s in scored if not s.protected),
            key=lambda s: s.score, reverse=True,
        )

        kept: List[_Seg] = list(protected)
        used = sum(estimate_tokens(s.text) for s in kept)

        for s in optional:
            cost = estimate_tokens(s.text)
            if used + cost <= budget or len(kept) < floor:
                kept.append(s); used += cost
        return kept


@dataclass
class _Seg:
    index: int
    text: str
    words: List[str]
    score: float
    protected: bool = False
    query_overlap: float = 0.0
