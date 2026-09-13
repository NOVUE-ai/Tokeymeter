"""
Query-aware prompt compression with a budget controller.

Built clean-room from the two state-of-the-art architectures (their INSIGHTS, not
their code):

  - LongLLMLingua (question-aware coarse-to-fine): compression should be driven by
    relevance to the actual QUESTION. Context chunks irrelevant to the query are
    pruned first; the answer-bearing chunk is preserved. This is where the RAG win
    comes from (the paper reports up to +21% accuracy at ~1/4 the tokens by
    filtering noise the model would otherwise get "lost in the middle" of).

  - LLMLingua (budget controller): different PARTS of a prompt tolerate different
    compression. Instructions need ~0-20% compression (preserve clarity), questions
    need ~0% (preserve intent), and demonstrations/context tolerate 60-80% because
    they are redundant. A single global ratio is wrong; the budget must be
    allocated per segment role.

This module combines both: it CLASSIFIES each segment by role, applies a
ROLE-SPECIFIC budget, and within the compressible (context) segments prunes by
IDF-weighted query relevance — so distinctive query-term matches survive and
generic filler is dropped.

It is EXTRACTIVE (keeps a faithful subset of original segments — every output token
appeared in the input), so the result is auditable and safe for code/SQL/tables,
which the literature says must never be abstractively rewritten. Pure stdlib,
fail-open (never raises; worst case is a no-op), and never drops a protected
instruction or the query.

This sits ALONGSIDE the existing SalienceCompressor (which stays unchanged for the
general/no-query case). Use this when the question/task is known — RAG, Q&A,
long-context — which is exactly where compression pays off.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from tokeymeter.engines.optimization.compression import CompressionResult
from tokeymeter.engines.economics.pricing import estimate_tokens

# Reuse the segment splitter + cues from salience to stay consistent.
from tokeymeter.engines.optimization.salience import (
    _WORD, _FENCE, _SENT_SPLIT, _QUESTION, _BOILERPLATE,
)


# Generic words that should NOT count as meaningful query overlap (they match
# everything, so they carry no relevance signal). Kept small and stdlib.
_STOPISH = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "as", "at", "by", "from",
    "how", "what", "why", "when", "where", "which", "who", "do", "does", "did",
    "can", "could", "would", "should", "will", "explain", "describe", "give",
    "please", "tell", "i", "you", "we", "they", "my", "your", "about", "into",
})


# Stricter instruction detection for the query-aware path. The shared _INSTRUCTION
# cue list is broad (it includes words like "first"/"answer"/"step" that also appear
# in ordinary prose), which wrongly protects narrative context. Here an instruction
# must look DIRECTIVE: an imperative opener, a 2nd-person directive, or an explicit
# task/format/rule statement — not merely containing a cue word.
_DIRECTIVE = re.compile(
    r"(?i)("
    r"^\s*(?:please\s+)?(?:do not|don'?t|never|always|ensure|make sure|use|provide|"
    r"return|output|answer|respond|write|explain|describe|summarize|list|format|"
    r"include|avoid|consider|note that|follow)\b"          # imperative opener
    r"|(?:^|\b)you (?:are|must|should|will|need to|have to)\b"  # 2nd-person directive
    r"|(?:^|\b)your (?:task|job|goal|role|response|answer|output)\b"
    r"|\b(?:instructions?|rules?|constraints?|requirements?|guidelines?)\s*:?\s*$"
    r"|\b(?:must|should|do not|don'?t|never|always)\s+(?:be|use|return|output|include|"
    r"follow|answer|provide|avoid|contain)\b"               # modal directive
    r")"
)


def _is_directive(seg: str) -> bool:
    return bool(_DIRECTIVE.search(seg.strip()))


class _SegmentRole:
    INSTRUCTION = "instruction"   # task/format/rule statements — protect
    QUESTION = "question"         # the actual question — protect
    CODE = "code"                 # fenced block — protect (never rewrite code)
    CONTEXT = "context"           # retrieved/background prose — compressible


@dataclass
class _QSeg:
    index: int
    text: str
    words: List[str]
    role: str
    relevance: float = 0.0        # IDF-weighted query relevance (context segments)
    info: float = 0.0             # mean self-information (tie-breaker)
    keep: bool = True


@dataclass
class QueryAwareCompressor:
    """Query-aware, budget-controlled extractive compressor.

    Args:
        query: the question/task the prompt must answer. REQUIRED for the
            query-aware path; without it, behaves conservatively (context kept).
        context_target_ratio: target fraction of CONTEXT tokens to keep (the budget
            for compressible context). Instructions/questions/code are exempt.
            0.5 = keep ~half the context (drop the least query-relevant half).
        min_context_keep: never drop a context segment whose relevance is at or
            above this percentile of context relevance (safety: always keep the
            clearly-relevant chunks regardless of budget).
        protect_top_k: always protect the K most query-relevant context segments
            (the answer-bearing chunks) even under an aggressive budget.
        drop_redundant: remove near-duplicate context segments first.
        dup_threshold: Jaccard above which two context segments are duplicates.
        min_segments: prompts with fewer segments are returned unchanged.
    """
    query: Optional[str] = None
    context_target_ratio: float = 0.5
    min_context_keep: float = 0.0
    protect_top_k: int = 2
    drop_redundant: bool = True
    dup_threshold: float = 0.82
    min_segments: int = 4
    relevance_fn: Optional[object] = None   # opt-in semantic relevance (see below)

    # ---- public API (Compressor protocol) ----
    def compress(self, text: str) -> CompressionResult:
        t0 = time.perf_counter()
        try:
            return self._compress_inner(text, t0)
        except Exception as e:  # never raise — fall open
            tb = estimate_tokens(text)
            return CompressionResult(
                before=text, after=text, tokens_before=tb, tokens_after=tb,
                ratio=1.0, duration_ms=(time.perf_counter() - t0) * 1000.0,
                method="query_aware", safe=False, extra={"error": type(e).__name__},
            )

    # ---- internals ----
    def _compress_inner(self, text: str, t0: float) -> CompressionResult:
        tokens_before = estimate_tokens(text)
        segments = self._segment(text)
        if len(segments) < self.min_segments:
            return self._noop(text, tokens_before, t0, reason="too_few_segments")

        # 1) IDF over the prompt's own segments (a token in few segments is
        #    distinctive; a token in every segment is generic).
        idf = self._segment_idf(segments)
        info = self._self_information(segments)
        qtokens = self._query_terms(self.query) if self.query else set()

        # 2) classify each segment by role + score context relevance to the query
        segs: List[_QSeg] = []
        for i, seg in enumerate(segments):
            segs.append(self._classify_and_score(seg, i, idf, info, qtokens))

        context = [s for s in segs if s.role == _SegmentRole.CONTEXT]
        protected = [s for s in segs if s.role != _SegmentRole.CONTEXT]

        # If there is no query or no compressible context, fall open (nothing to
        # safely gain — exactly the case the old test wrongly penalized).
        if not qtokens or not context:
            return self._noop(text, tokens_before, t0, reason="no_query_or_no_context")

        # 3) drop near-duplicate context (RAG chunks often overlap)
        if self.drop_redundant:
            context = self._dedupe_context(context)

        # 4) BUDGET CONTROLLER: keep the most query-relevant context up to the
        #    context token budget; always protect the top-K most relevant chunks.
        self._apply_budget(context)

        # 5) reassemble: protected segments + kept context, in original order
        keep_set = {s.index for s in protected} | {s.index for s in context if s.keep}
        kept_sorted = sorted((s for s in segs if s.index in keep_set),
                             key=lambda s: s.index)
        after = "\n".join(s.text for s in kept_sorted).strip()

        tokens_after = estimate_tokens(after)
        if not after or tokens_after >= tokens_before:
            return self._noop(text, tokens_before, t0, reason="no_gain")

        dropped = [s for s in context if not s.keep]
        return CompressionResult(
            before=text, after=after, tokens_before=tokens_before,
            tokens_after=tokens_after, ratio=tokens_after / max(tokens_before, 1),
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            method="query_aware", safe=True,
            extra={
                "segments_before": len(segments),
                "segments_after": len(kept_sorted),
                "context_total": len(context) + len(dropped),
                "context_dropped": len(dropped),
                "protected": len(protected),
                "roles": self._role_counts(segs),
            },
        )

    def _classify_and_score(self, seg: str, idx: int, idf: Dict[str, float],
                            info: Dict[str, float], qtokens: set) -> _QSeg:
        words = _WORD.findall(seg.lower())
        n = max(len(words), 1)

        # role classification (coarse): code > question > instruction > context.
        # Instruction requires a DIRECTIVE pattern (not just a cue word), so
        # narrative prose containing words like "first"/"answer" stays context.
        if seg.startswith("```"):
            role = _SegmentRole.CODE
        elif _QUESTION.search(seg):
            role = _SegmentRole.QUESTION
        elif _is_directive(seg):
            role = _SegmentRole.INSTRUCTION
        else:
            role = _SegmentRole.CONTEXT

        mean_info = sum(info.get(w, 0.0) for w in words) / n

        # IDF-weighted query relevance for CONTEXT segments: a match on a
        # distinctive query term (high IDF, not a stopword) counts far more than a
        # generic word. Normalized by the query's own IDF mass so it's 0..1-ish.
        relevance = 0.0
        if role == _SegmentRole.CONTEXT and qtokens:
            seg_set = set(words)
            matched = seg_set & qtokens
            if matched:
                num = sum(idf.get(w, 0.0) for w in matched)
                den = sum(idf.get(w, 0.0) for w in qtokens) or 1.0
                relevance = num / den
            # OPT-IN semantic relevance: when a relevance_fn (embedding cosine) is
            # supplied, use it to catch chunks that are relevant by MEANING but use
            # different words than the question (the lexical blind spot). We take
            # the MAX of lexical and semantic so neither signal can wrongly drop a
            # chunk the other finds relevant.
            if self.relevance_fn is not None and self.query:
                try:
                    sem = float(self.relevance_fn(self.query, seg))
                    relevance = max(relevance, sem)
                except Exception:
                    pass
            if _BOILERPLATE.search(seg):
                relevance *= 0.4

        return _QSeg(index=idx, text=seg, words=words, role=role,
                     relevance=relevance, info=mean_info)

    def _apply_budget(self, context: List[_QSeg]) -> None:
        """Keep the most query-relevant context up to the context token budget;
        always protect the top-K most relevant, and never drop a clearly-relevant
        chunk (relevance above the min_context_keep percentile)."""
        if not context:
            return
        total_ctx_tokens = sum(estimate_tokens(s.text) for s in context)
        budget = int(total_ctx_tokens * self.context_target_ratio)

        ranked = sorted(context, key=lambda s: (s.relevance, s.info), reverse=True)

        # protect the top-K most relevant chunks (answer-bearing) — but ONLY those
        # with meaningful relevance. Protecting a near-zero chunk just because it is
        # rank-2 would keep noise. The bar scales with the signal type.
        rel_bar = 1e-6 if self.relevance_fn is None else 0.15
        protected_idx = {s.index for s in ranked[: max(self.protect_top_k, 0)]
                         if s.relevance > rel_bar}

        # relevance floor: keep anything at/above this percentile no matter the budget
        floor_val = 0.0
        if self.min_context_keep > 0:
            rels = sorted((s.relevance for s in context))
            pos = min(len(rels) - 1, int(len(rels) * self.min_context_keep))
            floor_val = rels[pos]

        # With pure lexical scoring, irrelevant chunks score exactly 0 and should
        # be dropped regardless of leftover budget. With semantic scoring, cosine
        # gives small positive values to everything, so we treat anything at/below
        # a small epsilon as effectively irrelevant.
        eps = 1e-6 if self.relevance_fn is None else 0.15
        used = 0
        for s in ranked:
            cost = estimate_tokens(s.text)
            must_keep = (s.index in protected_idx) or (s.relevance >= floor_val and floor_val > 0)
            if must_keep:
                s.keep = True
                used += cost
            elif s.relevance <= eps:
                # effectively irrelevant: drop regardless of leftover budget — a
                # chunk that matches nothing in the question is noise, and keeping
                # it just to fill budget reintroduces the "lost in the middle" issue.
                s.keep = False
            elif used + cost <= budget:
                s.keep = True
                used += cost
            else:
                s.keep = False

    def _dedupe_context(self, context: List[_QSeg]) -> List[_QSeg]:
        kept: List[_QSeg] = []
        seen: List[set] = []
        # keep higher-relevance first so the duplicate we drop is the less relevant
        for s in sorted(context, key=lambda x: x.relevance, reverse=True):
            ws = set(s.words)
            dup = False
            for prev in seen:
                if not ws or not prev:
                    continue
                if len(ws & prev) / len(ws | prev) >= self.dup_threshold:
                    dup = True
                    break
            if not dup:
                kept.append(s); seen.append(ws)
        return kept

    # ---- shared helpers ----
    def _segment(self, text: str) -> List[str]:
        parts: List[str] = []
        last = 0
        for m in _FENCE.finditer(text):
            parts.extend(self._split_prose(text[last:m.start()]))
            parts.append(m.group(0))
            last = m.end()
        parts.extend(self._split_prose(text[last:]))
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _split_prose(prose: str) -> List[str]:
        if not prose.strip():
            return []
        return _SENT_SPLIT.split(prose)

    def _segment_idf(self, segments: List[str]) -> Dict[str, float]:
        """IDF over segments: log(N / df). Distinctive tokens (in few segments)
        get high IDF; tokens in every segment get ~0."""
        n = len(segments)
        df: Counter = Counter()
        for seg in segments:
            for w in set(_WORD.findall(seg.lower())):
                df[w] += 1
        return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def _self_information(self, segments: List[str]) -> Dict[str, float]:
        counts: Counter = Counter()
        total = 0
        for seg in segments:
            for w in _WORD.findall(seg.lower()):
                counts[w] += 1
                total += 1
        total = max(total, 1)
        return {w: -math.log2(c / total) for w, c in counts.items()}

    def _query_terms(self, query: str) -> set:
        """Distinctive query terms: drop generic stopish words so relevance is
        driven by the meaningful nouns/verbs of the question."""
        return {w for w in _WORD.findall((query or "").lower())
                if w not in _STOPISH and len(w) > 1}

    def _role_counts(self, segs: List[_QSeg]) -> Dict[str, int]:
        c: Counter = Counter(s.role for s in segs)
        return dict(c)

    def _noop(self, text: str, tb: int, t0: float, reason: str) -> CompressionResult:
        return CompressionResult(
            before=text, after=text, tokens_before=tb, tokens_after=tb,
            ratio=1.0, duration_ms=(time.perf_counter() - t0) * 1000.0,
            method="query_aware", safe=True, extra={"noop": True, "reason": reason},
        )

    def for_query(self, query: str) -> "QueryAwareCompressor":
        from dataclasses import replace
        return replace(self, query=query)


def make_embedding_relevance(encoder: object, threshold: float = 0.0) -> object:
    """Build an opt-in semantic relevance function for QueryAwareCompressor.

    Returns relevance(query, segment) -> float in 0..1 (cosine similarity of the
    query and segment embeddings, clamped to >= 0). This closes the lexical blind
    spot: a chunk relevant by MEANING but using different words than the question
    (synonyms, paraphrase) still scores high.

    `encoder` is any object with `.encode(text) -> vector` (e.g. the semantic
    cache's encoder / a sentence-transformers model). Heavy — opt-in only; the
    zero-dep core never requires it. Fail-safe: returns 0.0 on any error so a
    failing encoder never silently keeps or drops the wrong chunk on its own.
    """
    import math

    def _cos(a, b) -> float:
        try:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)
        except Exception:
            return 0.0

    # tiny cache so the query is encoded once per compression, not per segment
    _cache: dict = {}

    def relevance(query: str, segment: str) -> float:
        try:
            qv = _cache.get(query)
            if qv is None:
                qv = encoder.encode(query)
                _cache.clear()
                _cache[query] = qv
            sv = encoder.encode(segment)
            c = _cos(qv, sv)
            if c < threshold:
                return 0.0
            return max(0.0, min(1.0, c))
        except Exception:
            return 0.0

    return relevance
