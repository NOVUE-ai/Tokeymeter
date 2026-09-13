"""
Stage-2 semantic verification — the cross-encoder re-rank that makes the
semantic cache trustworthy.

THE PROBLEM IT SOLVES
A bi-encoder (Stage 1) encodes each prompt independently into a vector, then
compares by cosine similarity. That cannot reliably tell apart prompts that are
lexically/structurally similar but semantically DIFFERENT — e.g. "how do I read
a file" vs "how do I delete a file" score ~0.88 cosine and collide, so the cache
serves a WRONG answer. This is an inherent limitation of single-stage
vector-similarity caching (GPTCache documents the same failure).

THE FIX (documented state of the art: bi-encoder retrieve -> cross-encoder verify)
A cross-encoder reads BOTH prompts together in one forward pass and scores their
true relationship. Because it sees the pair jointly (not two independent vectors),
it discriminates near-misses cleanly: on the classic example the right match
scores ~+8 and the wrong one ~-4 — a gap cosine similarity cannot produce. We use
it as a VERIFICATION GATE: Stage 1 proposes a candidate; Stage 2 confirms the two
prompts are genuinely equivalent before the cached answer is served.

DESIGN PRINCIPLES (consistent with the engine doctrine)
- OPT-IN, never core. If the cross-encoder dependency/model is unavailable, this
  degrades to "no verifier" and the cache falls back to Stage-1 behavior. The
  zero-dependency core is never affected.
- FAIL-SAFE. On any verifier error or timeout, the gate REJECTS the hit (treats it
  as unverified -> cache miss -> real model call). A correctness gate must fail
  closed: when unsure, do NOT serve a possibly-wrong cached answer.
- LAZY. The model loads on first use, not import, so importing tokeymeter stays
  cheap and the cost is paid only when verification is actually enabled.

NOTE ON MODEL CHOICE
Off-the-shelf cross-encoders (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) are
trained for query->passage RELEVANCE, not query->query EQUIVALENCE. They work
well as a near-miss gate because a genuinely different question scores far lower
than a true paraphrase, but the absolute scores need calibration (hence the
`accept_threshold`, tuned on real pairs). A model fine-tuned on question-pair
equivalence (e.g. a quora-question-pairs cross-encoder) would be even sharper and
can be supplied via `model_name`. The clean path to a domain-tuned verifier is to
fine-tune on the customer's own accepted/rejected pairs — accumulated, per-tenant,
which is moat-aligned.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional, Tuple

log = logging.getLogger("tokeymeter.semantic.verify")

# Default cross-encoder: trained on Quora Question Pairs for DUPLICATE detection
# (predicts P(same question), 0..1) — the right task for cache equivalence. NOT
# ms-marco, which scores topical RELEVANCE and cannot tell "read a file" from
# "delete a file". Override via SemanticVerifier(model_name=...).
_DEFAULT_MODEL = "cross-encoder/quora-distilroberta-base"

# ── Antonym / alternative guard ──────────────────────────────────────────────
# Even a duplicate-detection model fails on ANTONYM near-misses: two sentences
# identical except for one opposite word ("LEFT join" vs "RIGHT join",
# "ascending" vs "descending") look like duplicate phrasings to it, so it scores
# them ~0.95. The single flipped word carries the whole meaning — the hardest
# case for any semantic model. A deterministic lexical guard catches exactly
# these: if the two prompts differ by members of a known opposite/alternative
# group, REJECT regardless of the model score. The guard is HIGH-PRECISION (it
# only fires on genuine opposite-word substitutions) so it never vetoes true
# paraphrases, which differ on function words, not opposites.

_WORD = re.compile(r"[a-z0-9_+]+")

# Function words: a difference in these is paraphrase noise, not a meaning flip.
# (Meaningful verbs like get/read/write are deliberately NOT here.)
_STOP = {
    "a", "an", "the", "is", "are", "do", "i", "how", "what", "whats", "to", "of",
    "in", "for", "on", "with", "my", "me", "can", "you", "please", "explain",
    "describe", "tell", "way", "and", "or", "using", "use", "that", "this", "it",
    "its", "be", "does", "did", "give", "show", "need", "want", "would", "should",
    "could", "there", "here", "about", "into", "from", "by", "as", "at", "but",
    "if", "then", "than", "so", "such", "via", "whether",
}

# Two DIFFERENT members of a group => different meaning. Curated for technical
# queries (the cache's domain). Extensible — add groups as new near-miss classes
# are observed (the eval loop surfaces them).
_OPPOSITE_GROUPS = [
    {"left", "right"},
    {"ascending", "descending"}, {"asc", "desc"},
    {"read", "delete", "create", "remove", "update", "modify", "append", "insert", "drop", "overwrite"},
    {"add", "subtract", "remove", "delete"},
    {"install", "uninstall"}, {"enable", "disable"}, {"open", "close"},
    {"push", "pop", "pull"}, {"encode", "decode"}, {"encrypt", "decrypt"},
    {"compress", "decompress"}, {"serialize", "deserialize"},
    {"min", "max"}, {"minimum", "maximum"}, {"first", "last"},
    {"start", "stop", "end", "begin", "pause"}, {"increment", "decrement"},
    {"lock", "unlock"}, {"connect", "disconnect"}, {"merge", "rebase"},
    {"get", "post", "put", "patch"},
    {"useeffect", "usestate", "usememo", "usecallback", "useref", "usecontext", "usereducer"},
    {"mysql", "mongodb", "postgresql", "postgres", "redis", "sqlite", "cassandra", "oracle"},
    {"tcp", "udp"}, {"flexbox", "grid"}, {"inner", "outer", "cross"},
    {"quicksort", "mergesort", "bubblesort", "heapsort", "insertionsort", "timsort"},
    {"stack", "queue", "heap", "deque"},
    {"http", "https", "ftp", "ssh"}, {"sql", "nosql"},
    {"synchronous", "asynchronous"}, {"sync", "async"},
    {"public", "private", "protected"}, {"true", "false"},
    {"client", "server"}, {"frontend", "backend"},
]


def _content_tokens(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _are_opposites(a: str, b: str) -> bool:
    if a == b:
        return False
    for g in _OPPOSITE_GROUPS:
        if a in g and b in g:
            return True
    return False


def lexical_veto(query: str, candidate: str) -> bool:
    """True => REJECT the match: the prompts differ by opposite/alternative
    content words (different meaning despite high lexical similarity). This is
    the deterministic guard for antonym near-misses that defeat the model."""
    qt, ct = _content_tokens(query), _content_tokens(candidate)
    if not qt or not ct:
        return False
    only_q = qt - ct
    only_c = ct - qt
    for wq in only_q:
        for wc in only_c:
            if _are_opposites(wq, wc):
                return True
    return False


class SemanticVerifier:
    """Stage-2 verification gate. Confirms a candidate cache match is a TRUE
    semantic equivalent of the query before the cached answer is served.

    Usage (inside the cache, after Stage 1 proposes candidates):
        verifier = SemanticVerifier()          # lazy; loads on first verify()
        if verifier.verify(query_prompt, cached_prompt):
            return cached_response             # confirmed same question
        # else: treat as a miss -> call the real model

    The verifier is OPTIONAL. If sentence-transformers / the model is unavailable,
    `available` is False and the cache should fall back to Stage-1-only behavior.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        accept_threshold: float = 0.5,
        max_pairs: int = 5,
        model: Optional[object] = None,
        use_lexical_guard: bool = True,
    ):
        """
        Args:
            model_name: cross-encoder checkpoint to load lazily. Default is the
                Quora duplicate-detection model (scores 0..1 = P(same question)).
                Ignored if a preloaded `model` is supplied.
            accept_threshold: minimum score to ACCEPT a match. For the Quora model
                (0..1) the default 0.5 cleanly separates duplicates (~0.9+) from
                different questions (~0.0). Calibrate on real pairs if needed.
            max_pairs: cap on candidates scored per lookup (protects latency).
            model: optional preloaded cross-encoder (dependency injection /
                testing). If given, no lazy load occurs.
            use_lexical_guard: when True (default), the deterministic antonym/
                alternative guard vetoes near-misses that differ by an opposite
                word (LEFT/RIGHT, ascending/descending) before the model is even
                consulted — catching the cases no semantic model handles.
        """
        self._model_name = model_name
        self._accept_threshold = float(accept_threshold)
        self._max_pairs = int(max_pairs)
        self._model = model
        self._use_lexical_guard = bool(use_lexical_guard)
        self._load_attempted = model is not None
        self._load_failed = False
        self._lock = threading.Lock()

    # ---- availability / lazy load ----

    @property
    def available(self) -> bool:
        """True if a verifier model is loaded or can be loaded. Triggers the
        lazy load on first access so callers can branch on it."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        self._ensure_model()
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_failed:
            return
        with self._lock:
            if self._model is not None or self._load_failed:
                return
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                log.debug("tokeymeter.verify: loaded cross-encoder %s", self._model_name)
            except Exception as e:
                # Graceful degradation: no verifier available -> Stage-1 fallback.
                self._load_failed = True
                log.debug("tokeymeter.verify: cross-encoder unavailable (%s); "
                          "verification disabled, falling back to Stage 1", e)

    # ---- the gate ----

    def score(self, query: str, candidate: str) -> Optional[float]:
        """Return the cross-encoder score for (query, candidate), or None if the
        verifier is unavailable. Higher = more likely the same question."""
        if not self.available:
            return None
        try:
            preds = self._model.predict([(query, candidate)])
            return float(preds[0])
        except Exception as e:
            log.debug("tokeymeter.verify: scoring error: %s", e)
            return None

    def verify(self, query: str, candidate: str) -> bool:
        """Confirm a single candidate is a true equivalent of the query.

        Two gates: (1) the deterministic antonym guard vetoes opposite-word
        near-misses outright; (2) the cross-encoder must then score the pair at
        or above `accept_threshold`. FAIL-SAFE: if the verifier is unavailable or
        errors, returns False (reject -> cache miss -> real call), because a
        correctness gate must never serve an UNVERIFIED possibly-wrong answer.
        """
        if not isinstance(query, str) or not isinstance(candidate, str):
            return False
        # Gate 1: deterministic antonym/alternative guard (catches what the model can't)
        if self._use_lexical_guard and lexical_veto(query, candidate):
            return False
        # Gate 2: the cross-encoder duplicate score
        s = self.score(query, candidate)
        if s is None:
            return False  # unavailable or error -> fail safe (reject)
        return s >= self._accept_threshold

    def best_verified(
        self, query: str, candidates: List[Tuple[str, object]]
    ) -> Optional[object]:
        """Given Stage-1 candidates [(cached_prompt, cached_response), ...],
        score each with the cross-encoder and return the response of the
        highest-scoring candidate that clears the accept threshold — or None if
        none qualify (a verified miss).

        This is the production entry point: Stage 1 hands the top-K candidates
        here, Stage 2 picks the one that is genuinely the same question (if any).
        Scoring is capped at `max_pairs` to bound latency.
        """
        if not candidates or not self.available:
            return None
        # Gate 1: drop candidates the antonym guard vetoes (opposite-word near-misses).
        pool = candidates
        if self._use_lexical_guard:
            pool = [c for c in candidates if not lexical_veto(query, c[0])]
            if not pool:
                return None  # every candidate was an opposite-word near-miss
        pairs_to_score = pool[: self._max_pairs]
        try:
            inputs = [(query, c[0]) for c in pairs_to_score]
            scores = self._model.predict(inputs)
        except Exception as e:
            log.debug("tokeymeter.verify: batch scoring error: %s", e)
            return None

        best_score = None
        best_response = None
        for (_, response), sc in zip(pairs_to_score, scores):
            sc = float(sc)
            if sc >= self._accept_threshold and (best_score is None or sc > best_score):
                best_score = sc
                best_response = response
        return best_response  # None if nothing cleared the threshold (verified miss)
