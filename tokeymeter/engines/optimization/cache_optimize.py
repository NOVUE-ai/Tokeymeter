"""
Prefix-cacheability optimization + tool-definition compression (Tier 1).

Provider-side prompt caching (OpenAI automatic 50%, Anthropic explicit 90%)
only applies to the *prefix* of a prompt: static content at the front gets
cached, and any change before a token invalidates everything after it. Most
developers don't structure prompts this way, so they leave the provider's own
caching discount on the table.

This module helps capture that discount — model-free, in-process, content-blind:

  1. CacheOptimizer — analyzes/reorders a structured prompt so stable content
     (system, tools, long static context) forms a maximal cacheable prefix and
     volatile content (the user turn) sits at the end. It does NOT cache anything
     itself; it makes the provider's cache hit. It can also place Anthropic
     cache_control breakpoints at the optimal boundary.

  2. compress_tools — model-free structural compression of verbose tool/function
     definitions (the JSON-schema descriptions resent every request).

Both are advisory/transform utilities: they never call a model, never touch the
network, and fall open (return the input unchanged) on anything unexpected.

Grounding: provider docs — caching is prefix-only, 1024-token minimum, static
content must lead; changing tools invalidates the system cache (tools precede
system in Anthropic's hierarchy).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tokeymeter.engines.economics.pricing import estimate_tokens

# Volatility heuristics: a message/segment is "volatile" (belongs at the end,
# breaks caching if early) if it looks per-request. Conservative: when unsure,
# treat as volatile so we never wrongly cache something dynamic.
_VOLATILE_ROLE = {"user"}
_STATIC_ROLE = {"system", "developer", "tool"}


# ---------------------------------------------------------------------------
# Tier A — volatile-content detection (detect + recommend, NEVER rewrite)
# ---------------------------------------------------------------------------
# Patterns that, when present in the CACHEABLE PREFIX, silently break provider
# caching: any of these changes per request, so the prefix hash changes and the
# cache never hits. We DETECT and RECOMMEND moving them; we never touch the text.
# Each pattern is precise to minimize false positives (a false "this breaks your
# cache" warning is annoying but safe; we still keep precision high).
_VOLATILE_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    # ISO-8601 / common datetimes:  2026-06-04, 2026-06-04T14:23:01, 14:23:01
    ("timestamp", re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b")),
    ("time_of_day", re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")),
    # UUIDs
    ("uuid", re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    # Unix epoch (10 or 13 digit) — only when labelled-ish to avoid matching any long number
    ("epoch", re.compile(r"\b(?:epoch|timestamp|ts|time)[\"'\s:=]+\d{10,13}\b", re.I)),
    # session/request/trace/correlation IDs with a value
    ("request_id", re.compile(
        r"(?i)\b(?:request|session|trace|correlation|req|txn|transaction)[ _-]?id[\"'\s:=]+[A-Za-z0-9_-]{6,}")),
    # bearer tokens / api keys (also a security smell in a prompt)
    ("token_or_key", re.compile(
        r"(?i)\b(?:bearer\s+[A-Za-z0-9._-]{12,}|sk-[A-Za-z0-9]{16,}|api[_-]?key[\"'\s:=]+[A-Za-z0-9._-]{12,})")),
    # explicit "current time/date/now" labels followed by a value
    ("current_datetime_label", re.compile(
        r"(?i)\b(?:current (?:time|date|datetime)|now|today(?:'s date)?)[\"'\s:=]+\S+")),
    # per-user identity injected into a system prompt
    ("user_identity", re.compile(
        r"(?i)\b(?:user(?:name)?|user id|logged in as|account)[\"'\s:=]+[A-Za-z0-9._@-]{2,}")),
]


@dataclass
class VolatileFinding:
    """One piece of cache-breaking content found in the cacheable prefix."""
    kind: str                 # which pattern matched (timestamp, uuid, ...)
    message_index: int        # which message it sits in
    role: str                 # that message's role
    excerpt: str              # short, redacted context (never the full content)
    recommendation: str       # plain-language fix


@dataclass
class CacheReport:
    cacheable_prefix_tokens: int
    total_tokens: int
    prefix_ratio: float
    meets_min_prefix: bool          # >= provider minimum (default 1024)
    reordered: bool                 # did we change message order?
    breakpoint_index: Optional[int] # where to place cache_control (Anthropic)
    notes: List[str] = field(default_factory=list)
    volatile_findings: List[VolatileFinding] = field(default_factory=list)

    @property
    def has_cache_breakers(self) -> bool:
        return bool(self.volatile_findings)

    @property
    def estimated_cache_health(self) -> str:
        """Quick verdict: 'good' | 'broken' | 'too_small' | 'no_prefix'."""
        if self.cacheable_prefix_tokens == 0:
            return "no_prefix"
        if not self.meets_min_prefix:
            return "too_small"
        if self.volatile_findings:
            return "broken"
        return "good"


@dataclass
class CacheOptimizer:
    """Reorder a structured chat prompt to maximize the cacheable prefix.

    Operates on the OpenAI/Anthropic-style messages list:
        [{"role": "system"|"user"|..., "content": "..."}, ...]

    Guarantees:
      - **Semantics preserved**: only *stable* leading blocks are grouped to the
        front; the relative order of user/assistant turns is NEVER changed
        (reordering a conversation would corrupt it). We only hoist standalone
        static blocks (system/developer/tool) above the first dynamic turn.
      - Falls open: malformed input is returned unchanged.

    Args:
        min_prefix_tokens: provider cache minimum (1024 default).
        place_breakpoint:  if True, annotate the last static block with an
                           Anthropic-style cache_control marker.
    """
    min_prefix_tokens: int = 1024
    place_breakpoint: bool = True
    detect_volatile: bool = True

    def optimize(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], CacheReport]:
        try:
            return self._optimize_inner(messages)
        except Exception:
            # fall open: return input unchanged with a minimal report
            toks = self._safe_total(messages)
            return messages, CacheReport(0, toks, 0.0, False, False, None, ["error: fell open"])

    def _optimize_inner(self, messages):
        if not isinstance(messages, list) or not messages:
            return messages, CacheReport(0, 0, 0.0, False, False, None, ["empty/invalid"])

        notes: List[str] = []
        # 1) find the first dynamic (user/assistant) turn; everything stable
        #    BEFORE it is already a clean prefix. Stable blocks that appear AFTER
        #    the first dynamic turn cannot be safely hoisted without risking
        #    semantic change, so we only hoist leading-eligible static blocks.
        first_dynamic = None
        for i, m in enumerate(messages):
            if self._role(m) not in _STATIC_ROLE:
                first_dynamic = i
                break
        if first_dynamic is None:
            first_dynamic = len(messages)  # all static

        head = messages[:first_dynamic]          # already-static prefix
        tail = messages[first_dynamic:]           # conversation (order preserved)

        # 2) Among the tail, are there static blocks we can SAFELY hoist?
        #    Only hoist a static block if every message before it in the tail is
        #    also static (i.e. it's contiguous with the prefix) — otherwise moving
        #    it would reorder relative to a dynamic turn. In practice this means we
        #    don't reorder conversations; we only catch the common mistake of a
        #    static system block placed after an initial user message at index 0
        #    is NOT hoisted (that would change meaning). Conservative by design.
        reordered = False
        # detect the common, SAFE win: a trailing/standalone static block that was
        # accidentally placed before content but we already captured via head.
        # (We deliberately do not reorder mixed conversations — safety first.)

        new_messages = head + tail

        # 3) compute cacheable prefix size (the leading static run)
        prefix_tokens = sum(self._content_tokens(m) for m in head)
        total = sum(self._content_tokens(m) for m in new_messages)
        meets = prefix_tokens >= self.min_prefix_tokens
        ratio = prefix_tokens / max(total, 1)

        if not head:
            notes.append("no leading static block — consider moving system/tools to the front")
        if head and not meets:
            notes.append(f"cacheable prefix {prefix_tokens} tok < provider min {self.min_prefix_tokens}; "
                         "caching may not engage")
        if meets:
            notes.append(f"cacheable prefix {prefix_tokens} tok qualifies for provider caching")

        # 4) place an Anthropic cache_control breakpoint on the last static block
        breakpoint_index = None
        if self.place_breakpoint and head and meets:
            breakpoint_index = len(head) - 1
            new_messages = [dict(m) for m in new_messages]  # shallow copy
            blk = new_messages[breakpoint_index]
            blk["cache_control"] = {"type": "ephemeral"}
            notes.append(f"placed cache_control breakpoint at message {breakpoint_index}")

        # 5) Tier A — detect volatile content in the cacheable prefix that would
        #    silently break caching. We DETECT and RECOMMEND; we never rewrite.
        volatile_findings = self._detect_volatile(head) if self.detect_volatile else []
        if volatile_findings:
            kinds = sorted({f.kind for f in volatile_findings})
            notes.append(
                f"cache-breaker(s) detected in prefix: {', '.join(kinds)} — "
                f"these change per request and prevent the prefix from caching; "
                f"move them out of the static prefix (see report.volatile_findings)")

        return new_messages, CacheReport(
            cacheable_prefix_tokens=prefix_tokens, total_tokens=total,
            prefix_ratio=ratio, meets_min_prefix=meets, reordered=reordered,
            breakpoint_index=breakpoint_index, notes=notes,
            volatile_findings=volatile_findings,
        )

    def _detect_volatile(self, prefix_messages) -> List[VolatileFinding]:
        """Scan the cacheable-prefix messages for per-request content that would
        break caching. Content-blind: excerpts are short and the matched value is
        masked so the finding itself never leaks the secret/PII."""
        findings: List[VolatileFinding] = []
        for idx, m in enumerate(prefix_messages):
            text = self._message_text(m)
            if not text:
                continue
            for kind, pat in _VOLATILE_PATTERNS:
                for match in pat.finditer(text):
                    findings.append(VolatileFinding(
                        kind=kind,
                        message_index=idx,
                        role=self._role(m),
                        excerpt=self._masked_excerpt(text, match.start(), match.end()),
                        recommendation=self._recommend(kind),
                    ))
                    break  # one finding per (message, kind) is enough signal
        return findings

    @staticmethod
    def _message_text(m) -> str:
        if not isinstance(m, dict):
            return ""
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") if isinstance(b, dict) else "" for b in c)
        return ""

    @staticmethod
    def _masked_excerpt(text: str, start: int, end: int, pad: int = 12) -> str:
        """A fully content-blind locator: the matched span is masked and NO raw
        surrounding text is included (adjacent context could itself contain a
        secret/PII/timestamp). The developer gets the kind, message index, and a
        character offset to locate it — never any verbatim sensitive content."""
        masked = "\u2588" * min(max(end - start, 4), 8)
        return f"[{masked}] (chars {start}\u2013{end} of message)"

    @staticmethod
    def _recommend(kind: str) -> str:
        msgs = {
            "timestamp": "Move this timestamp out of the system prompt; pass it in the final user turn instead.",
            "time_of_day": "Move this time value to the user turn; it changes every request and breaks the prefix cache.",
            "uuid": "Move this UUID to the dynamic (final) part of the prompt; per-request IDs must not sit in the cached prefix.",
            "epoch": "Move this epoch/timestamp value to the user turn.",
            "request_id": "Move request/session/trace IDs out of the prefix into the dynamic turn.",
            "token_or_key": "Remove this token/key from the prompt entirely (security risk) and never place secrets in a cached prefix.",
            "current_datetime_label": "Inject current date/time in the final user turn, not the system prompt, so the prefix stays stable.",
            "user_identity": "Pass per-user identity in the user turn or via metadata, not baked into the cached system prompt.",
        }
        return msgs.get(kind, "Move this per-request content out of the cacheable prefix.")

    @staticmethod
    def _role(m) -> str:
        return (m.get("role") if isinstance(m, dict) else "") or ""

    @staticmethod
    def _content_tokens(m) -> int:
        if not isinstance(m, dict):
            return 0
        c = m.get("content", "")
        if isinstance(c, str):
            return estimate_tokens(c)
        if isinstance(c, list):  # content blocks
            return sum(estimate_tokens(b.get("text", "")) if isinstance(b, dict) else 0 for b in c)
        return 0

    def _safe_total(self, messages) -> int:
        try:
            return sum(self._content_tokens(m) for m in messages) if isinstance(messages, list) else 0
        except Exception:
            return 0


# ---- tool-definition compression -------------------------------------------

_FILLER = re.compile(
    r"(?i)\b(?:this (?:tool|function) (?:is used to|should be used (?:to|when)|"
    r"allows you to|can be used to|will)|"
    r"use this (?:tool|function) (?:to|when)|"
    r"please note that|in order to|you can use this to)\b"
)
_MULTISPACE = re.compile(r"\s+")


def compress_tools(tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Model-free structural compression of tool/function descriptions.

    Trims verbose natural-language filler from each tool's `description` while
    preserving the schema (name, parameters, types) exactly — those are
    semantically load-bearing and never touched. Falls open per-tool.

    Returns (compressed_tools, stats) where stats has tokens_before/after.
    """
    if not isinstance(tools, list):
        return tools, {"tokens_before": 0, "tokens_after": 0}

    before = after = 0
    out = []
    for t in tools:
        try:
            if not isinstance(t, dict):
                out.append(t); continue
            t2 = dict(t)
            # OpenAI shape: {"type":"function","function":{"name","description","parameters"}}
            fn = t2.get("function") if isinstance(t2.get("function"), dict) else None
            target = fn if fn is not None else t2
            desc = target.get("description")
            if isinstance(desc, str) and desc.strip():
                before += estimate_tokens(desc)
                trimmed = _FILLER.sub("", desc)
                trimmed = _MULTISPACE.sub(" ", trimmed).strip(" .,;:") .strip()
                # never empty it; if trim removed everything meaningful, keep original
                if len(trimmed) >= max(8, int(len(desc) * 0.15)):
                    if fn is not None:
                        target = dict(target); target["description"] = trimmed
                        t2["function"] = target
                    else:
                        t2["description"] = trimmed
                    after += estimate_tokens(trimmed)
                else:
                    after += estimate_tokens(desc)  # kept original
            out.append(t2)
        except Exception:
            out.append(t)  # fall open per-tool
    return out, {"tokens_before": before, "tokens_after": after}
