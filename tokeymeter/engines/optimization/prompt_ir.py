"""Prompt IR (Phase 0) — a normalized, derived, immutable representation of a
prompt as an ordered list of typed spans.

This is the SUBSTRATE only. Per the design spec, Phase 0 does NOT wire the IR
into any optimizer; it ships the types, a deterministic heuristic parser, and a
byte-exact reconstruction guarantee, all behind equivalence + reconstruction
tests. No existing behavior changes.

Hard guarantees (each is a test target):
  * Reconstruction is byte-exact: reconstruct(parse(x)) == x for strings and
    for chat message arrays (content recovered, extra keys preserved).
  * Spans tile the input with no gaps or overlaps.
  * Misclassification fails SAFE — an unclassified span is treated as a
    verbatim, PII-sensitive, semantically-defining user query (does less
    optimization, never silent quality loss).

Zero external dependencies. Deterministic. No model, no network, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class SpanKind(str, Enum):
    SYSTEM_INSTRUCTION = "system_instruction"
    FEW_SHOT_EXAMPLE = "few_shot_example"
    RETRIEVED_CONTEXT = "retrieved_context"
    USER_QUERY = "user_query"
    CODE = "code"
    QUOTED = "quoted"
    TOOL_DEFINITION = "tool_definition"
    BOILERPLATE = "boilerplate"


class Origin(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    RETRIEVED = "retrieved"
    TOOL = "tool"
    MODEL = "model"


@dataclass(frozen=True)
class Permissions:
    """Per-span optimization permissions. Every field defaults to the SAFE
    value, so an unknown/unclassified span is maximally protected and identical
    to the USER_QUERY profile (see permissions_for / _DEFAULT_KIND).

    Note on `semantic_identity` defaulting True: including a span in semantic
    identity makes a semantic match STRICTER (more content must agree), which
    REDUCES false-positive cache serves — so True is the conservative default
    for a trust product, not a permissive one."""
    compressible: bool = False         # may a lossy compressor touch it?
    cacheable: bool = True             # participates in the exact cache key?
    semantic_identity: bool = True     # defines the prompt's meaning for matching?
    must_be_verbatim: bool = True      # must reach the model byte-for-byte?
    pii_sensitive: bool = True         # scan aggressively for PII?
    load_bearing: bool = True          # does correctness depend on it?


# Permission profiles per kind. The UNKNOWN/default profile equals USER_QUERY's:
# verbatim, PII-sensitive, semantically-defining, not compressible.
_PERMS: Dict[SpanKind, Permissions] = {
    SpanKind.SYSTEM_INSTRUCTION: Permissions(
        compressible=False, cacheable=True, semantic_identity=False,
        must_be_verbatim=True, pii_sensitive=False, load_bearing=True),
    SpanKind.FEW_SHOT_EXAMPLE: Permissions(
        compressible=True, cacheable=True, semantic_identity=False,
        must_be_verbatim=False, pii_sensitive=False, load_bearing=True),
    SpanKind.RETRIEVED_CONTEXT: Permissions(
        compressible=True, cacheable=False, semantic_identity=False,
        must_be_verbatim=False, pii_sensitive=True, load_bearing=True),
    SpanKind.USER_QUERY: Permissions(
        compressible=False, cacheable=True, semantic_identity=True,
        must_be_verbatim=True, pii_sensitive=True, load_bearing=True),
    SpanKind.CODE: Permissions(
        compressible=False, cacheable=True, semantic_identity=True,
        must_be_verbatim=True, pii_sensitive=False, load_bearing=True),
    SpanKind.QUOTED: Permissions(
        compressible=False, cacheable=True, semantic_identity=True,
        must_be_verbatim=True, pii_sensitive=True, load_bearing=True),
    SpanKind.TOOL_DEFINITION: Permissions(
        compressible=False, cacheable=True, semantic_identity=False,
        must_be_verbatim=True, pii_sensitive=False, load_bearing=True),
    SpanKind.BOILERPLATE: Permissions(
        compressible=True, cacheable=True, semantic_identity=False,
        must_be_verbatim=False, pii_sensitive=False, load_bearing=False),
}

# The conservative default for anything we cannot classify.
_DEFAULT_KIND = SpanKind.USER_QUERY


def permissions_for(kind: SpanKind) -> Permissions:
    """Return the permission profile for a span kind (safe default if unknown)."""
    return _PERMS.get(kind, _PERMS[_DEFAULT_KIND])


@dataclass(frozen=True)
class Span:
    span_id: int
    kind: SpanKind
    text: str
    origin: Origin
    permissions: Permissions
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptIR:
    """Immutable, derived representation of one prompt.

    `source_kind` is "string" or "messages". For "messages", `_messages_meta`
    holds each original message minus its content (so extra keys like "name"
    are preserved on reconstruction), and each span carries provenance
    {"message_index": i}.
    """
    spans: Tuple[Span, ...]
    source_kind: str
    parser_conf: float
    workload_tag: Optional[str] = None
    lineage: Optional[str] = None
    high_stakes: bool = False
    _messages_meta: Optional[Tuple[Dict[str, Any], ...]] = None

    # ---- convenience surfaces for FUTURE phases (pure; no side effects) ----
    def semantic_text(self) -> str:
        """Concatenation of spans that define semantic identity. Phase 3 will
        embed THIS rather than the whole prompt."""
        return "".join(s.text for s in self.spans if s.permissions.semantic_identity)

    def cacheable_text(self) -> str:
        """Concatenation of cacheable spans — volatile spans excluded. Phase 2
        will key on THIS."""
        return "".join(s.text for s in self.spans if s.permissions.cacheable)

    def compressible_spans(self) -> Tuple[Span, ...]:
        return tuple(s for s in self.spans if s.permissions.compressible)

    def is_single_span(self) -> bool:
        return len(self.spans) == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "parser_conf": self.parser_conf,
            "workload_tag": self.workload_tag,
            "lineage": self.lineage,
            "high_stakes": self.high_stakes,
            "spans": [
                {"span_id": s.span_id, "kind": s.kind.value, "origin": s.origin.value,
                 "text": s.text, "permissions": vars(s.permissions),
                 "provenance": s.provenance}
                for s in self.spans
            ],
        }


# Protected-region pattern — identical to tokeymeter/compression.py so the IR's
# segmentation is consistent with the existing proto-IR (_apply_outside_quotes).
_PRESERVED = re.compile(
    r"```[\s\S]*?```"          # fenced code block
    r"|`[^`\n]*`"              # inline code
    r'|"(?:\\.|[^"\\])*"'      # double-quoted string
    r"|'(?:\\.|[^'\\])*'"      # single-quoted string
)


def _kind_for_preserved(region: str) -> SpanKind:
    return SpanKind.CODE if region[:1] == "`" else SpanKind.QUOTED


def _segment_string(text: str, base_origin: Origin,
                    next_id: int) -> Tuple[List[Span], int, bool]:
    """Tile `text` into alternating plain / preserved spans. Returns
    (spans, next_id, found_structure). Tiling is gap-free so the concatenation
    of span texts equals `text` exactly."""
    spans: List[Span] = []
    pos = 0
    found = False
    for m in _PRESERVED.finditer(text):
        if m.start() > pos:
            chunk = text[pos:m.start()]
            spans.append(Span(next_id, _DEFAULT_KIND, chunk, base_origin,
                              permissions_for(_DEFAULT_KIND)))
            next_id += 1
        region = m.group()
        kind = _kind_for_preserved(region)
        spans.append(Span(next_id, kind, region, base_origin, permissions_for(kind)))
        next_id += 1
        found = True
        pos = m.end()
    if pos < len(text) or not spans:
        # trailing remainder, or an input with no matches at all (incl. "")
        chunk = text[pos:]
        spans.append(Span(next_id, _DEFAULT_KIND, chunk, base_origin,
                          permissions_for(_DEFAULT_KIND)))
        next_id += 1
    return spans, next_id, found


# Chat-role → (Origin, SpanKind). Coarse by design in Phase 0: precise
# conversation-history modeling (final-user-turn-is-query vs prior history) is a
# Phase 1 refinement. Permissions, not the label, drive optimization.
_ROLE_MAP: Dict[str, Tuple[Origin, SpanKind]] = {
    "system": (Origin.SYSTEM, SpanKind.SYSTEM_INSTRUCTION),
    "developer": (Origin.DEVELOPER, SpanKind.SYSTEM_INSTRUCTION),
    "user": (Origin.USER, SpanKind.USER_QUERY),
    "assistant": (Origin.MODEL, SpanKind.FEW_SHOT_EXAMPLE),
    "tool": (Origin.TOOL, SpanKind.TOOL_DEFINITION),
    "function": (Origin.TOOL, SpanKind.TOOL_DEFINITION),
}


def _looks_like_messages(obj: Any) -> bool:
    return (isinstance(obj, (list, tuple)) and len(obj) > 0
            and all(isinstance(m, dict) and "role" in m and "content" in m
                    and isinstance(m["content"], str) for m in obj))


def parse(obj: Union[str, List[Dict[str, Any]]], *,
          workload_tag: Optional[str] = None,
          lineage: Optional[str] = None,
          high_stakes: bool = False) -> PromptIR:
    """Parse a prompt (string or chat message array) into a PromptIR.

    Never raises: any unexpected input degrades to a single conservative
    USER_QUERY span over str(obj) with low parser_conf (the documented
    single-span fallback that guarantees behavior == today downstream).
    """
    try:
        if isinstance(obj, str):
            spans, _, found = _segment_string(obj, Origin.USER, 0)
            conf = 0.8 if found else 0.6
            return PromptIR(tuple(spans), "string", conf,
                            workload_tag, lineage, high_stakes)

        if _looks_like_messages(obj):
            spans: List[Span] = []
            meta: List[Dict[str, Any]] = []
            nid = 0
            for i, msg in enumerate(obj):
                role = str(msg.get("role", "")).lower()
                origin, kind = _ROLE_MAP.get(role, (Origin.USER, _DEFAULT_KIND))
                content = msg["content"]
                spans.append(Span(nid, kind, content, origin,
                                  permissions_for(kind),
                                  provenance={"message_index": i, "role": msg.get("role")}))
                nid += 1
                # preserve any non-content keys for byte-exact reconstruction
                meta.append({k: v for k, v in msg.items() if k != "content"})
            return PromptIR(tuple(spans), "messages", 1.0,
                            workload_tag, lineage, high_stakes,
                            _messages_meta=tuple(meta))

        # Unknown shape → conservative single-span fallback.
        return _single_span_fallback(obj, workload_tag, lineage, high_stakes)
    except Exception:
        return _single_span_fallback(obj, workload_tag, lineage, high_stakes)


def _single_span_fallback(obj: Any, workload_tag, lineage, high_stakes) -> PromptIR:
    text = obj if isinstance(obj, str) else str(obj)
    span = Span(0, _DEFAULT_KIND, text, Origin.USER, permissions_for(_DEFAULT_KIND))
    return PromptIR((span,), "string", 0.3, workload_tag, lineage, high_stakes)


def reconstruct(ir: PromptIR) -> Union[str, List[Dict[str, Any]]]:
    """Rebuild the original prompt from its spans. Byte-exact.

    Returns a string for string-sourced IRs and a list of message dicts for
    message-sourced IRs (content rebuilt from spans, extra keys preserved)."""
    if ir.source_kind == "messages" and ir._messages_meta is not None:
        # group span texts by message index, in order
        by_msg: Dict[int, List[str]] = {}
        for s in ir.spans:
            idx = s.provenance.get("message_index", 0)
            by_msg.setdefault(idx, []).append(s.text)
        out: List[Dict[str, Any]] = []
        for i, meta in enumerate(ir._messages_meta):
            msg = dict(meta)
            msg["content"] = "".join(by_msg.get(i, []))
            out.append(msg)
        return out
    return "".join(s.text for s in ir.spans)


def equivalent_single_span(text: str) -> PromptIR:
    """Build the explicit single-opaque-span IR for `text`. This is the state
    every optimizer must treat identically to the pre-IR raw string (the
    backward-compatibility contract)."""
    span = Span(0, _DEFAULT_KIND, text, Origin.USER, permissions_for(_DEFAULT_KIND))
    return PromptIR((span,), "string", 1.0)
