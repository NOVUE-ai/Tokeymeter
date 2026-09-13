"""Response content extraction — what an assistant turn actually produced.

WHY THIS EXISTS
---------------
The progress signal asks one question: are the agent's responses still novel?
Answering it requires the CONTENT of a response, and a provider object cannot
supply that by being stringified — an OpenAI `ChatCompletion` or an Anthropic
`Message` carries a unique request id, so `str(response)` differs on every call
even when the content is identical. Fingerprinting that would give a completely
stuck agent a perfect progress score.

So the node refuses to score a provider object unless it is handed a real
extractor. This module is that extractor for the two SDKs we ship wrappers
for, which is what makes the signal work out of the box instead of silently
reporting "not scored" for the most common shape in production.

TOOL CALLS ARE THE RESPONSE
---------------------------
This is the part that decides whether the signal works for agents at all. A
typical agent turn returns NO message text — it returns a tool call. If an
empty `content` meant "not scoreable", the progress signal would be blind to
the dominant agent architecture, which is precisely the workload it exists for.

So when a turn produced tool calls, the tool name and arguments ARE the
response: an agent invoking the same tool with the same arguments, turn after
turn, is exactly what being stuck looks like.

CONTENT-BLIND POSTURE IS UNCHANGED
----------------------------------
Everything extracted here is HASHED by the caller and discarded. Nothing is
stored, logged, or compared as text — the same posture the prompt fingerprint
has always had. This module returns text so a digest can be taken of it; it is
never a path by which content enters a record.

DEFENSIVE BY CONSTRUCTION
-------------------------
Provider SDKs change shape between versions, and a wrapper must never be the
reason a caller's request fails. Every accessor here is guarded and every
failure returns None, which the node reads as "not scored" — honest, and
strictly better than a fabricated number.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["openai_response_text", "anthropic_response_text"]

# A response is hashed from a bounded prefix, so there is no value in building
# a huge string out of many tool-call arguments before handing it over.
_MAX_PARTS = 32
_MAX_LEN = 65536


def _clip(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text[:_MAX_LEN]
    return text or None


def openai_response_text(response: Any) -> Optional[str]:
    """Assistant content from an OpenAI chat completion, or None.

    Falls back to the turn's tool calls when there is no message text, because
    a tool-calling turn IS the agent's output and an agent repeating the same
    call is the signal we are looking for.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None

        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return _clip(content)

        # No message text: the turn produced tool calls, which are its output.
        tool_calls = getattr(message, "tool_calls", None) or []
        parts = []
        for call in tool_calls[:_MAX_PARTS]:
            fn = getattr(call, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            args = getattr(fn, "arguments", None) if fn is not None else None
            if name or args:
                parts.append(f"{name or ''}({args or ''})")
        if parts:
            return _clip("\u0000".join(parts))

        # Some deployments still return the legacy single function_call.
        legacy = getattr(message, "function_call", None)
        if legacy is not None:
            name = getattr(legacy, "name", None)
            args = getattr(legacy, "arguments", None)
            if name or args:
                return _clip(f"{name or ''}({args or ''})")
        return None
    except Exception:
        return None


def anthropic_response_text(response: Any) -> Optional[str]:
    """Assistant content from an Anthropic message, or None.

    Anthropic returns a LIST of content blocks — text blocks and tool_use
    blocks can appear in the same turn — so every block contributes, in order.
    """
    try:
        blocks = getattr(response, "content", None)
        if isinstance(blocks, str):
            return _clip(blocks)
        if not blocks:
            return None
        parts = []
        for block in list(blocks)[:_MAX_PARTS]:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            # tool_use block: the tool and its input are the agent's output
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                name = getattr(block, "name", None)
                bin_ = getattr(block, "input", None)
                if name or bin_:
                    parts.append(f"{name or ''}({bin_ or ''})")
        if not parts:
            return None
        return _clip("\u0000".join(parts))
    except Exception:
        return None
