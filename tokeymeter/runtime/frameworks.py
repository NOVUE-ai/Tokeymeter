"""Framework integrations (W9) — any framework, governed the same way.

Because the runtime is in-process, integrating a framework means wrapping the
one point where that framework makes its model call. This module provides
those wrap points. The universal guarantee: ANY framework that can hand off a
callable (a function, a `.invoke`, a `.complete`, a `.chat`) is governed by
NOVUE with no bespoke integration — `wrap_callable` covers the long tail, and
named helpers make the common frameworks one line.

Design law: REACH only. Each helper builds a Runtime around the framework's
call site using the SHIPPED facade and adapters. No framework SDK is imported
at module load (they are optional); helpers duck-type or accept the call site
directly, so importing this module never requires LangChain or LlamaIndex to
be installed.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .facade import Runtime


def wrap_callable(fn: Callable[[str], Any], *,
                  config: Optional[Dict[str, Any]] = None,
                  model: str = "default", **kw: Any) -> Runtime:
    """The universal wrap point: ANY callable that maps a prompt string to a
    response becomes a governed Runtime. This is the guarantee that a
    framework we have never heard of works today.

        governed = wrap_callable(my_framework.run)
        governed.execute("hello")
    """
    return Runtime(call=fn, config=config, model=model, **kw)


def wrap_langchain_llm(llm: Any, *, config: Optional[Dict[str, Any]] = None,
                       model: str = "default", **kw: Any) -> Runtime:
    """Wrap a LangChain LLM/ChatModel. LangChain models expose `.invoke(str)`
    (and older `.predict`); we adapt whichever is present into the callable
    the Runtime governs. No langchain import required to define this.
    """
    invoke = getattr(llm, "invoke", None) or getattr(llm, "predict", None) \
        or getattr(llm, "__call__", None)
    if invoke is None:
        raise TypeError(
            "object is not a LangChain-style model: expected .invoke/.predict")

    def _call(prompt: str) -> Any:
        result = invoke(prompt)
        # LangChain chat models return a message with `.content`; normalize
        return getattr(result, "content", result)

    return Runtime(call=_call, config=config, model=model, **kw)


def wrap_llamaindex_llm(llm: Any, *, config: Optional[Dict[str, Any]] = None,
                        model: str = "default", **kw: Any) -> Runtime:
    """Wrap a LlamaIndex LLM. LlamaIndex LLMs expose `.complete(str)` returning
    an object with `.text`; adapt it into the governed callable."""
    complete = getattr(llm, "complete", None)
    if complete is None:
        raise TypeError(
            "object is not a LlamaIndex LLM: expected .complete")

    def _call(prompt: str) -> Any:
        result = complete(prompt)
        return getattr(result, "text", result)

    return Runtime(call=_call, config=config, model=model, **kw)


def governed_tool(fn: Callable[..., Any], *,
                  config: Optional[Dict[str, Any]] = None,
                  tool_name: str = "tool", **kw: Any) -> Callable[..., Any]:
    """Wrap an agent TOOL so its LLM-facing text is governed. Agent frameworks
    call tools with arbitrary signatures; this governs the tool's primary
    string argument (the model-facing content) while passing the rest through.
    Returns a drop-in replacement callable with the same signature shape.
    """
    runtime = Runtime(call=lambda p: p, config=config, model=tool_name, **kw)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # govern the first string arg (the model-facing content); the tool's
        # own logic runs on the governed (screened/optimized) text.
        if args and isinstance(args[0], str):
            governed_text = runtime.execute(args[0])
            return fn(governed_text, *args[1:], **kwargs)
        return fn(*args, **kwargs)

    _wrapped.__name__ = getattr(fn, "__name__", tool_name)
    return _wrapped
