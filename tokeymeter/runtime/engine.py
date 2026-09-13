"""Engine contract and registry.

An Engine is a named pipeline stage with optional phase methods. The kernel
invokes, per request: before_request (registration order), then exactly one
engine's execute (the one that declares handles_execution), then
after_response (REVERSE registration order — middleware unwinding), and
on_error on the failure path.

Engines are loosely coupled: they read/write only the shared request context
dict and services from the DI container.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional


class Engine:
    """Base engine. Subclasses set `name` and override phases they need."""

    name: str = "engine"
    handles_execution: bool = False

    def before_request(self, ctx: Dict[str, Any]) -> None:  # noqa: B027
        pass

    def execute(self, ctx: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def after_response(self, ctx: Dict[str, Any]) -> None:  # noqa: B027
        pass

    def on_error(self, ctx: Dict[str, Any], exc: BaseException) -> None:  # noqa: B027
        pass


class DuplicateEngine(ValueError):
    pass


class NoExecutionEngine(RuntimeError):
    pass


class EngineRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engines: List[Engine] = []

    def register(self, engine: Engine) -> Engine:
        with self._lock:
            if any(e.name == engine.name for e in self._engines):
                raise DuplicateEngine(engine.name)
            self._engines.append(engine)
        return engine

    def ordered(self) -> List[Engine]:
        with self._lock:
            return list(self._engines)

    def execution_engine(self) -> Engine:
        with self._lock:
            for e in self._engines:
                if e.handles_execution:
                    return e
        raise NoExecutionEngine(
            "no registered engine declares handles_execution=True"
        )

    def get(self, name: str) -> Optional[Engine]:
        with self._lock:
            for e in self._engines:
                if e.name == name:
                    return e
        return None
