"""Dependency-injection container for kernel services.

Core services (cache store, provider registry, telemetry sink, ...) are
registered by name; engines resolve by name so implementations can be swapped
(in-memory vs Redis) without touching engine logic — the doc's DI mandate.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional


class ServiceNotRegistered(KeyError):
    pass


class Container:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._factories: Dict[str, Callable[["Container"], Any]] = {}
        self._instances: Dict[str, Any] = {}

    def register_instance(self, name: str, instance: Any) -> None:
        with self._lock:
            self._instances[name] = instance
            self._factories.pop(name, None)

    def register_factory(
        self, name: str, factory: Callable[["Container"], Any]
    ) -> None:
        """Lazy singleton: factory runs once on first resolve."""
        with self._lock:
            self._factories[name] = factory
            self._instances.pop(name, None)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._instances or name in self._factories

    def resolve(self, name: str) -> Any:
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            factory = self._factories.get(name)
            if factory is None:
                raise ServiceNotRegistered(name)
            instance = factory(self)
            self._instances[name] = instance
            return instance

    @contextmanager
    def override(self, name: str, instance: Any) -> Iterator[None]:
        """Test seam: temporarily replace a service, restore on exit."""
        with self._lock:
            had_inst = name in self._instances
            prev_inst = self._instances.get(name)
            had_fact = name in self._factories
            prev_fact = self._factories.get(name)
            self._instances[name] = instance
            self._factories.pop(name, None)
        try:
            yield
        finally:
            with self._lock:
                if had_inst:
                    self._instances[name] = prev_inst
                else:
                    self._instances.pop(name, None)
                if had_fact and prev_fact is not None:
                    self._factories[name] = prev_fact
