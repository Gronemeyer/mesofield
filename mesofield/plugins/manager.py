"""Simple plugin infrastructure for Mesofield.

This module exposes a very small registry-based plugin system that mirrors the
register/get patterns used throughout the configuration layer.  The goal is to
keep the surface area tiny while still enabling runtime discovery of optional
components defined in experiment JSON files.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Type, TypeVar

from mesofield.utils._logger import get_logger

logger = get_logger("PluginManager")


class SupportsStop(Protocol):
    """Protocol describing the tiny surface exposed by plugins."""

    def attach(self, *, experiment_config: Any, data_manager: Any) -> None:  # pragma: no cover - typing hook
        ...

    def start(self) -> None:  # pragma: no cover - typing hook
        ...

    def stop(self) -> None:  # pragma: no cover - typing hook
        ...


@dataclass
class PluginSpec:
    """Light-weight container describing a configured plugin."""

    name: str
    enabled: bool
    module: Optional[str]
    config: Dict[str, Any]
    metadata: Dict[str, Any]


_P = TypeVar("_P", bound="Plugin")


class Plugin:
    """Base class for Mesofield plugins.

    Subclasses are expected to override :meth:`attach`, :meth:`start`, and
    :meth:`stop`.  The default implementation is intentionally minimal so that
    very small plugins can be authored without ceremony.
    """

    def __init__(self, *, name: str, config: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config: Dict[str, Any] = dict(config or {})
        self.metadata = dict(metadata or {})
        self._attached = False

    # lifecycle -------------------------------------------------------
    def attach(self, *, experiment_config: Any, data_manager: Any) -> None:
        self._attached = True

    def start(self) -> None:
        if not self._attached:
            raise RuntimeError(f"Plugin '{self.name}' has not been attached")

    def stop(self) -> None:
        self._attached = False


# Global class registry keyed by canonical plugin name.
_GLOBAL_REGISTRY: Dict[str, Type[Plugin]] = {}


def register_plugin(name: str) -> Callable[[Type[_P]], Type[_P]]:
    """Decorator used by plugins to register themselves.

    The decorator mirrors the registry pattern used by :class:`ExperimentConfig`
    so that plugins can be discovered without additional boilerplate.
    """

    def _wrap(cls: Type[_P]) -> Type[_P]:
        key = name.strip().lower()
        if key in _GLOBAL_REGISTRY:
            logger.warning("Plugin '%s' already registered; overriding with %s", key, cls)
        _GLOBAL_REGISTRY[key] = cls
        cls.plugin_name = key  # type: ignore[attr-defined]
        return cls

    return _wrap


class PluginManager:
    """Runtime plugin controller used by :class:`ExperimentConfig`.

    The manager keeps track of three concepts:
    * registry: Python classes registered via :func:`register_plugin`
    * specs: enabled plugins declared in experiment JSON files
    * instances: live plugin objects that have been attached to the experiment
    """

    def __init__(self, *, registry: Optional[Dict[str, Type[Plugin]]] = None) -> None:
        self._registry: Dict[str, Type[Plugin]] = registry or _GLOBAL_REGISTRY
        self._specs: Dict[str, PluginSpec] = {}
        self._instances: Dict[str, Plugin] = {}

    # configuration ---------------------------------------------------
    def clear(self) -> None:
        """Forget all configured and instantiated plugins."""

        self._specs.clear()
        self.shutdown()

    def configure_from_mapping(self, mapping: Dict[str, Dict[str, Any]]) -> None:
        """Load plugin declarations from a ``Plugins`` dictionary.

        Each entry should follow the structure::

            {
                "enabled": true,
                "module": "mesofield.plugins.mouseportal",
                "config": {...}
            }
        """

        for name, raw in mapping.items():
            if not isinstance(raw, dict):
                logger.warning("Plugin declaration for '%s' is not a mapping", name)
                continue
            enabled = bool(raw.get("enabled", False))
            module = raw.get("module")
            if module:
                try:
                    importlib.import_module(module)
                except Exception as exc:  # pragma: no cover - import failure path
                    logger.error("Failed to import plugin module '%s': %s", module, exc)
                    continue
            spec = PluginSpec(
                name=name.strip().lower(),
                enabled=enabled,
                module=module,
                config=dict(raw.get("config", {})),
                metadata=dict(raw),
            )
            if spec.enabled:
                self._specs[spec.name] = spec
            elif spec.name in self._specs:
                self._specs.pop(spec.name)

    # lookup ----------------------------------------------------------
    def get_config(self, name: str) -> Dict[str, Any]:
        spec = self._specs.get(name.strip().lower())
        return dict(spec.config) if spec else {}

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._specs

    def names(self) -> Iterable[str]:
        return list(self._specs.keys())

    # lifecycle -------------------------------------------------------
    def attach_all(self, *, experiment_config: Any, data_manager: Any) -> Iterable[Plugin]:
        """Instantiate and attach all configured plugins."""

        attached: list[Plugin] = []
        for name, spec in self._specs.items():
            if name in self._instances:
                attached.append(self._instances[name])
                continue
            cls = self._registry.get(name)
            if cls is None:
                logger.warning("No plugin registered under name '%s'", name)
                continue
            try:
                plugin = cls(name=name, config=spec.config, metadata=spec.metadata)
                plugin.attach(experiment_config=experiment_config, data_manager=data_manager)
                plugin.start()
            except Exception as exc:
                logger.error("Failed to start plugin '%s': %s", name, exc)
                continue
            self._instances[name] = plugin
            attached.append(plugin)
        return attached

    def shutdown(self) -> None:
        """Stop all active plugins."""

        for name, plugin in list(self._instances.items()):
            try:
                plugin.stop()
            except Exception as exc:  # pragma: no cover - shutdown path
                logger.error("Failed to stop plugin '%s': %s", name, exc)
            finally:
                self._instances.pop(name, None)

    # helpers ---------------------------------------------------------
    def instance(self, name: str) -> Optional[Plugin]:
        return self._instances.get(name.strip().lower())


__all__ = ["Plugin", "PluginManager", "register_plugin"]
