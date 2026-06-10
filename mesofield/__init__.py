"""Mesofield top-level package.

This module exposes the device-registration surface that binds a YAML
``type:`` string to a device class. There are three equivalent ways to make
a custom device available, in increasing order of decoupling — pick whichever
suits where your class lives:

1. **Decorator** (built-ins and modules you control)::

       from mesofield import DeviceRegistry

       @DeviceRegistry.register("lick_detector")
       class LickDetector(BaseSerialDevice): ...

2. **Programmatic** (house the class anywhere; register it from your own
   launch script before the hardware YAML is materialised)::

       from mesofield import register_device
       from my_lab.lick import LickDetector

       register_device(LickDetector, "lick_detector")

3. **Import string in the YAML** (no registration call at all; the class
   only has to be importable)::

       lick:
         type: my_lab.lick:LickDetector   # 'module:Class' or 'module.Class'

The remaining subpackages are not auto-imported; pull them in explicitly:

.. code-block:: python

    from mesofield.base import Procedure
    from mesofield.config import ExperimentConfig
    from mesofield.hardware import HardwareManager
"""

import importlib
from typing import Type, Callable, Dict, Optional, Any, TypeVar

T = TypeVar("T")

__all__ = ["DeviceRegistry", "register_device"]


class DeviceRegistry:
    """Registry mapping YAML ``type:`` strings to device classes.

    Hardware adapters register themselves via the
    :meth:`~DeviceRegistry.register` decorator (or the module-level
    :func:`register_device`); the
    :class:`~mesofield.hardware.HardwareManager` resolves them by string
    when materialising devices from a YAML file. A ``type:`` that isn't a
    registered short name but *looks* like an import string
    (``module:Class`` or ``module.Class``) is imported on demand, so a
    device class can live in any importable module without a registration
    call.
    """

    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, device_type: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator form: register a device class under ``device_type``.

        Also stamps ``registry_key`` onto the class so any device instance
        can report its YAML ``type:`` for hardware export.
        """
        def decorator(device_class: Type[T]) -> Type[T]:
            register_device(device_class, device_type)
            return device_class
        return decorator

    @classmethod
    def get_class(cls, device_type: str) -> Optional[Type[Any]]:
        """Resolve a YAML ``type:`` to a device class.

        Resolution order:

        1. A registered short name (decorator / :func:`register_device`).
        2. An import string — ``module:Class`` or dotted ``module.Class`` —
           imported on demand. Successful imports are cached under the
           original string so repeated lookups don't re-import.

        Returns ``None`` when neither resolves (HardwareManager turns that
        into an actionable error).
        """
        hit = cls._registry.get(device_type)
        if hit is not None:
            return hit
        resolved = _resolve_import_string(device_type)
        if resolved is not None:
            # Cache under the literal type string so we don't re-import, but
            # do NOT stamp registry_key — the import string is the identity.
            cls._registry[device_type] = resolved
        return resolved


def register_device(
    device_class: Type[T], type_key: Optional[str] = None
) -> Type[T]:
    """Register ``device_class`` under a YAML ``type:`` key.

    The programmatic twin of ``@DeviceRegistry.register``. Call it from your
    own launch script so a device class housed anywhere becomes available to
    ``hardware.yaml`` without editing mesofield::

        from mesofield import register_device
        from my_lab.lick import LickDetector
        register_device(LickDetector, "lick_detector")

    ``type_key`` defaults to the class's existing ``registry_key`` if it has
    one, otherwise the class name. Returns the class unchanged so it can also
    be used as a bare decorator (``@register_device``).
    """
    key = type_key or getattr(device_class, "registry_key", None) or device_class.__name__
    DeviceRegistry._registry[key] = device_class
    device_class.registry_key = key
    return device_class


def _resolve_import_string(ref: str) -> Optional[Type[Any]]:
    """Import a ``module:Class`` / ``module.Class`` reference, or ``None``.

    Only treats ``ref`` as an import string when it actually contains a
    module separator, so a plain unregistered short name (``"lick"``) falls
    through to a clean "not found" rather than a spurious import attempt.
    """
    if ":" not in ref and "." not in ref:
        return None
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
    else:
        module_name, _, attr = ref.rpartition(".")
    if not module_name or not attr:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    obj = getattr(module, attr, None)
    return obj if isinstance(obj, type) else None
