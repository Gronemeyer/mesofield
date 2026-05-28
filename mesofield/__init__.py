"""Mesofield top-level package.

This module exposes the :class:`DeviceRegistry` decorator used by hardware
adapters to declare themselves under a YAML ``type:`` key. The remaining
subpackages are not auto-imported; pull them in explicitly:

.. code-block:: python

    from mesofield.base import Procedure
    from mesofield.config import ExperimentConfig
    from mesofield.hardware import HardwareManager
"""

from typing import Type, Callable, Dict, Optional, Any, TypeVar

T = TypeVar("T")

class DeviceRegistry:
    """Registry mapping YAML ``type:`` strings to device classes.

    Hardware adapters register themselves via the
    :meth:`~DeviceRegistry.register` decorator; the
    :class:`~mesofield.hardware.HardwareManager` looks them up by string
    when materialising devices from a YAML file.
    """
    
    _registry: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register(cls, device_type: str) -> Callable[[Type[T]], Type[T]]:
        """Register a device class for a specific device type.

        The decorator also stamps ``registry_key`` onto the class so any
        device instance can report its YAML ``type:`` for hardware export.
        """
        def decorator(device_class: Type[T]) -> Type[T]:
            cls._registry[device_type] = device_class
            device_class.registry_key = device_type
            return device_class
        return decorator
    
    @classmethod
    def get_class(cls, device_type: str) -> Optional[Type[Any]]:
        """Get the device class for a specific device type."""
        return cls._registry.get(device_type)
