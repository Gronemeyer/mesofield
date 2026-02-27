from __future__ import annotations

from typing import Type, Callable, Any, TypeVar

T = TypeVar("T")

__all__ = ["DeviceRegistry"]


class DeviceRegistry:
    """Registry for device classes."""
    
    _registry: dict[str, type[Any]] = {}
    
    @classmethod
    def register(cls, device_type: str) -> Callable[[type[T]], type[T]]:
        """Register a device class for a specific device type."""
        def decorator(device_class: type[T]) -> type[T]:
            cls._registry[device_type] = device_class
            return device_class
        return decorator
    
    @classmethod
    def get_class(cls, device_type: str) -> type[Any] | None:
        """Get the device class for a specific device type."""
        return cls._registry.get(device_type)
