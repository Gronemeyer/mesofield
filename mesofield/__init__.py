from typing import Type, Callable, Dict, Optional, Any, TypeVar

T = TypeVar("T")

class DeviceRegistry:
    """Registry for device classes."""
    
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
