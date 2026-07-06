"""Name registry + config factory for :class:`FrameProcessor` subclasses.

Kept dependency-free (no import of :mod:`mesofield.processors.base`, which pulls
in PyQt6) so a subclass can register itself at import time without a circular
import. The factory only instantiates whatever class was registered.

A camera stanza in an ``experiment.json`` (or ``hardware.yaml``) can opt a
processor in per-experiment::

    "processors": [
        {"type": "frame_mean", "enabled": true, "plot": true,
         "label": "Mesoscope Frame Mean"}
    ]

Each entry is either a bare type name (``"frame_mean"``) or a mapping with a
``type`` plus constructor/plot options and an optional ``enabled`` toggle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

_PROCESSOR_REGISTRY: Dict[str, Type] = {}


def register_processor(name: str):
    """Class decorator registering a :class:`FrameProcessor` under ``name``.

    The name is what a config ``processors`` entry uses as its ``type``.
    """

    def wrap(cls: Type) -> Type:
        _PROCESSOR_REGISTRY[name] = cls
        return cls

    return wrap


def get_processor_class(name: str) -> Optional[Type]:
    """Return the registered class for ``name``, or ``None``."""
    return _PROCESSOR_REGISTRY.get(name)


def available_processors() -> List[str]:
    """Sorted list of registered processor type names."""
    return sorted(_PROCESSOR_REGISTRY)


def build_processor(
    spec: Any,
    camera: Optional[Any] = None,
    default_name: Optional[str] = None,
) -> Any:
    """Build a :class:`FrameProcessor` from a config ``spec``.

    ``spec`` is either a type-name string or a mapping with a ``type`` key.
    Remaining mapping keys are forwarded to the processor constructor (``plot``,
    ``sampling_rate``, and the recognized plot-styling kwargs). ``enabled`` is
    ignored here -- callers decide whether to build a disabled entry.

    Raises ``ValueError`` for an unknown/missing type and ``TypeError`` for a
    spec that is neither a string nor a mapping.
    """
    if isinstance(spec, str):
        type_key: Optional[str] = spec
        opts: Dict[str, Any] = {}
    elif isinstance(spec, dict):
        opts = dict(spec)
        type_key = opts.pop("type", None) or opts.pop("processor", None)
    else:
        raise TypeError(
            f"processor spec must be a str or mapping, got {type(spec).__name__}"
        )

    if not type_key:
        raise ValueError(f"processor spec {spec!r} is missing a 'type'")

    opts.pop("enabled", None)
    cls = get_processor_class(type_key)
    if cls is None:
        raise ValueError(
            f"Unknown processor type {type_key!r}. "
            f"Available: {available_processors()}"
        )

    name = opts.pop("name", None) or default_name
    return cls(name=name, camera=camera, **opts)
