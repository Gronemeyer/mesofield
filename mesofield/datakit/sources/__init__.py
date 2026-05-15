"""Registry of datakit data sources, populated lazily to dodge import cycles.

Three categories of source class:

- **Producer-paired parsers** live in ``mesofield/io/devices/*.py`` next to
  the device that writes the file. They are reached here via the producer
  class's ``Parser`` attribute (e.g. ``SerialWorker.Parser``).
- **Session-level parsers** (``configuration.csv``, ``notes.txt``,
  ``timestamps.csv``, ``dataqueue.csv``) live in ``sources/session/`` and
  ``sources/behavior/dataqueue.py`` because they are not written by a single
  device -- they're written by `DataManager` itself.
- **Analysis-derived parsers** (mesomap, DLC, suite2p, meso means) live in
  ``sources/analysis/`` and ``sources/camera/`` because they consume the
  output of intermediate processors, not a single producer. Those will move
  to ``mesofield.processing`` modules in a later step.

The producer-paired entries are pulled in lazily on first lookup; if they
were imported eagerly here, ``sources/__init__.py`` would reach into
``mesofield.io.devices``, which in turn imports back into this package via
``TimeseriesSource`` / ``SourceContext`` — a textbook circular import.
"""

# Session-level + analysis parsers (no dependency back on io.devices,
# so safe to import at module load time).
from .camera.mesoscope import MesoMetadataSource
from .camera.pupil import PupilMetadataSource
from .camera.suite2p import Suite2pV2
from .behavior.dataqueue import DataqueueSource
from .analysis.mesoscope import MesoMeanSource
from .analysis.mesomap import MesoMapSource
from .analysis.pupil import PupilDLCSource
from .session.config import SessionConfigSource
from .session.notes import SessionNotesSource
from .session.timestamps import SessionTimestampsSource
from .register import DataSource


_REGISTRY: dict[str, type[DataSource]] = {
    "meso_metadata": MesoMetadataSource,
    "pupil_metadata": PupilMetadataSource,
    "suite2p": Suite2pV2,
    "dataqueue": DataqueueSource,
    "meso_mean": MesoMeanSource,
    "mesomap": MesoMapSource,
    "pupil_dlc": PupilDLCSource,
    "session_config": SessionConfigSource,
    "notes": SessionNotesSource,
    "timestamps": SessionTimestampsSource,
}

# Producer-paired parsers added on first lookup; see module docstring.
_PRODUCER_PARSERS_LOADED = False


def _load_producer_parsers() -> None:
    global _PRODUCER_PARSERS_LOADED
    if _PRODUCER_PARSERS_LOADED:
        return
    from mesofield.io.devices.encoder import SerialWorker
    from mesofield.io.devices.treadmill import EncoderSerialInterface
    from mesofield.io.devices.psychopy_device import PsychoPyDevice

    _REGISTRY["wheel"] = SerialWorker.Parser
    _REGISTRY["treadmill"] = EncoderSerialInterface.Parser
    _REGISTRY["psychopy"] = PsychoPyDevice.Parser
    _PRODUCER_PARSERS_LOADED = True


def get_source_class(tag: str) -> type[DataSource]:
    """Return the source class for a tag (loading device parsers on first call)."""
    _load_producer_parsers()
    if tag not in _REGISTRY:
        raise KeyError(f"No source registered for tag '{tag}'")
    return _REGISTRY[tag]


def available_tags() -> tuple[str, ...]:
    """Return registered source tags in sorted order."""
    _load_producer_parsers()
    return tuple(sorted(_REGISTRY.keys()))


_LAZY_NAMES = {
    "SOURCE_REGISTRY": None,
    "WheelEncoder": "wheel",
    "TreadmillSource": "treadmill",
    "Psychopy": "psychopy",
}


def __getattr__(name: str):
    """Lazy access for SOURCE_REGISTRY and the producer-paired parser names.

    Defers `from mesofield.io.devices...` imports until the calling code
    actually wants the registry, by which point all modules have finished
    initial loading and the cycle no longer matters.
    """
    if name not in _LAZY_NAMES:
        raise AttributeError(name)
    _load_producer_parsers()
    if name == "SOURCE_REGISTRY":
        return _REGISTRY
    return _REGISTRY[_LAZY_NAMES[name]]


__all__ = [
    "MesoMetadataSource",
    "PupilMetadataSource",
    "Suite2pV2",
    "TreadmillSource",
    "DataqueueSource",
    "WheelEncoder",
    "Psychopy",
    "MesoMeanSource",
    "MesoMapSource",
    "PupilDLCSource",
    "SessionConfigSource",
    "SessionNotesSource",
    "SessionTimestampsSource",
    "SOURCE_REGISTRY",
    "get_source_class",
    "available_tags",
]
