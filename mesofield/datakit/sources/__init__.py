"""Explicit registry of datakit data sources."""

from .camera.mesoscope import MesoMetadataSource
from .camera.pupil import PupilMetadataSource
from .camera.suite2p import Suite2pV2
from .behavior.treadmill import TreadmillSource
from .behavior.dataqueue import DataqueueSource
from .behavior.wheel import WheelEncoder
from .behavior.psychopy import Psychopy
from .analysis.mesoscope import MesoMeanSource
from .analysis.mesomap import MesoMapSource
from .analysis.pupil import PupilDLCSource
from .session.config import SessionConfigSource
from .session.notes import SessionNotesSource
from .session.timestamps import SessionTimestampsSource
from .register import DataSource

SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "meso_metadata": MesoMetadataSource,
    "pupil_metadata": PupilMetadataSource,
    "suite2p": Suite2pV2,
    "treadmill": TreadmillSource,
    "dataqueue": DataqueueSource,
    "wheel": WheelEncoder,
    "psychopy": Psychopy,
    "meso_mean": MesoMeanSource,
    "mesomap": MesoMapSource,
    "pupil_dlc": PupilDLCSource,
    "session_config": SessionConfigSource,
    "notes": SessionNotesSource,
    "timestamps": SessionTimestampsSource,
}


def get_source_class(tag: str) -> type[DataSource]:
    """Return the source class for a tag."""
    if tag not in SOURCE_REGISTRY:
        raise KeyError(f"No source registered for tag '{tag}'")
    return SOURCE_REGISTRY[tag]


def available_tags() -> tuple[str, ...]:
    """Return registered source tags in sorted order."""
    return tuple(sorted(SOURCE_REGISTRY.keys()))

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