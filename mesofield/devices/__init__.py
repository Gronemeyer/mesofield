from .base import BaseDevice, BaseDataProducer, BaseSerialDevice

# Hardware-specific devices require rig-only dependencies (nidaqmx,
# pymmcore-plus, pyserial, etc.).  Import them lazily so analysis-only
# environments can still reach the base classes.
try:
    from .daq import Nidaq
except ImportError:
    Nidaq = None  # type: ignore[assignment,misc]

try:
    from .cameras import MMCamera, OpenCVCamera
except ImportError:
    MMCamera = None  # type: ignore[assignment,misc]
    OpenCVCamera = None  # type: ignore[assignment,misc]

try:
    from .encoder import SerialWorker
except ImportError:
    SerialWorker = None  # type: ignore[assignment,misc]

try:
    from .treadmill import EncoderSerialInterface
except ImportError:
    EncoderSerialInterface = None  # type: ignore[assignment,misc]

try:
    from .psychopy_device import PsychoPyDevice
except ImportError:
    PsychoPyDevice = None  # type: ignore[assignment,misc]

# Mocks have no rig-only dependencies; import them so their
# ``@DeviceRegistry.register`` decorators (``mock_wheel`` / ``mock_camera``)
# run, letting the ``dev`` rig and the GUI builder produce runnable configs
# without any custom procedure.py registration.
try:
    from .mocks import MockEncoderDevice, MockFrameProducer
except ImportError:
    MockEncoderDevice = None  # type: ignore[assignment,misc]
    MockFrameProducer = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseDevice",
    "BaseDataProducer",
    "BaseSerialDevice",
    "Nidaq",
    "MMCamera",
    "OpenCVCamera",
    "SerialWorker",
    "EncoderSerialInterface",
    "PsychoPyDevice",
    "MockEncoderDevice",
    "MockFrameProducer",
]
