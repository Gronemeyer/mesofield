"""
Protocol definitions for hardware instruments and data management.

This module defines the core interfaces that standardize behavior across
the mesofield project, allowing for interoperability between different
hardware instruments, data producers, and data consumers.

Protocol Implementation Notes
-----------------------------
When implementing these protocols, there are two approaches:

1. **Direct inheritance** (for regular classes without metaclass conflicts):

   .. code-block:: python

       class MySensor(DataAcquisitionDevice):
           def __init__(self):
               self._init_logger()
               # Implement required methods and attributes

2. **Duck typing** (for classes with existing inheritance or metaclass
   conflicts, e.g. ``QThread``):

   .. code-block:: python

       class MyQThreadSensor(QThread):
           def __init__(self):
               super().__init__()
               self._init_logger()
               self.device_type = "sensor"
               self.device_id = "my_sensor"

The second approach is necessary for Qt classes (``QObject``, ``QThread``,
``QWidget``) or any class that already uses a metaclass. Protocol checking
uses duck typing internally, so both approaches work with our system.
"""

import threading
from typing import Dict, List, Any, Optional, Protocol, TypeVar, Generic, runtime_checkable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.hardware import HardwareManager
    from mesofield.config import ExperimentConfig

T = TypeVar('T')

# These are the Protocol definitions - they are useful for static type checking
# and documentation, but should not be used for inheritance with classes that
# already have a metaclass (like QThread)


class Procedure(Protocol):
    """Protocol defining the standard interface for experiment procedures."""

    protocol: str
    experimenter: str
    config: "ExperimentConfig"
    data_dir: str

    def initialize_hardware(self) -> None:
        """Setup the experiment procedure.

        """
        ...

    def run(self) -> None:
        """Run the experiment procedure."""
        ...

    def save_data(self) -> None:
        """Save data from the experiment."""
        ...

    def cleanup(self) -> None:
        """Clean up after the experiment procedure."""
        ...

@runtime_checkable
class HardwareDevice(Protocol):
    """Protocol defining the standard interface for all hardware devices.

    Lifecycle: ``initialize`` -> ``arm`` -> ``start`` -> ``stop`` -> ``shutdown``.
    Every device exposes ``self.signals`` (a :class:`mesofield.signals.DeviceSignals`)
    carrying ``started``, ``finished``, and ``data`` emitters.
    """

    device_type: str
    device_id: str
    signals: Any  # DeviceSignals; typed as Any to avoid circular import

    def initialize(self) -> bool:
        """One-time setup (open ports, load configs).  Idempotent."""
        ...

    def arm(self, config: "ExperimentConfig") -> None:
        """Per-run preparation (writers, output paths, sequence build).

        Called by ``HardwareManager.arm_all`` immediately before
        ``start_all``.  Devices without per-run prep may no-op.
        """
        ...

    def stop(self):
        """Stop the device after a run."""
        ...

    def shutdown(self) -> None:
        """Close and clean up resources.

        Post-condition: after ``shutdown()`` returns, the device has stopped
        all threads, released its hardware/driver handles, and disconnected all
        of the signals it owns — no further emissions occur. Must be idempotent
        (``HardwareManager.deinitialize`` and other teardown paths may both
        reach a device). This lets a hot-reload tear a rig down without leaking
        threads or firing signals into deleted GUI widgets.
        """
        ...

    def status(self) -> Dict[str, Any]:
        """Get the current status of the device."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the hardware."""
        ...



@runtime_checkable
class DataProducer(HardwareDevice, Protocol):
    """Protocol for hardware that produces data streamed to the DataQueue."""

    sampling_rate: float  # in Hz
    data_type: str
    file_type: str
    bids_type: Optional[str] = None
    is_active: bool
    output_path: str
    metadata_path: Optional[str] = None

    def start(self) -> bool:
        """Start data acquisition.  Should emit ``signals.started``."""
        ...

    def stop(self) -> bool:
        """Stop data acquisition.  Should emit ``signals.finished``."""
        ...

    def save_data(self, path: Optional[str] = None):
        """Persist the captured data."""
        ...

    def get_data(self) -> Optional[Any]:
        """Return the latest data."""
        ...


@runtime_checkable
class FrameProcessor(Protocol):
    """Protocol for optional real-time per-frame consumers.

    A FrameProcessor subscribes to a :class:`DataProducer` camera's
    ``signals.frame`` (carrying ``(img, idx, device_ts)``) and emits a
    scalar result on its own ``signals.data`` and on a Qt-compatible
    ``valueUpdated(time, value)`` signal.  See
    :mod:`mesofield.processors` for the threaded reference base class.
    """

    name: str
    data_type: str
    sampling_rate: float

    def attach(self, camera: "DataProducer") -> None:
        """Subscribe to ``camera.signals.frame`` and start processing."""
        ...

    def detach(self) -> None:
        """Disconnect and stop the worker."""
        ...

    def compute(self, img: Any, idx: Any, ts: Any) -> Optional[float]:
        """Return a scalar for this frame, or ``None`` to skip."""
        ...


@runtime_checkable
class StimulusDevice(HardwareDevice, Protocol):
    """Protocol for stimulus-presentation devices (e.g. PsychoPy).

    Like :class:`HardwareDevice` but explicitly *not* a data producer:
    consumers should not expect ``data`` signal emissions and should not
    call ``save_data``/``get_data``.
    """

    def start(self) -> bool:
        """Begin stimulus presentation."""
        ...




@runtime_checkable
class DataConsumer(Protocol):
    """Protocol defining the interface for data-consuming components."""

    @property
    def name(self) -> str:
        """Return the name of the data consumer."""
        ...

    @property
    def get_supported_data_types(self) -> List[str]:
        """Return the types of data this consumer can process."""
        ...

    def process_data(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Process data with metadata.

        Args:
            data: The data to process.
            metadata: Metadata about the data, including source, timestamp, etc.

        Returns:
            bool: True if data was processed successfully, False otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Threading mixins
# ---------------------------------------------------------------------------

class ThreadedHardwareDevice:
    """
    Mixin for implementing the HardwareDevice protocol with Python's threading.

    This mixin provides the basic structure for a hardware device that uses
    Python's threading module. It handles the thread creation, starting, and stopping.

    Example:
        .. code-block:: python

            class MySensor(ThreadedHardwareDevice):
                device_type = "sensor"
                device_id = "my_sensor"

                def __init__(self, config=None):
                    super().__init__()
                    self.config = config or {}

                def initialize(self):
                    pass

                def _run(self):
                    while not self._stop_event.is_set():
                        # Do work
                        pass

                def get_status(self):
                    return {"active": not self._stop_event.is_set()}
    """

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._active = False

    def start(self) -> bool:
        """Start the device thread."""
        if self._active:
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        self._active = True
        return True

    def stop(self) -> bool:
        """Stop the device thread."""
        if not self._active:
            return True

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._active = False
        return True

    def close(self) -> None:
        """Close the device and clean up resources."""
        self.stop()

    def _run(self) -> None:
        """
        Main thread method to be overridden by subclasses.

        This method runs in a separate thread when start() is called.
        It should check self._stop_event periodically and exit if it's set.
        """
        raise NotImplementedError("Subclasses must implement _run()")


class AsyncioHardwareDevice:
    """
    Mixin for implementing the HardwareDevice protocol with asyncio.

    This mixin provides the basic structure for a hardware device that uses
    Python's asyncio module. It handles the task creation, starting, and cancellation.

    Example:
        .. code-block:: python

            class MySensor(AsyncioHardwareDevice):
                device_type = "sensor"
                device_id = "my_sensor"

                def __init__(self, loop=None, config=None):
                    super().__init__(loop)
                    self.config = config or {}

                def initialize(self):
                    pass

                async def _run(self):
                    while True:
                        if self._should_stop():
                            break
                        await asyncio.sleep(0.01)

                def get_status(self):
                    return {"active": self._task is not None and not self._task.done()}
    """

    def __init__(self, loop=None):
        import asyncio
        self._loop = loop or asyncio.get_event_loop()
        self._task = None
        self._stop_requested = False

    def start(self) -> bool:
        """Start the device task."""
        import asyncio
        if self._task is not None and not self._task.done():
            return True

        self._stop_requested = False
        self._task = asyncio.create_task(self._run())
        return True

    def stop(self) -> bool:
        """Stop the device task."""
        if self._task is None or self._task.done():
            return True

        self._stop_requested = True
        self._task.cancel()
        return True

    def close(self) -> None:
        """Close the device and clean up resources."""
        self.stop()

    def _should_stop(self) -> bool:
        """Check if the task should stop."""
        return self._stop_requested

    async def _run(self) -> None:
        """
        Main coroutine to be overridden by subclasses.

        This coroutine runs as a task when start() is called.
        It should check self._should_stop() periodically and exit if it returns True.
        """
        raise NotImplementedError("Subclasses must implement _run()")
