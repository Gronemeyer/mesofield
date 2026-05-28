from typing import Any, ClassVar, Dict, List

from dataclasses import dataclass
import threading 
from datetime import datetime
import time

import nidaqmx.system
import nidaqmx
from nidaqmx.constants import Edge

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger


@dataclass
class Nidaq:
    """
    NIDAQ hardware control device.
    
    This class implements the ControlDevice protocol via duck typing,
    providing all the necessary methods and attributes without inheritance.
    """
    device_name: str
    lines: str
    ctr: str
    io_type: str
    device_type: ClassVar[str] = "nidaq"
    device_id: str = "nidaq"
    bids_type: str = ""
    file_type: str = "csv"

    def __post_init__(self):
        self._started: datetime # Timestamp when the device was started
        self._stopped: datetime # Timestamp when the device was stopped
        self.signals = DeviceSignals()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}[{self.device_id}]")
        self.pulse_width = 0.001
        self.poll_interval = 0.01
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._exposure_times: list[float] = []

    def initialize(self) -> None:
        """Initialize the device."""
        pass

    def arm(self, config) -> None:
        """No per-run prep needed."""
        return None

    def test_connection(self):
        self.logger.info(f"Testing connection to NI-DAQ device: {self.device_name}")
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(f'{self.device_name}/{self.lines}')
                task.write(True)
                time.sleep(3)
                task.write(False)
            self.logger.info("NI-DAQ connection successful")
        except nidaqmx.DaqError as e:
            self.logger.error(f"NI-DAQ connection failed: {e}")
            
    def reset(self):
        self.logger.info(f"Resetting NIDAQ device")
        if self._thread and self._thread.is_alive():
            self.stop()
        nidaqmx.system.Device(self.device_name).reset_device()

    def start(self):
        # Configure and start the CI task
        self._ci = nidaqmx.Task()
        self._ci.ci_channels.add_ci_count_edges_chan(
            f"{self.device_name}/{self.ctr}", edge=Edge.RISING, initial_count=0
        )
        self._ci.start()
        self._started = datetime.now()
        # Configure the DO task (camera trigger)
        self._do = nidaqmx.Task()
        self._do.do_channels.add_do_chan(f"{self.device_name}/{self.lines}")

        # Launch background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.signals.started.emit()

    def _worker(self):
        prev_count = 0
        while not self._stop_event.is_set():
            # 1) Start Read count & timestamp
            cnt = self._ci.read()
            ts = time.time()
            dt = datetime.now()

            # 2) Trigger camera
            self._do.write(True)
            time.sleep(self.pulse_width)
            self._do.write(False)

            # 3) For each new edge, record the timestamp
            new_count = int(cnt)
            if new_count > prev_count:
                event_data = [ts] * (new_count - prev_count)
                with self._lock:
                    self._exposure_times.extend(event_data)
                self.signals.data.emit(cnt, ts)
                prev_count = new_count
            
            # 4) Wait before next trigger
            time.sleep(self.poll_interval)

        # Cleanup when stopping
        self._ci.stop()
        self._ci.close()
        self._do.close()
    
    def stop(self):
        """Signal the background thread to stop and wait for it.
        
        This will also reset the NIDAQ device.
        """
        self.logger.info(f"Resetting NIDAQ device")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._stopped = datetime.now()
        nidaqmx.system.Device(self.device_name).reset_device()
        self.signals.finished.emit()
    
    def shutdown(self) -> None:
        """Close the device."""
        self.stop()
    
    def get_data(self) -> list[float]:
        """Retrieve a copy of the host-time exposure timestamps."""
        with self._lock:
            return list(self._exposure_times)
