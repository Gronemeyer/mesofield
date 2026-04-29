import random
import time
import math
import serial
from typing import Dict, Any, Optional, ClassVar
from datetime import datetime

from PyQt6.QtCore import pyqtSignal, QThread

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger
# We're not importing DataAcquisitionDevice directly to avoid metaclass conflicts
# SerialWorker will implement the protocol through duck typing instead

class SerialWorker(QThread):
    """
    SerialWorker is a QThread subclass responsible for handling encoder data through two modes:
    development mode (generating simulated data) or serial mode (reading a real serial port).
    
    This class implements the DataAcquisitionDevice protocol through duck typing,
    providing all the necessary methods and attributes without direct inheritance.
    This approach avoids metaclass conflicts while still enabling usage with the DataManager.

    Signals:
    
        1. `serialDataReceived` (pyqtSignal(int)): Emits each time a new encoder reading is captured.
        2. `serialStreamStarted` (pyqtSignal()): Emits when the streaming thread starts running.
        3. `serialStreamStopped` (pyqtSignal()): Emits when the streaming thread stops running.
        4. `serialSpeedUpdated` (pyqtSignal(float, float)): Emits the elapsed time and current speed.
        
    Protocol Compliance:
        
        This class implements the DataAcquisitionDevice protocol attributes and methods:
        - device_type: ClassVar[str] - Type of device ("encoder")
        - device_id: str - Unique identifier for this device
        - config: Dict[str, Any] - Configuration parameters
        - data_rate: float - Data acquisition rate in Hz
        - initialize() - Initialize the device
        - start() -> bool - Start data acquisition
        - stop() -> bool - Stop data acquisition
        - close() - Clean up resources
        - get_status() -> Dict[str, Any] - Get device status
        - get_data() -> Any - Get latest data
    """
    
    # ===================== PyQt Signals ===================== #
    serialDataReceived = pyqtSignal(int) # Emits each time a new encoder reading is captured
    serialStreamStarted = pyqtSignal() # Emits when the streaming thread starts running
    serialStreamStopped = pyqtSignal() # Emits when the streaming thread stops running
    serialSpeedUpdated = pyqtSignal(float, float) # Emits the elapsed time (float) and current speed (float)
    # ======================================================== #
    
    # Hardware device interface properties
    device_id: str  # Unique identifier for this device
    device_type: ClassVar[str] = "encoder"
    file_type: str = "csv"
    bids_type: Optional[str] = "beh"
    data_rate: float = 0.0  # Will be calculated from sample_interval_ms
    _started: datetime  # Time when the device started recording
    _stopped: datetime  # Time when the device stopped recording
    
    def __init__(self, 
                 serial_port: str, 
                 baud_rate: int, 
                 sample_interval: int, 
                 wheel_diameter: float,
                 cpr: int,
                 development_mode: bool = True):
        
        super().__init__()
        self.signals = DeviceSignals()
        self.logger = get_logger(f"SerialWorker-{serial_port}")
        self.logger.debug(f"Initializing SerialWorker with serial port: {serial_port}, "
                         f"baud rate: {baud_rate}, sample interval: {sample_interval} ms, "
                         f"wheel diameter: {wheel_diameter} mm, cpr: {cpr}, "
                         f"development mode: {development_mode}")
        self.development_mode = development_mode
        self.device_id = f"encoder_{serial_port}" if not development_mode else "encoder_dev"
        self.output_path: str = ''  # Path to save recorded data
        # Create config dictionary for protocol compliance
        self.config = {
            "serial_port": serial_port,
            "baud_rate": baud_rate,
            "sample_interval_ms": sample_interval,
            "wheel_diameter": wheel_diameter,
            "cpr": cpr,
            "development_mode": development_mode
        }

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.sample_interval_ms = sample_interval
        self.diameter_mm = wheel_diameter
        self.cpr = cpr
        
        # Calculate data rate in Hz from sample interval
        self.data_rate = 1000.0 / sample_interval if sample_interval > 0 else 50.0

        self.init_data()
        
    def initialize(self) -> None:
        """Initialize the device. Required for HardwareDevice protocol."""
        self.init_data()
        
    def shutdown(self) -> None:
        """Close the device. Required for HardwareDevice protocol."""
        self.stop()
        
    def get_status(self) -> Dict[str, Any]:
        """Get device status. Required for HardwareDevice protocol."""
        return {
            "active": self.isRunning(),
            "data_rate": self.data_rate,
            "development_mode": self.development_mode
        }


    def init_data(self):
        self.stored_data = []
        self.times = []
        self.speeds = []
        self.clicks = []
        self.start_time = None

    def arm(self, config) -> None:
        """Per-run prep: clear buffers."""
        self.init_data()

    def start_recording(self) -> None:
        # Backwards-compat alias; ``start()`` is the standard entry point.
        self.start()

    def start(self):
        self.serialStreamStarted.emit()
        self.signals.started.emit()
        return super().start()

    def stop(self):
        self.requestInterruption()
        self.wait()
        self._stopped = datetime.now()
        self.serialStreamStopped.emit()
        self.signals.finished.emit()


    def run(self):
        self.init_data()
        self.start_time = time.time()
        self._started = datetime.now()
        try:
            if self.development_mode:
                self.run_development_mode()
            else:
                self.run_serial_mode()
        finally:
            self.logger.info("Encoder Stream stopped.")
            
            
    def run_development_mode(self):
        while not self.isInterruptionRequested():
            try:
                # Simulate receiving random encoder clicks
                clicks = random.randint(1, 10)  # Simulating random click values
                
                # Emit signals, store data, and push to the queue
                self.stored_data.append(clicks)  # Store data for later retrieval
                self.serialDataReceived.emit(clicks)  # GUI plotting
                self.signals.data.emit(clicks, time.time())
                
                # Optionally, simulate processing the data for speed calculation
                self.process_data(clicks)
            except Exception as e:
                print(f"Exception in DevelopmentSerialWorker: {e}")
                self.requestInterruption()
            self.msleep(self.sample_interval_ms)  # Sleep for sample interval to reduce CPU usage


    def run_serial_mode(self):
        """
        Runs a continuous loop to read integer data from the configured serial port. 
        Emits raw encoder clicks and send them to processed_data() method.

        Emits:
        
        - serialDataReceived (pyqtSignal(int)): Emits each time a new encoder reading is captured.
        - serialStreamStopped (pyqtSignal()): Emits when the streaming thread stops running.
    
        Raises:
        
            `serial.SerialException`: If there is an issue opening or reading from the serial port.
            `ValueError`: If non-integer values are encountered while reading data.
        """
        
        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            return
        
        try:
            while not self.isInterruptionRequested():
                try:
                    data = self.arduino.readline().decode('utf-8').strip()
                    if data:
                        clicks = int(data)
                        self.serialDataReceived.emit(clicks)  # GUI plotting
                        self.signals.data.emit(clicks, time.time())
                        self.process_data(clicks)
                except ValueError:
                    print(f"Non-integer data received: {data}")
                except serial.SerialException as e:
                    print(f"Serial exception: {e}")
                    self.requestInterruption()
                self.msleep(1)  # Sleep for 1ms to reduce CPU usage
        finally:
            if hasattr(self, 'arduino') and self.arduino is not None:
                try:
                    self.arduino.close()
                except Exception as e:
                    print(f"Exception while closing serial port: {e}")


    def process_data(self, position_change):
        try:
            # Use fixed delta_time based on sample interval
            delta_time = self.sample_interval_ms / 1000.0  # Convert milliseconds to seconds

            # Calculate speed
            speed = self.calculate_speed(position_change, delta_time)

            # Update data lists
            current_time = time.time()
            self.times.append(current_time - self.start_time)
            self.speeds.append(speed)
            self.clicks.append(position_change)

            # Emit a signal for speed update
            self.serialSpeedUpdated.emit((current_time - self.start_time), speed)
        except Exception as e:
            print(f"Exception in processData: {e}")


    def calculate_speed(self, delta_clicks, delta_time):
        """Calculates speed of a wheel with diameter_mm in meters/second
        """
        
        reverse = 1  # Placeholder for direction configuration
        diameter_m = self.diameter_mm / 1000.0 #convert millimeters to meters
        rotations = delta_clicks / self.cpr
        distance = reverse * rotations * (math.pi * diameter_m)  # Circumference * rotations
        speed = distance / delta_time
        return speed
    
        
    def get_data(self):
        from pandas import DataFrame

        clicks = self.clicks
        times = self.times
        speeds = self.speeds
        data = {
            'Clicks': clicks,
            'Time': times,
            'Speed': speeds
        }
        encoder_df = DataFrame(data)
        # Add start/stop timestamps to every row
        encoder_df['Started'] = self._started
        encoder_df['Stopped'] = self._stopped
        return encoder_df
    
    def clear_data(self):
        self.stored_data = []
        self.times = []
        self.speeds = []
        self.start_time = time.time()
    
    def save_data(self, path: str):
        """Save the recorded data to a file."""
        self.output_path = path
        try:
            df = self.get_data()
            df.to_csv(self.output_path, index=False)
            self.logger.debug(f"Data saved to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")

    def __repr__(self):
        class_name = self.__class__.__name__
        module_name = self.__module__
        parent_classes = [cls.__name__ for cls in self.__class__.__bases__]
        return (
            f"<{class_name} {parent_classes} from {module_name}>"
        )

# Usage Example:
# Replace the original SerialWorker instantiation with SerialWorker in development mode
# encoder = SerialWorker(cfg=your_config, development_mode=True)
# encoder.start()
