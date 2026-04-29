#!/usr/bin/env python3
"""
encoder_interface.py

This module provides a serial interface to a Teensy board running the EncoderInterfaceT4 firmware.
It abstracts the serial communication and encoder data parsing to allow easy integration into larger systems.

Firmware Data Format:
  - With SHOW_MICROS defined:
      "micros,distance (mm),speed (mm/s)"
  - Without SHOW_MICROS defined:
      "distance (mm),speed (mm/s)"
      
Supported Commands:
  | Command | Description                                  |
  |---------|----------------------------------------------|
  | '?'     | Print version and header info                |
  | 'c'     | Initiate speed output calibration            |

Usage Example:
    def process_encoder_data(data):
        print(data)

    encoder_interface = EncoderSerialInterface('/dev/ttyACM0', data_callback=process_encoder_data)
    encoder_interface.start()
    
    # Send a command to get header information.
    encoder_interface.send_command('?')
    
    # Run until interrupted...
"""
from typing import Any, Callable, ClassVar, Dict, List

import serial
import time
import logging
import csv
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from PyQt6.QtCore import pyqtSignal, QThread


@dataclass
class EncoderData:
    distance: float
    speed: float
    timestamp: Optional[int] = None

    def __repr__(self):
        return (f"EncoderData(timestamp={self.timestamp}, "
                f"distance={self.distance:.3f} mm, speed={self.speed:.3f} mm/s)")

from mesofield import DeviceRegistry
from mesofield.signals import DeviceSignals

@DeviceRegistry.register("encoder")
class EncoderSerialInterface(QThread):
    
    serialDataReceived = pyqtSignal(int)  # Emits the parsed EncoderData
    serialStreamStarted = pyqtSignal()         # Emits when streaming starts
    serialStreamStopped = pyqtSignal()         # Emits when streaming stops
    serialSpeedUpdated = pyqtSignal(float, float)  # Emits elapsed time and current speed
    device_id: str = "treadmill"  # Default device ID, can be overridden
    device_type: ClassVar[str] = "encoder"
    file_type: str = "csv"
    bids_type: Optional[str] = "beh"
    _started: datetime  # Timestamp when the interface started
    _stopped: datetime # Timestamp when the interface stopped
    
    def __init__(self, port: str, baudrate: int = 192000):
        super().__init__()
        self.signals = DeviceSignals()
        self.logger = logging.getLogger("EncoderSerialInterface")
        #self.device_id: str
        self.serial_port = port
        self.baud_rate = baudrate
        self.output_path: str = ''  # Path to save recorded data
        
        self._recording = False
        self.session_data = []
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
        except serial.SerialException as e:
            self.logger.error(f"Failed to open serial port {port}: {e}")
            raise

        self.logger.info(f"EncoderSerialInterface initialized on port {port} with baudrate {baudrate}")

    def arm(self, config) -> None:
        """Per-run prep: clear buffered samples."""
        self.session_data = []

    def start_recording(self, file_path: Optional[str] = None):
        self._recording = True
        self._started = datetime.now()
        if not self.isRunning():
            self.start()
        if self.isRunning():
            # Flush the input buffer to discard any backlog of serial data.
            self.ser.reset_input_buffer()
            self.output_path = file_path
            self.session_data = []
            self.logger.debug(f"Recording started. Data will be stored to {file_path}")
        else:
            self.recording = False
            self.logger.warning("Cannot start recording: Serial interface is not running.")

    def start(self):
        self.serialStreamStarted.emit()
        self.signals.started.emit()
        super().start()

    def stop(self):
        if self._recording:
            self._recording = False
            self._stopped = datetime.now()
        else:
            self.logger.warning("Recording not active; nothing to stop.")
        self.signals.finished.emit()

    def run(self):
        self.logger.info(f"Serial read thread started.")
        while not self.isInterruptionRequested():
            try:
                raw_line = self.ser.readline()
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8', errors='replace').strip()
                if line:
                    data = self._parse_line(line)
                    if data:
                        #self.serialDataReceived.emit(data.timestamp)
                        self.signals.data.emit(data, datetime.now().timestamp())
                        self.serialSpeedUpdated.emit(data.timestamp or 0, data.speed)
                        # Record data if recording is active.
                        if self._recording:
                            self.session_data.append(data)
            except serial.SerialException as e:
                self.logger.error(f"Serial error: {e}")
                break
            except Exception as ex:
                self.logger.error(f"Unexpected error: {ex}")
        self.logger.info(f"Exiting serial read loop.")

    def save_data(self, path: Optional[str] = None):
        """
        Save the recorded data to a CSV file.
        This method is called when recording is stopped.
        """
        if path:
            self.output_path = path
        if not self.session_data:
            self.logger.warning("No data recorded to save.")
            return
        
        with open(self.output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'distance_mm', 'speed_mm'])
            for data in self.session_data:
                writer.writerow([data.timestamp, data.distance, data.speed])
        
        #self.logger.info(f"Recorded data saved to {self.output_path}")

    def get_data(self) -> list[EncoderData]:
        """
        Get the recorded data.
        This method returns the data collected during the recording session.
        """
        return list(self.session_data)

    def _parse_line(self, line: str) -> Optional[EncoderData]:
        parts = line.split(',')
        try:
            if len(parts) == 3:
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                return EncoderData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 2:
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
                return EncoderData(distance=distance, speed=speed)
            else:
                self.logger.debug(f"Ignored non-data line: {line}")
                return None
        except ValueError:
            self.logger.debug(f"Failed to parse line: {line}")
            return None

    def send_command(self, command: str):
        if self.ser.is_open:
            self.ser.write(command.encode('utf-8'))
            self.logger.info(f"Sent command: {command}")
        else:
            self.logger.warning(f"Serial port not open; command not sent.")

    def shutdown(self):
        self.requestInterruption()
        self.wait()
        if self.ser.is_open:
            self.ser.close()
        self.serialStreamStopped.emit()
        self.logger.info(f"Serial interface stopped and port closed.")

def main():
    """
    Demonstration of how to use the EncoderSerialInterface.
    """
    def process_encoder_data(data: EncoderData):
        print(data)
    
    port = 'COM6'
    encoder_interface = EncoderSerialInterface(port)
    encoder_interface.start()

    try:
        encoder_interface.send_command('?')
        while True:
            time.sleep(1)
    finally:
        encoder_interface.stop()


if __name__ == '__main__':
    main()
