"""Teensy treadmill encoder over USB-serial.

Firmware Data Format:
  - With SHOW_MICROS defined:  ``"micros,distance(mm),speed(mm/s)"``
  - Without SHOW_MICROS:       ``"distance(mm),speed(mm/s)"``

Supported Commands:
  | Command | Description                                  |
  |---------|----------------------------------------------|
  | '?'     | Print version and header info                |
  | 'c'     | Initiate speed output calibration            |

Built on :class:`mesofield.devices.base.BaseSerialDevice`.  Each parsed
line is recorded as a dict ``{"distance": float, "speed": float,
"device_us": int|None}`` so that the default
:meth:`BaseDataProducer.save_data` writes a 4-column CSV
(``timestamp,distance,speed,device_us``).
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

from mesofield import DeviceRegistry
from mesofield.devices.base import BaseSerialDevice


@DeviceRegistry.register("encoder")
class EncoderSerialInterface(BaseSerialDevice):
    """Teensy encoder/treadmill device.

    Constructor accepts either a cfg dict (``BaseSerialDevice``-style) or
    legacy positional/keyword args ``(port, baudrate)`` for backward
    compatibility with :mod:`mesofield.hardware`.
    """

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"
    default_baudrate: ClassVar[int] = 192_000

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = {}
        else:
            cfg = dict(cfg)
        if port is not None:
            cfg.setdefault("port", port)
        if baudrate is not None:
            cfg.setdefault("baudrate", baudrate)
        cfg.setdefault("id", "treadmill")

        super().__init__(cfg, **kwargs)

        # Optional Qt adapter for GUI live-preview signals.  Lazy import
        # so headless sessions don't require PyQt6.
        self._qt_adapter = None
        self.serialDataReceived = None
        self.serialSpeedUpdated = None
        try:
            from mesofield.gui.qt_device_adapter import QtDeviceAdapter

            self._qt_adapter = QtDeviceAdapter(self)
            self.serialDataReceived = self._qt_adapter.serialDataReceived
            self.serialSpeedUpdated = self._qt_adapter.serialSpeedUpdated
        except Exception:
            self.logger.debug("Qt adapter unavailable; running headless.")

    # -- BaseSerialDevice hooks ----------------------------------------
    def parse_line(
        self, line: bytes
    ) -> Optional[Tuple[Dict[str, Any], Optional[float]]]:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        parts = text.split(",")
        try:
            if len(parts) == 3:
                device_us = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
            elif len(parts) == 2:
                device_us = None
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
            else:
                self.logger.debug("Ignored non-data line: %r", text)
                return None
        except ValueError:
            self.logger.debug("Failed to parse line: %r", text)
            return None

        return {"distance": distance, "speed": speed, "device_us": device_us}, None

    # -- Convenience wrapper for firmware commands ---------------------
    def send_command(self, command: str) -> bytes:
        """Send a single-character firmware command (no newline)."""
        return self.send_line(command, newline=b"")
