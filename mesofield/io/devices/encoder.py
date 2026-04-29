"""Wheel encoder over USB-serial.

Subclass of :class:`mesofield.devices.base.BaseSerialDevice` that reads
integer click counts (one per line) from an Arduino-style firmware.
Emitted payload is the raw click count (``int``); speed and distance
are derived in analysis from the wheel diameter and CPR carried in the
device config.

Constructor preserves the legacy keyword API
(``serial_port``, ``baud_rate``, ``sample_interval``,
``wheel_diameter``, ``cpr``, ``development_mode``) for compatibility
with :mod:`mesofield.hardware`.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

from mesofield import DeviceRegistry
from mesofield.devices.base import BaseSerialDevice


@DeviceRegistry.register("wheel")
class SerialWorker(BaseSerialDevice):
    """Arduino wheel-encoder device."""

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        serial_port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        sample_interval: Optional[int] = None,
        wheel_diameter: Optional[float] = None,
        cpr: Optional[int] = None,
        development_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = {}
        else:
            cfg = dict(cfg)
        if serial_port is not None:
            cfg.setdefault("port", serial_port)
        if baud_rate is not None:
            cfg.setdefault("baudrate", baud_rate)
        if sample_interval is not None:
            cfg.setdefault("sample_interval_ms", sample_interval)
        if wheel_diameter is not None:
            cfg.setdefault("wheel_diameter", wheel_diameter)
        if cpr is not None:
            cfg.setdefault("cpr", cpr)
        if development_mode is not None:
            cfg.setdefault("development_mode", development_mode)
        cfg.setdefault("id", "encoder")

        super().__init__(cfg, **kwargs)

        # Passive analysis metadata (consumed downstream, not by the
        # acquisition loop).
        self.sample_interval_ms: Optional[int] = self.cfg.get("sample_interval_ms")
        self.wheel_diameter: Optional[float] = self.cfg.get("wheel_diameter")
        self.cpr: Optional[int] = self.cfg.get("cpr")

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
    def parse_line(self, line: bytes) -> Optional[Tuple[int, Optional[float]]]:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            return int(text), None
        except ValueError:
            self.logger.debug("Non-integer line: %r", text)
            return None
