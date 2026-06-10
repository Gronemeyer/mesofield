"""Shared helper classes for the mesofield test suite.

Importable as ``from _helpers import MockSerial, StubConfig, SignalCounter``
(``tests/`` is on the path via ``pythonpath`` in pyproject's pytest config).
Fixtures live in ``conftest.py``; plain doubles live here so they can be
imported at module scope for parametrization.
"""

from __future__ import annotations

import threading


class MockSerial:
    """Stand-in for ``serial.Serial``: replays canned ``bytes`` lines and
    records writes."""

    def __init__(self, lines: tuple[bytes, ...] = ()):
        self._lines = list(lines)
        self.written: list = []

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        pass


class StubConfig:
    """Minimal ExperimentConfig double: a duration and a path factory —
    enough for ``device.arm(config)`` and ``set_writer``."""

    def __init__(self, tmp_path, duration: float = 0.25):
        self._tmp = tmp_path
        self.sequence_duration = duration

    def make_path(self, name, ext, bids_type=None, *_args):
        return str(self._tmp / f"{name}.{ext}")

    @staticmethod
    def build_sequence(_device):
        return None


class SignalCounter:
    """Counts a device's DeviceSignals emissions and exposes a wait-able
    ``finished`` event. Construct with the device to attach."""

    def __init__(self, device):
        self.started = 0
        self.finished = 0
        self.errors: list[BaseException] = []
        self.finished_event = threading.Event()
        # Keep refs alive (psygnal holds the callables, but be explicit).
        self._cbs = [self._on_started, self._on_finished, self._on_error]
        device.signals.started.connect(self._on_started)
        device.signals.finished.connect(self._on_finished)
        device.signals.error.connect(self._on_error)

    def _on_started(self):
        self.started += 1

    def _on_finished(self):
        self.finished += 1
        self.finished_event.set()

    def _on_error(self, exc):
        self.errors.append(exc)

    def wait_finished(self, timeout: float = 5.0) -> bool:
        return self.finished_event.wait(timeout)
