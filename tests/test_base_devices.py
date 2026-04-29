"""Tests for the authorable device base classes.

Covers:
- ``BaseDevice`` lifecycle + signal emission + status fields.
- ``BaseDataProducer.record`` writes to buffer and emits ``signals.data``.
- ``BaseDataProducer.save_data`` produces a valid CSV.
- ``BaseSerialDevice`` polling + send_line + setup_serial hook against
  a mock serial port; development_mode launches without hardware.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import pytest

from mesofield.devices import (
    BaseDataProducer,
    BaseDevice,
    BaseSerialDevice,
)


# ---------------------------------------------------------------------------
# BaseDevice
# ---------------------------------------------------------------------------


def test_base_device_lifecycle_and_signals() -> None:
    dev = BaseDevice({"id": "alpha", "primary": True})
    started: list = []
    finished: list = []
    dev.signals.started.connect(lambda: started.append(True))
    dev.signals.finished.connect(lambda: finished.append(True))

    assert dev.device_id == "alpha"
    assert dev.is_primary is True
    assert dev.is_running is False

    dev.start()
    assert dev.is_running is True
    assert started == [True]

    dev.stop()
    assert dev.is_running is False
    assert finished == [True]

    status = dev.status()
    assert status["device_id"] == "alpha"
    assert status["is_primary"] is True
    assert status["started"] is not None
    assert status["stopped"] is not None


def test_base_device_defaults_when_no_cfg() -> None:
    dev = BaseDevice()
    assert dev.is_primary is False
    assert dev.cfg == {}


# ---------------------------------------------------------------------------
# BaseDataProducer
# ---------------------------------------------------------------------------


def test_record_buffers_and_emits() -> None:
    dev = BaseDataProducer({"id": "p1"})
    seen: list = []
    dev.signals.data.connect(lambda payload, ts: seen.append((payload, ts)))

    ts1 = dev.record({"x": 1})
    dev.record("hello", ts=42.0)

    assert len(dev.get_data()) == 2
    assert dev.get_data()[1] == (42.0, "hello")
    assert seen == [({"x": 1}, ts1), ("hello", 42.0)]


def test_save_data_writes_csv(tmp_path: Path) -> None:
    dev = BaseDataProducer({"id": "p2"})
    dev.record("a", ts=1.0)
    dev.record("b", ts=2.0)

    target = tmp_path / "out.csv"
    written = dev.save_data(str(target))
    assert written == str(target)

    rows = list(csv.reader(target.open()))
    assert rows[0] == ["timestamp", "payload"]
    assert rows[1] == ["1.0", "a"]
    assert rows[2] == ["2.0", "b"]


def test_save_data_with_dict_payloads_fans_to_columns(tmp_path: Path) -> None:
    dev = BaseDataProducer({"id": "p2b"})
    dev.record({"distance": 1.5, "speed": 0.2}, ts=1.0)
    dev.record({"distance": 3.0, "speed": 0.4}, ts=2.0)

    target = tmp_path / "multi.csv"
    dev.save_data(str(target))

    rows = list(csv.reader(target.open()))
    assert rows[0] == ["timestamp", "distance", "speed"]
    assert rows[1] == ["1.0", "1.5", "0.2"]
    assert rows[2] == ["2.0", "3.0", "0.4"]


def test_save_data_with_mixed_keys_unions_columns(tmp_path: Path) -> None:
    dev = BaseDataProducer({"id": "p2c"})
    dev.record({"a": 1}, ts=1.0)
    dev.record({"a": 2, "b": 9}, ts=2.0)

    target = tmp_path / "mixed.csv"
    dev.save_data(str(target))

    rows = list(csv.reader(target.open()))
    assert rows[0] == ["timestamp", "a", "b"]
    assert rows[1] == ["1.0", "1", ""]
    assert rows[2] == ["2.0", "2", "9"]


def test_arm_clears_buffer_and_calls_make_path() -> None:
    dev = BaseDataProducer({"id": "p3"})
    dev.record("stale")
    assert dev.get_data()

    class FakeCfg:
        def __init__(self) -> None:
            self.calls: list = []

        def make_path(self, name, ext, bids):
            self.calls.append((name, ext, bids))
            return f"/tmp/{name}.{ext}"

    cfg = FakeCfg()
    dev.arm(cfg)
    assert dev.get_data() == []
    assert dev.output_path == "/tmp/p3.csv"
    assert cfg.calls == [("p3", "csv", "beh")]


# ---------------------------------------------------------------------------
# BaseSerialDevice
# ---------------------------------------------------------------------------


class _MockSerial:
    """Minimal stand-in for ``serial.Serial`` for unit tests."""

    def __init__(self, lines=()):
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


class _IntParser(BaseSerialDevice):
    poll_interval = 0.005

    def parse_line(self, line: bytes):
        text = line.strip()
        if not text:
            return None
        return int(text), None


def test_serial_device_polling_with_mock_port() -> None:
    dev = _IntParser({"id": "mock", "port": "MOCK"})
    dev._serial = _MockSerial([b"10\n", b"20\n", b"30\n"])
    dev.start()

    deadline = time.time() + 1.0
    while time.time() < deadline and len(dev.get_data()) < 3:
        time.sleep(0.01)

    dev.stop()
    assert [p for _t, p in dev.get_data()] == [10, 20, 30]


def test_serial_device_requires_port_when_not_development() -> None:
    dev = _IntParser({"id": "fake2", "development_mode": False})
    with pytest.raises(ValueError):
        dev.initialize()


def test_send_line_in_development_mode_is_noop() -> None:
    dev = _IntParser({"id": "fake3", "development_mode": True})
    assert dev.send_line("HELLO") == b"HELLO\n"
    assert dev.send_line(b"RAW\n") == b"RAW\n"


def test_send_line_writes_to_mock_port() -> None:
    dev = _IntParser({"id": "fake5", "port": "MOCK"})
    mock = _MockSerial()
    dev._serial = mock
    dev.send_line("PUFF 50")
    assert mock.written == [b"PUFF 50\n"]


def test_setup_serial_hook_runs_after_open() -> None:
    calls: list = []

    class _HandshakeDev(BaseSerialDevice):
        def parse_line(self, line):
            return None

        def setup_serial(self):
            calls.append(self._serial)
            self.send_line("HELLO")

    dev = _HandshakeDev({"id": "h", "port": "MOCK"})
    # Stub the real port-open path: pre-set _serial so initialize() short-circuits
    # past the pyserial import, then invoke setup_serial directly to mirror the
    # real lifecycle order.
    mock = _MockSerial()
    dev._serial = mock
    dev.setup_serial()
    assert calls == [mock]
    assert mock.written == [b"HELLO\n"]


def test_send_line_without_init_raises() -> None:
    dev = _IntParser({"id": "fake4", "development_mode": False, "port": "COMX"})
    # Did not call initialize(), real port not opened.
    with pytest.raises(RuntimeError):
        dev.send_line("HI")


# ---------------------------------------------------------------------------
# Teensy example
# ---------------------------------------------------------------------------


def test_teensy_example_loads_in_development_mode() -> None:
    """The shipped Teensy example must import and instantiate without hardware."""
    import importlib.util
    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "examples" / "teensy_pulse_generator.py"
    assert example.is_file()

    spec = importlib.util.spec_from_file_location("teensy_example", example)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    teensy = mod.TeensyPulseGenerator(
        {"id": "teensy", "development_mode": True}
    )
    teensy.start()
    try:
        # Commands should be no-ops, not exceptions.
        teensy.set_frequency(10.0)
        teensy.start_pulses()
        teensy.query_status()
        teensy.stop_pulses()
        assert teensy.frequency_hz == 10.0
    finally:
        teensy.stop()
