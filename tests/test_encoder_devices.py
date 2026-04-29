"""Tests for EncoderSerialInterface (treadmill) and SerialWorker (wheel encoder)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from mesofield.io.devices.encoder import SerialWorker
from mesofield.io.devices.treadmill import EncoderSerialInterface


class _MockSerial:
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


def _drain(dev, n: int, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline and len(dev.get_data()) < n:
        time.sleep(0.01)


def test_treadmill_parses_three_column_lines_to_dict() -> None:
    dev = EncoderSerialInterface(port="MOCK")
    dev._serial = _MockSerial([b"1234,5.5,12.3\n", b"5678,7.0,15.0\n"])
    dev.start()
    _drain(dev, 2)
    dev.stop()

    samples = dev.get_data()
    assert len(samples) == 2
    _ts, payload = samples[0]
    assert payload == {"distance": 5.5, "speed": 12.3, "device_us": 1234}


def test_treadmill_parses_two_column_lines() -> None:
    dev = EncoderSerialInterface(port="MOCK")
    dev._serial = _MockSerial([b"5.5,12.3\n"])
    dev.start()
    _drain(dev, 1)
    dev.stop()

    _ts, payload = dev.get_data()[0]
    assert payload == {"distance": 5.5, "speed": 12.3, "device_us": None}


def test_treadmill_save_data_writes_four_columns(tmp_path: Path) -> None:
    dev = EncoderSerialInterface(port="MOCK")
    dev._serial = _MockSerial([b"1234,5.5,12.3\n", b"5678,7.0,15.0\n"])
    dev.start()
    _drain(dev, 2)
    dev.stop()

    target = tmp_path / "treadmill.csv"
    dev.save_data(str(target))

    rows = list(csv.reader(target.open()))
    assert rows[0] == ["timestamp", "distance", "speed", "device_us"]
    assert rows[1][1:] == ["5.5", "12.3", "1234"]
    assert rows[2][1:] == ["7.0", "15.0", "5678"]


def test_treadmill_send_command_no_newline() -> None:
    dev = EncoderSerialInterface(port="MOCK")
    mock = _MockSerial()
    dev._serial = mock
    dev.send_command("?")
    assert mock.written == [b"?"]


def test_serial_worker_records_int_clicks() -> None:
    dev = SerialWorker(serial_port="MOCK", baud_rate=115200)
    dev._serial = _MockSerial([b"3\n", b"5\n", b"8\n"])
    dev.start()
    _drain(dev, 3)
    dev.stop()

    assert [p for _t, p in dev.get_data()] == [3, 5, 8]


def test_serial_worker_passive_metadata_preserved() -> None:
    dev = SerialWorker(
        serial_port="MOCK",
        baud_rate=115200,
        sample_interval=20,
        wheel_diameter=120.0,
        cpr=2048,
        development_mode=True,
    )
    assert dev.sample_interval_ms == 20
    assert dev.wheel_diameter == 120.0
    assert dev.cpr == 2048
    assert dev.development_mode is True
