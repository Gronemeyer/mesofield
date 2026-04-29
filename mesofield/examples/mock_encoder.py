"""Synthetic encoder device for headless / GUI-only iteration.

Produces random click counts on a daemon thread without any serial
hardware.  Subclass :class:`mesofield.devices.base.BaseDataProducer`
directly because there is no physical port to manage.

Usage in ``hardware.yaml``::

    encoder:
      type: mock
      sample_interval_ms: 100

Or programmatically::

    from mesofield.examples.mock_encoder import MockEncoderDevice
    dev = MockEncoderDevice({"id": "encoder", "sample_interval_ms": 50})
    dev.start()
    ...
    dev.stop()
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any, ClassVar, Dict, Optional

from mesofield.devices.base import BaseDataProducer


class MockEncoderDevice(BaseDataProducer):
    """Synthetic encoder that records random click counts."""

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.sample_interval_s: float = float(
            self.cfg.get("sample_interval_ms", 100)
        ) / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.record(random.randint(1, 10))
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MockEncoder-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return super().stop()
