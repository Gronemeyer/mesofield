from __future__ import annotations

from dataclasses import dataclass

import ast
import bisect
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class PlaybackEvent:
    """Lightweight record representing a single queued datapoint."""

    elapsed: float
    packet_ts: datetime
    device_ts: float | datetime | None
    device_id: str
    payload: Any


class DataqueuePlayback:
    """Time-aligned playback helper for ``dataqueue.csv`` logs.

    This class stays decoupled from the GUI and hardware layers: clients
    register callbacks (either global or per-device) and drive playback by
    calling :meth:`start`, :meth:`stop`, or :meth:`scrub`.
    """

    def __init__(
        self,
        events: Sequence[PlaybackEvent],
        *,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        if not events:
            raise ValueError("Playback requires at least one event")
        if speed <= 0:
            raise ValueError("speed must be positive")

        self._events = sorted(events, key=lambda e: e.elapsed)
        self.speed = speed
        self.loop = loop

        self._listeners: dict[str | None, list[Callable[[PlaybackEvent], None]]] = {}

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._position = 0
        self._last_elapsed = 0.0

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        speed: float = 1.0,
        loop: bool = False,
        payload_parser: Callable[[str], Any] | None = None,
    ) -> "DataqueuePlayback":
        """Create a playback instance from a ``dataqueue.csv`` path."""

        events = list(
            _iter_queue_csv(
                path,
                payload_parser=payload_parser,
            )
        )
        return cls(events, speed=speed, loop=loop)

    @property
    def duration(self) -> float:
        """Total playback span in seconds."""

        return self._events[-1].elapsed - self._events[0].elapsed

    def add_listener(
        self, callback: Callable[[PlaybackEvent], None], device_id: str | None = None
    ) -> None:
        """Register a callback for all events or a specific device."""

        self._listeners.setdefault(device_id, []).append(callback)

    def start(self, *, blocking: bool = False) -> None:
        """Begin playback. If ``blocking`` is ``True``, run in current thread."""

        self._stop_event.clear()
        if blocking:
            self._run()
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and wait for the thread to end."""

        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=1)

    def set_speed(self, speed: float) -> None:
        """Update playback speed at runtime."""

        if speed <= 0:
            raise ValueError("speed must be positive")

        with self._lock:
            self.speed = speed
            self._wake_event.set()

    def scrub(self, *, elapsed: float | None = None, fraction: float | None = None) -> PlaybackEvent:
        """Jump to a position within the playback timeline.

        Provide either ``elapsed`` seconds from start or a ``fraction`` between
        0 and 1. The next dispatched event will be the first event whose
        ``elapsed`` time is greater than or equal to the requested position.
        Returns the event now positioned at (for preview in calling code).
        """

        if elapsed is None and fraction is None:
            raise ValueError("Provide either elapsed or fraction for scrub")
        if fraction is not None and not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")

        with self._lock:
            if elapsed is None:
                elapsed = self.duration * fraction + self._events[0].elapsed
            else:
                elapsed = elapsed + self._events[0].elapsed

            timestamps = [evt.elapsed for evt in self._events]
            idx = bisect.bisect_left(timestamps, elapsed)
            idx = min(idx, len(self._events) - 1)
            self._position = idx
            self._last_elapsed = self._events[idx - 1].elapsed if idx > 0 else self._events[0].elapsed
            self._wake_event.set()
            return self._events[idx]

    # internal helpers -----------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if self._position >= len(self._events):
                    if self.loop:
                        self._position = 0
                        self._last_elapsed = self._events[0].elapsed
                    else:
                        break

                event = self._events[self._position]
                delay = max(0.0, (event.elapsed - self._last_elapsed) / self.speed)
                self._last_elapsed = event.elapsed
                self._position += 1

            if delay:
                self._wake_event.clear()
                self._wake_event.wait(timeout=delay)
            if self._stop_event.is_set():
                break

            self._dispatch(event)

    def _dispatch(self, event: PlaybackEvent) -> None:
        for callback in self._listeners.get(None, []):
            callback(event)
        for callback in self._listeners.get(event.device_id, []):
            callback(event)


def _iter_queue_csv(
    path: str | Path,
    *,
    payload_parser: Callable[[str], Any] | None = None,
) -> Iterable[PlaybackEvent]:
    path = Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        first_elapsed: float | None = None
        for row in reader:
            queue_elapsed = float(row["queue_elapsed"])
            if first_elapsed is None:
                first_elapsed = queue_elapsed
            packet_ts = datetime.fromisoformat(row["packet_ts"])
            device_ts = _parse_device_ts(row.get("device_ts", ""))
            payload_raw = row.get("payload", "")
            payload = (
                payload_parser(payload_raw)
                if payload_parser
                else _default_payload_parser(payload_raw)
            )

            yield PlaybackEvent(
                elapsed=queue_elapsed - first_elapsed,
                packet_ts=packet_ts,
                device_ts=device_ts,
                device_id=row["device_id"],
                payload=payload,
            )


def _default_payload_parser(payload: str) -> Any:
    try:
        return ast.literal_eval(payload)
    except Exception:
        return payload


def _parse_device_ts(raw: str | None) -> float | datetime | None:
    if raw in {"", "None", None}:
        return None

    try:
        return float(raw)
    except ValueError:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
