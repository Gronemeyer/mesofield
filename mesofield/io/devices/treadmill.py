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

The producer (`EncoderSerialInterface`) writes the CSV; the matching
ingest-side parser (`TreadmillSource`) lives at the bottom of this
module so producer and parser sit in the same file.
`EncoderSerialInterface.Parser` resolves to `TreadmillSource` for
manifest-driven dispatch.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mesofield import DeviceRegistry
from mesofield.devices.base import BaseSerialDevice
from mesofield.datakit.config import settings
from mesofield.datakit.sources.register import SourceContext, TimeseriesSource


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


# ---------------------------------------------------------------------------
# Parser (the ingest-side counterpart). Lives in the same file so producer-
# side payload changes are forced to confront parser-side regexes.
# Bound below as `EncoderSerialInterface.Parser`.


class TreadmillSource(TimeseriesSource):
    """Load treadmill samples aligned to the experiment window."""
    tag = "treadmill"
    patterns = ("**/*_treadmill.csv", "**/*_treadmill_data.csv")
    camera_tag = None

    timestamp_column = "timestamp"
    distance_columns = ("distance_mm", "distance")
    speed_columns = ("speed_mm", "speed_mm_s", "speed")
    output_distance_column = "distance_mm"
    output_speed_column = "speed_mm"

    dataqueue_queue_column = settings.timeline.queue_column
    dataqueue_device_id_column = "device_id"
    dataqueue_payload_column = "payload"

    dataqueue_payload_prefix = "EncoderData"
    timeline_master_patterns = ("dhyana", "mesoscope")
    _encoder_ts_re = re.compile(r"timestamp\s*=\s*(\d+)", re.IGNORECASE)

    def build_timeseries(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        context = self._require_context(context)
        df = pd.read_csv(path)
        dist = next((c for c in self.distance_columns if c in df.columns), None)
        speed = next((c for c in self.speed_columns if c in df.columns), None)
        if self.timestamp_column not in df.columns or dist is None or speed is None:
            raise ValueError(f"Expected timestamp/distance/speed columns in {path}")
        df = df.rename(columns={dist: self.output_distance_column, speed: self.output_speed_column})

        dq_path = context.path_for("dataqueue")
        aligned, meta = self.extract_treadmill_aligned(
            df,
            dq_path,
            window=context.experiment_window,
            dataqueue_frame=context.dataqueue_frame,
        )
        t = aligned["time_elapsed_s"].to_numpy(dtype=np.float64)
        meta = {
            "source_file": str(path),
            "dataqueue_file": str(dq_path) if dq_path is not None else None,
            "n_samples": len(aligned),
            **meta,
        }
        return t, aligned, meta

    # --- alignment --------------------------------

    def extract_treadmill_aligned(
        self,
        treadmill_df: pd.DataFrame,
        dq_path: Path | None,
        *,
        window: tuple[float, float] | None = None,
        dataqueue_frame: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        t0, t1 = window or self._window(dq_path, dataqueue_frame)
        duration = float(t1 - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid camera window duration from dataqueue: start={t0}, end={t1}")

        enc = self._encoder_rows(dq_path, dataqueue_frame)
        a, b = self._fit(enc["encoder_ts"].to_numpy(), enc["queue_elapsed"].to_numpy())

        ts = pd.to_numeric(treadmill_df[self.timestamp_column], errors="coerce").to_numpy(dtype=np.float64)
        time_s = a * ts + b - t0

        mask = np.isfinite(time_s) & (time_s >= 0.0) & (time_s <= duration)
        if not np.any(mask):
            raise ValueError("No treadmill samples fall inside the Dhyana window")

        aligned = treadmill_df.loc[mask].copy()
        aligned["time_elapsed_s"] = time_s[mask]
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        return aligned, {
            "source_method": "dataqueue_align_mvp",
            "experiment_window": {"start": float(t0), "end": float(t1)},
            "alignment": {"a": float(a), "b": float(b)},
        }

    # --- dataqueue utilities -----------------------------------------------------

    def _window(
        self,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
    ) -> tuple[float, float]:
        dq = dataqueue_frame
        if dq is None:
            if dq_path is None:
                raise FileNotFoundError("TreadmillSource: dataqueue path not available")
            dq = pd.read_csv(
                dq_path,
                usecols=[self.dataqueue_queue_column, self.dataqueue_device_id_column],
                low_memory=False,
            )
        device = dq.get(self.dataqueue_device_id_column, pd.Series(dtype=str)).astype(str)

        mask = np.zeros(len(dq), dtype=bool)
        for pattern in self.timeline_master_patterns:
            mask |= device.str.contains(pattern, case=False, na=False, regex=False).to_numpy()

        rows = pd.to_numeric(dq.loc[mask, self.dataqueue_queue_column], errors="coerce").dropna()
        if len(rows) < 2:
            raise ValueError("Could not find >=2 master camera rows in dataqueue")
        return float(rows.iloc[0]), float(rows.iloc[-1])

    def _encoder_rows(
        self,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        dq = dataqueue_frame
        if dq is None:
            if dq_path is None:
                raise FileNotFoundError("TreadmillSource: dataqueue path not available")
            dq = pd.read_csv(
                dq_path,
                usecols=[self.dataqueue_queue_column, self.dataqueue_payload_column],
                low_memory=False,
            )
        payload = dq[self.dataqueue_payload_column].astype(str)

        # strict parse only from EncoderData payloads with timestamp=...
        mask = payload.str.contains(self.dataqueue_payload_prefix, na=False)
        if not np.any(mask):
            raise ValueError("No EncoderData payloads found in dataqueue")

        enc = dq.loc[mask, [self.dataqueue_queue_column, self.dataqueue_payload_column]].copy()
        enc["encoder_ts"] = enc[self.dataqueue_payload_column].apply(self._parse_encoder_ts)

        enc = enc.dropna(subset=["encoder_ts", self.dataqueue_queue_column])
        enc = enc.rename(columns={self.dataqueue_queue_column: "queue_elapsed"})
        enc["queue_elapsed"] = pd.to_numeric(enc["queue_elapsed"], errors="coerce")
        enc = enc.dropna(subset=["queue_elapsed"])

        if len(enc) < 2:
            raise ValueError("Insufficient encoder samples for alignment")
        return enc

    def _parse_encoder_ts(self, payload: str) -> int | None:
        m = self._encoder_ts_re.search(payload or "")
        return int(m.group(1)) if m else None

    def _fit(self, encoder_ts: np.ndarray, queue_elapsed: np.ndarray) -> tuple[float, float]:
        x = np.asarray(encoder_ts, dtype=np.float64)
        y = np.asarray(queue_elapsed, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            raise ValueError("Insufficient encoder samples for alignment")

        # Centered affine fit (stable for large microsecond timestamps)
        x0 = float(x.mean())
        y0 = float(y.mean())
        xc = x - x0
        yc = y - y0
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            raise ValueError("Degenerate encoder timestamps for alignment")

        a = float(np.dot(xc, yc) / denom)
        b = float(y0 - a * x0)

        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Encoder alignment fit returned non-finite coefficients")
        return a, b


# Manifest-driven dispatch: SOURCE_REGISTRY["treadmill"] resolves to
# EncoderSerialInterface.Parser.
EncoderSerialInterface.Parser = TreadmillSource
