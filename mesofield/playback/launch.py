from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .dataqueue import DataqueuePlayback, PlaybackEvent
from .media import (
    Mp4FrameSource,
    TiffFrameSource,
    discover_media_paths,
    load_treadmill_trace,
)


@dataclass
class PlaybackContext:
    """Bundle of artifacts needed to drive playback."""

    experiment_dir: Path
    selection: Tuple[str, str, str]
    dataqueue_path: Path
    playback: DataqueuePlayback
    meso_path: Path | None = None
    pupil_path: Path | None = None
    treadmill_path: Path | None = None


def discover_playback_context(
    experiment_dir: str | Path, *, speed: float = 1.0, loop: bool = False
) -> PlaybackContext:
    """Locate a recorded session and build a playback context.

    The search prefers BIDS-organized experiments discoverable via
    :class:`ExperimentData`, but falls back to a simple recursive search for
    ``dataqueue.csv`` when necessary.
    """

    root = Path(experiment_dir)
    exp = None
    meso_path: Path | None = None
    pupil_path: Path | None = None
    treadmill_path: Path | None = None
    try:
        from mesofield.data.proc.load import ExperimentData

        exp = ExperimentData(root)
    except Exception:
        exp = None

    if exp is not None and "dataqueue" in exp.data.columns:
        for idx, row in exp.data.iterrows():
            dq_path = row.get("dataqueue")
            if dq_path and Path(dq_path).exists():
                meso_path = Path(row.get("meso_tiff")) if row.get("meso_tiff") else None
                pupil_path = Path(row.get("pupil_mp4")) if row.get("pupil_mp4") else None
                treadmill_path = Path(row.get("encoder")) if row.get("encoder") else None
                return PlaybackContext(
                    experiment_dir=root,
                    selection=idx,
                    dataqueue_path=Path(dq_path),
                    playback=DataqueuePlayback.from_csv(
                        dq_path, speed=speed, loop=loop
                    ),
                    meso_path=meso_path,
                    pupil_path=pupil_path,
                    treadmill_path=treadmill_path,
                )

    dq_candidates = sorted(root.rglob("dataqueue.csv"))
    if not dq_candidates:
        raise FileNotFoundError(
            f"No dataqueue.csv found under {experiment_dir}; cannot start playback"
        )

    dummy_index = ("unknown", "unknown", "unknown")
    dq_path = dq_candidates[0]
    if not meso_path and not pupil_path and not treadmill_path:
        meso_path, pupil_path, treadmill_path = discover_media_paths(root)
    return PlaybackContext(
        experiment_dir=root,
        selection=dummy_index,
        dataqueue_path=dq_path,
        playback=DataqueuePlayback.from_csv(dq_path, speed=speed, loop=loop),
        meso_path=meso_path,
        pupil_path=pupil_path,
        treadmill_path=treadmill_path,
    )


def launch_playback_app(context: PlaybackContext) -> None:
    """Start a minimal Qt application for playback control."""

    from PyQt6.QtCore import Qt, pyqtSignal, QObject
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    try:
        import pyqtgraph as pg
    except Exception:
        pg = None

    class _PlaybackBridge(QObject):
        """Bridge DataqueuePlayback callbacks onto Qt signals."""

        event_ready = pyqtSignal(object)

        def __init__(self, playback: DataqueuePlayback):
            super().__init__()
            playback.add_listener(self.event_ready.emit)

    class PlaybackWindow(QWidget):
        """Minimal controller window for dataqueue playback."""

        def __init__(self, context: PlaybackContext):
            super().__init__()
            self.context = context
            self.bridge = _PlaybackBridge(context.playback)
            self.bridge.event_ready.connect(self._on_event)
            self._tread_times = None
            self._tread_vals = None
            self._live_times: list[float] = []
            self._live_vals: list[float] = []
            self._last_elapsed = 0.0

            self.setWindowTitle("Mesofield Playback")

            layout = QVBoxLayout(self)

            self.status = QLabel(
                f"Session: {context.selection} | {context.dataqueue_path.name}"
            )
            layout.addWidget(self.status)

            self.video_layout = QHBoxLayout()
            layout.addLayout(self.video_layout)
            self.sources: list[tuple[object, QLabel, QLabel]] = []
            if context.meso_path:
                meso_box = QVBoxLayout()
                meso_view = QLabel("Loading mesoscope…")
                meso_view.setMinimumSize(320, 240)
                meso_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
                meso_view.setStyleSheet("background: #111; color: #aaa;")
                meso_view.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
                )
                meso_info = QLabel("Mesoscope")
                meso_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                meso_box.addWidget(meso_view)
                meso_box.addWidget(meso_info)
                meso_widget = QWidget()
                meso_widget.setLayout(meso_box)
                self.video_layout.addWidget(meso_widget)
                source = TiffFrameSource(
                    context.meso_path, duration_hint=context.playback.duration
                )
                meso_info.setText(
                    f"{context.meso_path.name} • {source.fps:.2f} fps"
                    if source.fps
                    else context.meso_path.name
                )
                self.sources.append((source, meso_view, meso_info))
            if context.pupil_path:
                pupil_box = QVBoxLayout()
                pupil_view = QLabel("Loading pupil…")
                pupil_view.setMinimumSize(320, 240)
                pupil_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pupil_view.setStyleSheet("background: #111; color: #aaa;")
                pupil_view.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
                )
                pupil_info = QLabel("Pupil camera")
                pupil_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pupil_box.addWidget(pupil_view)
                pupil_box.addWidget(pupil_info)
                pupil_widget = QWidget()
                pupil_widget.setLayout(pupil_box)
                self.video_layout.addWidget(pupil_widget)
                source = Mp4FrameSource(
                    context.pupil_path, duration_hint=context.playback.duration
                )
                pupil_info.setText(
                    f"{context.pupil_path.name} • {source.fps:.2f} fps"
                    if source.fps
                    else context.pupil_path.name
                )
                self.sources.append((source, pupil_view, pupil_info))

            self.start_btn = QPushButton("Start")
            self.stop_btn = QPushButton("Stop")
            self.start_btn.clicked.connect(self.context.playback.start)
            self.stop_btn.clicked.connect(self.context.playback.stop)
            layout.addWidget(self.start_btn)
            layout.addWidget(self.stop_btn)

            speed_row = QHBoxLayout()
            speed_row.addWidget(QLabel("Playback speed"))
            self.speed_selector = QComboBox()
            for factor in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
                self.speed_selector.addItem(f"{factor}x", factor)
            try:
                idx = [self.speed_selector.itemData(i) for i in range(self.speed_selector.count())].index(
                    round(self.context.playback.speed, 2)
                )
                self.speed_selector.setCurrentIndex(idx)
            except ValueError:
                pass
            self.speed_selector.currentIndexChanged.connect(self._change_speed)
            speed_row.addWidget(self.speed_selector)
            layout.addLayout(speed_row)

            self.scrubber = QSlider(Qt.Orientation.Horizontal)
            self.scrubber.setMinimum(0)
            self.scrubber.setMaximum(1000)
            self.scrubber.sliderReleased.connect(self._scrub)
            layout.addWidget(self.scrubber)

            self.event_label = QLabel("Waiting for events…")
            layout.addWidget(self.event_label)

            self.speed_label = QLabel("Speed: --")
            layout.addWidget(self.speed_label)

            if pg is not None and context.treadmill_path:
                times, vals = load_treadmill_trace(context.treadmill_path)
                if times.size and vals.size:
                    self._tread_times = times
                    self._tread_vals = vals
                    self.trace_plot = pg.PlotWidget()
                    self.trace_plot.setBackground("#111")
                    self.trace_plot.showGrid(x=True, y=True, alpha=0.2)
                    self.trace_plot.setLabel("left", "Speed")
                    self.trace_plot.setLabel("bottom", "Time", units="s")
                    pen = pg.mkPen("#00e5ff", width=2)
                    self.trace_curve = self.trace_plot.plot(times, vals, pen=pen)
                    self.trace_live_curve = self.trace_plot.plot([], [], pen=pg.mkPen("#ff00aa", width=2))
                    self.trace_marker = self.trace_plot.addLine(x=0, pen=pg.mkPen("#ffcc00", width=2))
                    self.trace_plot.setMaximumHeight(220)
                    self.trace_plot.setSizePolicy(
                        QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Fixed,
                    )
                    layout.addWidget(self.trace_plot)

            self._update_media(0.0)

        def _scrub(self) -> None:
            fraction = self.scrubber.value() / 1000
            event = self.context.playback.scrub(fraction=fraction)
            self._update_label(event)
            self._update_media(event.elapsed)

        def _on_event(self, event: PlaybackEvent) -> None:
            self._update_label(event)
            fraction = 0.0
            if self.context.playback.duration:
                fraction = event.elapsed / self.context.playback.duration
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(int(fraction * 1000))
            self.scrubber.blockSignals(False)
            self._update_media(event.elapsed)
            self._update_speed(event)

        def _update_label(self, event: PlaybackEvent) -> None:
            self.event_label.setText(
                f"{event.device_id} @ {event.elapsed:.3f}s | payload: {event.payload}"
            )

        def _update_media(self, elapsed: float) -> None:
            self._last_elapsed = elapsed
            if not self.sources or not self.context.playback.duration:
                return

            fraction = max(0.0, min(1.0, elapsed / self.context.playback.duration))
            for source, label, info in self.sources:
                frame = source.frame_at_fraction(fraction)
                if frame is None:
                    continue
                pixmap = source.to_pixmap(frame, target_size=label.size())
                if pixmap is not None:
                    label.setPixmap(pixmap)
            if self._tread_times is not None and self._tread_vals is not None:
                self._update_treadmill_trace(elapsed)

        def _update_speed(self, event: PlaybackEvent) -> None:
            maybe_speed = _extract_speed(event.payload)
            if maybe_speed is not None:
                self._live_times.append(event.elapsed)
                self._live_vals.append(maybe_speed)
            if hasattr(self, "trace_live_curve"):
                if maybe_speed is not None:
                    self.trace_live_curve.setData(self._live_times, self._live_vals)
                    self.speed_label.setText(f"{event.device_id} speed: {maybe_speed:.2f}")
                    self.trace_marker.setValue(event.elapsed)
                elif self._tread_times is not None:
                    # Fall back to treadmill trace even if payload lacks speed
                    self._update_treadmill_trace(self._last_elapsed)

        def _change_speed(self, index: int) -> None:
            factor = self.speed_selector.itemData(index)
            if factor:
                self.context.playback.set_speed(float(factor))

        def _update_treadmill_trace(self, elapsed: float) -> None:
            """Animate the treadmill trace up to the current elapsed time."""

            if self._tread_times is None or self._tread_vals is None:
                return

            idx = int(np.searchsorted(self._tread_times, elapsed, side="right"))
            if idx <= 0:
                self.trace_live_curve.setData([], [])
                self.trace_marker.setValue(elapsed)
                return

            live_times = self._tread_times[:idx]
            live_vals = self._tread_vals[:idx]
            self.trace_live_curve.setData(live_times, live_vals)
            self.trace_marker.setValue(elapsed)
            current_speed = float(live_vals[-1])
            self.speed_label.setText(f"Treadmill speed: {current_speed:.2f}")

    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(context)
    window.show()
    app.exec()


def _extract_speed(payload) -> float | None:
    """Best-effort extraction of numeric speed from playback payloads."""

    try:
        if isinstance(payload, (int, float)):
            return float(payload)
        if isinstance(payload, dict):
            for key in ("speed", "value", "v", "velocity"):
                if key in payload:
                    return float(payload[key])
        if isinstance(payload, (list, tuple)) and payload:
            return float(payload[0])
    except Exception:
        return None
    return None

