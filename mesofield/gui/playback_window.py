"""Dedicated, seek-based playback window for recorded mesofield sessions.

The window is a read-only viewer built on the headless :mod:`mesofield.playback`
model. It is fully decoupled from live acquisition: there is no ``Procedure``,
no device threads and no clock — a timeline position drives "show the nearest
frame per stream", which makes play / pause / scrubbing trivial and lets each
stream fail independently.

Layout::

    [ Folder… | Subject ▾ | Session ▾ | Task ▾ ]      (selectors)
    [   camera grid: one pyqtgraph ImageView per stream   ]
    [   treadmill speed plot with a moving cursor (opt)    ]
    [ ▶/⏸ | speed | loop | <----- scrub ----->  mm:ss ]   (transport)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mesofield.gui import theme
from mesofield.playback import (
    CameraStream,
    PlaybackSession,
    SessionRef,
    discover_recordings,
    load_session,
)
from mesofield.utils._logger import get_logger

logger = get_logger("PlaybackWindow")

_TICK_MS = 33  # ~30 Hz UI refresh
_SCRUB_COALESCE_MS = 40  # cap scrub-triggered renders to ~25 Hz


def _orient_for_view(frame: Optional[np.ndarray], is_video: bool) -> Optional[np.ndarray]:
    """Orient a recorded frame for pyqtgraph's default (col-major) ImageView.

    Recorded arrays are ``[row(y), col(x)]``; pyqtgraph's default treats axis 0
    as x, so we transpose the spatial axes. Video frames arrive BGR (OpenCV) and
    are converted to RGB.
    """
    if frame is None:
        return None
    arr = frame
    if is_video and arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR -> RGB
    if arr.ndim == 2:
        arr = arr.T
    elif arr.ndim == 3:
        arr = np.transpose(arr, (1, 0, 2))
    return np.ascontiguousarray(arr)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


class _CameraTile(QWidget):
    """One camera panel: a caption + an image-only pyqtgraph ImageView."""

    def __init__(self, device_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.device_id = device_id
        self._first_shown = False
        self._last_idx = -1
        self._shown_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.caption = QLabel(device_id)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.caption)

        self.image = pg.ImageView()
        # Strip the ImageView chrome down to just the image.
        for widget in (self.image.ui.histogram, self.image.ui.roiBtn, self.image.ui.menuBtn):
            widget.hide()
        self.image.ui.roiPlot.hide()
        layout.addWidget(self.image, 1)

    def show_frame(self, frame: np.ndarray, is_video: bool, idx: int, total: int) -> None:
        if idx == self._last_idx:
            return
        disp = _orient_for_view(frame, is_video)
        if disp is None:
            return
        auto = self._shown_count == 1
        # Auto-range / level on the second displayed frame so the initial
        # frame paints quickly while still establishing stable view settings.
        self.image.setImage(
            disp, autoLevels=auto, autoRange=auto, autoHistogramRange=auto
        )
        self._first_shown = True
        self._shown_count += 1
        self._last_idx = idx
        self.caption.setText(f"{self.device_id}   [{idx + 1}/{total}]")

    def show_placeholder(self, reason: str) -> None:
        self.caption.setText(f"{self.device_id} — {reason}")
        self._shown_count = 0
        self._last_idx = -1
        self.image.clear()

    def bind(self, device_id: str) -> None:
        """Reuse this tile for a different stream (no widget churn)."""
        self.device_id = device_id
        self._first_shown = False
        self._shown_count = 0
        self._last_idx = -1
        self.caption.setText(device_id)
        self.image.clear()


class PlaybackWindow(QMainWindow):
    """Top-level playback viewer for an experiment directory."""

    def __init__(
        self,
        experiment_dir: Optional[str | Path] = None,
        *,
        speed: float = 1.0,
        loop: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mesofield Playback")

        self._index: Dict[str, Dict[str, Dict[str, SessionRef]]] = {}
        self._session: Optional[PlaybackSession] = None
        self._streams: Dict[str, CameraStream] = {}
        # A reusable pool of camera tiles — never destroyed between sessions, so
        # switching never tears down ImageView widgets (the source of the crash).
        self._tile_pool: List[_CameraTile] = []
        self._tile_by_id: Dict[str, _CameraTile] = {}
        self._pos_ms: float = 0.0
        self._duration_ms: int = 0
        self._last_tick: float = 0.0
        self._cursor: Optional[pg.InfiniteLine] = None
        self._experiment_dir: Optional[Path] = None
        self._is_scrubbing = False
        self._resume_after_scrub = False
        self._pending_scrub_ms: Optional[int] = None

        self._build_ui()

        self.spin_speed.setValue(float(speed))
        self.chk_loop.setChecked(bool(loop))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._flush_scrub)

        if experiment_dir:
            self.load_experiment(experiment_dir)

    # ---- UI construction ----------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- selector bar ---
        bar = QHBoxLayout()
        self.btn_folder = QPushButton("Folder…")
        self.btn_folder.setToolTip("Choose a different experiment directory")
        self.btn_folder.clicked.connect(self._choose_folder)
        self.lbl_dir = QLabel("")
        self.lbl_dir.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.cb_subject = QComboBox()
        self.cb_session = QComboBox()
        self.cb_task = QComboBox()
        self.cb_subject.currentTextChanged.connect(self._populate_sessions)
        self.cb_session.currentTextChanged.connect(self._populate_tasks)
        self.cb_task.currentTextChanged.connect(self._load_current)
        bar.addWidget(self.btn_folder)
        bar.addWidget(QLabel("Subject:"))
        bar.addWidget(self.cb_subject)
        bar.addWidget(QLabel("Session:"))
        bar.addWidget(self.cb_session)
        bar.addWidget(QLabel("Task:"))
        bar.addWidget(self.cb_task)
        bar.addStretch(1)
        bar.addWidget(self.lbl_dir, 2)
        root.addLayout(bar)

        # --- status (shown only when something is wrong) ---
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setVisible(False)
        root.addWidget(self.lbl_status)

        # --- camera grid ---
        self.grid_container = QWidget()
        self.grid = QGridLayout(self.grid_container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.grid_container, 1)

        # --- treadmill plot ---
        self.plot = pg.PlotWidget()
        self.plot.setMaximumHeight(170)
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setLabel("left", "speed", units="mm/s")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setVisible(False)
        root.addWidget(self.plot)

        # --- transport bar ---
        transport = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.btn_play.toggled.connect(self._on_play_toggled)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8.0)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSuffix("×")
        self.spin_speed.setToolTip("Playback speed")

        self.chk_loop = QCheckBox("Loop")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_scrub)
        self.slider.sliderPressed.connect(self._on_scrub_started)
        self.slider.sliderReleased.connect(self._on_scrub_finished)

        self.lbl_time = QLabel("00:00.000 / 00:00.000")

        transport.addWidget(self.btn_play)
        transport.addWidget(self.spin_speed)
        transport.addWidget(self.chk_loop)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.lbl_time)
        root.addLayout(transport)

    # ---- experiment / selection ---------------------------------------

    def load_experiment(self, experiment_dir: str | Path) -> None:
        """Discover recordings under ``experiment_dir`` and populate selectors."""
        self._experiment_dir = Path(experiment_dir)
        self.lbl_dir.setText(str(self._experiment_dir))
        try:
            self._index = discover_recordings(self._experiment_dir)
        except Exception as exc:
            logger.error("Discovery failed for %s: %s", experiment_dir, exc)
            self._index = {}

        if not self._index:
            self._set_status(f"No recordings found under {self._experiment_dir}")
            for cb in (self.cb_subject, self.cb_session, self.cb_task):
                cb.blockSignals(True)
                cb.clear()
                cb.blockSignals(False)
            self._teardown_session()
            return

        self._set_status("")
        self._populate_subjects()

    def _choose_folder(self) -> None:
        start = str(self._experiment_dir or "")
        directory = QFileDialog.getExistingDirectory(
            self, "Select experiment directory", start
        )
        if directory:
            self.load_experiment(directory)

    def _populate_subjects(self) -> None:
        self.cb_subject.blockSignals(True)
        self.cb_subject.clear()
        self.cb_subject.addItems(sorted(self._index))
        self.cb_subject.blockSignals(False)
        self._populate_sessions()

    def _populate_sessions(self) -> None:
        subject = self.cb_subject.currentText()
        sessions = sorted(self._index.get(subject, {}))
        self.cb_session.blockSignals(True)
        self.cb_session.clear()
        self.cb_session.addItems(sessions)
        self.cb_session.blockSignals(False)
        self._populate_tasks()

    def _populate_tasks(self) -> None:
        subject = self.cb_subject.currentText()
        session = self.cb_session.currentText()
        tasks = sorted(self._index.get(subject, {}).get(session, {}))
        self.cb_task.blockSignals(True)
        self.cb_task.clear()
        self.cb_task.addItems(tasks)
        self.cb_task.blockSignals(False)
        self._load_current()

    def _load_current(self) -> None:
        subject = self.cb_subject.currentText()
        session = self.cb_session.currentText()
        task = self.cb_task.currentText()
        try:
            ref = self._index[subject][session][task]
        except KeyError:
            return
        self._load_ref(ref)

    # ---- session loading ----------------------------------------------

    def _load_ref(self, ref: SessionRef) -> None:
        # Stop any running playback and release the previous session's files.
        self._stop_playback()
        self._teardown_session()

        try:
            session = load_session(ref)
        except Exception as exc:
            logger.error("Failed to load %s: %s", ref.label, exc)
            self._set_status(f"Failed to load {ref.label}: {exc}")
            return

        self._session = session
        self._streams = {c.device_id: c for c in session.cameras}
        self._assign_tiles(session.cameras)
        self._rebuild_treadmill(session)

        self._pos_ms = 0.0
        self._duration_ms = int(max(0.0, session.duration_s) * 1000)
        playable = any(c.error is None and c.n_frames > 0 for c in session.cameras)

        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self._duration_ms))
        self.slider.setValue(0)
        self.slider.setEnabled(playable and self._duration_ms > 0)
        self.slider.blockSignals(False)
        self.btn_play.setEnabled(playable and self._duration_ms > 0)

        if not playable:
            self._set_status(f"{ref.label}: no playable camera streams")
        else:
            self._set_status("")
        self._render(0.0)

    def _assign_tiles(self, cameras: List[CameraStream]) -> None:
        """Bind the reusable tile pool to ``cameras`` — never destroys widgets."""
        self._detach_tiles()
        self._tile_by_id = {}
        if not cameras:
            return
        # Grow the pool if this session has more cameras than any prior one.
        while len(self._tile_pool) < len(cameras):
            self._tile_pool.append(_CameraTile(""))
        cols = max(1, math.ceil(math.sqrt(len(cameras))))
        for i, cam in enumerate(cameras):
            tile = self._tile_pool[i]
            tile.bind(cam.device_id)
            if cam.error is not None:
                tile.show_placeholder(cam.error)
            self.grid.addWidget(tile, i // cols, i % cols)
            tile.show()
            self._tile_by_id[cam.device_id] = tile

    def _detach_tiles(self) -> None:
        """Remove pooled tiles from the grid and clear them, keeping them alive."""
        for tile in self._tile_pool:
            self.grid.removeWidget(tile)
            tile.image.clear()
            tile.hide()
        self._tile_by_id = {}

    def _rebuild_treadmill(self, session: PlaybackSession) -> None:
        self.plot.clear()
        self._cursor = None
        track = session.treadmill
        if track is None or track.time_s.size == 0:
            self.plot.setVisible(False)
            return
        self.plot.plot(track.time_s, track.speed, pen=pg.mkPen(theme.ACCENT, width=1))
        self._cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(theme.ACCENT_HI, width=1)
        )
        self._cursor.setPos(0.0)
        self.plot.addItem(self._cursor)
        # Scrub range is [0, duration]; samples before t=0 stay off-view.
        self.plot.setXRange(0.0, max(session.duration_s, 1e-6))
        self.plot.setVisible(True)

    def _teardown_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._streams = {}
        self._detach_tiles()
        self.plot.clear()
        self.plot.setVisible(False)
        self._cursor = None

    # ---- transport -----------------------------------------------------

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and self._session is not None and self._duration_ms > 0:
            # Restart from the beginning if we're parked at the end.
            if self._pos_ms >= self._duration_ms:
                self._pos_ms = 0.0
            self._last_tick = time.monotonic()
            self._timer.start(_TICK_MS)
            self.btn_play.setText("⏸ Pause")
        else:
            self._timer.stop()
            self.btn_play.setText("▶ Play")

    def _on_tick(self) -> None:
        if self._session is None or self._duration_ms <= 0:
            return
        # Advance by *real* elapsed time so a slow frame read drops frames
        # instead of backing up the event loop (keeps high-speed playback from
        # accumulating jitter / drift).
        now = time.monotonic()
        self._pos_ms += (now - self._last_tick) * 1000.0 * self.spin_speed.value()
        self._last_tick = now
        if self._pos_ms >= self._duration_ms:
            if self.chk_loop.isChecked():
                self._pos_ms = 0.0
            else:
                self._pos_ms = float(self._duration_ms)
                self.btn_play.setChecked(False)  # also stops the timer
        self.slider.blockSignals(True)
        self.slider.setValue(int(self._pos_ms))
        self.slider.blockSignals(False)
        self._render(self._pos_ms / 1000.0)

    def _on_scrub(self, value: int) -> None:
        # Programmatic updates block signals; user scrubbing can emit a high
        # event rate, so coalesce and render at a bounded cadence.
        self._pending_scrub_ms = int(value)
        if not self._is_scrubbing:
            self._flush_scrub()
            return
        if not self._scrub_timer.isActive():
            self._scrub_timer.start(_SCRUB_COALESCE_MS)

    def _on_scrub_started(self) -> None:
        self._is_scrubbing = True
        self._resume_after_scrub = self.btn_play.isChecked()
        if self._resume_after_scrub:
            self.btn_play.setChecked(False)

    def _on_scrub_finished(self) -> None:
        self._is_scrubbing = False
        if self._scrub_timer.isActive():
            self._scrub_timer.stop()
        self._flush_scrub()
        if self._resume_after_scrub and self._duration_ms > 0:
            self.btn_play.setChecked(True)
        self._resume_after_scrub = False

    def _flush_scrub(self) -> None:
        if self._pending_scrub_ms is None:
            return
        self._pos_ms = float(self._pending_scrub_ms)
        self._pending_scrub_ms = None
        self._render(self._pos_ms / 1000.0)

    # ---- rendering -----------------------------------------------------

    def _render(self, t_s: float) -> None:
        if self._session is None:
            return
        for result in self._session.seek(t_s):
            tile = self._tile_by_id.get(result.device_id)
            stream = self._streams.get(result.device_id)
            if tile is None or stream is None:
                continue
            if result.frame is not None:
                tile.show_frame(result.frame, stream.is_video, result.index, stream.n_frames)
        self.lbl_time.setText(
            f"{_fmt_time(t_s)} / {_fmt_time(self._duration_ms / 1000.0)}"
        )
        if self._cursor is not None:
            self._cursor.setPos(t_s)

    # ---- misc ----------------------------------------------------------

    def _stop_playback(self) -> None:
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)
        self._timer.stop()
        if self._scrub_timer.isActive():
            self._scrub_timer.stop()
        self.btn_play.setText("▶ Play")

    def _set_status(self, message: str) -> None:
        self.lbl_status.setText(message)
        self.lbl_status.setVisible(bool(message))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        self._scrub_timer.stop()
        self._teardown_session()
        super().closeEvent(event)
