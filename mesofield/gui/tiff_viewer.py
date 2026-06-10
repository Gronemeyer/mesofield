import os
import json
from typing import List, Tuple, Optional

import numpy as np
import tifffile
import pyqtgraph as pg

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QSlider,
    QProgressBar,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QMessageBox,
)

from mesofield.utils._logger import get_logger
logger = get_logger("TiffViewer")

# Optimize pyqtgraph settings
pg.setConfigOptions(useOpenGL=True, antialias=False)

# Shared thread pool for background tasks
thread_pool = QThreadPool()
thread_pool.setMaxThreadCount(4)


class ROIWorkerSignals(QObject):
    """
    Signals for ROIWorker.
    finished: emits (index, time_series)
    progress: emits (index, percent_complete)
    """
    finished = pyqtSignal(int, np.ndarray)
    progress = pyqtSignal(int, int)


class ROIWorker(QRunnable):
    """
    Worker that computes the mean intensity over time for a given ROI.
    Automatically clips the ROI mask to image bounds.
    """
    def __init__(
        self,
        index: int,
        filepath: str,
        dtype_str: str,
        shape: Tuple[int, int, int],
        x0: int,
        y0: int,
        mask: np.ndarray,
        chunk: int = 1000,
        frame_start: int = 1,
        frame_end: Optional[int] = None,
    ):
        super().__init__()
        self.index = index
        self.filepath = filepath
        self.dtype_str = dtype_str
        self.shape = shape  # (frames, height, width)
        self.x0 = x0
        self.y0 = y0
        self.mask = mask.astype(bool)
        self.chunk = chunk
        # Inclusive start, exclusive end, both 0-indexed.
        self.frame_start = max(0, int(frame_start))
        self.frame_end = int(frame_end) if frame_end is not None else int(shape[0])
        self.frame_end = max(self.frame_start + 1, min(self.frame_end, int(shape[0])))
        self.signals = ROIWorkerSignals()

    def run(self) -> None:
        # Memory-map the TIFF for fast access
        mmap = np.memmap(
            self.filepath,
            mode='r',
            dtype=np.dtype(self.dtype_str),
            shape=self.shape
        )
        total_frames = self.frame_end - self.frame_start
        img_h, img_w = self.shape[1], self.shape[2]

        # Compute clipped ROI bounds
        h_mask, w_mask = self.mask.shape
        y1 = min(self.y0 + h_mask, img_h)
        x1 = min(self.x0 + w_mask, img_w)
        mask_clipped = self.mask[:(y1 - self.y0), :(x1 - self.x0)]
        mask_sum = mask_clipped.sum()

        # Prepare result array
        result = np.empty(total_frames, dtype=float)

        # Process in chunks to update progress
        for start in range(self.frame_start, self.frame_end, self.chunk):
            end = min(self.frame_end, start + self.chunk)
            block = mmap[
                start:end,
                self.y0:y1,
                self.x0:x1
            ]
            # Compute sums within clipped mask
            sums = (block * mask_clipped).sum(axis=(1, 2))
            idx0 = start - self.frame_start
            length = end - start
            result[idx0:idx0 + length] = sums / mask_sum

            percent = int((idx0 + length) * 100 / total_frames)
            self.signals.progress.emit(self.index, percent)

        self.signals.finished.emit(self.index, result)

class AlignmentWorkerSignals(QObject):
    """Signals for AlignmentWorker: result(lags, correlation_values)"""
    result = pyqtSignal(np.ndarray, np.ndarray)


class AlignmentWorker(QRunnable):
    """
    Worker that computes normalized cross-correlation between two time series.
    """
    def __init__(self, series_list: List[np.ndarray]) -> None:
        super().__init__()
        self.series_list = series_list
        self.signals = AlignmentWorkerSignals()

    def run(self) -> None:
        s0, s1 = self.series_list[0], self.series_list[1]
        a = (s0 - s0.mean()) / s0.std()
        b = (s1 - s1.mean()) / s1.std()
        corr = np.correlate(a, b, mode='full')
        lags = np.arange(-len(a) + 1, len(a))
        self.signals.result.emit(lags, corr)


class EnhancedROIWorkerSignals(QObject):
    """
    Enhanced signals for ROIWorker with ΔF/F calculations.
    finished: emits (index, raw_series, df_f_series)
    progress: emits (index, percent_complete)
    """
    finished = pyqtSignal(int, np.ndarray, np.ndarray)
    progress = pyqtSignal(int, int)


class EnhancedROIWorker(QRunnable):
    """
    Enhanced worker that computes ROI analysis with ΔF/F calculations.
    """
    def __init__(
        self,
        index: int,
        filepath: str,
        dtype_str: str,
        shape: Tuple[int, int, int],
        x0: int,
        y0: int,
        mask: np.ndarray,
        baseline_frames: int = 100,
        chunk: int = 1000,
        frame_start: int = 1,
        frame_end: Optional[int] = None,
    ):
        super().__init__()
        self.index = index
        self.filepath = filepath
        self.dtype_str = dtype_str
        self.shape = shape  # (frames, height, width)
        self.x0 = x0
        self.y0 = y0
        self.mask = mask.astype(bool)
        self.baseline_frames = baseline_frames
        self.chunk = chunk
        self.frame_start = max(0, int(frame_start))
        self.frame_end = int(frame_end) if frame_end is not None else int(shape[0])
        self.frame_end = max(self.frame_start + 1, min(self.frame_end, int(shape[0])))
        self.signals = EnhancedROIWorkerSignals()

    def run(self) -> None:
        # Memory-map the TIFF for fast access
        mmap = np.memmap(
            self.filepath,
            mode='r',
            dtype=np.dtype(self.dtype_str),
            shape=self.shape
        )
        total_frames = self.frame_end - self.frame_start
        img_h, img_w = self.shape[1], self.shape[2]

        # Compute clipped ROI bounds
        h_mask, w_mask = self.mask.shape
        y1 = min(self.y0 + h_mask, img_h)
        x1 = min(self.x0 + w_mask, img_w)
        mask_clipped = self.mask[:(y1 - self.y0), :(x1 - self.x0)]
        mask_sum = mask_clipped.sum()
        
        if mask_sum == 0:
            # Empty mask, return zeros
            zeros = np.zeros(total_frames, dtype=float)
            self.signals.finished.emit(self.index, zeros, zeros)
            return

        # Prepare result array
        raw_series = np.empty(total_frames, dtype=float)

        # Process in chunks to update progress
        for start in range(self.frame_start, self.frame_end, self.chunk):
            end = min(self.frame_end, start + self.chunk)

            # Load data block
            data_block = mmap[start:end, self.y0:y1, self.x0:x1].astype(np.float32)

            # Compute mean for each frame
            for i in range(len(data_block)):
                frame = data_block[i]
                frame_pixels = frame[mask_clipped]
                raw_series[start - self.frame_start + i] = frame_pixels.mean()

            percent = int((start - self.frame_start + len(data_block)) * 100 / total_frames)
            self.signals.progress.emit(self.index, percent)

        # Compute baseline (median of first N frames)
        baseline_end = min(self.baseline_frames, len(raw_series))
        baseline = np.median(raw_series[:baseline_end])
        
        # Compute ΔF/F
        df_f_series = (raw_series - baseline) / baseline

        self.signals.finished.emit(self.index, raw_series, df_f_series)


class PixelTraceWorkerSignals(QObject):
    finished = pyqtSignal(int, int, int, np.ndarray)  # x, y, bin, series


class PixelTraceWorker(QRunnable):
    """Compute the mean time-series over an NxN window around (x, y).

    Runs off the GUI thread; uses a fresh np.memmap so it does not block
    or contend with the viewer's display memmap.
    """
    def __init__(self, filepath: str, dtype_str: str, shape: Tuple[int, int, int],
                 x: int, y: int, bin_size: int,
                 frame_start: int = 0, frame_end: Optional[int] = None):
        super().__init__()
        self.filepath = filepath
        self.dtype_str = dtype_str
        self.shape = shape
        self.x = int(x)
        self.y = int(y)
        self.bin_size = max(1, int(bin_size))
        self.frame_start = max(0, int(frame_start))
        self.frame_end = int(frame_end) if frame_end is not None else int(shape[0])
        self.frame_end = max(self.frame_start + 1, min(self.frame_end, int(shape[0])))
        self.signals = PixelTraceWorkerSignals()

    def run(self) -> None:
        mmap = np.memmap(self.filepath, mode='r',
                         dtype=np.dtype(self.dtype_str), shape=self.shape)
        n_frames, h, w = self.shape
        half = self.bin_size // 2
        y0 = max(0, self.y - half)
        x0 = max(0, self.x - half)
        y1 = min(h, y0 + self.bin_size)
        x1 = min(w, x0 + self.bin_size)
        s, e = self.frame_start, self.frame_end
        if self.bin_size == 1:
            series = np.asarray(mmap[s:e, self.y, self.x], dtype=float)
        else:
            block = mmap[s:e, y0:y1, x0:x1]
            series = block.mean(axis=(1, 2)).astype(float)
        self.signals.finished.emit(self.x, self.y, self.bin_size, series)


class PixelTraceWindow(QWidget):
    """Pop-up window showing a single pixel/bin time-series. Closable."""
    closed = pyqtSignal(object)  # emits self on close

    def __init__(self, x: int, y: int, bin_size: int, series: np.ndarray,
                 color: str = 'y', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Pixel ({x}, {y})  {bin_size}x{bin_size}")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(640, 320)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel(
            f"<b>Pixel</b> ({x}, {y}) &nbsp; <b>Bin</b> {bin_size}x{bin_size} "
            f"&nbsp; <b>Frames</b> {len(series)}"
        ))
        header.addStretch(1)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        header.addWidget(self.btn_close)
        layout.addLayout(header)

        self.plot = pg.PlotWidget()
        self.plot.plot(np.arange(len(series)), series, pen=color)
        self.plot.setLabel('bottom', 'Frame')
        self.plot.setLabel('left', 'Intensity')
        layout.addWidget(self.plot)

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)


class TiffViewer(QWidget):
    """
    Main application widget for interactive TIFF viewing and ROI analysis.
    """
    def __init__(self, initial_dir: Optional[str] = None, procedure=None):
        super().__init__()
        self.filepath: Optional[str] = None
        self.mmap: Optional[np.memmap] = None
        self.rois: List[Tuple[str, pg.ROI]] = []
        self.results: dict = {}
        self.df_f_results: dict = {}
        self.colors = ['r', 'g', 'b', 'c', 'm']
        # Optional reference to the running Procedure so we can refuse to open
        # files that are part of an in-progress recording.
        self._procedure = procedure
        self._initial_dir: Optional[str] = initial_dir
        # Pixel-trace state
        self._pixel_windows: List[PixelTraceWindow] = []
        self._pixel_marker = None  # transient pg.RectROI showing last pick

        self._setup_ui()
        self._connect_signals()
        self.setWindowTitle("TIFF ROI Viewer with ΔF/F Analysis")

    def _setup_ui(self) -> None:
        """Initialize and arrange all UI components."""
        main_layout = QVBoxLayout(self)

        # --- Control panel ---
        ctrl_panel = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_panel)
        
        # File controls
        self.btn_open = QPushButton("Open TIFF…")
        ctrl_layout.addWidget(self.btn_open)
        
        # ROI controls
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Rect", "Ellipse", "Polygon"])
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_clear_roi = QPushButton("Clear ROIs")
        self.btn_save_roi = QPushButton("Save ROIs…")
        self.btn_load_roi = QPushButton("Load ROIs…")
        
        for widget in [self.combo_shape, self.btn_add_roi, self.btn_clear_roi, 
                      self.btn_save_roi, self.btn_load_roi]:
            ctrl_layout.addWidget(widget)
        
        # Analysis controls
        self.btn_compute = QPushButton("Compute ROIs")
        self.btn_export_svg = QPushButton("Export SVG…")
        self.btn_export_svg.setEnabled(False)  # Disabled until traces are computed
        self.chk_df_f = QCheckBox("ΔF/F Analysis")
        self.chk_df_f.setChecked(True)
        self.chk_corr = QCheckBox("Compute Correlation")
        self.chk_corr.setChecked(True)
        self.lbl_align = QLabel("")

        # Pixel-trace controls
        self.btn_pick_pixel = QPushButton("Pick Pixel")
        self.btn_pick_pixel.setCheckable(True)
        self.btn_pick_pixel.setToolTip(
            "Click a pixel on the image to plot its time-series.\n"
            "Use the bin selector to average an NxN window around the click."
        )
        self.combo_bin = QComboBox()
        for n in (1, 2, 3, 5, 7, 9):
            self.combo_bin.addItem(f"{n}x{n}", n)
        self.combo_bin.setCurrentIndex(2)  # default 3x3

        for widget in [self.btn_compute, self.btn_export_svg, self.chk_df_f, self.chk_corr,
                       self.btn_pick_pixel, QLabel("Bin:"), self.combo_bin, self.lbl_align]:
            ctrl_layout.addWidget(widget)
        
        main_layout.addWidget(ctrl_panel)

        # --- Analysis parameters ---
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout(params_group)
        
        # Baseline frames
        params_layout.addWidget(QLabel("Baseline Frames:"), 0, 0)
        self.spin_baseline = QSpinBox()
        self.spin_baseline.setRange(10, 1000)
        self.spin_baseline.setValue(100)
        params_layout.addWidget(self.spin_baseline, 0, 1)

        # Frame range (1-indexed, inclusive on both ends) — applied to all
        # ROI / ΔF/F / pixel-trace computations.
        params_layout.addWidget(QLabel("First Frame:"), 0, 2)
        self.spin_first_frame = QSpinBox()
        self.spin_first_frame.setRange(1, 1)
        self.spin_first_frame.setValue(1)
        params_layout.addWidget(self.spin_first_frame, 0, 3)

        params_layout.addWidget(QLabel("Last Frame:"), 0, 4)
        self.spin_last_frame = QSpinBox()
        self.spin_last_frame.setRange(1, 1)
        self.spin_last_frame.setValue(1)
        params_layout.addWidget(self.spin_last_frame, 0, 5)

        main_layout.addWidget(params_group)

        # --- Image display ---
        self.img_view = pg.ImageView()
        self.view_box = self.img_view.getView()
        self.img_item = self.img_view.getImageItem()
        main_layout.addWidget(self.img_view)

        # --- Slider and progress bar ---
        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.1, 240.0)
        self.spin_fps.setDecimals(1)
        self.spin_fps.setValue(30.0)
        self.spin_fps.setSuffix(" fps")
        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(self.spin_fps)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        playback_layout.addWidget(self.slider, 1)
        main_layout.addLayout(playback_layout)

        # Playback timer
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # --- Time-series plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setVisible(False)
        main_layout.addWidget(self.plot_widget)

        # --- Correlation plot ---
        self.corr_widget = pg.PlotWidget(title="Cross-Correlation")
        self.corr_widget.setVisible(False)
        main_layout.addWidget(self.corr_widget)

    def _connect_signals(self) -> None:
        """Wire UI events to their handlers."""
        self.btn_open.clicked.connect(self.open_file)
        self.slider.valueChanged.connect(self.display_frame)
        self.btn_add_roi.clicked.connect(self.add_roi)
        self.btn_clear_roi.clicked.connect(self.clear_rois)
        self.btn_save_roi.clicked.connect(self.save_rois)
        self.btn_load_roi.clicked.connect(self.load_rois)
        self.btn_compute.clicked.connect(self.compute_rois)
        self.btn_export_svg.clicked.connect(self.export_svg)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        self.btn_pick_pixel.toggled.connect(self._on_pick_pixel_toggled)
        # Frame-range interlock so first <= last.
        self.spin_first_frame.valueChanged.connect(self._on_first_frame_changed)
        self.spin_last_frame.valueChanged.connect(self._on_last_frame_changed)
        # Mouse click in the image scene -> pixel trace (when picking enabled).
        # pyqtgraph's GraphicsScene exposes sigMouseClicked at runtime even
        # though Qt's QGraphicsScene type stubs do not advertise it.
        self.view_box.scene().sigMouseClicked.connect(self._on_image_clicked)  # type: ignore[attr-defined]

    def _is_recording_active(self) -> bool:
        """Return True if any camera attached to the procedure is running."""
        proc = self._procedure
        if proc is None:
            return False
        try:
            cams = proc.config.hardware.cameras
        except Exception:
            return False
        for cam in cams or ():
            if getattr(cam, "is_active", False):
                return True
        return False

    def _recording_dir(self) -> Optional[str]:
        """Return the active experiment's output directory, if any."""
        proc = self._procedure
        if proc is None:
            return None
        cfg = getattr(proc, "config", None)
        if cfg is None:
            return None
        for attr in ("bids_dir", "save_dir", "data_dir"):
            d = getattr(cfg, attr, None)
            if d:
                try:
                    return os.path.abspath(d)
                except Exception:
                    return None
        return None

    def _is_inside_recording_dir(self, path: str) -> bool:
        rec_dir = self._recording_dir()
        if not rec_dir:
            return False
        try:
            abs_path = os.path.abspath(path)
            return os.path.normcase(abs_path).startswith(
                os.path.normcase(rec_dir + os.sep)
            )
        except Exception:
            return False

    def open_file(self) -> None:
        """Open a TIFF file and memory-map it for fast access."""
        start_dir = self._initial_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF", start_dir, "*.tif *.tiff"
        )
        if not path:
            return

        # Refuse to open files that belong to an in-progress recording.
        # The active experiment's writer may still hold the file open and
        # mapping/reading it concurrently risks corruption or a crash.
        if self._is_recording_active() and self._is_inside_recording_dir(path):
            QMessageBox.warning(
                self,
                "Recording in progress",
                "This file is inside the active experiment's output directory "
                "while a recording is running.\n\n"
                "Wait until acquisition finishes before opening it.",
            )
            return

        # Clear stale ROIs/results and release the previous memmap before
        # loading a new stack. ROIs reference the old image item and the
        # previous numpy.memmap holds a file handle that must be released
        # (especially on Windows) to avoid a crash on reopen.
        self.clear_rois()
        if self.mmap is not None:
            try:
                del self.mmap
            except Exception:
                pass
            self.mmap = None

        self.filepath = path
        # Read-only memmap so we never collide with an experiment writer.
        self.mmap = tifffile.memmap(path, mode='r')
        self._initial_dir = os.path.dirname(path)
        total_frames = self.mmap.shape[0]
        # Reset so the first frame of the new stack auto-ranges/levels once.
        self._first_shown = False
        # Block slider signals while reconfiguring its range to avoid
        # display_frame firing with a transient/out-of-range index.
        self.slider.blockSignals(True)
        self.slider.setRange(1, total_frames - 1)
        self.slider.setValue(1)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)
        self.btn_play.setEnabled(True)
        # Configure frame-range spinboxes for the new stack.
        # Default: skip the first frame (matches previous behavior) and
        # include everything through the last.
        self.spin_first_frame.blockSignals(True)
        self.spin_last_frame.blockSignals(True)
        self.spin_first_frame.setRange(1, total_frames)
        self.spin_last_frame.setRange(1, total_frames)
        self.spin_first_frame.setValue(min(2, total_frames))
        self.spin_last_frame.setValue(total_frames)
        self.spin_first_frame.blockSignals(False)
        self.spin_last_frame.blockSignals(False)
        self.display_frame(1)

    def display_frame(self, index: int) -> None:
        """Display a single frame from the TIFF stack."""
        if self.mmap is None:
            return
        image = np.asarray(self.mmap[index])
        first = (index == 1) and not getattr(self, "_first_shown", False)
        # Preserve the user's current zoom/pan and contrast on subsequent
        # frames; only auto-range/level on the very first frame of a stack.
        self.img_view.setImage(
            image,
            autoLevels=first,
            autoRange=first,
            autoHistogramRange=first,
        )
        if first:
            self._first_shown = True

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and self.mmap is not None:
            self._play_timer.start(max(1, int(1000.0 / self.spin_fps.value())))
            self.btn_play.setText("Pause")
        else:
            self._play_timer.stop()
            self.btn_play.setText("Play")

    def _on_fps_changed(self, fps: float) -> None:
        if self._play_timer.isActive():
            self._play_timer.start(max(1, int(1000.0 / fps)))

    def _advance_frame(self) -> None:
        if self.mmap is None:
            return
        nxt = self.slider.value() + 1
        if nxt > self.slider.maximum():
            nxt = self.slider.minimum()
        self.slider.setValue(nxt)

    # ---- Pixel-trace picking ----------------------------------------------
    def _on_pick_pixel_toggled(self, on: bool) -> None:
        """Switch the image cursor to indicate pick mode."""
        if on:
            self.img_view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.img_view.unsetCursor()

    def _on_first_frame_changed(self, val: int) -> None:
        if val > self.spin_last_frame.value():
            self.spin_last_frame.setValue(val)

    def _on_last_frame_changed(self, val: int) -> None:
        if val < self.spin_first_frame.value():
            self.spin_first_frame.setValue(val)

    def _frame_range(self) -> Tuple[int, int]:
        """Return (frame_start, frame_end) as 0-indexed half-open range."""
        s = max(0, self.spin_first_frame.value() - 1)
        e = self.spin_last_frame.value()  # 1-indexed inclusive == 0-indexed exclusive
        if self.mmap is not None:
            n = int(self.mmap.shape[0])
            e = min(e, n)
            s = min(s, max(0, e - 1))
        return s, e

    def _on_image_clicked(self, ev) -> None:
        """Handle a click on the image while pixel-pick mode is active."""
        if not self.btn_pick_pixel.isChecked():
            return
        if self.mmap is None or self.filepath is None:
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        # Map scene click -> image (pixel) coords
        try:
            scene_pos = ev.scenePos()
            img_pt = self.img_item.mapFromScene(scene_pos)
        except Exception:
            return
        x = int(img_pt.x())
        y = int(img_pt.y())
        h, w = self.mmap.shape[1], self.mmap.shape[2]
        if not (0 <= x < w and 0 <= y < h):
            return
        ev.accept()

        bin_size = int(self.combo_bin.currentData() or 1)

        # Draw a marker showing the picked region on the image.
        if self._pixel_marker is not None:
            try:
                self.view_box.removeItem(self._pixel_marker)
            except Exception:
                pass
            self._pixel_marker = None
        half = bin_size // 2
        rx = max(0, x - half)
        ry = max(0, y - half)
        marker = pg.RectROI([rx, ry], [bin_size, bin_size],
                            pen=pg.mkPen('y', width=1), movable=False, resizable=False)
        # Disable the default handle so it's just a visual marker.
        for handle in list(marker.getHandles()):
            marker.removeHandle(handle)
        self.view_box.addItem(marker)
        self._pixel_marker = marker

        # Compute trace off the GUI thread.
        fs, fe = self._frame_range()
        worker = PixelTraceWorker(
            self.filepath, self.mmap.dtype.str, self.mmap.shape,
            x, y, bin_size,
            frame_start=fs, frame_end=fe,
        )
        worker.signals.finished.connect(self._on_pixel_trace_ready)
        thread_pool.start(worker)

    def _on_pixel_trace_ready(self, x: int, y: int, bin_size: int,
                              series: np.ndarray) -> None:
        """Show the pixel trace in a closable popup window."""
        color = self.colors[len(self._pixel_windows) % len(self.colors)]
        win = PixelTraceWindow(x, y, bin_size, series, color=color, parent=self)
        win.closed.connect(self._on_pixel_window_closed)
        win.show()
        self._pixel_windows.append(win)

    def _on_pixel_window_closed(self, win) -> None:
        if win in self._pixel_windows:
            self._pixel_windows.remove(win)
    # -----------------------------------------------------------------------

    def add_roi(self) -> None:
        """Add a new ROI of the selected shape to the image view."""
        shape_type = self.combo_shape.currentText()
        idx = len(self.rois)
        color = self.colors[idx % len(self.colors)]

        if shape_type == "Rect":
            roi = pg.RectROI([20, 20], [100, 100], pen=color)
        elif shape_type == "Ellipse":
            roi = pg.EllipseROI([20, 20], [100, 100], pen=color)
        else:
            pts = [[20, 20], [120, 20], [120, 120], [20, 120]]
            roi = pg.PolyLineROI(pts, closed=True, pen=color)

        self.view_box.addItem(roi)
        self.rois.append((shape_type, roi))

    def clear_rois(self) -> None:
        """Remove all ROIs and reset plots/labels."""
        for _, roi in self.rois:
            self.view_box.removeItem(roi)
        self.rois.clear()
        # Also remove the transient pixel-pick marker, if any.
        if self._pixel_marker is not None:
            try:
                self.view_box.removeItem(self._pixel_marker)
            except Exception:
                pass
            self._pixel_marker = None
        self.plot_widget.clear()
        self.plot_widget.setVisible(False)
        self.corr_widget.clear()
        self.corr_widget.setVisible(False)
        self.results.clear()
        self.df_f_results.clear()
        self.lbl_align.clear()
        self.progress.setVisible(False)
        self.btn_export_svg.setEnabled(False)

    def save_rois(self) -> None:
        """Export ROI definitions to a JSON file."""
        if not self.rois:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs", "rois.json", "JSON Files (*.json)"
        )
        if not path:
            return

        export_data = []
        for shape_type, roi in self.rois:
            info: dict = {'type': shape_type}
            if shape_type in ("Rect", "Ellipse"):
                pos = roi.pos()
                size = roi.size()
                info['pos'] = [float(pos.x()), float(pos.y())]
                info['size'] = [float(size.x()), float(size.y())]
            else:
                info['points'] = roi.getState()['points']
            export_data.append(info)

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Saved {len(export_data)} ROIs to {path}")

    def load_rois(self) -> None:
        """Import ROI definitions from a JSON file and render them."""
        if self.mmap is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self.clear_rois()

        with open(path) as f:
            roi_list = json.load(f)

        for entry in roi_list:
            shape_type = entry['type']
            idx = len(self.rois)
            color = self.colors[idx % len(self.colors)]

            if shape_type in ("Rect", "Ellipse"):
                x, y = entry['pos']
                w, h = entry['size']
                if shape_type == "Rect":
                    roi = pg.RectROI([x, y], [w, h], pen=color)
                else:
                    roi = pg.EllipseROI([x, y], [w, h], pen=color)
            else:
                pts = entry['points']
                roi = pg.PolyLineROI(pts, closed=True, pen=color)

            self.view_box.addItem(roi)
            self.rois.append((shape_type, roi))

        logger.info(f"Loaded {len(self.rois)} ROIs from {path}")

    def export_svg(self) -> None:
        """Export individual ROI traces as minimalistic SVG plots for use in Illustrator."""
        if not self.results and not self.df_f_results:
            logger.warning("No ROI traces to export. Compute ROIs first.")
            return
            
        # Let user choose a directory instead of a single file
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for SVG Export"
        )
        if not directory:
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import os
            
            # Set up matplotlib for clean, minimalistic SVG export
            matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep fonts as text for editing
            matplotlib.rcParams['font.family'] = 'Arial'
            matplotlib.rcParams['font.size'] = 10
            matplotlib.rcParams['axes.linewidth'] = 1.0
            matplotlib.rcParams['lines.linewidth'] = 1.5
            matplotlib.rcParams['axes.spines.top'] = False
            matplotlib.rcParams['axes.spines.right'] = False
            matplotlib.rcParams['xtick.direction'] = 'out'
            matplotlib.rcParams['ytick.direction'] = 'out'
            
            # Determine which data to plot
            use_df_f = self.chk_df_f.isChecked() and self.df_f_results
            data_dict = self.df_f_results if use_df_f else self.results
            y_label = "ΔF/F" if use_df_f else "Intensity"
            
            # Color mapping to match pyqtgraph colors
            color_map = {
                'r': '#E74C3C',  # Softer red
                'g': '#27AE60',  # Softer green
                'b': '#3498DB',  # Softer blue
                'c': '#17A2B8',  # Softer cyan
                'm': '#8E44AD'   # Softer magenta
            }
            
            exported_files = []
            
            # Create individual plots for each ROI
            for idx in sorted(data_dict.keys()):
                series = data_dict[idx]
                x_data = np.arange(len(series))
                color_key = self.colors[idx % len(self.colors)]
                color_hex = color_map.get(color_key, '#2C3E50')
                
                # Create minimalistic figure
                fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
                
                # Plot the trace with clean styling
                ax.plot(x_data, series, 
                       color=color_hex, 
                       linewidth=1.5,
                       alpha=0.8)
                
                # Minimalistic styling
                ax.set_xlabel("Frame", fontsize=10, color='#2C3E50')
                ax.set_ylabel(y_label, fontsize=10, color='#2C3E50')
                
                # Remove top and right spines (already set in rcParams but ensuring)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#7F8C8D')
                ax.spines['bottom'].set_color('#7F8C8D')
                ax.spines['left'].set_linewidth(0.8)
                ax.spines['bottom'].set_linewidth(0.8)
                
                # Clean tick styling
                ax.tick_params(colors='#7F8C8D', labelsize=9, width=0.8, length=4)
                ax.tick_params(axis='both', which='minor', length=2, width=0.5)
                
                # Minimal grid (very subtle)
                ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3, color='#BDC3C7')
                
                # Tight layout with minimal margins
                plt.tight_layout(pad=0.3)
                
                # Save individual SVG
                filename = f"ROI_{idx+1:02d}_trace.svg"
                filepath = os.path.join(directory, filename)
                plt.savefig(filepath, format='svg', bbox_inches='tight', 
                           facecolor='white', edgecolor='none', 
                           pad_inches=0.05)
                plt.close()
                
                exported_files.append(filename)
            
            logger.info(f"Exported {len(exported_files)} individual ROI traces to {directory}")
            logger.info(f"Files: {', '.join(exported_files)}")
            
        except ImportError:
            logger.error("matplotlib is required for SVG export. Please install: pip install matplotlib")
        except Exception as e:
            logger.error(f"Failed to export SVG: {e}")

    def compute_rois(self) -> None:
        """Run ROI mean-series calculations with ΔF/F and optional correlation."""
        if self.mmap is None or not self.rois or self.filepath is None:
            return
            
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.plot_widget.clear()
        self.plot_widget.setVisible(True)
        self.results.clear()
        self.df_f_results.clear()

        baseline_frames = self.spin_baseline.value()
        fs, fe = self._frame_range()

        for idx, (_, roi) in enumerate(self.rois):
            h, w = self.mmap.shape[1], self.mmap.shape[2]
            ones = np.ones((h, w), dtype=np.uint8)
            mask_array = roi.getArrayRegion(
                ones, self.img_view.getImageItem(), returnMappedCoords=False
            )
            # Convert to boolean mask
            if isinstance(mask_array, tuple):
                mask = np.asarray(mask_array[0]).astype(bool)
            else:
                mask = np.asarray(mask_array).astype(bool)
                
            x0 = int(roi.pos().x())
            y0 = int(roi.pos().y())

            if self.chk_df_f.isChecked():
                worker = EnhancedROIWorker(
                    idx, self.filepath, self.mmap.dtype.str,
                    self.mmap.shape, x0, y0, mask,
                    baseline_frames=baseline_frames,
                    frame_start=fs, frame_end=fe,
                )
                worker.signals.progress.connect(
                    lambda _, pct: self.progress.setValue(pct)
                )
                worker.signals.finished.connect(self._on_enhanced_roi_finished)
            else:
                worker = ROIWorker(
                    idx, self.filepath, self.mmap.dtype.str,
                    self.mmap.shape, x0, y0, mask,
                    frame_start=fs, frame_end=fe,
                )
                worker.signals.progress.connect(
                    lambda _, pct: self.progress.setValue(pct)
                )
                worker.signals.finished.connect(self._on_roi_finished)
            
            thread_pool.start(worker)

    def _on_enhanced_roi_finished(self, idx: int, raw_series: np.ndarray, 
                                 df_f_series: np.ndarray) -> None:
        """Handle completion of an enhanced ROIWorker."""
        color = self.colors[idx % len(self.colors)]
        
        # Plot ΔF/F line
        x_data = np.arange(len(df_f_series))
        
        # Plot mean line
        self.plot_widget.plot(
            x_data, df_f_series, pen=color,
            name=f"ROI {idx+1} ΔF/F"
        )
        
        self.results[idx] = raw_series
        self.df_f_results[idx] = df_f_series

        all_done = len(self.results) == len(self.rois)
        if all_done:
            self.btn_export_svg.setEnabled(True)
            
        if all_done and self.chk_corr.isChecked() and len(self.rois) > 1:
            self.progress.setRange(0, 0)
            # Use ΔF/F data for correlation if available
            data_for_corr = [self.df_f_results.get(0, self.results[0]), 
                           self.df_f_results.get(1, self.results[1])]
            align_worker = AlignmentWorker(data_for_corr)
            align_worker.signals.result.connect(self._on_aligned)
            thread_pool.start(align_worker)
        elif all_done:
            self.progress.setVisible(False)

    def _on_roi_finished(self, idx: int, series: np.ndarray) -> None:
        """Handle completion of a basic ROIWorker."""
        self.plot_widget.plot(
            series, pen=self.colors[idx % len(self.colors)],
            name=f"ROI {idx+1}"
        )
        self.results[idx] = series

        all_done = len(self.results) == len(self.rois)
        if all_done:
            self.btn_export_svg.setEnabled(True)
            
        if all_done and self.chk_corr.isChecked() and len(self.rois) > 1:
            self.progress.setRange(0, 0)
            align_worker = AlignmentWorker(
                [self.results[0], self.results[1]]
            )
            align_worker.signals.result.connect(self._on_aligned)
            thread_pool.start(align_worker)
        elif all_done:
            self.progress.setVisible(False)

    def _on_aligned(self, lags: np.ndarray, corr: np.ndarray) -> None:
        """Plot correlation results and display peak lag."""
        self.corr_widget.clear()
        self.corr_widget.plot(lags, corr, pen='y')
        peak = lags[corr.argmax()]
        self.lbl_align.setText(f"Lag = {peak} frames")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)


def main() -> None:
    from mesofield.gui import theme
    app = QApplication([])
    theme.apply_theme(app)
    viewer = TiffViewer()
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
