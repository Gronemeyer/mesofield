import sys
import json
import logging
from typing import List, Tuple, Optional

import numpy as np
import tifffile
import pyqtgraph as pg

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, Qt, QPointF
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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TiffViewer")

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
        chunk: int = 1000
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
        self.signals = ROIWorkerSignals()

    def run(self) -> None:
        # Memory-map the TIFF for fast access
        mmap = np.memmap(
            self.filepath,
            mode='r',
            dtype=np.dtype(self.dtype_str),
            shape=self.shape
        )
        total_frames = self.shape[0] - 1  # skip the first frame
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
        for start in range(1, self.shape[0], self.chunk):
            end = min(self.shape[0], start + self.chunk)
            block = mmap[
                start:end,
                self.y0:y1,
                self.x0:x1
            ]
            # Compute sums within clipped mask
            sums = (block * mask_clipped).sum(axis=(1, 2))
            idx0 = start - 1
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
        chunk: int = 1000
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
        self.signals = EnhancedROIWorkerSignals()

    def run(self) -> None:
        # Memory-map the TIFF for fast access
        mmap = np.memmap(
            self.filepath,
            mode='r',
            dtype=np.dtype(self.dtype_str),
            shape=self.shape
        )
        total_frames = self.shape[0] - 1  # skip the first frame
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
        for start in range(1, self.shape[0], self.chunk):
            end = min(self.shape[0], start + self.chunk)
            
            # Load data block
            data_block = mmap[start:end, self.y0:y1, self.x0:x1].astype(np.float32)
            
            # Compute mean for each frame
            for i in range(len(data_block)):
                frame = data_block[i]
                frame_pixels = frame[mask_clipped]
                raw_series[start - 1 + i] = frame_pixels.mean()

            percent = int((start - 1 + len(data_block)) * 100 / total_frames)
            self.signals.progress.emit(self.index, percent)

        # Compute baseline (median of first N frames)
        baseline_end = min(self.baseline_frames, len(raw_series))
        baseline = np.median(raw_series[:baseline_end])
        
        # Compute ΔF/F
        df_f_series = (raw_series - baseline) / baseline

        self.signals.finished.emit(self.index, raw_series, df_f_series)


class TiffViewer(QWidget):
    """
    Main application widget for interactive TIFF viewing and ROI analysis.
    """
    def __init__(self):
        super().__init__()
        self.filepath: Optional[str] = None
        self.mmap: Optional[np.memmap] = None
        self.rois: List[Tuple[str, pg.ROI]] = []
        self.results: dict = {}
        self.df_f_results: dict = {}
        self.colors = ['r', 'g', 'b', 'c', 'm']

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
        
        for widget in [self.btn_compute, self.btn_export_svg, self.chk_df_f, self.chk_corr, self.lbl_align]:
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
        
        main_layout.addWidget(params_group)

        # --- Image display ---
        self.img_view = pg.ImageView()
        self.view_box = self.img_view.getView()
        self.img_item = self.img_view.getImageItem()
        main_layout.addWidget(self.img_view)

        # --- Slider and progress bar ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        main_layout.addWidget(self.slider)

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

    def open_file(self) -> None:
        """Open a TIFF file and memory-map it for fast access."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF", "", "*.tif *.tiff"
        )
        if not path:
            return
        self.filepath = path
        self.mmap = tifffile.memmap(path)
        total_frames = self.mmap.shape[0]
        self.slider.setRange(1, total_frames - 1)
        self.slider.setValue(1)
        self.slider.setEnabled(True)
        self.display_frame(1)

    def display_frame(self, index: int) -> None:
        """Display a single frame from the TIFF stack."""
        if self.mmap is None:
            return
        image = np.asarray(self.mmap[index])
        self.img_view.setImage(image, autoLevels=(index == 1))

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
                    baseline_frames=baseline_frames
                )
                worker.signals.progress.connect(
                    lambda _, pct: self.progress.setValue(pct)
                )
                worker.signals.finished.connect(self._on_enhanced_roi_finished)
            else:
                worker = ROIWorker(
                    idx, self.filepath, self.mmap.dtype.str,
                    self.mmap.shape, x0, y0, mask
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
    app = QApplication([])
    viewer = TiffViewer()
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
