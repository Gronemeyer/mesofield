from contextlib import suppress
from typing import Tuple, Union, Literal
from time import perf_counter
import numpy as np
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressBar,
    QLabel,
    QWidget,
    QSizePolicy,
)
from threading import Lock

class ImagePreview(QWidget):
    """
    A PyQt widget that displays images from a `CMMCorePlus` instance (mmcore).

    This widget displays images from a single `CMMCorePlus` instance,
    updating the display in real-time as new images are captured.

    The image is displayed using PyQt's `QLabel` and `QPixmap`, allowing for efficient
    rendering without external dependencies like VisPy.

    **Parameters**
    ----------
    parent : QWidget, optional
        The parent widget. Defaults to `None`.
    mmcore : CMMCorePlus
        The `CMMCorePlus` instance from which images will be displayed.
        Represents the microscope control core.
    use_with_mda : bool, optional
        If `True`, the widget will update during Multi-Dimensional Acquisitions (MDA).
        If `False`, the widget will not update during MDA. Defaults to `True`.

    **Attributes**
    ----------
    clims : Union[Tuple[float, float], Literal["auto"]]
        The contrast limits for the image display. If set to `"auto"`, the widget will
        automatically adjust the contrast limits based on the image data.
    cmap : str
        The colormap to use for the image display. Currently set to `"grayscale"`.

    **Notes**
    -----
    - **Image Display**: Uses a `QLabel` widget to display the image.
      The image is set to scale to fit the label size (`setScaledContents(True)`).

    - **Image Conversion**: Converts images from the `CMMCorePlus` instance to `uint8`
      and scales them appropriately for display using `QImage` and `QPixmap`.

    - **Event Handling**: Connects to various events emitted by the `CMMCorePlus` instance:
        - `imageSnapped`: Emitted when a new image is snapped.
        - `continuousSequenceAcquisitionStarted` and `sequenceAcquisitionStarted`: Emitted when
          a sequence acquisition starts.
        - `sequenceAcquisitionStopped`: Emitted when a sequence acquisition stops.
        - `exposureChanged`: Emitted when the exposure time changes.
        - `frameReady` (MDA): Emitted when a new frame is ready during MDA.

    - **Thread Safety**: Uses a threading lock (`Lock`) to ensure thread-safe access to
      shared resources, such as the current frame. UI updates are performed in the main
      thread using Qt's signals and slots mechanism, ensuring thread safety.

    - **Timer for Updates**: A `QTimer` is used to periodically update the image
      from the core. The timer interval can be adjusted based on the exposure time,
      ensuring that updates occur at appropriate intervals.

    - **Contrast Limits and Colormap**: Allows setting contrast limits (`clims`) and
      colormap (`cmap`) for the image. Currently, only grayscale images are supported.
      The `clims` can be set to a tuple `(min, max)` or `"auto"` for automatic adjustment.

    - **Usage with MDA**: The `use_with_mda` parameter determines whether the widget updates
      during Multi-Dimensional Acquisitions. If set to `False`, the widget will not update
      during MDA runs.

    **Examples**
    --------
    ```python
    from pymmcore_plus import CMMCorePlus
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

    # Initialize a CMMCorePlus instance
    mmc = CMMCorePlus()

    # Set up the application and main window
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout(window)

    # Create the ImagePreview widget
    image_preview = ImagePreview(mmcore=mmc)

    # Add the widget to the layout
    layout.addWidget(image_preview)
    window.show()

    # Start the Qt event loop
    app.exec()
    ```

    **Methods**
    -------
    - `clims`: Property to get or set the contrast limits of the image.
    - `cmap`: Property to get or set the colormap of the image.

    **Initialization Parameters**
    ----------
    parent : QWidget, optional
        The parent widget for this widget.
    mmcore : CMMCorePlus
        The `CMMCorePlus` instance to be used for image acquisition.
    use_with_mda : bool, optional
        Flag to determine if the widget should update during MDA sequences.

    **Raises**
    ------
    ValueError
        If `mmcore` is not provided.

    **Private Methods**
    ----------------
    These methods handle internal functionality:

    - `_disconnect()`: Disconnects all connected signals from the `CMMCorePlus` instance.
    - `_on_streaming_start()`: Starts the streaming timer when a sequence acquisition starts.
    - `_on_streaming_stop()`: Stops the streaming timer when the sequence acquisition stops.
    - `_on_exposure_changed(device, value)`: Adjusts the timer interval when the exposure changes.
    - `_on_streaming_timeout()`: Called periodically by the timer to fetch and display new images.
    - `_on_image_snapped(img)`: Handles new images snapped outside of sequences.
    - `_on_frame_ready(event)`: Handles new frames ready during MDA.
    - `_display_image(img)`: Converts and displays the image in the label.
    - `_adjust_image_data(img)`: Scales image data to `uint8` for display.
    - `_convert_to_qimage(img)`: Converts a NumPy array to a `QImage` for display.

    **Usage Notes**
    ------------
    - **Initialization**: Provide an initialized and configured `CMMCorePlus` instance.
    - **Thread Safety**: UI updates are performed in the main thread. Ensure that heavy computations are offloaded to avoid blocking the UI.
    - **Customization**: You can adjust the `clims` and `cmap` properties to customize the image display.

    **Performance Considerations**
    --------------------------
    - **Frame Rate**: The default timer interval is set to 10 milliseconds. Adjust the interval based on your performance needs.
    - **Resource Management**: Disconnect signals properly by ensuring the `_disconnect()` method is called when the widget is destroyed.

    """

    def __init__(self, parent: QWidget = None, *,
                 _clims: Union[Tuple[float, float], Literal["auto"]] = (0, 255),
                 use_with_mda: bool = True,
                 mmcore: CMMCorePlus | None = None,
                 image_payload=None):
        super().__init__(parent=parent)
        if mmcore is None and image_payload is None:
            raise ValueError(
                "ImagePreview requires either an mmcore CMMCorePlus instance "
                "or an image_payload pyqtSignal emitting numpy frames."
            )
        self._mmcore = mmcore
        self._use_with_mda = use_with_mda
        self._clims = _clims
        self._cmap: str = "grayscale"
        self._current_frame = None
        self._frame_lock = Lock()

        # Set up image label
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setScaledContents(False)  # Keep aspect ratio

        # Set up layout with an image view and an optional progress bar
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.image_label)

        # Progress bar shown during MDA acquisitions
        self.progress_bar = QProgressBar()
        #self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.layout().addWidget(self.progress_bar)

        # Per-viewer FPS display for static live views.
        self.fps_label = QLabel("FPS: --.-")
        self.fps_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.layout().addWidget(self.fps_label)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._fps_window_start = perf_counter()
        self._fps_frame_count = 0
        self._fps_value = 0.0

        # Set up timer
        self.streaming_timer = QTimer(parent=self)
        self.streaming_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.streaming_timer.setInterval(10)  # Default interval; adjust as needed
        self.streaming_timer.timeout.connect(self._on_streaming_timeout)

        if self._mmcore is not None:
            # Connect events for the mmcore
            ev = self._mmcore.events
            #ev.imageSnapped.connect(self._on_image_snapped)
            ev.continuousSequenceAcquisitionStarted.connect(self._on_streaming_start)
            ev.sequenceAcquisitionStarted.connect(self._on_streaming_start)
            ev.sequenceAcquisitionStopped.connect(self._on_streaming_stop)
            ev.exposureChanged.connect(self._on_exposure_changed)

            enev = self._mmcore.mda.events
            enev.frameReady.connect(self._on_frame_ready)
            enev.sequenceStarted.connect(self._on_sequence_started)
            enev.sequenceFinished.connect(self._on_sequence_finished)
            enev.sequenceCanceled.connect(self._on_sequence_finished)

        # Optional non-mmcore frame source (e.g. OpenCVCamera.image_ready).
        if image_payload is not None and hasattr(image_payload, "connect"):
            image_payload.connect(
                self._on_external_frame, type=Qt.ConnectionType.QueuedConnection
            )

        self._progress_total = 0
        self._progress_count = 0

        self.destroyed.connect(self._disconnect)

    def _disconnect(self) -> None:
        if self._mmcore is None:
            return
        # Disconnect events for the mmcore
        ev = self._mmcore.events
        with suppress(TypeError):
            ev.imageSnapped.disconnect()
            ev.continuousSequenceAcquisitionStarted.disconnect()
            ev.sequenceAcquisitionStarted.disconnect()
            ev.sequenceAcquisitionStopped.disconnect()
            ev.exposureChanged.disconnect()

        enev = self._mmcore.mda.events
        with suppress(TypeError):
            enev.frameReady.disconnect()
            enev.sequenceStarted.disconnect()
            enev.sequenceFinished.disconnect()
            enev.sequenceCanceled.disconnect()

    def _on_streaming_start(self) -> None:
        self._reset_fps_counter()
        if not self.streaming_timer.isActive():
            self.streaming_timer.start()

    def _on_streaming_stop(self) -> None:
        # Stop the streaming timer
        if self._mmcore is None or not self._mmcore.isSequenceRunning():
            self.streaming_timer.stop()
            self.fps_label.setText(f"FPS: {self._fps_value:.1f}")

    def _on_exposure_changed(self, device: str, value: str) -> None:
        # Adjust timer interval if needed
        exposure = self._mmcore.getExposure() or 10
        interval = int(exposure) or 10
        self.streaming_timer.setInterval(interval)

    def _on_streaming_timeout(self) -> None:
        frame = None
        new_frames = 0
        if self._mmcore is None:
            with self._frame_lock:
                if self._current_frame is not None:
                    frame = self._current_frame
                    self._current_frame = None
                    new_frames = 1
        elif not self._mmcore.mda.is_running():
            frame, new_frames = self._pop_latest_live_frame()
        else:
            with self._frame_lock:
                if self._current_frame is not None:
                    frame = self._current_frame
                    self._current_frame = None
        # Update the image if a frame is available
        if frame is not None:
            self._display_image(frame)
            if new_frames:
                self._update_fps_counter(new_frames)

    def _on_image_snapped(self, img: np.ndarray) -> None:
        self._update_image(img)
        self._display_image(img)
        self._update_fps_counter(1)

    def _on_external_frame(self, img: np.ndarray) -> None:
        """Handle frames coming from a non-mmcore producer (e.g. OpenCVCamera).

        The frame is stashed under the lock and displayed on the next timer
        tick to keep all draw calls on the GUI thread.
        """
        if img is None:
            return
        with self._frame_lock:
            self._current_frame = img
        if not self.streaming_timer.isActive():
            self._reset_fps_counter()
            self.streaming_timer.start()

    def _on_frame_ready(self, img: np.ndarray) -> None:
        frame = img
        with self._frame_lock:
            self._current_frame = frame
        self._update_fps_counter(1)
        if self.progress_bar.isVisible():
            self._progress_count = min(self._progress_count + 1, self._progress_total)
            self.progress_bar.setValue(self._progress_count)
            # Update the text to "current/total"
            self.progress_bar.setFormat(f"{self._progress_count}/{self._progress_total}")

    def _on_sequence_started(self, sequence, metadata) -> None:
        self._reset_fps_counter()
        self._progress_total = len(list(sequence.iter_events()))
        self._progress_count = 0
        self.progress_bar.setRange(0, self._progress_total)
        self.progress_bar.setValue(0)
        # Show "0/total" initially
        self.progress_bar.setFormat(f"{self._progress_count}/{self._progress_total}")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(True)

    def _on_sequence_finished(self, *_) -> None:
        self.progress_bar.setValue(self._progress_total)
        self.progress_bar.setFormat(f"{self._progress_total}/{self._progress_total}")
        # Hide after a short delay
        QTimer.singleShot(500, lambda: self.progress_bar.setVisible(False))

    def _reset_fps_counter(self) -> None:
        self._fps_window_start = perf_counter()
        self._fps_frame_count = 0
        self._fps_value = 0.0
        self.fps_label.setText("FPS: --.-")

    def _update_fps_counter(self, count: int = 1) -> None:
        if count <= 0:
            return
        self._fps_frame_count += count
        elapsed = perf_counter() - self._fps_window_start
        if elapsed < 0.5:
            return
        self._fps_value = self._fps_frame_count / elapsed
        self.fps_label.setText(f"FPS: {self._fps_value:.1f}")
        self._fps_window_start = perf_counter()
        self._fps_frame_count = 0

    def _pop_latest_live_frame(self) -> tuple[np.ndarray | None, int]:
        """Pop all newly queued live frames and return the latest one and count.

        FPS should reflect true incoming frames, not timer refresh ticks.
        """
        frame = None
        new_frames = 0
        with suppress(RuntimeError, IndexError, TypeError, AttributeError):
            remaining = int(self._mmcore.getRemainingImageCount())
            for _ in range(remaining):
                try:
                    frame, _ = self._mmcore.popNextImageAndMD()
                except Exception:
                    frame = self._mmcore.popNextImage()
                new_frames += 1

        # Fallback for display continuity only; does not increment FPS.
        if frame is None:
            with suppress(RuntimeError, IndexError):
                frame = self._mmcore.getLastImage()

        return frame, new_frames

    def _display_image(self, img: np.ndarray) -> None:
        if img is None:
            return
        qimage = self._convert_to_qimage(img)
        if qimage is not None:
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def _update_image(self, img: np.ndarray) -> None:
        # Update the current frame
        with self._frame_lock:
            self._current_frame = img

    def _adjust_image_data(self, img: np.ndarray) -> np.ndarray:
        # NOTE: This is the default implementation for grayscale images
        # NOTE: This is the most processor-intensive part of this widget

        # Color frames pass through unchanged (handled in _convert_to_qimage)
        if img.ndim == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
            return img

        # Ensure the image is in float format for scaling
        img = img.astype(np.float32, copy=False)

        # Apply contrast limits
        if self._clims == "auto":
            min_val, max_val = np.min(img), np.max(img)
        else:
            min_val, max_val = self._clims

        # Avoid division by zero
        scale = 255.0 / (max_val - min_val) if max_val != min_val else 255.0

        # Scale to 0-255
        img = np.clip((img - min_val) * scale, 0, 255).astype(np.uint8, copy=False)
        return img

    def _convert_to_qimage(self, img: np.ndarray) -> QImage:
        """Convert a NumPy array to QImage."""
        if img is None:
            return None
        img = self._adjust_image_data(img)
        img = np.ascontiguousarray(img)
        height, width = img.shape[:2]

        if img.ndim == 2:
            # Grayscale image
            bytes_per_line = width
            qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            # Color image (BGR from OpenCV) -> RGB888
            if img.shape[2] == 4:
                rgb = img[..., [2, 1, 0]]  # drop alpha
            else:
                rgb = img[..., ::-1]       # BGR -> RGB
            rgb = np.ascontiguousarray(rgb)
            bytes_per_line = 3 * width
            qimage = QImage(
                rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
            # Keep the underlying buffer alive for the life of the QImage
            qimage._np_data = rgb  # type: ignore[attr-defined]
        else:
            return None

        return qimage

    @property
    def clims(self) -> Union[Tuple[float, float], Literal["auto"]]:
        """Get the contrast limits of the image."""
        return self._clims

    @clims.setter
    def clims(self, clims: Union[Tuple[float, float], Literal["auto"]] = (0, 255)) -> None:
        """Set the contrast limits of the image.

        Parameters
        ----------
        clims : tuple[float, float], or "auto"
            The contrast limits to set.
        """
        self._clims = clims

    @property
    def cmap(self) -> str:
        """Get the colormap (lookup table) of the image."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: str = "grayscale") -> None:
        """Set the colormap (lookup table) of the image.

        Parameters
        ----------
        cmap : str
            The colormap to use.
        """
        self._cmap = cmap


import pyqtgraph as pg
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
import numpy as np
from threading import Lock
from contextlib import suppress
from typing import Tuple, Union, Literal
from pymmcore_plus import CMMCorePlus

pg.setConfigOptions(imageAxisOrder='row-major', useOpenGL=True)

class InteractivePreview(pg.ImageView):
    def __init__(self, parent=None, mmcore=None, use_with_mda=True, image_payload=None):
        super().__init__(parent=parent)
        self._mmcore: CMMCorePlus = mmcore
        self._use_with_mda = use_with_mda
        self._clims: Union[Tuple[float, float], Literal["auto"]] = (0, 65535)
        self._current_frame = np.zeros((512, 512), dtype=np.uint8)
        self._display_image(self._current_frame)
        self._cmap: str = "grayscale"
        self._current_frame = None
        self._frame_lock = Lock()

        self._fps_window_start = perf_counter()
        self._fps_frame_count = 0
        self._fps_value = 0.0
        self._fps_text = pg.TextItem("FPS: --.-", color=(255, 255, 0), anchor=(1, 0))
        self.view.addItem(self._fps_text)

        if image_payload is not None:
            image_payload.connect(self._on_image_payload)

        if self._mmcore is not None:
            self._mmcore.events.imageSnapped.connect(self._on_image_snapped)
            self._mmcore.events.continuousSequenceAcquisitionStarted.connect(self._on_streaming_start)
            self._mmcore.events.sequenceAcquisitionStarted.connect(self._on_streaming_start)
            self._mmcore.events.sequenceAcquisitionStopped.connect(self._on_streaming_stop)
            self._mmcore.events.exposureChanged.connect(self._on_exposure_changed)

            enev = self._mmcore.mda.events
            enev.frameReady.connect(self._on_image_payload, type=Qt.ConnectionType.QueuedConnection)
            if self._use_with_mda:
                self._mmcore.mda.events.frameReady.connect(self._on_frame_ready)

            self.streaming_timer = QTimer(parent=self)
            self.streaming_timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.streaming_timer.setInterval(10)
            self.streaming_timer.timeout.connect(self._on_streaming_timeout)

        self.destroyed.connect(self._disconnect)

    def _disconnect(self) -> None:
        if self._mmcore:
            ev = self._mmcore.events
            with suppress(TypeError):
                ev.imageSnapped.disconnect()
                ev.continuousSequenceAcquisitionStarted.disconnect()
                ev.sequenceAcquisitionStarted.disconnect()
                ev.sequenceAcquisitionStopped.disconnect()
                ev.exposureChanged.disconnect()
            enev = self._mmcore.mda.events
            with suppress(TypeError):
                enev.frameReady.disconnect()

    def _on_streaming_start(self) -> None:
        self._reset_fps_counter()
        if not self.streaming_timer.isActive():
            self.streaming_timer.start()

    def _on_streaming_stop(self) -> None:
        if not self._mmcore.isSequenceRunning():
            self.streaming_timer.stop()
            self._fps_text.setText(f"FPS: {self._fps_value:.1f}")

    def _on_exposure_changed(self, device: str, value: str) -> None:
        exposure = self._mmcore.getExposure() or 10
        self.streaming_timer.setInterval(int(exposure) or 10)

    def _on_frame_ready(self, img: np.ndarray) -> None:
        with self._frame_lock:
            self._current_frame = img

    def _on_streaming_timeout(self) -> None:
        frame = None
        new_frames = 0
        if not self._mmcore.mda.is_running():
            frame, new_frames = self._pop_latest_live_frame()
        else:
            with self._frame_lock:
                if self._current_frame is not None:
                    frame = self._current_frame
                    self._current_frame = None
        if frame is not None:
            self._display_image(frame)
            if new_frames:
                self._update_fps_counter(frame.shape, count=new_frames)

    def _on_image_snapped(self, img: np.ndarray) -> None:
        with self._frame_lock:
            self._current_frame = img
        self._display_image(img)
        self._update_fps_counter(img.shape, count=1)

    def _on_image_payload(self, img: np.ndarray) -> None:
        if img is None:
            return
        #img = self._adjust_image_data(img)
        self.setImage(img.T, 
                      autoHistogramRange=False, 
                      autoRange=False, 
                      levelMode='mono', 
                      autoLevels=(self._clims == "auto"),
                      )
        self._update_fps_counter(img.shape, count=1)

    def _display_image(self, img: np.ndarray) -> None:
        if img is None:
            return
        img = self._adjust_image_data(img)
        self.setImage(img.T, 
                      autoHistogramRange=False, 
                      autoRange=False, 
                      levelMode='mono', 
                      autoLevels=(self._clims == "auto"),
                      )

    def _reset_fps_counter(self) -> None:
        self._fps_window_start = perf_counter()
        self._fps_frame_count = 0
        self._fps_value = 0.0
        self._fps_text.setText("FPS: --.-")

    def _update_fps_counter(self, image_shape: tuple[int, ...], count: int = 1) -> None:
        if count <= 0:
            return
        self._fps_frame_count += count
        elapsed = perf_counter() - self._fps_window_start
        if elapsed >= 0.5:
            self._fps_value = self._fps_frame_count / elapsed
            self._fps_text.setText(f"FPS: {self._fps_value:.1f}")
            self._fps_window_start = perf_counter()
            self._fps_frame_count = 0
        if image_shape:
            self._fps_text.setPos(max(image_shape[1] - 1, 0), 0)

    def _pop_latest_live_frame(self) -> tuple[np.ndarray | None, int]:
        """Pop all newly queued live frames and return the latest one and count."""
        frame = None
        new_frames = 0
        with suppress(RuntimeError, IndexError, TypeError, AttributeError):
            remaining = int(self._mmcore.getRemainingImageCount())
            for _ in range(remaining):
                try:
                    frame, _ = self._mmcore.popNextImageAndMD()
                except Exception:
                    frame = self._mmcore.popNextImage()
                new_frames += 1

        # Fallback for display continuity only; does not increment FPS.
        if frame is None:
            with suppress(RuntimeError, IndexError):
                frame = self._mmcore.getLastImage()

        return frame, new_frames

    def _adjust_image_data(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        if self._clims == "auto":
            min_val, max_val = np.min(img), np.max(img)
        else:
            min_val, max_val = self._clims
        scale = 255.0 / (max_val - min_val) if max_val != min_val else 255.0
        img = np.clip((img - min_val) * scale, 0, 255).astype(np.uint8, copy=False)
        return img

    # @property
    # def clims(self) -> Union[Tuple[float, float], Literal["auto"]]:
    #     return self._clims

    # @clims.setter
    # def clims(self, clims: Union[Tuple[float, float], Literal["auto"]] = "auto") -> None:
    #     self._clims = clims
    #     if self._current_frame is not None:
    #         self._display_image(self._current_frame)

    # @property
    # def cmap(self) -> str:
    #     return self._cmap

    # @cmap.setter
    # def cmap(self, cmap: str = "grayscale") -> None:
    #     self._cmap = cmap