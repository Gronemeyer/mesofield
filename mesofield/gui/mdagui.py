from contextlib import suppress

from pymmcore_plus import CMMCorePlus

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from pymmcore_widgets import MDAWidget

from mesofield.data.writer import CustomWriter
from mesofield.gui.viewer import ImagePreview, InteractivePreview
from mesofield.utils._logger import get_logger


_logger = get_logger(__name__)


class CameraButtons(QWidget):
    """Snap + Live toggle wired through the BaseCamera contract.

    Replaces pymmcore-widgets' ``SnapButton`` / ``LiveButton`` (which only
    worked for mmcore-backed cameras) with two buttons that call
    ``cam.snap()`` / ``cam.start_live()`` / ``cam.stop_live()``.  Those
    methods are defined on :class:`mesofield.devices.base_camera.BaseCamera`
    and implemented by every camera subclass, so the GUI no longer cares
    whether the underlying backend is Micro-Manager, OpenCV, or the
    synthetic :class:`MockFrameProducer`.

    The widget delegates frame display to the camera's existing signal
    plumbing:

    - MMCamera: ``cam.snap()`` calls ``mmcore.snap()`` which fires the
      ``imageSnapped`` event the :class:`ImagePreview` is already wired
      to; ``cam.start_live()`` triggers ``startContinuousSequenceAcquisition``
      and frames flow through the MDA events.
    - OpenCVCamera / MockFrameProducer: ``snap()`` and the live loop emit
      ``cam.image_ready`` (a ``pyqtSignal(np.ndarray)``) which the
      ``ImagePreview`` subscribes to via ``image_payload``.
    """

    def __init__(self, cam, preview=None) -> None:
        super().__init__()
        self.cam = cam
        self.preview = preview
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.snap_btn = QPushButton("Snap")
        self.snap_btn.setToolTip(f"cam.snap()  [{cam.backend} backend]")
        self.snap_btn.clicked.connect(self._on_snap)
        layout.addWidget(self.snap_btn)

        self.live_btn = QPushButton("Live")
        self.live_btn.setCheckable(True)
        self.live_btn.setToolTip(
            f"toggle cam.start_live() / cam.stop_live()  [{cam.backend} backend]"
        )
        self.live_btn.toggled.connect(self._on_live_toggled)
        layout.addWidget(self.live_btn)

        # Runtime auto-contrast toggle for the live viewer. The "off" state
        # restores a viewer-appropriate manual range (8-bit static preview
        # vs 16-bit interactive preview).
        self._manual_clims = (
            (0, 65535) if isinstance(preview, InteractivePreview) else (0, 255)
        )
        self.auto_contrast_cb = QCheckBox("Auto-contrast")
        self.auto_contrast_cb.setChecked(bool(getattr(cam, "auto_contrast", True)))
        self.auto_contrast_cb.setToolTip("Toggle live-viewer auto-contrast")
        self.auto_contrast_cb.toggled.connect(self._on_auto_contrast_toggled)
        layout.addWidget(self.auto_contrast_cb)
        # Sync the viewer's contrast mode to the checkbox's initial state.
        self._on_auto_contrast_toggled(self.auto_contrast_cb.isChecked())

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the snap + live controls (used during acquisition)."""
        self.snap_btn.setEnabled(enabled)
        self.live_btn.setEnabled(enabled)

    def _on_auto_contrast_toggled(self, checked: bool) -> None:
        if self.preview is None:
            return
        try:
            self.preview.clims = "auto" if checked else self._manual_clims
        except Exception as exc:
            _logger.warning(
                "auto-contrast toggle failed on %s: %s", self.cam.device_id, exc
            )

    def _on_snap(self) -> None:
        try:
            self.cam.snap()
        except Exception as exc:
            _logger.warning("snap failed on %s: %s", self.cam.device_id, exc)

    def _on_live_toggled(self, checked: bool) -> None:
        try:
            if checked:
                self.cam.start_live()
                self.live_btn.setText("Stop Live")
            else:
                self.cam.stop_live()
                self.live_btn.setText("Live")
        except Exception as exc:
            _logger.warning(
                "live toggle failed on %s: %s", self.cam.device_id, exc
            )
            # Reset the toggle state if start/stop failed.
            self.live_btn.blockSignals(True)
            self.live_btn.setChecked(not checked)
            self.live_btn.setText("Live")
            self.live_btn.blockSignals(False)

class CustomMDAWidget(MDAWidget):
    def run_mda(self) -> None:
        """Run the MDA sequence experiment."""
        # in case the user does not press enter after editing the save name.
        self.save_info.save_name.editingFinished.emit()

        sequence = self.value()

        # technically, this is in the metadata as well, but isChecked is more direct
        if self.save_info.isChecked():
            save_path = self._update_save_path_from_metadata(
                sequence, update_metadata=True
            )
        else:
            save_path = None

        # run the MDA experiment asynchronously
        self._mmc.run_mda(sequence, output=CustomWriter(save_path))

class MDA(QWidget):
    """
    The `MDAWidget` provides a GUI to construct a `useq.MDASequence` object.
    This object describes a full multi-dimensional acquisition;
    In this example, we set the `MDAWidget` parameter `include_run_button` to `True`,
    meaning that a `run` button is added to the GUI. When pressed, a `useq.MDASequence`
    is first built depending on the GUI values and is then passed to the
    `CMMCorePlus.run_mda` to actually execute the acquisition.
    For details of the corresponding schema and methods, see
    https://github.com/pymmcore-plus/useq-schema and
    https://github.com/pymmcore-plus/pymmcore-plus.

    """

    def __init__(self, procedure) -> None:
        """

        The layout adapts the viewer based on the number of cores:

        Single Core Layout:

            +----------------------------------------+
            | Live Viewer                            |
            | +-----------------+-----------------+  |
            | | [Snap Button]   |  [Live Button}  |  |
            | +-----------------+-----------------+  |
            | |                                   |  |
            | |            Image Preview          |  |
            | |                                   |  |
            | +-----------------------------------+  |
            +----------------------------------------+
                
        Dual Core Layout:

            +-----------------------------------------------+
            |   Live Viewer                                 |
            |  +---------------------+-------------------+  |
            |  |      Core 1         |      Core 2       |  |
            |  +---------------------+-------------------+  |
            |  |     [Buttons]       |     [Buttons]     |  |
            |  +---------------------+-------------------+  |
            |  |                     |                   |  |
            |  |  Image Preview 1    |  Image Preview 2  |  |
            |  |                     |                   |  |
            |  +---------------------+-------------------+  |
            +-----------------------------------------------+

        """
        super().__init__()
        # get the CMMCore instance and load the default config
        self.procedure = procedure
        config = procedure.config
        self.cameras = config.hardware.cameras
        self.mmcores = tuple(cam.core for cam in self.cameras)
        self._viewer_type = config.hardware._viewer
        # instantiate the MDAWidget
        #self.mda = MDAWidget(mmcore=self.mmcores[0])
        # ----------------------------------Auto-set MDASequence and save_info----------------------------------#
        #self.mda.setValue(config.pupil_sequence)
        #self.mda.save_info.setValue({'save_dir': r'C:/dev', 'save_name': 'file', 'format': 'ome-tiff', 'should_save': True})
        # -------------------------------------------------------------------------------------------------------#
        self.setLayout(QHBoxLayout())

        live_viewer = QGroupBox()
        live_viewer.setLayout(QVBoxLayout())
        live_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        buttons = QGroupBox()
        buttons.setLayout(QHBoxLayout())

        cores_groupbox = QGroupBox(f"{self.__module__}.{self.__class__.__name__}: Live Viewer")
        cores_groupbox.setLayout(QHBoxLayout())

        # Per-preview minimum width budget. Each ImagePreview otherwise floors
        # at 512px; laid out left-to-right, three VGA cameras forced the window
        # past a 16:9 screen and pushed the ConfigController off the edge. Size
        # each preview so all cameras plus the right-hand config panel fit the
        # available screen instead.
        preview_min = self._preview_min_size(len(self.cameras))

        # Track the per-camera button widgets so acquisition can lock them out.
        self._camera_buttons: list[CameraButtons] = []
        # Track previews so `cleanup` can disconnect them from the (longer-lived)
        # cameras before this widget is deleteLater()'d on a hardware reload.
        self._previews: list = []
        self._procedure_events = getattr(procedure, "events", None)

        for cam in self.cameras:
            # Per-core container
            core_box = QGroupBox(title=str(cam.name))
            core_box.setLayout(QVBoxLayout())
            core_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            auto_contrast = getattr(cam, "auto_contrast", True)

            # Preview widget: ImagePreview drives display.
            # - mmcore-backed cameras wire the preview to mmcore events;
            # - everyone else feeds it via the camera's `image_ready` Qt
            #   signal (an attribute on every BaseCamera subclass).
            if cam.viewer == "static":
                if isinstance(cam.core, CMMCorePlus):
                    preview = ImagePreview(
                        mmcore=cam.core,
                        _clims='auto' if auto_contrast else (0, 255),
                        min_size=preview_min,
                    )
                else:
                    image_signal = getattr(cam, "image_ready", None)
                    if image_signal is None and cam.core is not None:
                        image_signal = getattr(cam.core, "image_ready", None)
                    preview = ImagePreview(
                        mmcore=None,
                        image_payload=image_signal,
                        progress_payload=getattr(cam, "progress", None),
                        _clims='auto' if auto_contrast else (0, 255),
                        min_size=preview_min,
                    )
            else:
                # Interactive / pyqtgraph viewer.
                image_signal = getattr(cam, "image_ready", None)
                if image_signal is None and cam.core is not None:
                    image_signal = getattr(cam.core, "image_ready", None)
                preview = InteractivePreview(
                    image_payload=image_signal, min_size=preview_min
                )

            # Unified snap + live buttons -- driven by BaseCamera methods.
            # Works for MMCamera (via mmcore), OpenCVCamera (capture thread),
            # and MockFrameProducer (synthetic frames). Non-mmcore cameras
            # used to auto-start on widget creation; now the user clicks
            # "Live" to start, matching mmcore camera UX.
            self._previews.append(preview)
            cam_buttons = CameraButtons(cam, preview)
            self._camera_buttons.append(cam_buttons)
            core_box.layout().addWidget(cam_buttons)
            core_box.layout().addWidget(preview)
            cores_groupbox.layout().addWidget(core_box)

        # Add the cores_groupbox once, not once per camera (the old code
        # added it inside the loop, producing duplicate top-level widgets).
        self.layout().addWidget(cores_groupbox)

        # Lock the Snap/Live buttons for the duration of a Procedure run.
        # Bound-method connections auto-disconnect when this QWidget is
        # destroyed (the widget is rebuilt on every hardware reload).
        events = getattr(procedure, "events", None)
        if events is not None:
            events.procedure_started.connect(self._on_acquisition_started)
            events.procedure_finished.connect(self._on_acquisition_finished)
            events.procedure_error.connect(self._on_acquisition_finished)

    @staticmethod
    def _preview_min_size(n_cameras: int) -> int:
        """Per-preview minimum edge (px) that keeps the window on one screen.

        Previews sit in a single horizontal row, so their minimum widths add up
        and (with ``QLayout.SizeConstraint.SetMinimumSize`` on the main window)
        force the window's minimum width. The default 512px floor overflows a
        16:9 screen once a third VGA camera is added, pushing the right-hand
        ConfigController off the edge. Divide the available screen width — minus
        a reserve for the config panel — across the cameras, capped at 512 so the
        common one/two-camera layouts are unchanged and floored so a preview
        never becomes uselessly small.
        """
        DEFAULT = 512        # unchanged single-/dual-camera behaviour
        FLOOR = 256          # never shrink a preview below this
        PANEL_RESERVE = 480  # right-hand tabs (ConfigController) + margins
        if n_cameras <= 1:
            return DEFAULT

        screen = QApplication.primaryScreen()
        if screen is None:
            return DEFAULT
        avail = int(screen.availableGeometry().width() * 0.95) - PANEL_RESERVE
        if avail <= 0:
            return FLOOR
        return max(FLOOR, min(DEFAULT, avail // n_cameras))

    def cleanup(self) -> None:
        """Tear down every preview and procedure-event subscription.

        Called by ``MainWindow._build_acquisition_ui`` before ``deleteLater()``
        so previews disconnect from their cameras (which outlive this widget)
        and the procedure-event slots don't fire into a deleted MDA on the next
        run. Idempotent and exception-safe.
        """
        for preview in self._previews:
            try:
                preview.cleanup()
            except Exception:
                pass
        self._previews.clear()

        events = self._procedure_events
        if events is not None:
            for sig, slot in (
                (getattr(events, "procedure_started", None), self._on_acquisition_started),
                (getattr(events, "procedure_finished", None), self._on_acquisition_finished),
                (getattr(events, "procedure_error", None), self._on_acquisition_finished),
            ):
                if sig is not None:
                    with suppress(TypeError, RuntimeError):
                        sig.disconnect(slot)

    def _on_acquisition_started(self, *_args) -> None:
        self.set_acquisition_active(True)

    def _on_acquisition_finished(self, *_args) -> None:
        self.set_acquisition_active(False)

    def set_acquisition_active(self, active: bool) -> None:
        """Lock out the Snap/Live buttons while a Procedure run is in progress."""
        for cam_buttons in self._camera_buttons:
            cam_buttons.set_enabled(not active)




