from pymmcore_plus import CMMCorePlus

from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)

from pymmcore_widgets import (
    MDAWidget,
    ExposureWidget,
    LiveButton,
    SnapButton,
)

from mesofield.data.writer import CustomWriter
from mesofield.gui.viewer import ImagePreview, InteractivePreview

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

    def __init__(self, config) -> None:
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

        for cam in self.cameras:
            # Per-core container
            core_box = QGroupBox(title=str(cam.name))
            core_box.setLayout(QVBoxLayout())
            core_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            auto_contrast = getattr(cam, "auto_contrast")

            # Preview widget based on cam.viewer
            if cam.viewer == "static":
                if isinstance(cam.core, CMMCorePlus):
                    preview = ImagePreview(
                        mmcore=cam.core,
                        _clims='auto' if auto_contrast else (0, 255),
                    )

                    # Buttons row
                    btn_box = QWidget()
                    btn_box.setLayout(QHBoxLayout())
                    snap_btn = SnapButton(mmcore=cam.core)
                    live_btn = LiveButton(mmcore=cam.core)
                    btn_box.layout().addWidget(snap_btn)
                    btn_box.layout().addWidget(live_btn)
                    core_box.layout().addWidget(btn_box)
                    core_box.layout().addWidget(preview)
                    cores_groupbox.layout().addWidget(core_box)
                    self.layout().addWidget(cores_groupbox)
                else:
                    preview = InteractivePreview(
                        image_payload=cam.core.image_ready,
                        auto_contrast=auto_contrast,
                    )
                    cam.core.start()
                    core_box.layout().addWidget(preview)
                    cores_groupbox.layout().addWidget(core_box)




