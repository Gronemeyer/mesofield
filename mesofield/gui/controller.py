import os
from datetime import datetime
from contextlib import suppress
import threading

from qtpy.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QInputDialog,
    QStyle,
    QFormLayout,
    QSpinBox,
    QCheckBox,
)
from PyQt6.QtGui import QIcon
from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig
    from mesofield.protocols import Procedure

from .dynamic_controller import DynamicController

class ConfigFormWidget(QWidget):
    """Map each config key to an appropriate editor in a form layout."""

    @staticmethod
    def _text_for_editor(key: str, value) -> str:
        if value is None:
            return ""
        if key == "led_pattern" and isinstance(value, list):
            return "".join(str(v) for v in value)
        return str(value)

    def _commit_text_editor(self, key: str, editor: QLineEdit) -> None:
        text = editor.text().strip() if key == "led_pattern" else editor.text()
        try:
            self._registry.set(key, text)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Invalid value", f"{key}: {e}")
            editor.setText(self._text_for_editor(key, self._registry.get(key)))

    def __init__(self, registry, keys=None):
        super().__init__()
        self._registry = registry
        form = QFormLayout(self)
        if keys is None:
            keys = self._registry.keys()
        self._keys = list(keys)
        # create editor per config key with initial values and two-way binding
        for key in self._keys:
            type_hint = self._registry.get_metadata(key).get("type")
            value = self._registry.get(key)
            choices = self._registry.get_choices(key)
            if key == "session" and not choices:
                # Session is a zero-padded BIDS string (e.g. "02"). Edit it as
                # a stepper while preserving the "%02d" string format on commit.
                # When choices are registered for "session" the dropdown branch
                # below is used instead.
                editor = QSpinBox()
                editor.setRange(0, 999)
                try:
                    editor.setValue(int(value))
                except (TypeError, ValueError):
                    editor.setValue(0)
                editor.valueChanged.connect(
                    lambda val, k=key: self._registry.set(k, f"{val:02d}")
                )
            elif choices and key != "led_pattern":
                # Key has registered choices — render a dropdown
                editor = QComboBox()
                editor.addItems([str(c) for c in choices])
                current = str(value) if value is not None else ""
                idx = editor.findText(current)
                if idx >= 0:
                    editor.setCurrentIndex(idx)
                editor.currentTextChanged.connect(lambda text, k=key: self._registry.set(k, text))
            elif type_hint is int:
                editor = QSpinBox()
                editor.setRange(-1_000_000, 1_000_000)
                editor.setValue(int(value or 0))
                editor.valueChanged.connect(lambda val, k=key: self._registry.set(k, val))
            elif type_hint is bool:
                editor = QCheckBox()
                editor.setChecked(bool(value))
                editor.toggled.connect(lambda checked, k=key: self._registry.set(k, checked))
            else:
                editor = QLineEdit()
                editor.setText(self._text_for_editor(key, value))
                if key == "led_pattern":
                    editor.setPlaceholderText("e.g. 422222442 or [\"4\",\"2\",\"2\"]")
                editor.editingFinished.connect(lambda k=key, e=editor: self._commit_text_editor(k, e))
            form.addRow(key, editor)
        self.config_hint_text = QLabel("<i>Values persist upon Procedure completion or with the \"Save\" button</i>")
        form.addRow(self.config_hint_text)


    @property
    def keys(self):
        """Return the list of registry keys displayed in the form."""
        return list(self._keys)


class ConfigController(QWidget):
    """
    The ConfigController widget displays subject parameters loaded from the
    experiment JSON and allows selection of the current subject.
    
    The object connects to the Micro-Manager Core object instances and the Config object.
    
    The ConfigController widget emits signals to notify other widgets when the configuration is updated
    and when the record button is pressed.
    
    Public Methods:
    ----------------
    record(): 
        triggers the MDA sequence with the configuration parameters


    Private Methods:
    ----------------
    _load_subject():
        updates the configuration form for the selected subject
    _test_led():
        tests the LED pattern by sending a test sequence to the Arduino-Switch device
    _stop_led(): 
        stops the LED pattern by sending a stop sequence to the Arduino-Switch device
    _add_note(): 
        opens a dialog to get a note from the user and save it to the ExperimentConfig.notes list
    
    """
    # ==================================== Signals ===================================== #
    configUpdated = pyqtSignal(object)
    recordStarted = pyqtSignal(str)
    # ------------------------------------------------------------------------------------- #
    def __init__(self, procedure: 'Procedure', display_keys=None):
        super().__init__()
        self.config: ExperimentConfig = procedure.config 
        self.procedure = procedure
        if display_keys is None and hasattr(self.config, "display_keys"):
            display_keys = self.config.display_keys
        self.display_keys = list(display_keys) if display_keys is not None else None

        # Create main layout
        layout = QVBoxLayout(self)
        self.setFixedWidth(500)

        # ==================================== GUI Widgets ===================================== #
        # Button to open the BIDS directory in the system file explorer
        self.open_bids_button = QPushButton("Open BIDS Directory")
        layout.addWidget(self.open_bids_button)
        self.open_bids_button.setToolTip("Open the procedure.config.bids_dir in your file explorer")

        # subject selection dropdown
        self.subject_dropdown_label = QLabel('Select Subject:')
        self.subject_dropdown = QComboBox()
        self.add_subject_button = QPushButton("+ Subject")
        self.add_subject_button.setToolTip("Add a new subject to experiment.json")
        self.add_parameter_button = QPushButton("+ Parameter")
        self.add_parameter_button.setToolTip(
            "Add a new parameter applied to every subject and made editable in DisplayKeys"
        )
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.subject_dropdown_label)
        sub_layout.addWidget(self.subject_dropdown)
        sub_layout.addWidget(self.add_subject_button)
        sub_layout.addWidget(self.add_parameter_button)
        layout.addLayout(sub_layout)

        # BIDS filename preview — updates live as subject/session/task change
        self.filename_preview_label = QLabel()
        self.filename_preview_label.setWordWrap(True)
        self.filename_preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        from mesofield.gui import theme
        self.filename_preview_label.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-family: {theme.MONO_FONT};"
        )
        layout.addWidget(self.filename_preview_label)

        self.config_model = ConfigFormWidget(self.procedure.config, keys=self.display_keys)

        self._populate_subjects()
        self._change_subject(0)

        # Register live updates for the filename preview. Callbacks fire from
        # ConfigRegister.set() — which is what ConfigFormWidget editors call.
        for key in ("subject", "session", "task"):
            self.config.register_callback(key, lambda _k, _v: self._update_filename_preview())
        self._update_filename_preview()
        
        # 4. Record button to start the MDA sequence
        self.record_button = QPushButton("Record")

        # Tint the standard play icon red
        play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        pix = play_icon.pixmap(24, 24)
        mask = pix.createMaskFromColor(Qt.transparent)
        pix.fill(Qt.GlobalColor.red)
        pix.setMask(mask)
        self.record_button.setIcon(QIcon(pix))

        from mesofield.gui import theme
        self.record_button.setStyleSheet(theme.record_button_qss())
        self.record_button.setToolTip("Start Recording (MDA Sequence)")
        self.record_button.setShortcut("Ctrl+R")  # Set shortcut for recording

        # 5. Abort button to safely stop a running Procedure
        self.abort_button = QPushButton("Abort")
        stop_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        self.abort_button.setIcon(stop_icon)
        self.abort_button.setToolTip("Abort the running Procedure (safe stop + save)")
        self.abort_button.setEnabled(False)

        # 6. Add Note button to add a note to the configuration
        self.add_note_button = QPushButton("Add Note")

        # 7. Save button to persist edits to experiment.json on demand
        self.save_button = QPushButton("Save")
        save_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.save_button.setIcon(save_icon)
        self.save_button.setToolTip("Save edits to experiment.json now")

        # Group the Record, Abort, Add Note and Save buttons horizontally
        self._action_buttons_layout = QHBoxLayout()
        self._action_buttons_layout.addWidget(self.record_button)
        self._action_buttons_layout.addWidget(self.abort_button)
        self._action_buttons_layout.addWidget(self.add_note_button)
        self._action_buttons_layout.addWidget(self.save_button)
        layout.addLayout(self._action_buttons_layout)

        # Absorb extra vertical space here so the form + primary action buttons
        # stay anchored to the top, and hardware-specific controls stay pinned
        # to the bottom edge regardless of window height.
        layout.addStretch(1)

        # Dynamic hardware-specific controls (pinned to bottom of the panel)
        self.dynamic_controller = DynamicController(self.procedure.config, parent=self)
        layout.addWidget(self.dynamic_controller)
        # ------------------------------------------------------------------------------------- #

        # ============ Callback connections between widget values and functions ================ #

        self.subject_dropdown.currentIndexChanged.connect(self._change_subject) # When the subject is changed, update the config form
        self.record_button.clicked.connect(self.record)
        self.abort_button.clicked.connect(self._abort)
        self.add_note_button.clicked.connect(self._add_note)
        self.add_subject_button.clicked.connect(self._add_subject)
        self.add_parameter_button.clicked.connect(self._add_parameter)
        self.save_button.clicked.connect(self._save_config)
        self.open_bids_button.clicked.connect(self._open_bids_directory)

        # Toggle Record/Abort availability from the live procedure lifecycle.
        # Bound-method connections auto-disconnect when this widget (a
        # QObject) is destroyed on a config reload.
        events = getattr(self.procedure, "events", None)
        if events is not None:
            events.procedure_started.connect(self._on_run_started)
            events.procedure_finished.connect(self._on_run_finished)
            events.procedure_error.connect(self._on_run_finished)

        # Connect dynamic controls using constants defined in DynamicController.
        # Snap and PsychoPy launch are handled elsewhere (CameraButtons in
        # mdagui.py, PsychoPyDevice.arm() respectively).
        dynamic_buttons = [
            (DynamicController.LED_TEST_BTN, self._test_led),
            (DynamicController.STOP_BTN, self._stop_led),
            (DynamicController.NIDAQ_BTN, self._test_nidaq),
        ]
        for btn_attr, handler in dynamic_buttons:
            if hasattr(self.dynamic_controller, btn_attr):
                getattr(self.dynamic_controller, btn_attr).clicked.connect(handler)

        # ------------------------------------------------------------------------------------- #
    # ------------------------------- Introspection Helpers --------------------------- #
    def displayed_values(self) -> dict:
        """Return the configuration values currently shown in the form widget."""
        if not hasattr(self, "config_model"):
            return {}
        return {k: self.config.get(k) for k in self.config_model.keys}

    def set_display_keys(self, keys=None):
        """Set which configuration keys should be displayed in the form."""
        self.display_keys = list(keys) if keys is not None else None
        old_form = getattr(self, 'config_model', None)
        if old_form:
            new_form = ConfigFormWidget(self.config, keys=self.display_keys)
            self.config_model = new_form
            layout = self.layout()
            idx = layout.indexOf(old_form)
            layout.insertWidget(idx, new_form)
            layout.removeWidget(old_form)
            old_form.deleteLater()


    # ============================== Public Class Methods ============================================ #

    def record(self):
        """Run the experimental procedure or fallback to legacy MDA sequence."""
        self._stop_live_streams()
        # If a procedure is available, use it for the experimental workflow
        if self.procedure is not None:
            try:
                # Run the procedure in a separate thread to avoid blocking the GUI
                self.procedure_thread = threading.Thread(target=self.procedure.run())
                self.procedure_thread.start()
                
                # Signal that recording has started
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.recordStarted.emit(timestamp)
                return
            except Exception as e:
                QMessageBox.critical(self, "Procedure Error", f"Failed to run procedure: {str(e)}")
                return

    def _abort(self):
        """Safely stop the running Procedure (stops hardware, saves data)."""
        if self.procedure is None:
            return
        self.abort_button.setEnabled(False)
        try:
            self.procedure.cleanup()
        except Exception as e:
            QMessageBox.critical(self, "Abort Error", f"Failed to abort procedure: {e}")

    def _on_run_started(self, *_args) -> None:
        self.record_button.setEnabled(False)
        self.abort_button.setEnabled(True)

    def _on_run_finished(self, *_args) -> None:
        self.record_button.setEnabled(True)
        self.abort_button.setEnabled(False)

    def _stop_live_streams(self) -> None:
        """Ensure any live/sequence streams are halted before starting acquisition."""
        cores = getattr(self.config, "_cores", ())
        for core in cores:
            with suppress(Exception):
                if hasattr(core, "isSequenceRunning") and core.isSequenceRunning():
                    core.stopSequenceAcquisition()

    #-----------------------------------------------------------------------------------------------#

    #============================== Private Class Methods ==========================================#

    def _open_bids_directory(self):
        """Open the BIDS directory in the system file explorer."""
        path = self.config.bids_dir
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Warning", "No BIDS directory selected or it does not exist.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _populate_subjects(self):
        self.subject_dropdown.clear()
        for sub in self.config.subjects.keys():
            self.subject_dropdown.addItem(sub)

    def _change_subject(self, index):
        subject_id = self.subject_dropdown.currentText()
        if not subject_id:
            return
        try:
            self.config.select_subject(subject_id)
        except Exception as e:
            print(e)
            return

        old_form = getattr(self, 'config_model', None)
        new_form = ConfigFormWidget(self.config, keys=self.display_keys)
        self.config_model = new_form
        if old_form:
            layout = self.layout()
            idx = layout.indexOf(old_form)
            layout.insertWidget(idx, new_form)
            layout.removeWidget(old_form)
            old_form.deleteLater()
        self._update_filename_preview()

    def _update_filename_preview(self):
        """Render the BIDS filename template for the currently selected subject."""
        if not hasattr(self, "filename_preview_label"):
            return
        subject = self.config.get("subject") or "?"
        session = self.config.get("session") or "?"
        task = self.config.get("task") or "?"
        template = (
            f"Filename template: "
            f"YYYYMMDD_HHMMSS_sub-{subject}_ses-{session}_task-{task}_<suffix>.<ext>"
        )
        self.filename_preview_label.setText(template)

    def _add_subject(self):
        """Prompt for a new subject ID, add it to the config, and select it."""
        subject_id, ok = QInputDialog.getText(self, "Add Subject", "Subject ID:")
        if not ok or not subject_id.strip():
            return
        try:
            self.config.add_subject(subject_id.strip())
        except ValueError as e:
            QMessageBox.warning(self, "Add Subject", str(e))
            return

        self.subject_dropdown.blockSignals(True)
        self._populate_subjects()
        self.subject_dropdown.blockSignals(False)
        idx = self.subject_dropdown.findText(subject_id.strip())
        if idx >= 0:
            self.subject_dropdown.setCurrentIndex(idx)
        else:
            self._change_subject(0)

    def _add_parameter(self):
        """Prompt for a new parameter (name, type, default) and apply to all subjects."""
        name, ok = QInputDialog.getText(self, "Add Parameter", "Parameter name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        type_label, ok = QInputDialog.getItem(
            self, "Add Parameter", f"Type for '{name}':", ["str", "int", "bool"], 0, False
        )
        if not ok:
            return

        if type_label == "int":
            value, ok = QInputDialog.getInt(self, "Add Parameter", f"Default value for '{name}':", 0)
            if not ok:
                return
            default = int(value)
            type_hint = int
        elif type_label == "bool":
            choice, ok = QInputDialog.getItem(
                self, "Add Parameter", f"Default value for '{name}':", ["False", "True"], 0, False
            )
            if not ok:
                return
            default = choice == "True"
            type_hint = bool
        else:
            text, ok = QInputDialog.getText(self, "Add Parameter", f"Default value for '{name}':")
            if not ok:
                return
            default = text
            type_hint = str

        try:
            self.config.add_parameter(name, default, type_hint)
        except ValueError as e:
            QMessageBox.warning(self, "Add Parameter", str(e))
            return

        self.set_display_keys(self.config.display_keys)

    def _save_config(self):
        """Persist current displayed values to experiment.json on demand."""
        path = getattr(self.config, "_json_file_path", "")
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Save", "No experiment.json is loaded.")
            return
        try:
            self.config.save_json()
        except Exception as e:
            QMessageBox.critical(self, "Save", f"Failed to save: {e}")
            return
        self.save_button.setToolTip(
            f"Saved {datetime.now().strftime('%H:%M:%S')} — {path}"
        )

    def _test_led(self):
        """
        Test the LED pattern by sending a test sequence to the Arduino-Switch device.
        """
        try:
            led_pattern = self.config.led_pattern
            cam = self.config.hardware.Dhyana
            if hasattr(cam, "start_led_sequence"):
                cam.start_led_sequence(led_pattern)
            else:
                cam.core.getPropertyObject('Arduino-Switch', 'State').loadSequence(led_pattern)
                cam.core.getPropertyObject('Arduino-Switch', 'State').setValue(4) # seems essential to initiate serial communication
                cam.core.getPropertyObject('Arduino-Switch', 'State').startSequence()
            print("LED test pattern sent successfully.")
        except Exception as e:
            print(f"Error testing LED pattern: {e}")
            
    def _stop_led(self):
        """
        Stop the LED pattern by sending a stop sequence to the Arduino-Switch device.
        """
        try:
            cam = self.config.hardware.Dhyana
            if hasattr(cam, "stop_led_sequence"):
                cam.stop_led_sequence()
            else:
                cam.core.getPropertyObject('Arduino-Switch', 'State').stopSequence()
            print("LED test pattern stopped successfully.")
        except Exception as e:
            print(f"Error stopping LED pattern: {e}")
    
    def _add_note(self):
        """
        Open a dialog to get a note from the user, save it to the
        ExperimentConfig.notes list, and push it onto the live DataQueue so it
        is timestamped into dataqueue.csv.
        """
        now = datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        text, ok = QInputDialog.getText(self, 'Add Note', 'Enter your note:')
        if ok and text:
            note_with_timestamp = f"{time}: {text}"
            self.config.notes.append(note_with_timestamp)
            data_manager = getattr(self.procedure, "data", None)
            if data_manager is not None and getattr(data_manager, "queue", None) is not None:
                data_manager.queue.push("notes", text, timestamp=now)

    def _test_nidaq(self):
        """
        PUlse the nidaq device to test its functionality.
        """
        self.procedure.config.hardware.get_device('nidaq').start()
    # ----------------------------------------------------------------------------------------------- #




