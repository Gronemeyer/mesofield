import os
import json
import subprocess
import time
from datetime import datetime
import threading
import numpy as np

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
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QDialog,
    QStyle,
    QFormLayout, 
    QLineEdit, 
    QSpinBox, 
    QCheckBox
)
from PyQt6.QtGui import QIcon
from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig
    from mesofield.protocols import Procedure

from mesofield.subprocesses.mouseportal import MousePortal

from mesofield.gui import ConfigTableModel
from .dynamic_controller import DynamicController

class ConfigFormWidget(QWidget):
    """Map each config key to an appropriate editor in a form layout."""

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
            if type_hint is int:
                editor = QSpinBox()
                editor.setRange(-1_000_000, 1_000_000)
                editor.setValue(int(value or 0))
                editor.valueChanged.connect(lambda val, k=key: self._registry.set(k, val))
            elif type_hint is bool:
                editor = QCheckBox()
                editor.setChecked(bool(value))
                editor.toggled.connect(lambda checked, k=key: self._registry.set(k, checked))
            elif type_hint is list: #create a dropdown for keys registered in the ExperimentConfig as list types
                editor = QComboBox()
                editor.addItems(value)
                editor.currentTextChanged.connect(lambda text, k=key: self._registry.set(k, text))
            else:
                editor = QLineEdit()
                editor.setText(str(value))
                editor.textChanged.connect(lambda text, k=key: self._registry.set(k, text))
            form.addRow(key, editor)

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
        self.config: ExperimentConfig = procedure.config #type: ignore
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
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.subject_dropdown_label)
        sub_layout.addWidget(self.subject_dropdown)
        layout.addLayout(sub_layout)

        self.config_model = ConfigFormWidget(self.procedure.config, keys=self.display_keys)
        
        self._populate_subjects()
        self._change_subject(0)
        
        # 4. Record button to start the MDA sequence
        self.record_button = QPushButton("Record")

        # Tint the standard play icon red
        play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        pix = play_icon.pixmap(24, 24)
        mask = pix.createMaskFromColor(Qt.transparent)
        pix.fill(Qt.GlobalColor.red)
        pix.setMask(mask)
        self.record_button.setIcon(QIcon(pix))

        # Use default background, no custom color
        self.record_button.setStyleSheet("""
            QPushButton {
            background-color: #424242; /* Dark Grey */
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            }
            QPushButton:hover {
            background-color: #616161;
            }
            QPushButton:pressed {
            background-color: #212121;
            }
        """)        
        layout.addWidget(self.record_button)
        self.record_button.setToolTip("Start Recording (MDA Sequence)")
        self.record_button.setShortcut("Ctrl+R")  # Set shortcut for recording

        # 5. Add Note button to add a note to the configuration
        self.add_note_button = QPushButton("Add Note")
        layout.addWidget(self.add_note_button)

        # Button to launch external MousePortal subprocess
        self.mouseportal_button = QPushButton("Launch MousePortal")
        layout.addWidget(self.mouseportal_button)

        # Simple prototype button to launch runportal.py directly
        self.prototype_portal_button = QPushButton("Launch Portal (Prototype)")
        self.prototype_portal_button.setStyleSheet("""
            QPushButton {
            background-color: #2E7D32; /* Green background */
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            }
            QPushButton:hover {
            background-color: #388E3C;
            }
            QPushButton:pressed {
            background-color: #1B5E20;
            }
        """)
        layout.addWidget(self.prototype_portal_button)

        # Keep reference to process instance
        self._mouseportal_process: Optional[MousePortal] = None
        self._prototype_process = None  # Can be subprocess.Popen, bool, or None

        # Dynamic hardware-specific controls
        self.dynamic_controller = DynamicController(self.procedure.config, parent=self)
        layout.addWidget(self.dynamic_controller)
        # ------------------------------------------------------------------------------------- #

        # ============ Callback connections between widget values and functions ================ #

        self.subject_dropdown.currentIndexChanged.connect(self._change_subject) # When the subject is changed, update the config form
        self.record_button.clicked.connect(self.record)
        self.add_note_button.clicked.connect(self._add_note)
        self.mouseportal_button.clicked.connect(self._toggle_mouseportal)
        self.prototype_portal_button.clicked.connect(self._toggle_prototype_portal)
        self.open_bids_button.clicked.connect(self._open_bids_directory)

        # Connect dynamic controls using constants defined in DynamicController
        dynamic_buttons = [
            (DynamicController.LED_TEST_BTN, self._test_led),
            (DynamicController.STOP_BTN, self._stop_led),
            (DynamicController.SNAP_BTN, lambda: self._save_snapshot(self._mmc1.snap())),
            (DynamicController.NIDAQ_BTN, self._test_nidaq),
            #(DynamicController.PSYCHOPY_BTN, self.launch_psychopy),
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

    def _test_led(self):
        """
        Test the LED pattern by sending a test sequence to the Arduino-Switch device.
        """
        try:
            led_pattern = self.config.led_pattern
            self.config.hardware.Dhyana.core.getPropertyObject('Arduino-Switch', 'State').loadSequence(led_pattern)
            self.config.hardware.Dhyana.core.getPropertyObject('Arduino-Switch', 'State').loadSequence(led_pattern)
            self.config.hardware.Dhyana.core.getPropertyObject('Arduino-Switch', 'State').setValue(4) # seems essential to initiate serial communication
            self.config.hardware.Dhyana.core.getPropertyObject('Arduino-Switch', 'State').startSequence()
            print("LED test pattern sent successfully.")
        except Exception as e:
            print(f"Error testing LED pattern: {e}")
            
    def _stop_led(self):
        """
        Stop the LED pattern by sending a stop sequence to the Arduino-Switch device.
        """
        try:
            self._mmc1.getPropertyObject('Arduino-Switch', 'State').stopSequence()
            print("LED test pattern stopped successfully.")
        except Exception as e:
            print(f"Error stopping LED pattern: {e}")
    
    def _add_note(self):
        """
        Open a dialog to get a note from the user and save it to the ExperimentConfig.notes list.
        """
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text, ok = QInputDialog.getText(self, 'Add Note', 'Enter your note:')
        if ok and text:
            note_with_timestamp = f"{time}: {text}"
            self.config.notes.append(note_with_timestamp)

    def _test_nidaq(self):
        """
        PUlse the nidaq device to test its functionality.
        """
        self.procedure.config.hardware.get_device('nidaq').start()

    def _toggle_mouseportal(self) -> None:
        """Launch or terminate the external MousePortal process."""
        if self._mouseportal_process is None:
            self._mouseportal_process = MousePortal(
                self.config, data_manager=getattr(self.procedure, "data", None)
            )

        if self._mouseportal_process.is_running:
            self._mouseportal_process.end()
            self.mouseportal_button.setText("Launch MousePortal")
        else:
            self._mouseportal_process.start()
            self.mouseportal_button.setText("End MousePortal")

    def _toggle_prototype_portal(self) -> None:
        """Minimal MousePortal launcher."""
        if self._prototype_process:
            # Send escape key to terminate MousePortal
            try:
                if hasattr(self._prototype_process, 'stdin') and self._prototype_process.stdin:
                    self._prototype_process.stdin.write(b'\x1b')  # ESC key
                    self._prototype_process.stdin.flush()
                self._prototype_process.terminate()  # Backup termination
            except:
                pass  # Ignore errors
            self._prototype_process = None
            self.prototype_portal_button.setText("Launch Portal (Prototype)")
            return
        
        # Get paths from config
        cfg = self.config.plugins["mouseportal"]["config"]
        python_exe = cfg["env_path"]
        script_path = cfg["script_path"]
        
        # Create config file
        runtime_path = os.path.join(os.getcwd(), "mouseportal_runtime.json")
        with open(runtime_path, "w") as f:
            json.dump({k: v for k, v in cfg.items() if k not in ["env_path", "script_path"]}, f)
        
        # Launch it (non-blocking) with stdin pipe for sending commands
        self._prototype_process = subprocess.Popen([python_exe, script_path, "--cfg", runtime_path], 
                                                  cwd=os.path.dirname(script_path),
                                                  creationflags=subprocess.CREATE_NEW_CONSOLE,
                                                  stdin=subprocess.PIPE)
        
        self.prototype_portal_button.setText("Stop Portal (Prototype)")
    # ----------------------------------------------------------------------------------------------- #




