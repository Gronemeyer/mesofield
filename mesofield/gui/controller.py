import numpy as np
import os
import subprocess #for PsychoPy Subprocess
import datetime

from qtpy.QtCore import Qt
from PyQt6.QtCore import pyqtSignal, QProcess
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTableWidget,
    QHeaderView,
    QFileDialog,
    QTableWidgetItem,
    QMessageBox,
    QInputDialog,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QTabWidget,
)
from PyQt6.QtGui import QImage, QPixmap


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig
    from pymmcore_plus import CMMCorePlus

class ConfigController(QWidget):
    """AcquisitionEngine object for the napari-mesofield plugin.
    The object connects to the Micro-Manager Core object instances and the Config object.

    The ConfigController widget is a QWidget that allows the user to select a save directory,
    load a JSON configuration file, and edit the configuration parameters in a table.
    The user can also trigger the MDA sequence with the current configuration parameters.
    
    The ConfigController widget emits signals to notify other widgets when the configuration is updated
    and when the record button is pressed.
    
    Public Methods:
    ----------------
    save_config(): 
        saves the current configuration to a JSON file
    
    record(): 
        triggers the MDA sequence with the configuration parameters
    
    launch_psychopy(): 
        launches the PsychoPy experiment as a subprocess with ExperimentConfig parameters
    
    show_popup(): 
        shows a popup message to the user
    
    Private Methods:
    ----------------
    _select_directory(): 
        opens a dialog to select a directory and update the GUI accordingly
    _get_json_file_choices(): 
        returns a list of JSON files in the current directory
    _update_config(): 
        updates the experiment configuration from a new JSON file
    _on_table_edit(): 
        updates the configuration parameters when the table is edited
    _refresh_config_table(): 
        refreshes the configuration table to reflect current parameters
    _test_led(): 
        tests the LED pattern by sending a test sequence to the Arduino-Switch device
    _stop_led(): 
        stops the LED pattern by sending a stop sequence to the Arduino-Switch device
    _add_note(): 
        opens a dialog to get a note from the user and save it to the ExperimentConfig.notes list
    
    """
    # ==================================== Signals ===================================== #
    configUpdated = pyqtSignal(object)
    recordStarted = pyqtSignal()
    # ------------------------------------------------------------------------------------- #
    
    def __init__(self, cfg: 'ExperimentConfig'):
        super().__init__()
        self.mmcores = cfg._cores
        # TODO: Add a check for the number of cores, and adjust rest of controller accordingly

        self.config = cfg
        if len(self.mmcores) == 1:
            self._mmc: CMMCorePlus = self.mmcores[0]
        elif len(self.mmcores) == 2:
            self._mmc1: CMMCorePlus = self.mmcores[0]
            self._mmc2: CMMCorePlus = self.mmcores[1]

        self.psychopy_process = None

        # Create main layout
        layout = QVBoxLayout(self)
        self.setFixedWidth(500)

        # ==================================== GUI Widgets ===================================== #

        # 1. Selecting a save directory
        self.directory_label = QLabel('Select Save Directory:')
        self.directory_line_edit = QLineEdit()
        self.directory_line_edit.setReadOnly(True)
        self.directory_button = QPushButton('Browse')

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.directory_label)
        dir_layout.addWidget(self.directory_line_edit)
        dir_layout.addWidget(self.directory_button)

        layout.addLayout(dir_layout)

        # 2. Dropdown Widget for JSON configuration files
        self.json_dropdown_label = QLabel('Select JSON Config:')
        self.json_dropdown = QComboBox()

        json_layout = QHBoxLayout()
        json_layout.addWidget(self.json_dropdown_label)
        json_layout.addWidget(self.json_dropdown)

        layout.addLayout(json_layout)

        # 3. Table widget to display the configuration parameters loaded from the JSON
        config_label = QLabel('Experiment Config:')
        config_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(config_label)
        
        # Add description text
        description = QLabel("Edit parameters directly in the table below, or use the Advanced Editor for specialized widgets")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add button for advanced parameter editor
        self.advanced_editor_button = QPushButton("Advanced Parameter Editor")
        layout.addWidget(self.advanced_editor_button)
        self.advanced_editor_button.clicked.connect(self.show_advanced_editor)
        
        # Configuration table
        self.config_table = QTableWidget()
        self.config_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.config_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.config_table)

        # 4. Record button to start the MDA sequence
        self.record_button = QPushButton('Record')
        layout.addWidget(self.record_button)
        
        # 5. Test LED button to test the LED pattern
        self.test_led_button = QPushButton("Test LED")
        layout.addWidget(self.test_led_button)
        
        # 6. Stop LED button to stop the LED pattern
        self.stop_led_button = QPushButton("Stop LED")
        layout.addWidget(self.stop_led_button)
        
        # 7. Add Note button to add a note to the configuration
        self.add_note_button = QPushButton("Add Note")
        layout.addWidget(self.add_note_button)

        # 7. Add a snap image button for self._mmc1.snap() 
        self.snap_button = QPushButton("Snap Image")
        layout.addWidget(self.snap_button)

        # ------------------------------------------------------------------------------------- #

        # ============ Callback connections between widget values and functions ================ #

        self.directory_button.clicked.connect(self._select_directory)
        self.json_dropdown.currentIndexChanged.connect(self._update_config)
        self.config_table.cellChanged.connect(self._on_table_edit)
        self.record_button.clicked.connect(self.record)
        self.test_led_button.clicked.connect(self._test_led)
        self.stop_led_button.clicked.connect(self._stop_led)
        self.add_note_button.clicked.connect(self._add_note)
        self.snap_button.clicked.connect(lambda: self._save_snapshot(self._mmc1.snap()))

        # ------------------------------------------------------------------------------------- #

        # Initialize the config table
        self._refresh_config_table()

    # ============================== Public Class Methods ============================================ #

    def _save_snapshot(self, image: np.ndarray):
        """Creates a PyQt popup window for saving the snapped image."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        dialog = QDialog(self)
        dialog.setWindowTitle("Save Snapped Image")
        layout = QVBoxLayout(dialog)

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        layout.addWidget(canvas)
        
        # Save button
        save_button = QPushButton("Save", dialog)
        layout.addWidget(save_button)

        save_button.clicked.connect(lambda: self._save_image(image, dialog))

        dialog.exec()

    def _save_image(self, image: np.ndarray, dialog: QDialog):
        """Save the snapped image to the specified directory with a unique filename."""

        # Generate a unique filename with a timestamp
        file_path = self.config.make_path(suffix="snapped", extension="png", bids_type="func")

        # Save the image as a PNG file using matplotlib
        import matplotlib.pyplot as plt

        plt.imsave(file_path, image, cmap='gray')

        # Close the dialog
        dialog.accept()

    def record(self):
        """Run the MDA sequence with the global Config object parameters loaded from JSON."""
        from mesofield.io import CustomWriter
        import threading
        import useq
        # TODO: Add a check for the MDA sequence and pupil sequence
        # TODO: Fix this ugly logic :)
        if len(self.mmcores) == 1:
            pupil_sequence = useq.MDASequence(metadata=self.config.hardware.nidaq.__dict__,
                                            time_plan={"interval": 0, "loops": self.config.num_pupil_frames})
            thread = threading.Thread(target=self._mmc.run_mda, args=(pupil_sequence,), kwargs={'output': CustomWriter(self.config.make_path("pupil", "ome.tiff", bids_type="func"))})
        elif len(self.mmcores) == 2:        
            thread1 = threading.Thread(target=self._mmc1.run_mda, 
                                       args=(self.config.meso_sequence,), 
                                       kwargs={'output': CustomWriter(self.config.make_path("meso", "ome.tiff", bids_type="func"))})
            thread2 = threading.Thread(target=self._mmc2.run_mda, 
                                       args=(self.config.pupil_sequence,), 
                                       kwargs={'output': CustomWriter(self.config.make_path("pupil", "ome.tiff", bids_type="func"))})

        # Wait for spacebar press if start_on_trigger is True
        if self.config.start_on_trigger == "True":
            self.launch_psychopy()
            self.show_popup()

        if len(self.mmcores) == 1:
            thread.start()
        elif len(self.mmcores) == 2:        
            thread1.start()
            thread2.start()
        self.config.hardware.encoder.start()
        self.recordStarted.emit() # Signals to start the MDA sequence to notify other widgets

    def launch_psychopy(self):
        """Launches a PsychoPy experiment as a subprocess with the current ExperimentConfig parameters."""
        from mesofield.subprocesses import psychopy

        self.psychopy_process = psychopy.launch(self.config, self)
        self.psychopy_process.finished.connect(self._handle_process_finished)

    def _handle_process_finished(self, exit_code, exit_status):
        from PyQt6.QtCore import QProcess

        """Handle the finished state of the PsychoPy subprocess."""
        if self.psychopy_process.state() != QProcess.NotRunning:
            self.psychopy_process.kill()
            self.psychopy_process = None
        self.psychopy_process.deleteLater()
        print(f"PsychoPy process finished with exit code {exit_code} and exit status {exit_status}")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.event_loop.quit()
    
    def show_popup(self):
        msg_box = QMessageBox()
        msg_box.setText("Press spacebar to start recording.")
        #msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    def save_config(self):
        """ Save the current configuration to a JSON file """
        self.config.save_parameters()
        
    #TODO: add breakdown method

    #-----------------------------------------------------------------------------------------------#
    
    #============================== Private Class Methods ==========================================#

    def _select_directory(self):
        """Open a dialog to select a directory and update the GUI accordingly."""
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.directory_line_edit.setText(directory)
            self._get_json_file_choices(directory)

    def _get_json_file_choices(self, path):
        """Return a list of JSON files in the current directory."""
        import glob
        self.config.save_dir = path
        try:
            json_files = glob.glob(os.path.join(path, "*.json"))
            self.json_dropdown.clear()
            self.json_dropdown.addItems(json_files)
        except Exception as e:
            print(f"Error getting JSON files from directory: {path}\n{e}")

    def _update_config(self, index):
        """Update the experiment configuration from a new JSON file."""
        json_path_input = self.json_dropdown.currentText()

        if json_path_input and os.path.isfile(json_path_input):
            try:
                # Load the JSON configuration
                self.config.load_json(json_path_input)
                
                # Setup configuration (configures engines, etc.)
                self.config.setup_configuration(json_path_input)
                
                # Refresh the GUI table with the new parameters
                self._refresh_config_table()
                
                # Show a success message
                QMessageBox.information(self, "Configuration Loaded", 
                                      f"Successfully loaded configuration from:\n{json_path_input}")
            except Exception as e:
                error_msg = f"Trouble updating ExperimentConfig:\n{json_path_input}\nConfiguration not updated."
                print(f"{error_msg}\n{e}")
                QMessageBox.critical(self, "Error Loading Configuration", error_msg)

    def _on_table_edit(self, row, column):
        """Update the configuration parameters when the table is edited."""
        # Only process edits to the value column (column 1)
        if column != 1:
            return
            
        try:
            # Get parameter key and new value
            if self.config_table.item(row, 0) and self.config_table.item(row, 1):
                key = self.config_table.item(row, 0).text()
                value_str = self.config_table.item(row, 1).text()
                
                # Get parameter type and try to convert the value accordingly
                type_str = self.config_table.item(row, 2).text() if self.config_table.item(row, 2) else None
                
                # Convert the value string based on its type
                if type_str == 'bool':
                    # Handle boolean values
                    value = value_str.lower() in ('true', 'yes', 't', 'y', '1')
                elif type_str == 'int':
                    # Handle integer values
                    try:
                        value = int(value_str)
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Value", 
                                          f"'{value_str}' is not a valid integer for parameter '{key}'.")
                        return
                elif type_str == 'float':
                    # Handle float values
                    try:
                        value = float(value_str)
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Value", 
                                          f"'{value_str}' is not a valid float for parameter '{key}'.")
                        return
                elif type_str == 'list':
                    # Handle list values
                    try:
                        import json
                        value = json.loads(value_str)
                        if not isinstance(value, list):
                            value = [value]
                    except json.JSONDecodeError:
                        # If not valid JSON, try simple comma-separated list
                        value = [item.strip() for item in value_str.split(',')]
                else:
                    # For all other types, use the string value as is
                    value = value_str
                
                # Update the parameter directly through the dynamic property system
                setattr(self.config, key, value)
                
                # Emit signal to notify other components of the change
                self.configUpdated.emit(self.config)
                
        except Exception as e:
            error_msg = f"Error updating parameter: {e}"
            print(error_msg)
            QMessageBox.critical(self, "Error Updating Parameter", error_msg)

    def _refresh_config_table(self):
        """Refresh the configuration table to reflect current parameters."""
        # Get UI schema which contains enhanced parameter information
        schema = self.config.get_ui_schema()
        
        # Convert schema to a DataFrame-like structure for display
        parameters = []
        values = []
        descriptions = []
        types = []
        
        for key, info in schema.items():
            parameters.append(key)
            values.append(str(info['value']))
            descriptions.append(info['description'])
            types.append(info['type'])
        
        # Prepare table
        self.config_table.blockSignals(True)  # Prevent signals while updating the table
        self.config_table.clear()
        self.config_table.setRowCount(len(parameters))
        self.config_table.setColumnCount(4)  # Parameter, Value, Type, Description
        self.config_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Description"])
        
        # Fill table with data
        for i in range(len(parameters)):
            # Parameter name - read only
            param_item = QTableWidgetItem(parameters[i])
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.config_table.setItem(i, 0, param_item)
            
            # Value - editable
            value_item = QTableWidgetItem(values[i])
            self.config_table.setItem(i, 1, value_item)
            
            # Type - read only
            type_item = QTableWidgetItem(types[i])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.config_table.setItem(i, 2, type_item)
            
            # Description - read only
            desc_item = QTableWidgetItem(descriptions[i])
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.config_table.setItem(i, 3, desc_item)
        
        # Resize columns to content
        self.config_table.resizeColumnsToContents()
        self.config_table.blockSignals(False)  # Re-enable signals
        
        # Emit signal to listeners
        self.configUpdated.emit(self.config)
        
    def _test_led(self):
        """
        Test the LED pattern by sending a test sequence to the Arduino-Switch device.
        """
        try:
            led_pattern = self.config.led_pattern
            self.config.hardware.Dhyana.core.getPropertyObject('Arduino-Switch', 'State').loadSequence(led_pattern)
            self._mmc1.getPropertyObject('Arduino-Switch', 'State').loadSequence(led_pattern)
            self._mmc1.getPropertyObject('Arduino-Switch', 'State').setValue(4) # seems essential to initiate serial communication
            self._mmc1.getPropertyObject('Arduino-Switch', 'State').startSequence()
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
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text, ok = QInputDialog.getText(self, 'Add Note', 'Enter your note:')
        if ok and text:
            note_with_timestamp = f"{time}: {text}"
            self.config.notes.append(note_with_timestamp)

    def show_advanced_editor(self):
        """
        Show the advanced parameter editor dialog.
        """
        dialog = ParameterEditorDialog(self.config, self)
        if dialog.exec() == QDialog.Accepted:
            # Refresh the config table to reflect any changes
            self._refresh_config_table()
            # Notify other components
            self.configUpdated.emit(self.config)

    # ----------------------------------------------------------------------------------------------- #

class ParameterEditorDialog(QDialog):
    """
    Advanced parameter editor dialog that creates appropriate widgets for
    different parameter types based on the UI schema.
    """
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.schema = config.get_ui_schema()
        self.widgets = {}
        
        self.setWindowTitle("Advanced Parameter Editor")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Add tabs for different parameter categories
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Group parameters by category
        categories = {}
        for key, info in self.schema.items():
            category = info.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append((key, info))
        
        # Create a tab for each category
        for category, params in categories.items():
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Add scrollable area for many parameters
            scroll = QWidget()
            scroll_layout = QVBoxLayout(scroll)
            
            # Add parameter widgets
            for key, info in params:
                self._add_parameter_widget(scroll_layout, key, info)
            
            # Make scroll area scrollable if needed
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(scroll)
            
            tab_layout.addWidget(scroll_area)
            self.tabs.addTab(tab, category.capitalize())
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
    
    def _add_parameter_widget(self, layout, key, info):
        """
        Add an appropriate widget for the parameter based on its UI schema.
        """
        # Create container for the parameter
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 10, 0, 10)
        
        # Create label with parameter name and description
        label = QLabel(f"<b>{key}</b>")
        if info.get('description'):
            label.setToolTip(info.get('description'))
        container_layout.addWidget(label)
        
        # Create appropriate widget based on ui_widget type
        widget_type = info.get('ui_widget', 'text')
        value = info.get('value')
        
        if widget_type == 'check':
            # Boolean checkbox
            widget = QCheckBox("Enabled")
            widget.setChecked(bool(value))
            widget.stateChanged.connect(lambda state, k=key: self._on_check_changed(k, state))
        
        elif widget_type == 'combo':
            # Dropdown select
            widget = QComboBox()
            options = info.get('options', [])
            if options:
                widget.addItems([str(opt) for opt in options])
                if value is not None:
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
            widget.currentTextChanged.connect(lambda text, k=key: self._on_combo_changed(k, text))
        
        elif widget_type == 'spin':
            # Numeric spinner
            if isinstance(value, float) or info.get('type') == float:
                widget = QDoubleSpinBox()
                widget.setDecimals(3)
                step = info.get('step', 0.1)
            else:
                widget = QSpinBox()
                step = info.get('step', 1)
            
            # Set limits if specified
            min_val = info.get('min_value')
            if min_val is not None:
                widget.setMinimum(float(min_val) if isinstance(min_val, str) else min_val)
            else:
                widget.setMinimum(-1000000)
                
            max_val = info.get('max_value')
            if max_val is not None:
                widget.setMaximum(float(max_val) if isinstance(max_val, str) else max_val)
            else:
                widget.setMaximum(1000000)
            
            # Set step size
            widget.setSingleStep(float(step) if isinstance(step, str) else step)
            
            # Set current value
            if value is not None:
                widget.setValue(float(value) if isinstance(value, str) else value)
            
            widget.valueChanged.connect(lambda val, k=key: self._on_spin_changed(k, val))
        
        elif widget_type == 'list':
            # List editor
            widget = QLineEdit()
            if isinstance(value, list):
                # Convert list to comma-separated string
                widget.setText(", ".join(str(item) for item in value))
            else:
                widget.setText(str(value))
            widget.editingFinished.connect(lambda k=key, w=widget: self._on_list_changed(k, w.text()))
        
        else:
            # Default text input
            widget = QLineEdit()
            widget.setText(str(value) if value is not None else "")
            widget.editingFinished.connect(lambda k=key, w=widget: self._on_text_changed(k, w.text()))
        
        # Add widget to container
        container_layout.addWidget(widget)
        
        # If there's a description, add it as a label
        if info.get('description'):
            desc_label = QLabel(info.get('description'))
            desc_label.setStyleSheet("color: gray; font-size: 10px;")
            desc_label.setWordWrap(True)
            container_layout.addWidget(desc_label)
        
        # Store widget for later reference
        self.widgets[key] = widget
        
        # Add container to layout
        layout.addWidget(container)
    
    def _on_check_changed(self, key, state):
        """Handle checkbox state change."""
        value = state == Qt.CheckState.Checked
        setattr(self.config, key, value)
    
    def _on_combo_changed(self, key, text):
        """Handle combobox selection change."""
        setattr(self.config, key, text)
    
    def _on_spin_changed(self, key, value):
        """Handle spinner value change."""
        setattr(self.config, key, value)
    
    def _on_text_changed(self, key, text):
        """Handle text field editing."""
        setattr(self.config, key, text)
    
    def _on_list_changed(self, key, text):
        """Handle list field editing."""
        # Parse comma-separated list
        items = [item.strip() for item in text.split(',')]
        setattr(self.config, key, items)
    
    def accept(self):
        """Save all changes when dialog is accepted."""
        # All changes have already been applied during editing
        super().accept()
        
    def reject(self):
        """Discard changes when dialog is canceled."""
        super().reject()




