import sys
import pytest
#pytest.skip("Skipping GUI dependent tests", allow_module_level=True)
from PyQt6.QtWidgets import (
    QWidget,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QFileDialog,
)
from PyQt6.QtCore import QProcess

from mesofield.config import ExperimentConfig
from mesofield.devices.subprocesses import psychopy 
import sys
import pickle
import base64
import pickle, base64
import os

""" This test_script is used to test a launching a prebuilt Psychopy experiment script 
as a subprocess which epxects parameters piped from a parent process. 

Below is the custom-code in a Psychopy experiment that accepts the sysargs:

# Run 'Before Experiment' code from get_input_arguments
#=============================== Custom Codeblock jgronemeyer =====================================#
#add system argument functionality
#Get command line arguments passed if present
import sys
import dill
import base64

sys.path.append(r'C:/dev/mesofield') # Add the path to the mesofield package for the config object
b64_serialized_config = sys.argv[1] #Get the base64 encoded serialized config from the command line
serialized_config = base64.b64decode(b64_serialized_config) # Decode it from base64 to the original serialized bytes
config = dill.loads(serialized_config) # Use dill to load the ExperimentConfig object

# Debugging: print the configuration
print("Decoded ExperimentConfig object:")
print(config)

sysarg_subject_id = config.subject  
sysarg_session_id = config.session  
sysarg_save_dir = config.save_dir 
nTrials = config.num_trials
sysarg_save_path = config.save_path
#==================================================================================================#
"""

def launch_experiment(python_path, experiment_path, subject, session, save_dir, num_trials, filename):
    
    args = [
        f'{python_path}',
        f'{experiment_path}',
        f'{subject}',
        f'{session}',
        f'{save_dir}',
        f'{num_trials}',
        f'{filename}'
    ]
    
    # Create and start the QProcess
    psychopy_process = QProcess()
    psychopy_process.start(args[0], args[1:])
    
    return psychopy_process

def launch_experiment_pickle(cfg: ExperimentConfig):
    """
    Launch the experiment by passing a pickled ExperimentConfig via the command line.

    The ExperimentConfig object is serialized with pickle and encoded in base64 to be safely
    delivered as a single command line argument.

    Example for the subprocess to unpickle the config:
    ```python
    if len(sys.argv) > 1:
        # Assume the pickled config is the second argument (first argument after the script)
        pickled_cfg_str = sys.argv[1]
        cfg = pickle.loads(base64.b64decode(pickled_cfg_str))
        # Now use 'cfg' as your configuration object
    ```
    """

    # Serialize and encode the ExperimentConfig
    pickled_bytes = pickle.dumps(cfg)
    pickled_str = base64.b64encode(pickled_bytes).decode('utf-8')

    # Use the python and experiment paths stored in the configuration
    args = [
        cfg.python_path,      # Path to the Python interpreter
        cfg.experiment_path,  # Path to the experiment script
        pickled_str           # Encoded pickled ExperimentConfig object
    ]

    psychopy_process = QProcess()
    psychopy_process.start(args[0], args[1:])

    return psychopy_process

class PsychopyGui(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PsychoPy Launcher")
        form_layout = QFormLayout()

        self.python_path_edit = QLineEdit(r"C:\\Program Files\\PsychoPy\\python.exe")
        self.experiment_path_edit = QLineEdit(r"D:\Experiment Types\Checkerboard Experiment\CheckerBar_vis_build-v0.8.py")
        self.subject_edit = QLineEdit("Subject1")
        self.session_edit = QLineEdit("Session1")
        self.save_dir_edit = QLineEdit(r"C:\\")
        self.trials_edit = QLineEdit("10")
        self.file_name_edit = QLineEdit(r"D:\Experiment Types\Checkerboard Experiment\data\filename_test2")

        form_layout.addRow("Python Path:", self.python_path_edit)
        form_layout.addRow("Experiment Path:", self.experiment_path_edit)
        form_layout.addRow("Subject:", self.subject_edit)
        form_layout.addRow("Session:", self.session_edit)
        form_layout.addRow("Save Dir:", self.save_dir_edit)
        form_layout.addRow("Num Trials:", self.trials_edit)
        form_layout.addRow("File Name:", self.file_name_edit)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.on_run_clicked)
        form_layout.addRow(run_button)

        self.setLayout(form_layout)

    def on_run_clicked(self):
        python_path = self.python_path_edit.text()
        experiment_path = self.experiment_path_edit.text()
        subject = self.subject_edit.text()
        session = self.session_edit.text()
        save_dir = self.save_dir_edit.text()
        num_trials = self.trials_edit.text()
        filename = self.file_name_edit.text()

        try:
            self.process = launch_experiment(
                python_path,
                experiment_path,
                subject,
                session,
                save_dir,
                num_trials,
                filename
            )
        except Exception as e:
            print(f"Error launching PsychoPy: {e}")

class DillPsychopyGui(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initConnections()

    def initUI(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'dev.yaml')
        self.config = ExperimentConfig(yaml_path)

        self.setWindowTitle("PsychoPy Launcher")
        layout = QHBoxLayout()
        
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
        
        layout.addWidget(self.json_dropdown_label)
        layout.addWidget(self.json_dropdown)
        layout.addWidget(self.directory_line_edit)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(run_button)

        self.setLayout(layout)
    
    def initConnections(self):
        self.json_dropdown.currentIndexChanged.connect(self._update_config)
        self.directory_button.clicked.connect(self._select_directory)
    
    def _select_directory(self):
        """Open a dialog to select a directory and update the GUI accordingly."""
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.directory_line_edit.setText(directory)
            self._get_json_file_choices(directory)


    def _get_json_file_choices(self, path):
        """Return a list of JSON files in the current directory."""
        import glob
        try:
            json_files = glob.glob(os.path.join(path, "*.json"))
            self.json_dropdown.clear()
            self.json_dropdown.addItems(json_files)
        except Exception as e:
            print(f"Error getting JSON files from directory: {path}\n{e}")
        self.config.save_dir = path


    def _update_config(self, index):
        """Update the experiment configuration from a new JSON file."""
        self.config.load_json(self.json_dropdown.currentText())

    def on_run_clicked(self):
        self.psychopy_process = psychopy.launch(self.config, self)


def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    gui = DillPsychopyGui()
    gui.show()
    sys.exit(app.exec())
    
    # process = launch_experiment(
    #     python_path=r"C:\\Program Files\\PsychoPy\\python.exe",
    #     experiment_path=r"D:\Experiment Types\Checkerboard Experiment\CheckerBar_vis_build-v0.8.py",
    #     subject="Subject1",
    #     session="Session1",
    #     save_dir=r"C:\\",
    #     num_trials="1",
    #     filename=r"D:\Experiment Types\Checkerboard Experiment\data\filename_test2"
    #     )
    


if __name__ == "__main__":
    main()