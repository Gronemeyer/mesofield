import os
import sys
import json
import warnings
import yaml
import datetime

import pathlib
import pandas as pd

try:
    from pymmcore_plus import CMMCorePlus
    import useq
    from mesofield.engines import DevEngine, MesoEngine, PupilEngine
    from mesofield.io.encoder import SerialWorker
    from mesofield.io.arducam import VideoThread
    from mesofield.io import SerialWorker
except ImportError:
    # These are not required for unpickling.
    pass

from mesofield.hardware import HardwareManager

class ExperimentConfig:
    """## Generate and store parameters loaded from a JSON file. 
    
    #### Example Usage:
    ```python
    config = ExperimentConfig()
    # create dict and pandas DataFrame from JSON file path:
    config.load_parameters('path/to/json_file.json')
        
    config._save_dir = './output'
    config.subject = '001'
    config.task = 'TestTask'
    config.notes.append('This is a test note.')

    # Update a parameter
    config.update_parameter('new_param', 'test_value')

    # Save parameters and notes
    config.save_parameters()
    ```
    """

    def __init__(self, path: str):
        self._parameters: dict = {}
        self._json_file_path = ''
        self._output_path = ''
        self._save_dir = ''
        self.module_path = r'C:/dev/mesofield'

        self.hardware = HardwareManager(path)
        
        self.notes: list = []    

    @property
    def _cores(self):# -> tuple[CMMCorePlus, ...]:
        """Return the tuple of CMMCorePlus instances from the hardware cameras."""
        return tuple(cam.core for cam in self.hardware.cameras if hasattr(cam, 'core'))

    @property
    def encoder(self):# -> SerialWorker:
        return self.hardware.encoder

    @property
    def save_dir(self) -> str:
        return os.path.join(self._save_dir, 'data')

    @save_dir.setter
    def save_dir(self, path: str):
        if isinstance(path, str):
            self._save_dir = os.path.abspath(path)
        else:
            print(f"ExperimentConfig: \n Invalid save directory path: {path}")

    @property
    def subject(self) -> str:
        return self._parameters.get('subject', 'sub')

    @property
    def session(self) -> str:
        return self._parameters.get('session', 'ses')

    @property
    def task(self) -> str:
        return self._parameters.get('task', 'task')

    @property
    def start_on_trigger(self) -> bool:
        return self._parameters.get('start_on_trigger', False)
    
    @property
    def sequence_duration(self) -> int:
        return int(self._parameters.get('duration', 60))
    
    @property
    def trial_duration(self) -> int:
        return int(self._parameters.get('trial_duration', None))
    
    @property
    def num_meso_frames(self) -> int:
        return int(self.hardware.Dhyana.fps * self.sequence_duration) 
    
    @property
    def num_pupil_frames(self) -> int:
        return int((self.hardware.ThorCam.fps * self.sequence_duration)) + 100 
    
    @property
    def num_trials(self) -> int:
        return int(self.sequence_duration / self.trial_duration)  
    
    @property
    def parameters(self) -> dict:
        return self._parameters
    
    @property
    def meso_sequence(self) -> useq.MDASequence:
        return useq.MDASequence(time_plan={"interval": 0, "loops": self.num_meso_frames})
    
    @property
    def pupil_sequence(self) -> useq.MDASequence:
        return useq.MDASequence(time_plan={"interval": 0, "loops": self.num_pupil_frames})
    
    @property
    def bids_dir(self) -> str:
        """ Dynamic construct of BIDS directory path """
        bids = os.path.join(
            f"sub-{self.subject}",
            f"ses-{self.session}",
        )
        return os.path.abspath(os.path.join(self.save_dir, bids))

    # Property to compute the full file path, handling existing files
    @property
    def meso_file_path(self):
        return self._generate_unique_file_path(suffix="meso", extension="ome.tiff", bids_type="func")

    # Property for pupil file path, if needed
    @property
    def pupil_file_path(self):
        return self._generate_unique_file_path(suffix="pupil", extension="ome.tiff", bids_type="func")

    @property
    def notes_file_path(self):
        return self._generate_unique_file_path(suffix="notes", extension="txt")
    
    @property
    def encoder_file_path(self):
        return self._generate_unique_file_path(suffix="encoder-data", extension="csv", bids_type='beh')
    
    @property
    def dataframe(self):
        data = {'Parameter': list(self._parameters.keys()),
                'Value': list(self._parameters.values())}
        return pd.DataFrame(data)
    
    @property
    def json_path(self):
        return self._json_file_path
    
    @property
    def psychopy_filename(self) -> str:
        """Path to PsychoPy script within the Experiment Directory.

        NOTE: Requires the JSON file to have a 'psychopy_filename' key
        
        Returns:
            str: path to psychopy.py experiment script
        """
        
        filename = self._parameters.get('psychopy_filename', 'experiment.py')
        # in case the .json file does not include the .py file extension (silly but probable)
        if not filename.endswith('.py'):
            filename += '.py'
        # search for the psychopy filename in the Experiment directory (ie. where the JSON file was loaded from)
        for root, _, files in os.walk(self._save_dir):
            if filename in files:
                return os.path.join(root, filename)
        return os.path.join(self._save_dir, filename)
    
    @psychopy_filename.setter
    def psychopy_filename(self, value: str) -> None:
        self._parameters['psychopy_filename'] = value

    @property
    def psychopy_path(self) -> str:
        return os.path.join(self._save_dir, self.psychopy_filename)
    
    @property
    def psychopy_save_path(self) -> str:
        return os.path.join(self._save_dir, f"data/sub-{self.subject}/ses-{self.session}/beh/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sub-{self.subject}_ses-{self.session}_task-{self.task}_psychopy")
    
    @property
    def psychopy_parameters(self) -> dict:
        return {
            'subject': self.subject,
            'session': self.session,
            'save_dir': self.save_dir,
            'num_trials': self.num_trials,
            'save_path': self.psychopy_save_path
        }
    
    @property
    def led_pattern(self) -> list[str]:
        return self._parameters.get('led_pattern', ['4', '4', '2', '2'])
    
    @led_pattern.setter
    def led_pattern(self, value: list) -> None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("led_pattern string must be a valid JSON list")
        if isinstance(value, list):
            self._parameters['led_pattern'] = [str(item) for item in value]
        else:
            raise ValueError("led_pattern must be a list or a JSON string representing a list")
    
    # Helper method to generate a unique file path
    def _generate_unique_file_path(self, suffix: str, extension: str, bids_type: str = None):
        """ Generates unique filename with timestamp and BIDS indentifiers
        
        ```python
            ExperimentConfig._generate_unique_file_path("images", "jpg", "func")
            print(unique_path)
            `20250110_123456_sub-001_ses-01_task-example_images.jpg`
        ```
        """
        file = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sub-{self.subject}_ses-{self.session}_task-{self.task}_{suffix}.{extension}"

        if bids_type is None:
            bids_path = self.bids_dir
        else:
            bids_path = os.path.join(self.bids_dir, bids_type)
            
        os.makedirs(bids_path, exist_ok=True)
        base, ext = os.path.splitext(file)
        counter = 1
        file_path = os.path.join(bids_path, file)
        while os.path.exists(file_path):
            file_path = os.path.join(bids_path, f"{base}_{counter}{ext}")
            counter += 1
        return file_path
        
    def load_parameters(self, json_file_path) -> None:
        """ Load parameters from a JSON file path into the config object. 
        """
        self._json_file_path = json_file_path 
        self.save_dir = os.path.dirname(os.path.abspath(json_file_path))
        try:
            with open(json_file_path, 'r') as f: 
                self._parameters = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    def update_parameter(self, key, value) -> None:
        self._parameters[key] = value
        
    def list_parameters(self) -> pd.DataFrame:
        """ Create a DataFrame from the ExperimentConfig properties 
        """
        properties = [prop for prop in dir(self.__class__) if isinstance(getattr(self.__class__, prop), property)]
        exclude_properties = {'dataframe', 'parameters', 'json_path', "_cores", "meso_sequence", "pupil_sequence", "psychopy_path", "encoder"}
        data = {prop: getattr(self, prop) for prop in properties if prop not in exclude_properties}
        return pd.DataFrame(data.items(), columns=['Parameter', 'Value'])
                
    def save_wheel_encoder_data(self, data):
        """ Save the wheel encoder data to a CSV file 
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)
           
        try:
            data.to_csv(self.encoder_file_path, index=False)
            print(f"Encoder data saved to {self.encoder_file_path}")
        except Exception as e:
            print(f"Error saving encoder data: {e}")
            
    def save_configuration(self):
        """ Save the configuration parameters to a CSV file 
        """
        params_path = self._generate_unique_file_path(suffix="configuration", extension="csv")
        try:
            params = self.list_parameters()
            params.to_csv(params_path, index=False)
            print(f"Configuration saved to {params_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
            
        try:
            with open(self.notes_file_path, 'w') as f:
                f.write('\n'.join(self.notes))
                print(f"Notes saved to {self.notes_file_path}")
        except Exception as e:
            print(f"Error saving notes: {e}")
            
class ConfigLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.parameters = self.load_parameters(file_path)
        self.set_attributes()

    def load_parameters(self, file_path: str) -> dict:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith(('.yaml', '.yml')):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format")

    def set_attributes(self):
        for key, value in self.parameters.items():
            setattr(self, key, value)

