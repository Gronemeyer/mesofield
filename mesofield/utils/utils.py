"""Miscellaneous diagnostic helpers.

These utilities are not imported by the runtime; they are convenience
functions for use in ad-hoc scripts, notebooks, and the embedded IPython
console. ``nidaqmx`` is imported lazily inside the NI-DAQ helpers so this
module loads cleanly on machines without NI-DAQmx installed.
"""

import serial.tools.list_ports  # pip install pyserial
import requests
import pymmcore_plus
from useq import MDASequence
import pandas as pd


def test_arduino_connection():
    try:
        arduino = serial.Serial('COM4', 9600)
        arduino.close()
        print("Arduino connection successful!")
    except serial.SerialException:
        print("Failed to connect to Arduino on COM4.")
        
#TODO Save config for each session
#TODO Add a button to save the current configuration to a JSON file
#TODO Auto-fps calculation based on the number of frames and duration
def get_fps(mmc: pymmcore_plus.CMMCorePlus):
    ## There is a problem when this function is called more than once in succession, freezing the program
    """
    Calculate the frames per second (FPS) based on the number of frames and duration of an MDA sequence.
        - num_frames = num_trials × trial_time(5 seconds) × framerate (45 fps)
        - num_trials = num_frames / (trial_time * framerate) (255 frames for a 5 seconds trial at 45 fps)
        - Total duration = num_frames / framerate or num_trials * trial_time
        - num_frames = num_trials × trial_time(5 seconds) × framerate (45 fps)

    """
    from pymmcore_plus import Metadata
    core = mmc
    frames = 120
    core.run_mda(
        MDASequence(time_plan={"interval": 0, "loops": frames}), 
        block=True
    )
    
    duration = core.mda._time_elapsed()
    fps = frames / duration
    return fps

### Utility functions for querying serial ports and USB IDs ###

def load_metadata_from_json(json_file_path) -> pd.DataFrame:
    """load metadata from a JSON file as a Pandas Dataframe."""
    try:
        metadata_df = pd.read_json(json_file_path)
        return metadata_df
    except Exception as e:
        print(f"Error loading metadata from JSON file: {e}")
        return pd.DataFrame()

def list_serial_ports():
    """
    List all available serial ports on the system
    """
    # Get a list of all available serial ports
    ports = serial.tools.list_ports.comports()
    
    # Check if there are any ports available
    if not ports:
        print("No serial ports found.")
        return
    
    # Print details of each available serial port
    for port in ports:
        print(f"Device: {port.device}, Description: {port.description}, HWID: {port.hwid}")



def download_usb_ids():
    """
    Download the USB IDs file from the internet resources 
    """
    url = "http://www.linux-usb.org/usb.ids"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print("Failed to download USB IDs")
        return None

def parse_usb_ids(usb_ids_content):
    """
    Parse the USB IDs file and return a dictionary of vendor IDs and product IDs
    """

    vendor_ids = {}
    current_vendor_id = ""
    for line in usb_ids_content.splitlines():
        if not line or line.startswith("#") or line.startswith("\t\t"):
            continue
        if line.startswith("\t"):
            product_id, product_name = line.strip().split(" ", 1)
            vendor_ids[current_vendor_id]["products"][product_id] = product_name
        else:
            vendor_id, vendor_name = line.split(" ", 1)
            current_vendor_id = vendor_id
            vendor_ids[vendor_id] = {"name": vendor_name, "products": {}}
    return vendor_ids

def identify_device(vendor_id, product_id, usb_ids):
    """
    Identify the vendor and product of a USB device using the USB IDs file
    """

    vendor = usb_ids.get(vendor_id, {}).get("name", "Unknown Vendor")
    product = usb_ids.get(vendor_id, {}).get("products", {}).get(product_id, "Unknown Product")
    return vendor, product

def list_serial_ports_with_vendors(usb_ids):
    """List serial ports and resolve their vendor / product names.

    Args:
        usb_ids: Dictionary produced by :func:`parse_usb_ids`.
    """
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        hwid_parts = port.hwid.split()
        vid_pid = [part for part in hwid_parts if part.startswith("VID:PID=")]
        if vid_pid:
            vid, pid = vid_pid[0].replace("VID:PID=", "").split(":")
            vendor, product = identify_device(vid, pid, usb_ids)
            print(f"Device: {port.device}, Description: {port.description}, Vendor: {vendor}, Product: {product}")
        else:
            print(f"Device: {port.device}, Description: {port.description}, HWID: {port.hwid}")

### NI-DAQ utility functions ###

def list_nidaq_devices():
    """List all connected NI-DAQ devices."""
    import nidaqmx  # lazy: not all rigs have NI-DAQmx installed
    system = nidaqmx.system.System.local()
    return [device.name for device in system.devices]


def read_analog_input(device_name, channel='ai0'):
    """Read a single analog input from a specified channel."""
    import nidaqmx
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(f"{device_name}/{channel}")
        return task.read()


def test_nidaq_connection(device_name):
    """Test connection to a specified NI-DAQ device."""
    import nidaqmx
    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{device_name}/ai0")
        return True
    except nidaqmx.DaqError:
        return False
    
    
### MMCore configuration-check utility functions (check settings of devices connected to Micro-Manager via loaded configuration.cfg file) ###

def sanity_check(mmc):
    """
    Perform a sanity check to ensure that the Core is properly
    initialized and that the camera is connected. 

    This function lists: 
    - the loaded devices
    - configuration groups
    - current MMCore configuration settings 
    - camera settings.

    """
    print("Sanity Check:")
    print("=============")
    
    # ==== List all devices loaded by Micro-Manager ==== #
    devices = mmc.getLoadedDevices()
    print("Loaded Devices:")
    for device in devices:
        print(f" - {device}: {mmc.getDeviceDescription(device)}")
    
    # ==== Display configuration groups ==== #
    config_groups = mmc.getAvailableConfigGroups()
    print("\nConfiguration Groups:")
    for group in config_groups:
        print(f" - {group}")
        configs = mmc.getAvailableConfigs(group)
        for config in configs:
            print(f"    - {config}")
    
    # ==== Display current configuration ==== #
    print("\nCurrent Configuration:")
    # Get the current configuration settings for each group
    for group in config_groups:
        print(f" - {group}: {mmc.getCurrentConfig(group)}")
    
    # ==== Display camera settings ==== #
    camera_device = mmc.getCameraDevice()
    if camera_device:
        print(f"\nCamera Device: {camera_device}")
        print(f" - Exposure: {mmc.getExposure()} ms")
        print(f" - Pixel Size: {mmc.getPixelSizeUm()} um")
    else:
        print("\nNo camera device found.")

    
    print("\nOther Information:")
    print(f" - Image Width: {mmc.getImageWidth()}")
    print(f" - Image Height: {mmc.getImageHeight()}")
    print(f" - Bit Depth: {mmc.getImageBitDepth()}")