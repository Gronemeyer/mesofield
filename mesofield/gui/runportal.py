#!/usr/bin/env python3
"""
Infinite Corridor using Panda3D

This script creates an infinite corridor effect with user-controlled forward/backward movement.

Features:
- Configurable parameters loaded from JSON
- Infinite corridor effect
- User-controlled movement
- [real-time] Data logging (timestamp, position, velocity)

The corridor consists of left, right, ceiling, and floor segments.
It uses the Panda3D CardMaker API to generate flat geometry for the corridor's four faces.
An infinite corridor/hallway effect is simulated by recycling the front segments to the back when the player moves forward. 


Configuration parameters are loaded from a JSON file "conf.json".

Author: Jacob Gronemeyer
Date: 2025-07-28
Version: 0.3
"""

import json
import sys
import csv
import os
import time
import serial
import threading
import queue
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog, GraphicsPipe
from direct.showbase import DirectObject
from direct.fsm.FSM import FSM
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from portal_socket import PortalServer

# ─── Fix for running as subprocess ─────────────────────────────────────────────────────
# https://raw.githubusercontent.com/panda3d/panda3d/release/1.10.x/panda/src/doc/howto.use_config.txt
# https://stackoverflow.com/questions/73900341/what-is-the-path-of-config-prc-files-in-panda3d


# Configure Panda3D before any ShowBase initialization
from panda3d.core import loadPrcFileData
    
# Set essential graphics configuration
loadPrcFileData('', 'load-display pandagl')
loadPrcFileData('', 'aux-display pandadx9') 
loadPrcFileData('', 'aux-display pandadx8')
loadPrcFileData('', 'aux-display tinydisplay')
loadPrcFileData('', 'window-title MousePortal')

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.
    
    Parameters:
        config_file (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)


def _restore_windows_path(path: str) -> str:
    """Convert Mesofield-normalized POSIX paths back to native Windows form."""
    if os.name != "nt":
        return path
    if len(path) >= 4 and path[0] == "/" and path[1].isalpha() and path[2] == "/":
        drive = path[1].upper()
        rest = path[3:]
        normalized = rest.replace("/", os.sep)
        if normalized:
            return f"{drive}:{os.sep}{normalized}"
        return f"{drive}:{os.sep}"
    return path


def _restore_config_paths(payload: Any) -> Any:
    """Recursively restore any normalized filesystem paths within a config payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            payload[key] = _restore_config_paths(value)
        return payload
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            payload[idx] = _restore_config_paths(value)
        return payload
    if isinstance(payload, str):
        return _restore_windows_path(payload)
    return payload

class DataLogger:
    """
    Logs movement data to a CSV file.
    """
    def __init__(self, filename):
        """
        Initialize the data logger.
        
        Args:
            filename (str): Path to the CSV file.
        """
        self.filename = filename
        self.fieldnames = ['timestamp', 'position', 'velocity']
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def log(self, timestamp, position, velocity):
        self.writer.writerow({'timestamp': timestamp, 'position': position, 'velocity': velocity})
        self.file.flush()

    def close(self):
        self.file.close()

class EventLogger:
    """Save event markers with timing information."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.fieldnames = ["time_sent", "time_received", "delta", "position", "event_name"]
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def log(self, time_sent: float | None, time_received: float, position: float, name: str) -> None:
        delta = time_received - time_sent if time_sent is not None else None
        self.writer.writerow({
            'time_sent': time_sent,
            'time_received': time_received,
            'delta': delta,
            'position': position,
            'event_name': name,
        })
        self.file.flush()
    def close(self) -> None:
        self.file.close()


@dataclass
class OpenLoopSegment:
    """Single stage within an open-loop trial."""

    duration: float
    gain: float = 0.0
    bias: float = 0.0
    label: Optional[str] = None


class TrialController:
    """Abstract base for runtime control of corridor velocity."""

    mode: str = "idle"

    def start(self) -> None:
        """Prepare controller state before a trial begins."""

    def stop(self) -> None:
        """Cleanup when the trial ends."""

    def compute_velocity(self, input_speed: float, dt: float) -> float:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"mode": self.mode}


class ClosedLoopTrial(TrialController):
    """Directly map treadmill velocity to camera velocity."""

    mode = "closed_loop"

    def compute_velocity(self, input_speed: float, dt: float) -> float:  # noqa: ARG002
        return input_speed


class OpenLoopTrial(TrialController):
    """Cycle through a scripted set of gain/bias glitches."""

    mode = "open_loop"

    def __init__(self, schedule: List[OpenLoopSegment], loop: bool = True) -> None:
        self.schedule = schedule
        self.loop = loop
        self.active = False
        self._index = 0
        self._time_left = 0.0
        self._current: Optional[OpenLoopSegment] = None

    def start(self) -> None:
        self.active = True
        self._index = 0
        self._current = self.schedule[0] if self.schedule else None
        self._time_left = self._current.duration if self._current else 0.0

    def stop(self) -> None:
        self.active = False

    def compute_velocity(self, input_speed: float, dt: float) -> float:
        if not self.schedule:
            return 0.0
        if not self._current or not self.active:
            self.start()
        self._advance(dt)
        current = self._current or self.schedule[0]
        return current.gain * input_speed + current.bias

    def _advance(self, dt: float) -> None:
        if not self._current:
            return
        self._time_left -= dt
        while self._time_left <= 0.0 and self.schedule:
            next_index = self._index + 1
            if next_index >= len(self.schedule):
                if self.loop:
                    next_index = 0
                else:
                    next_index = len(self.schedule) - 1
            self._index = next_index
            self._current = self.schedule[self._index]
            self._time_left += max(self._current.duration, 1e-6)

    def info(self) -> Dict[str, Any]:
        segment = self._current
        if not segment and self.schedule:
            segment = self.schedule[self._index]
        info: Dict[str, Any] = {
            "mode": self.mode,
            "active": self.active,
            "loop": self.loop,
            "segment_count": len(self.schedule),
        }
        if segment:
            info.update({
                "segment_index": self._index,
                "segment_label": segment.label,
                "gain": segment.gain,
                "bias": segment.bias,
                "remaining": max(self._time_left, 0.0),
            })
        return info


@dataclass
class OpenLoopSegment:
    """Single stage within an open-loop trial."""

    duration: float
    gain: float = 0.0
    bias: float = 0.0
    label: Optional[str] = None


DEFAULT_OPEN_LOOP_PATTERN = [
    {"duration": 1.5, "gain": 1.0, "bias": 0.0, "label": "follow"},
    {"duration": 0.6, "gain": 0.0, "bias": 3.5, "label": "glitch_forward"},
    {"duration": 0.5, "gain": 0.0, "bias": 0.0, "label": "freeze"},
    {"duration": 0.7, "gain": -0.3, "bias": 0.0, "label": "reverse"},
    {"duration": 1.2, "gain": 0.5, "bias": 0.0, "label": "half_gain"},
]
@dataclass
class EncoderData:
    """ Represents a single encoder reading."""
    timestamp: int
    distance: float
    speed: float

    def __repr__(self):
        return (f"EncoderData(timestamp={self.timestamp}, "
                f"distance={self.distance:.3f} mm, speed={self.speed:.3f} mm/s)")

class CommandListener:
    """Background thread to read commands from STDIN."""
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        for line in sys.stdin:
            self.queue.put(line.strip())

    def get(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

class ExperimentFSM(FSM):
    def __init__(self, owner):
        super().__init__("ExperimentFSM")
        self.owner = owner

    def enterIdle(self):
        self.owner.update_state_text("Idle")
        print("STATE IDLE")

    def enterRunning(self):
        self.owner.update_state_text("Running")
        print("STATE RUNNING")

    def exitRunning(self):
        self.owner.update_state_text("Idle")
        print("STATE STOPPED")



class Corridor:
    """
    Class for generating infinite corridor geometric rendering
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the corridor by creating segments for each face.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        self.base = base
        self.segment_length: float = config["segment_length"]
        self.corridor_width: float = config["corridor_width"]
        self.wall_height: float = config["wall_height"]
        self.num_segments: int = config["num_segments"]
        self.left_wall_texture: str = config["left_wall_texture"]
        self.right_wall_texture: str = config["right_wall_texture"]
        self.ceiling_texture: str = config["ceiling_texture"]
        self.floor_texture: str = config["floor_texture"]
        
        # Create a parent node for all corridor segments.
        self.parent: NodePath = base.render.attachNewNode("corridor")
        
        # Separate lists for each face.
        self.left_segments: list[NodePath] = []
        self.right_segments: list[NodePath] = []
        self.ceiling_segments: list[NodePath] = []
        self.floor_segments: list[NodePath] = []
        
        self.build_segments()
        
    def build_segments(self) -> None:
        """ 
        Build the initial corridor segments using CardMaker.
        """
        # two prerendered backward segments, then the forward segments
        hallway_segments = [-2, -1] + list(range(self.num_segments))

        for i in hallway_segments:
            segment_start: float = i * self.segment_length
            
            # ─── Left Wall ─────────────────────────────────────────────────────
            # Create a card with dimensions (segment_length x wall_height),
            # position it at x = -corridor_width/2 and rotate it so the face is inward.
            cm_left: CardMaker = CardMaker("left_wall")
            # The card is generated in the XY plane; here we use X (length) and Z (height).
            cm_left.setFrame(0, self.segment_length, 0, self.wall_height)
            left_node: NodePath = self.parent.attachNewNode(cm_left.generate())
            # Position the left wall at x = -corridor_width/2 and at the starting Y position
            left_node.setPos(-self.corridor_width / 2, segment_start, 0)
            # Rotate to face inward (rotate around Z axis by 90°)
            # This maps the card's original X (now wall height) to the Z axis and Y remains.
            left_node.setHpr(90, 0, 0)
            self.apply_texture(left_node, self.left_wall_texture)
            self.left_segments.append(left_node)
            
            # ─── Right Wall ─────────────────────────────────────────────────────
            cm_right: CardMaker = CardMaker("right_wall")
            cm_right.setFrame(0, self.segment_length, 0, self.wall_height)
            right_node: NodePath = self.parent.attachNewNode(cm_right.generate())
            right_node.setPos(self.corridor_width / 2, segment_start, 0)
            right_node.setHpr(-90, 0, 0) # Rotate to face inward (rotate around Z axis by -90°)
            self.apply_texture(right_node, self.right_wall_texture)
            self.right_segments.append(right_node)

            # ─── Ceiling (Top) ─────────────────────────────────────────────────
            cm_ceiling: CardMaker = CardMaker("ceiling")
            # The ceiling card covers the corridor width and one segment length.
            cm_ceiling.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            ceiling_node: NodePath = self.parent.attachNewNode(cm_ceiling.generate())
            ceiling_node.setPos(0, segment_start, self.wall_height)
            ceiling_node.setHpr(0, 90, 0)
            self.apply_texture(ceiling_node, self.ceiling_texture)
            self.ceiling_segments.append(ceiling_node)

            # ─── Floor (Bottom) ─────────────────────────────────────────────────
            cm_floor: CardMaker = CardMaker("floor")
            cm_floor.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            floor_node: NodePath = self.parent.attachNewNode(cm_floor.generate())
            floor_node.setPos(0, segment_start, 0)
            floor_node.setHpr(0, -90, 0)
            self.apply_texture(floor_node, self.floor_texture)
            self.floor_segments.append(floor_node)
            
    def apply_texture(self, node: NodePath, texture_path: str) -> None:
        texture: Texture = self.base.loader.loadTexture(texture_path)
        node.setTexture(texture)
        
    def set_texture(self, face: str, texture_path: str) -> None:
        lists = {
            "left": self.left_segments,
            "right": self.right_segments,
            "ceiling": self.ceiling_segments,
            "floor": self.floor_segments,
        }
        segs = lists.get(face.lower())
        if not segs:
            return
        texture = self.base.loader.loadTexture(texture_path)
        for seg in segs:
            seg.setTexture(texture)

    def recycle_segment(self, direction: str) -> None:
        """
        Recycle the front segments by repositioning them to the end of the corridor.
        This is called when the player has advanced by one segment length.
        """
        
        if direction == "forward":
            # Calculate new base Y position from the last segment in the left wall.
            new_y: float = self.left_segments[-1].getY() + self.segment_length
            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(0)
            left_seg.setY(new_y)
            self.left_segments.append(left_seg)
            
            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(0)
            right_seg.setY(new_y)
            self.right_segments.append(right_seg)
            
            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(0)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.append(ceiling_seg)
            
            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(0)
            floor_seg.setY(new_y)
            self.floor_segments.append(floor_seg)

        elif direction == "backward":
            new_y = self.left_segments[0].getY() - self.segment_length
            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(-1)
            left_seg.setY(new_y)
            self.left_segments.insert(0, left_seg)

            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(-1)
            right_seg.setY(new_y)
            self.right_segments.insert(0, right_seg)

            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(-1)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.insert(0, ceiling_seg)

            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(-1)
            floor_seg.setY(new_y)
            self.floor_segments.insert(0, floor_seg)
            
class FogEffect:
    """
    Parameters:
        base (ShowBase): The Panda3D base instance.
        fog_color (tuple): RGB color for the fog (default is white).
        near_distance (float): The near distance where the fog starts.
        far_distance (float): The far distance where the fog completely obscures the scene.
    """

    def __init__(self, base: ShowBase, fog_color, density):
        self.base = base
        self.fog = Fog("fog")
        base.setBackgroundColor(fog_color)
        
        # Set fog color.
        self.fog.setColor(*fog_color)
        
        # Set the density for the fog.
        self.fog.setExpDensity(density)
        
        # Attach the fog to the root node to affect the entire scene.
        render.setFog(self.fog)


class SerialInputManager(DirectObject.DirectObject):
    """
    This class abstracts the serial connection and starts a thread that listens
    for serial data.
    """
    def __init__(self, serial_port: str, baudrate: int = 57600, messenger: DirectObject = None) -> None:
        self._port = serial_port
        self._baud = baudrate
        try:
            self.serial = serial.Serial(self._port, self._baud, timeout=1)
        except serial.SerialException as e:
            print(f"{self.__class__}: I failed to open serial port {self._port}: {e}")
            raise
        self.accept('readSerial', self._store_data)
        self.data = EncoderData(0, 0.0, 0.0)
        self.messenger = messenger

    def _store_data(self, data: EncoderData):
        self.data = data

    def _read_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the serial port."""
        # Read a line from the Teensy board
        raw_line = self.serial.readline()

        # Decode and strip newline characters
        line = raw_line.decode('utf-8', errors='replace').strip()
        if line:
            data = self._parse_line(line)
            if data:
                self.messenger.send("readSerial", [data])

        return Task.cont

    def _parse_line(self, line: str) -> EncoderData | None:
        """
        Expected line formats:
          - "timestamp,distance,speed"  or
          - "distance,speed"
        """
        parts = line.split(',')
        try:
            if len(parts) == 3:
                # Format: timestamp, distance, speed
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                return EncoderData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 2:
                # Format: distance, speed
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
                return EncoderData(distance=distance, speed=speed)
            else:
                # Likely a header or message line (non-data)
                return None
        except ValueError:
            # Non-numeric data (e.g., header info)
            return None

    
class DummyInputManager:
    """Stand-in for SerialInputManager when running without hardware."""
    def __init__(self):
        self.data = EncoderData(0, 0.0, 0.0)
    def _read_serial(self, task: Task) -> Task:
        return Task.cont


class MousePortal(ShowBase):
    """
    Main application class for Mouse Portal's infinite corridor.
    """
    def __init__(
        self,
        config_file,
        dev: bool = False,
        *,
        socket_host: Optional[str] = None,
        socket_port: Optional[int] = None,
        enable_socket: bool = True,
        status_interval: float = 0.1,
    ) -> None:
        """
        Initialize the application, load configuration, set up the camera, user input,
        corridor geometry, and add the update task.
        """
        ShowBase.__init__(self)
        
        # Load configuration from JSON (direct option)
        # config: Dict[str, Any] = load_config("conf.json")
        # Load configuration (init option for testing)
        with open(config_file, 'r') as f:
            self.cfg: Dict[str, Any] = load_config(config_file)

        self.default_trial_type = str(self.cfg.get("default_trial_type", "closed_loop")).lower()
        if self.default_trial_type not in {"open_loop", "closed_loop"}:
            self.default_trial_type = "closed_loop"
        self.open_loop_schedule = self._load_open_loop_schedule(self.cfg)
        self.open_loop_repeat = bool(self.cfg.get("open_loop_repeat", True))
        self.closed_loop_controller = ClosedLoopTrial()
        self.controller: TrialController = self.closed_loop_controller
        self.trial_mode: str = "idle"
        self.trial_started_at: Optional[float] = None
        self.current_input_speed: float = 0.0

        cfg_port = self.cfg.get("socket_port", 8765)
        try:
            cfg_port = int(cfg_port)
        except (TypeError, ValueError):
            cfg_port = 8765
        host = socket_host or self.cfg.get("socket_host", "127.0.0.1")
        port = socket_port if socket_port is not None else cfg_port
        self.socket_server: Optional[PortalServer] = None
        self._socket_status_interval = max(status_interval, 0.01)
        self._last_status_sent = 0.0
        if enable_socket:
            self.socket_server = PortalServer(host, port)
            self.socket_server.start()
            print(f"SOCKET LISTEN {host}:{port}")
            sys.stdout.flush()

        # ─── Window Properties ─────────────────────────────────────────────────────
        # Get the display width and height for both monitors
        pipe = self.win.getPipe()
        display_width = pipe.getDisplayWidth()
        display_height = pipe.getDisplayHeight()

        # Set window properties to span across both monitors
        wp: WindowProperties = WindowProperties()
        wp.setSize(1920, 1280)  # Double the width for two
        wp.set_origin(display_width, 0)
        self.dev = dev
        self.win.requestProperties(wp)
        self.setFrameRateMeter(True)
        # Disable default mouse-based camera control for mapped input
        self.disableMouse()
        
        # ─── Camera Setup ──────────────────────────────────────────────────────────
        self.camera_position: float = 0.0
        self.camera_velocity: float = 0.0
        self.speed_scaling: float = self.cfg.get("speed_scaling", 5.0)
        self.camera_height: float = self.cfg.get("camera_height", 2.0)  
        self.camera.setPos(0, self.camera_position, self.camera_height)
        self.camera.setHpr(0, 0, 0)
        
        # ─── User Input ────────────────────────────────────────────────────────────
        self.key_map: Dict[str, bool] = {"forward": False, "backward": False}
        self.accept("arrow_up", self.set_key, ["forward", True])
        self.accept("arrow_up-up", self.set_key, ["forward", False])
        self.accept("arrow_down", self.set_key, ["backward", True])
        self.accept("arrow_down-up", self.set_key, ["backward", False])
        self.accept('escape', self.userExit)

        # ─── Treadmill Input ─────────────────────────────────────────────────────
        if self.dev:
            self.treadmill = DummyInputManager()
        else:
            self.treadmill = SerialInputManager(serial_port=self.cfg["serial_port"], 
                                                messenger=self.messenger)

        # ─── Corridor Setup ─────────────────────────────────────────────────────
        self.corridor: Corridor = Corridor(self, self.cfg)
        self.segment_length: float = self.cfg["segment_length"]
        
        # Variable to track movement since last recycling.
        self.distance_since_recycle: float = 0.0
        
        # Movement speed (units per second).
        self.movement_speed: float = 10.0
        
        # ─── Fog Effect ─────────────────────────────────────────────────────────────
        self.fog_effect = FogEffect(self, 
                                    density= self.cfg["fog_density"], 
                                    fog_color=(0.5, 0.5, 0.5))
        
        # ─── Data and Event Logging ────────────────────────────────────────────────
        self.data_logger = DataLogger(self.cfg["data_logging_file"])
        self.event_logger = EventLogger(self.cfg.get("event_log_file", "event_markers.csv"))

        # ─── TaskMgr and FSM ──────────────────────────────────────────────────────
        self.taskMgr.add(self.update, "updateTask")
        self.command_listener = CommandListener()
        self.fsm = ExperimentFSM(self)
        self.state_name = "Idle"
        self.fsm.request("Idle")
        if self.dev:
            self.state_text = OnscreenText(text="State: Idle", 
                                           pos=(-1.3, 0.9), 
                                           scale=0.07, 
                                           align=TextNode.ALeft)
        self.events = []
        self._shutdown_requested = False
        self.taskMgr.add(self.process_commands, "commandTask")

        
        # self.taskMgr.setupTaskChain("serialInputDevice", numThreads = 1, tickClock = None,
        #                threadPriority = None, frameBudget = None,
        #                frameSync = True, timeslicePriority = None)
        if not self.dev:
            self.taskMgr.add(self.treadmill._read_serial, name="readSerial")

        if dev:
            # In development mode, enable verbose logging for debugging.
            self.messenger.toggleVerbose()
            self.accept("v", self.messenger.toggle_verbose)

    def _default_open_loop_schedule(self) -> List[OpenLoopSegment]:
        return [
            OpenLoopSegment(
                duration=float(entry.get("duration", 1.0)),
                gain=float(entry.get("gain", 0.0)),
                bias=float(entry.get("bias", 0.0)),
                label=entry.get("label"),
            )
            for entry in DEFAULT_OPEN_LOOP_PATTERN
        ]

    def _load_open_loop_schedule(self, cfg: Dict[str, Any]) -> List[OpenLoopSegment]:
        schedule_cfg = cfg.get("open_loop_schedule")
        if not schedule_cfg:
            return self._default_open_loop_schedule()
        schedule: List[OpenLoopSegment] = []
        for entry in schedule_cfg:
            if not isinstance(entry, dict):
                continue
            try:
                duration = float(entry.get("duration", 0))
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            gain_value = entry.get("gain")
            bias_value = entry.get("bias", entry.get("offset", 0.0))
            if "velocity" in entry:
                bias_value = entry.get("velocity", bias_value)
                gain_value = entry.get("gain", 0.0)
            base_gain = 0.0 if "velocity" in entry else 1.0
            try:
                gain = float(gain_value) if gain_value is not None else base_gain
            except (TypeError, ValueError):
                gain = base_gain
            try:
                bias = float(bias_value) if bias_value is not None else 0.0
            except (TypeError, ValueError):
                bias = 0.0
            schedule.append(OpenLoopSegment(duration=duration, gain=gain, bias=bias, label=entry.get("label")))
        if not schedule:
            return self._default_open_loop_schedule()
        return schedule

    def _parse_open_loop_segments(self, raw_segments: Any) -> List[OpenLoopSegment]:
        if not isinstance(raw_segments, list):
            return []
        segments: List[OpenLoopSegment] = []
        last_gain = 1.0
        last_bias = 0.0
        for entry in raw_segments:
            if not isinstance(entry, dict):
                continue
            try:
                duration = float(entry.get("duration", 0))
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            raw_gain = entry.get("gain")
            raw_bias = entry.get("bias", entry.get("offset"))
            if "velocity" in entry and entry["velocity"] is not None:
                raw_bias = entry["velocity"]
                raw_gain = entry.get("gain", 0.0)
            if raw_gain is None:
                gain = last_gain if segments else 1.0
            else:
                try:
                    gain = float(raw_gain)
                except (TypeError, ValueError):
                    gain = last_gain if segments else 1.0
            if raw_bias is None:
                bias = last_bias if segments else 0.0
            else:
                try:
                    bias = float(raw_bias)
                except (TypeError, ValueError):
                    bias = last_bias if segments else 0.0
            segments.append(OpenLoopSegment(duration=duration, gain=gain, bias=bias, label=entry.get("label")))
            last_gain = gain
            last_bias = bias
        return segments

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    def _create_open_loop_controller(
        self,
        schedule: Optional[List[OpenLoopSegment]] = None,
        *,
        loop: Optional[bool] = None,
    ) -> OpenLoopTrial:
        source = schedule if schedule else self.open_loop_schedule
        if not source:
            source = self._default_open_loop_schedule()
        controller_schedule = [
            OpenLoopSegment(duration=segment.duration, gain=segment.gain, bias=segment.bias, label=segment.label)
            for segment in source
        ]
        loop_flag = self.open_loop_repeat if loop is None else loop
        return OpenLoopTrial(controller_schedule, loop=loop_flag)

    def _start_trial(self, trial_type: str, sent_time: Optional[float], payload: Dict[str, Any]) -> Dict[str, Any]:
        requested = (trial_type or "closed_loop").lower()
        if requested == "open_loop":
            segments_payload = payload.get("segments") or payload.get("schedule")
            schedule_override: Optional[List[OpenLoopSegment]] = None
            if segments_payload is not None:
                parsed = self._parse_open_loop_segments(segments_payload)
                schedule_override = parsed or None
            loop_value = payload.get("loop")
            if loop_value is None:
                loop_value = payload.get("repeat")
            loop_override: Optional[bool] = None
            if loop_value is not None:
                loop_override = self._coerce_bool(loop_value, self.open_loop_repeat)
            controller = self._create_open_loop_controller(schedule_override, loop=loop_override)
        else:
            requested = "closed_loop"
            controller = self.closed_loop_controller

        if self.controller is not controller:
            self.controller.stop()
        self.controller = controller
        self.controller.start()
        self.trial_mode = requested
        self.trial_started_at = time.time()

        event_name = payload.get("event_name") or f"start_{requested}"
        self.mark_event(event_name, sent_time)
        self.fsm.request("Running")
        self._push_status(force=True)
        info = {
            "state": self.state_name,
            "trial_mode": self.trial_mode,
            "controller": self.controller.info(),
        }
        return info

    def _stop_trial(self, sent_time: Optional[float]) -> Dict[str, Any]:
        if self.trial_mode != "idle":
            self.mark_event(f"stop_{self.trial_mode}", sent_time)
        self.controller.stop()
        self.controller = self.closed_loop_controller
        self.controller.start()
        self.trial_mode = "idle"
        self.trial_started_at = None
        self.fsm.request("Idle")
        self._push_status(force=True)
        info = {
            "state": self.state_name,
            "trial_mode": self.trial_mode,
            "controller": self.controller.info(),
        }
        return info

    def userExit(self):
        self.data_logger.close()
        self.event_logger.close()
        if self.socket_server:
            self.socket_server.close()
        super().userExit()

    def set_key(self, key: str, value: bool) -> None:
        self.key_map[key] = value
        
    def update_state_text(self, state: str):
        self.state_name = state
        if self.dev and hasattr(self, "state_text"):
            self.state_text.setText(f"State: {state}")

    def update(self, task: Task) -> Task:
        """
        Update the camera's position based on user input and recycle corridor segments
        when the player moves forward beyond one segment.
        
        Parameters:
            task (Task): The Panda3D task instance.
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        dt: float = globalClock.getDt()

        if self.dev:
            if self.key_map["forward"]:
                input_speed = self.speed_scaling
            elif self.key_map["backward"]:
                input_speed = -self.speed_scaling
            else:
                input_speed = 0.0
        else:
            input_speed = self.treadmill.data.speed

        self.current_input_speed = input_speed
        self.camera_velocity = self.controller.compute_velocity(input_speed, dt)
        move_distance: float = self.camera_velocity * dt
        self.camera_position += move_distance
        self.camera.setPos(0, self.camera_position, self.camera_height)
        
        # Recycle corridor segments when the camera moves beyond one segment length
        # ─── Forward Movement -----> ───────────────────────────────────────────────────
        # Recycle segments from the back to the front
        if move_distance > 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle >= self.segment_length:
                self.corridor.recycle_segment(direction="forward")
                self.distance_since_recycle -= self.segment_length
        # ─── Backward Movement <----- ──────────────────────────────────────────────────
        # Recycle segments from the front to the back
        elif move_distance < 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle <= -self.segment_length:
                self.corridor.recycle_segment(direction="backward")
                self.distance_since_recycle += self.segment_length
        
        # Log movement data (timestamp, position, velocity)
        self.data_logger.log(time.time(), self.camera_position, self.camera_velocity)
        self._push_status()
        
        print(f"STATUS {time.time()} {self.camera_position:.3f} {self.camera_velocity:.3f}")
        sys.stdout.flush()
        return Task.cont

    def _push_status(self, force: bool = False) -> None:
        if not self.socket_server or not self.socket_server.is_client_connected():
            return
        now = time.time()
        if force or (now - self._last_status_sent) >= self._socket_status_interval:
            status = {
                "type": "status",
                "time": now,
                "position": self.camera_position,
                "velocity": self.camera_velocity,
                "state": self.state_name,
                "trial_mode": self.trial_mode,
                "input_velocity": self.current_input_speed,
                "controller": self.controller.info(),
            }
            if self.trial_started_at is not None:
                status["trial_elapsed"] = now - self.trial_started_at
            self.socket_server.send_message(status)
            self._last_status_sent = now

    def process_commands(self, task: Task) -> Task:
        while True:
            cmd = self.command_listener.get()
            if not cmd:
                break
            self._handle_text_command(cmd)

        if self.socket_server:
            while True:
                message = self.socket_server.get_message()
                if not message:
                    break
                self._handle_socket_message(message)

        if self._shutdown_requested:
            self._shutdown_requested = False
            self.userExit()
            return Task.done

        return Task.cont

    def _handle_text_command(self, cmd: str) -> None:
        parts = cmd.split()
        if not parts:
            return
        name = parts[0].lower()
        payload: Dict[str, Any] = {}
        sent_time: Optional[float] = None

        if name in {"start_trial", "stop_trial"} and len(parts) > 1:
            for token in parts[1:]:
                token_lower = token.lower()
                try:
                    sent_time = float(token)
                    continue
                except ValueError:
                    pass
                if name == "start_trial" and token_lower in {"open_loop", "closed_loop"}:
                    payload["trial_type"] = token_lower
                elif name == "start_trial" and "event_name" not in payload:
                    payload["event_name"] = f"start_{token_lower}"
        elif name == "set_texture" and len(parts) >= 3:
            payload["face"] = parts[1]
            payload["texture"] = parts[2]
        elif name == "mark_event":
            if len(parts) > 1:
                payload["name"] = parts[1]
            if len(parts) > 2:
                try:
                    sent_time = float(parts[2])
                except ValueError:
                    sent_time = None

        self._execute_command(name, sent_time, payload, source="stdin")

    def _handle_socket_message(self, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "error":
            print(f"SOCKET ERROR {message.get('message')} raw={message.get('raw')}")
            sys.stdout.flush()
            return
        if msg_type == "ping":
            if self.socket_server:
                reply = {"type": "pong"}
                if "client_time" in message:
                    reply["client_time"] = message["client_time"]
                if "request_id" in message:
                    reply["request_id"] = message["request_id"]
                self.socket_server.send_message(reply)
            return

        command = message.get("command") or msg_type
        if not command:
            return

        sent_time: Optional[float] = None
        for key in ("sent_time", "client_time", "timestamp"):
            value = message.get(key)
            if isinstance(value, (int, float)):
                sent_time = float(value)
                break

        try:
            success, info = self._execute_command(command, sent_time, message, source="socket")
        except Exception as exc:  # pragma: no cover - defensive
            success = False
            info = {"error": f"exception:{exc}"}
            print(f"SOCKET COMMAND ERROR {command}: {exc}")
            sys.stdout.flush()

        status = "ok" if success else "error"
        if self.socket_server:
            ack_payload: Dict[str, Any] = {"type": "ack", "command": command, "status": status}
            if sent_time is not None:
                ack_payload["sent_time"] = sent_time
            for key in ("request_id", "message_id", "id", "correlation_id"):
                if key in message:
                    ack_payload[key] = message[key]
            ack_payload.update(info)
            self.socket_server.send_message(ack_payload)

    def _execute_command(
        self,
        command: str,
        sent_time: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        *,
        source: str = "stdin",
    ) -> tuple[bool, Dict[str, Any]]:
        payload = payload or {}
        cmd = command.lower()

        if cmd == "start_trial":
            trial_type = payload.get("trial_type") or payload.get("mode") or self.default_trial_type
            info = self._start_trial(trial_type, sent_time, payload)
            return True, info

        if cmd == "stop_trial":
            info = self._stop_trial(sent_time)
            return True, info

        if cmd == "start_open_loop":
            info = self._start_trial("open_loop", sent_time, payload)
            return True, info

        if cmd == "start_closed_loop":
            info = self._start_trial("closed_loop", sent_time, payload)
            return True, info

        if cmd == "set_texture":
            face = payload.get("face") or payload.get("surface")
            texture = payload.get("texture") or payload.get("path")
            if not face or not texture:
                return False, {"error": "missing_face_or_texture"}
            self.corridor.set_texture(face, texture)
            return True, {"face": face, "texture": texture, "trial_mode": self.trial_mode}

        if cmd == "mark_event":
            name = payload.get("name") or payload.get("event") or "event"
            self.mark_event(name, sent_time)
            return True, {"event": name, "trial_mode": self.trial_mode, "controller": self.controller.info()}

        if cmd in {"end", "shutdown", "quit", "exit"}:
            self._shutdown_requested = True
            return True, {"state": self.state_name}

        if cmd == "get_status":
            self._push_status(force=True)
            return True, {"state": self.state_name, "trial_mode": self.trial_mode, "controller": self.controller.info()}

        if cmd == "ping" and source == "socket":
            if self.socket_server:
                self.socket_server.send_message({"type": "pong"})
            return True, {}

        return False, {"error": f"unknown_command:{cmd}"}

    def mark_event(self, name: str, sent_time: float | None = None):
        recv = time.time()
        delta = recv - sent_time if sent_time is not None else None
        self.events.append({'name': name, 'time_sent': sent_time, 'time_received': recv, 'position': self.camera_position, 'delta': delta})
        self.event_logger.log(sent_time, recv, self.camera_position, name)
        if self.socket_server and self.socket_server.is_client_connected():
            self.socket_server.send_message({
                "type": "event",
                "name": name,
                "time_sent": sent_time,
                "time_received": recv,
                "delta": delta,
                "position": self.camera_position,
                "trial_mode": self.trial_mode,
                "controller": self.controller.info(),
            })
        print(f'EVENT {name} {recv} {self.camera_position} {delta}')
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.json")
    parser.add_argument("--dev", action="store_true", help="Run without hardware")
    parser.add_argument("--socket-host", default="127.0.0.1", help="Host interface for the control socket")
    parser.add_argument("--socket-port", type=int, default=8765, help="TCP port for the control socket")
    parser.add_argument("--no-socket", action="store_true", help="Disable the control socket server")
    parser.add_argument("--status-interval", type=float, default=0.1, help="Seconds between status messages")
    args = parser.parse_args()
    app = MousePortal(
        args.cfg,
        dev=args.dev,
        socket_host=args.socket_host,
        socket_port=args.socket_port,
        enable_socket=not args.no_socket,
        status_interval=args.status_interval,
    )
    app.run()
