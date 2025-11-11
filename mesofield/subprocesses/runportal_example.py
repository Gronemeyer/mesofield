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

import copy
import json
import sys
import csv
import os
import time
import serial
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Optional, List, Mapping
from dataclasses import dataclass, field

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog, GraphicsPipe, Filename
from direct.showbase import DirectObject
from direct.fsm.FSM import FSM
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from portal_socket import PortalServer

try:  # Support running as ``python -m mesofield.gui.runportal`` and standalone
    from . import portal_protocol  # type: ignore[import-error]
except ImportError:  # pragma: no cover - executed when run as a script
    import portal_protocol  # type: ignore[import-not-found]

try:  # Local planner utilities are optional when running standalone
    from mesofield.protocols.experiment_logic import StructuredTrial, build_structured_trials
except ImportError:  # pragma: no cover - executed when run as a script or out of tree
    try:
        from .experiment_logic import StructuredTrial, build_structured_trials  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - executed when run fully standalone
        try:
            from experiment_logic import StructuredTrial, build_structured_trials  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - graceful fallback when helpers are unavailable
            StructuredTrial = None  # type: ignore[assignment]

            def build_structured_trials(definition: Dict[str, Any]):  # type: ignore[override]
                return [], {}

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
    """Load configuration parameters from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)

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
        # Safe default passthrough so that accidental use of the base class does not
        # crash the portal loop. Subclasses override this with specific behaviour.
        return input_speed

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


def _plan_normalise_mode(value: Optional[str], default: str) -> str:
    if not value:
        return default
    return value.strip().lower() if value.strip().lower() in {"open_loop", "closed_loop"} else default


def _plan_coerce_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result <= 0:
        return default
    return result


def _plan_clone_segments(raw: Any) -> Optional[list[Dict[str, Any]]]:
    if not isinstance(raw, list):
        return None
    cloned: list[Dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            cloned.append(dict(entry))
    return cloned or None


@dataclass
class TrialDefinition:
    index: int
    label: str
    mode: str
    duration: Optional[float] = None
    segments: Optional[list[Dict[str, Any]]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    source: str = "plan"
    sequence_index: Optional[int] = None

    def merged(
        self,
        *,
        index: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
        mode_hint: Optional[str] = None,
        source: Optional[str] = None,
        default_mode: str,
    ) -> "TrialDefinition":
        payload: Dict[str, Any] = {}
        if overrides:
            payload.update(overrides)
        if "trial" in payload and isinstance(payload["trial"], dict):
            nested = dict(payload.pop("trial"))
            nested.update(payload)
            payload = nested

        mode = _plan_normalise_mode(
            payload.get("mode") or payload.get("trial_type") or mode_hint or self.mode,
            default_mode,
        )
        label = payload.get("label") or payload.get("event_name") or self.label
        if not isinstance(label, str) or not label.strip():
            label = f"trial_{index or self.index}"
        duration = _plan_coerce_float(payload.get("duration"), default=self.duration)
        seg_override = payload.get("segments") or payload.get("schedule")
        segments = _plan_clone_segments(seg_override) if seg_override is not None else None
        if segments is None:
            segments = _plan_clone_segments(self.segments)
        parameters: Dict[str, Any] = dict(self.parameters or {})
        param_override = payload.get("parameters")
        if isinstance(param_override, dict):
            parameters.update(param_override)
        for key, value in payload.items():
            if key in {
                "mode",
                "trial_type",
                "duration",
                "segments",
                "schedule",
                "parameters",
                "label",
                "event_name",
                "trial",
            }:
                continue
            parameters.setdefault(key, value)
        return TrialDefinition(
            index=index if index is not None else self.index,
            label=label,
            mode=mode,
            duration=duration,
            segments=segments,
            parameters=parameters,
            source=source or self.source,
            sequence_index=self.sequence_index,
        )

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "trial_type": self.mode,
            "event_name": f"trial_start:{self.label}",
            "trial_label": self.label,
            "trial_index": self.index,
            "trial_source": self.source,
        }
        if self.sequence_index is not None:
            payload["trial_sequence_index"] = self.sequence_index
        if self.duration is not None:
            payload["duration"] = self.duration
        if self.segments is not None:
            payload["segments"] = _plan_clone_segments(self.segments)
        if self.parameters:
            payload["parameters"] = dict(self.parameters)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "mode": self.mode,
            "duration": self.duration,
            "parameters": dict(self.parameters or {}),
            "source": self.source,
            "sequence_index": self.sequence_index,
        }


def _plan_from_structured_definition(cfg: Dict[str, Any], default_mode: str) -> List[TrialDefinition]:
    """Convert the new block/trial/routine format into :class:`TrialDefinition` objects."""

    if build_structured_trials is None:
        return []
    if not isinstance(cfg, dict) or "blocks" not in cfg:
        return []

    try:
        structured_trials, metadata = build_structured_trials(cfg)
    except Exception:
        return []

    if not structured_trials:
        return []

    required_keys = metadata.get("required_keys")
    timing_cfg = metadata.get("timing") if isinstance(metadata.get("timing"), dict) else {}
    schema_version = metadata.get("schema_version")
    rng_seed = metadata.get("rng_seed")

    trials: List[TrialDefinition] = []
    for entry in structured_trials:
        parameters: Dict[str, Any] = {
            "plan_source": "structured_blocks",
        }
        if entry.block_name:
            parameters["block_name"] = entry.block_name
        if entry.routines:
            parameters["routines"] = copy.deepcopy(entry.routines)
        parameters["structured_definition"] = copy.deepcopy(entry.definition)
        if required_keys:
            parameters["required_keys"] = list(required_keys)
        if timing_cfg:
            parameters["timing"] = dict(timing_cfg)
        if schema_version is not None:
            parameters["plan_schema_version"] = schema_version
        if rng_seed is not None:
            parameters["plan_rng_seed"] = rng_seed

        clean_parameters = {
            key: value
            for key, value in parameters.items()
            if value not in (None, [], {})
        }

        trials.append(
            TrialDefinition(
                index=entry.sequence_index,
                label=entry.label,
                mode=_plan_normalise_mode(entry.mode, default_mode),
                duration=entry.duration,
                parameters=clean_parameters,
                source="structured_plan",
                sequence_index=entry.sequence_index,
            )
        )

    return trials


class ExperimentPlan:
    def __init__(
        self,
        trials: List[TrialDefinition],
        *,
        inter_trial_interval: float = 0.0,
        auto_advance: bool = False,
        auto_start: bool = False,
        loop: bool = False,
        default_mode: str = "closed_loop",
    ) -> None:
        self._trials = trials
        self.inter_trial_interval = max(0.0, inter_trial_interval)
        self.auto_advance = auto_advance
        self.auto_start = auto_start
        self.loop = loop
        self.default_mode = default_mode
        self.reset()

    @classmethod
    def from_config(cls, cfg: Any, *, default_mode: str = "closed_loop") -> Optional["ExperimentPlan"]:
        if not isinstance(cfg, dict):
            return None

        trials: List[TrialDefinition] = _plan_from_structured_definition(cfg, default_mode)

        if not trials:
            trials_cfg = cfg.get("trials")
            if isinstance(trials_cfg, list):
                for idx, entry in enumerate(trials_cfg, start=1):
                    if not isinstance(entry, dict):
                        continue
                    mode = _plan_normalise_mode(entry.get("mode"), default_mode)
                    label = entry.get("label")
                    if not isinstance(label, str) or not label.strip():
                        label = f"trial_{idx}"
                    duration = _plan_coerce_float(entry.get("duration"))
                    segments = _plan_clone_segments(entry.get("segments") or entry.get("schedule"))
                    parameters: Dict[str, Any] = {}
                    param_section = entry.get("parameters")
                    if isinstance(param_section, dict):
                        parameters.update(param_section)
                    for key, value in entry.items():
                        if key in {"mode", "label", "duration", "segments", "schedule", "parameters"}:
                            continue
                        parameters.setdefault(key, value)
                    trials.append(
                        TrialDefinition(
                            index=idx,
                            label=label,
                            mode=mode,
                            duration=duration,
                            segments=segments,
                            parameters=parameters,
                            source="config",
                            sequence_index=idx,
                        )
                    )

        if not trials:
            count = cfg.get("trial_count")
            try:
                count = int(count)
            except (TypeError, ValueError):
                count = 0
            if count and count > 0:
                duration = _plan_coerce_float(cfg.get("trial_duration"))
                mode = _plan_normalise_mode(cfg.get("mode") or cfg.get("trial_mode"), default_mode)
                label_template = cfg.get("trial_label_template")
                if not isinstance(label_template, str) or not label_template.strip():
                    label_template = "trial_{index}"
                base_parameters: Dict[str, Any] = {}
                param_section = cfg.get("parameters")
                if isinstance(param_section, dict):
                    base_parameters.update(param_section)
                for idx in range(1, count + 1):
                    label = label_template.format(index=idx)
                    trials.append(
                        TrialDefinition(
                            index=idx,
                            label=label,
                            mode=mode,
                            duration=duration,
                            parameters=dict(base_parameters),
                            source="config",
                            sequence_index=idx,
                        )
                    )

        if not trials:
            return None

        return cls(
            trials,
            inter_trial_interval=_plan_coerce_float(cfg.get("inter_trial_interval"), default=0.0) or 0.0,
            auto_advance=bool(cfg.get("auto_advance", False)),
            auto_start=bool(cfg.get("auto_start", False)),
            loop=bool(cfg.get("loop", False)),
            default_mode=default_mode,
        )

    def reset(self) -> None:
        self._cursor = 0
        self._global_index = 0

    def has_remaining(self) -> bool:
        if not self._trials:
            return False
        if self.loop:
            return True
        return self._cursor < len(self._trials)

    def remaining_trials(self) -> Optional[int]:
        if not self._trials:
            return 0
        if self.loop:
            return None
        return max(0, len(self._trials) - self._cursor)

    def next_trial(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        mode_hint: Optional[str] = None,
        source: str = "plan",
    ) -> Optional[TrialDefinition]:
        if not self._trials:
            return None
        if self._cursor >= len(self._trials):
            if not self.loop:
                return None
            self._cursor = 0
        base = self._trials[self._cursor]
        self._cursor += 1
        self._global_index += 1
        derived = base.merged(
            index=self._global_index,
            overrides=overrides,
            mode_hint=mode_hint,
            source=source,
            default_mode=self.default_mode,
        )
        return derived

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
    """Class for generating infinite corridor geometric rendering."""

    def __init__(
        self,
        base: ShowBase,
        config: Dict[str, Any],
        *,
        asset_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the corridor by creating segments for each face.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        self.base = base
        self.asset_dir = Path(asset_dir) if asset_dir else None
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
            
    def _resolve_texture_path(self, texture_path: str) -> Filename:
        if not texture_path:
            return Filename()
        candidate = Path(texture_path)
        if not candidate.is_absolute():
            if self.asset_dir:
                candidate = self.asset_dir / candidate
            else:
                candidate = candidate.resolve()
        return Filename.from_os_specific(str(candidate.resolve()))

    def apply_texture(self, node: NodePath, texture_path: str) -> None:
        panda_path = self._resolve_texture_path(texture_path)
        texture: Texture = self.base.loader.loadTexture(panda_path)
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
        panda_path = self._resolve_texture_path(texture_path)
        texture = self.base.loader.loadTexture(panda_path)
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
            self.cfg = load_config(config_file)

        asset_dir_value = self.cfg.get("asset_dir")
        self.asset_dir = Path(asset_dir_value).expanduser().resolve() if asset_dir_value else None

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

        self._plan_state: str = "idle"
        self._loaded_plan_id: Optional[str] = None

        self._setup_experiment_plan()

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

        # Set window properties based on configuration
        window_width = int(self.cfg.get("window_width", display_width))
        window_height = int(self.cfg.get("window_height", display_height))
        if window_width <= 0:
            window_width = display_width
        if window_height <= 0:
            window_height = display_height

        origin_x = self.cfg.get("window_origin_x")
        if origin_x is None:
            origin_x = display_width
        origin_y = self.cfg.get("window_origin_y", 0)
        try:
            origin_x = int(origin_x)
        except (TypeError, ValueError):
            origin_x = display_width
        try:
            origin_y = int(origin_y)
        except (TypeError, ValueError):
            origin_y = 0

        wp: WindowProperties = WindowProperties()
        wp.setSize(window_width, window_height)
        wp.set_origin(origin_x, origin_y)

        if bool(self.cfg.get("fullscreen", False)):
            wp.setFullscreen(True)

        self.dev = True
        self.win.requestProperties(wp)
        self.setFrameRateMeter(bool(self.cfg.get("show_frame_rate_meter", True)))
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
        self.corridor = Corridor(self, self.cfg, asset_dir=self.asset_dir)
        self.segment_length: float = self.cfg["segment_length"]
        
        # Variable to track movement since last recycling.
        self.distance_since_recycle: float = 0.0

        # Movement speed (units per second).
        self.movement_speed = float(self.cfg.get("movement_speed", 10.0))
        
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

        if self.experiment_plan and self.experiment_plan.auto_start:
            self._schedule_next_trial(
                self.experiment_plan.inter_trial_interval,
                reason="auto_start",
            )

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

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------
    def _setup_experiment_plan(self) -> None:
        plan_cfg_raw = self.cfg.get("experiment")
        plan_cfg: Optional[Dict[str, Any]] = None
        if isinstance(plan_cfg_raw, dict):
            try:
                plan_cfg = json.loads(json.dumps(plan_cfg_raw))
            except (TypeError, ValueError):
                plan_cfg = dict(plan_cfg_raw)
        self.experiment_plan: Optional[ExperimentPlan] = (
            ExperimentPlan.from_config(plan_cfg, default_mode=self.default_trial_type)
            if plan_cfg
            else None
        )
        self.active_trial: Optional[Dict[str, Any]] = None
        self._active_trial_definition: Optional[TrialDefinition] = None
        self._trial_timer_task: Optional[Task] = None
        self._scheduled_start_task: Optional[Task] = None
        self._scheduled_start_reason: Optional[str] = None
        if self.experiment_plan:
            self.experiment_plan.reset()
            if self._plan_state not in {"scheduled", "running", "paused"}:
                self._plan_state = "loaded"
        else:
            self._plan_state = "idle"

    def _augment_with_plan_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info["plan_state"] = self._plan_state
        if self._loaded_plan_id:
            info["plan_id"] = self._loaded_plan_id
        else:
            info.pop("plan_id", None)
        if self.experiment_plan:
            remaining = self.experiment_plan.remaining_trials()
            if remaining is not None:
                info["remaining_trials"] = remaining
            else:
                info.pop("remaining_trials", None)
        else:
            info.pop("remaining_trials", None)
        return info

    def _schedule_next_trial(self, delay: float, *, reason: str = "auto") -> None:
        if not self.experiment_plan:
            return
        self._cancel_next_trial_schedule()
        delay_seconds = max(0.0, float(delay))
        self._scheduled_start_task = self.taskMgr.doMethodLater(
            delay_seconds,
            self._auto_start_next_trial,
            f"MousePortalAutoTrial-{time.time():.3f}",
        )
        self._scheduled_start_reason = reason
        if self._plan_state not in {"running", "paused"}:
            self._plan_state = "scheduled"

    def _cancel_next_trial_schedule(self) -> None:
        if self._scheduled_start_task is not None:
            try:
                self._scheduled_start_task.remove()
            except Exception:
                self.taskMgr.remove(self._scheduled_start_task)
            self._scheduled_start_task = None
        self._scheduled_start_reason = None
        if self.experiment_plan and self._plan_state == "scheduled":
            self._plan_state = "loaded"

    def _auto_start_next_trial(self, task: Task) -> Task:
        self._scheduled_start_task = None
        self._scheduled_start_reason = None
        if self.experiment_plan:
            success, info = self._start_planned_trial({}, None, source="auto")
            if not success:
                print(f"NO MORE TRIALS: {info}")
        return Task.done

    def _start_planned_trial(
        self,
        overrides: Optional[Dict[str, Any]],
        sent_time: Optional[float],
        *,
        source: str,
    ) -> tuple[bool, Dict[str, Any]]:
        if not self.experiment_plan:
            return False, {"error": "no_plan"}
        self._cancel_next_trial_schedule()
        effective_overrides = dict(overrides or {})
        mode_hint = effective_overrides.get("mode") or effective_overrides.get("trial_type")
        definition = self.experiment_plan.next_trial(
            effective_overrides if effective_overrides else None,
            mode_hint=mode_hint,
            source=source,
        )
        if not definition:
            if not self.experiment_plan.has_remaining():
                self._plan_state = "completed"
            return False, {"error": "no_trials_remaining"}
        self._active_trial_definition = definition
        payload = definition.to_payload()
        if effective_overrides:
            if effective_overrides.get("event_name"):
                payload["event_name"] = effective_overrides["event_name"]
            for key in ("duration", "segments", "schedule"):
                if key in effective_overrides and effective_overrides[key] is not None:
                    payload[key] = effective_overrides[key]
            if effective_overrides.get("trial_label"):
                payload["trial_label"] = effective_overrides["trial_label"]
        info = self._start_trial(definition.mode, sent_time, payload, source=source)
        info.setdefault("trial", definition.to_dict())
        remaining = self.experiment_plan.remaining_trials()
        if remaining is not None:
            info["remaining_trials"] = remaining
        self._plan_state = "running"
        self._augment_with_plan_info(info)
        return True, info

    def _set_active_trial_metadata(self, payload: Mapping[str, Any], *, source: str) -> None:
        metadata: Dict[str, Any] = {
            "label": payload.get("trial_label"),
            "index": payload.get("trial_index"),
            "sequence_index": payload.get("trial_sequence_index"),
            "mode": self.trial_mode,
            "source": payload.get("trial_source", source),
            "parameters": dict(payload.get("parameters") or {}),
            "planned_duration": payload.get("duration") or payload.get("planned_duration"),
            "started_at": self.trial_started_at,
        }
        if self._active_trial_definition:
            metadata.setdefault("label", self._active_trial_definition.label)
            metadata.setdefault("index", self._active_trial_definition.index)
            metadata.setdefault("sequence_index", self._active_trial_definition.sequence_index)
            metadata.setdefault("mode", self._active_trial_definition.mode)
            if not metadata.get("parameters"):
                metadata["parameters"] = dict(self._active_trial_definition.parameters or {})
        if self.experiment_plan:
            remaining = self.experiment_plan.remaining_trials()
            if remaining is not None:
                metadata["remaining_plan_trials"] = remaining
            metadata["plan_auto_advance"] = self.experiment_plan.auto_advance
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        clean_metadata.setdefault("mode", self.trial_mode)
        clean_metadata.setdefault("label", f"{self.trial_mode}_trial")
        self.active_trial = clean_metadata

    def _schedule_trial_end(self, payload: Mapping[str, Any]) -> None:
        duration_value = payload.get("duration") or payload.get("planned_duration")
        if duration_value is None:
            return
        try:
            seconds = float(duration_value)
        except (TypeError, ValueError):
            return
        if seconds <= 0:
            return
        self._cancel_trial_timer()
        self._trial_timer_task = self.taskMgr.doMethodLater(
            seconds,
            self._auto_stop_current_trial,
            f"MousePortalTrialTimer-{time.time():.3f}",
        )

    def _cancel_trial_timer(self) -> None:
        if self._trial_timer_task is not None:
            try:
                self._trial_timer_task.remove()
            except Exception:
                self.taskMgr.remove(self._trial_timer_task)
            self._trial_timer_task = None

    def _auto_stop_current_trial(self, task: Task) -> Task:
        self._trial_timer_task = None
        self._stop_trial(None, source="auto_timer")
        return Task.done

    def _start_trial(
        self,
        trial_type: str,
        sent_time: Optional[float],
        payload: Dict[str, Any],
        *,
        source: str = "manual",
    ) -> Dict[str, Any]:
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

        self._cancel_trial_timer()
        if source != "auto":
            self._cancel_next_trial_schedule()

        if self.controller is not controller:
            self.controller.stop()
        self.controller = controller
        self.controller.start()
        self.trial_mode = requested
        self.trial_started_at = time.time()

        self._set_active_trial_metadata(payload, source=source)

        event_name = payload.get("event_name") or f"start_{requested}"
        extras = {"trial": dict(self.active_trial)} if self.active_trial else None
        self.mark_event(event_name, sent_time, extra=extras)
        self.fsm.request("Running")
        self._push_status(force=True)
        self._schedule_trial_end(payload)
        info = {
            "state": self.state_name,
            "trial_mode": self.trial_mode,
            "controller": self.controller.info(),
        }
        if self.active_trial:
            info["trial"] = dict(self.active_trial)
        return info

    def _stop_trial(
        self,
        sent_time: Optional[float],
        *,
        source: str = "manual",
    ) -> Dict[str, Any]:
        self._cancel_trial_timer()
        if source != "auto_advance":
            self._cancel_next_trial_schedule()

        trial_snapshot: Optional[Dict[str, Any]] = None
        if self.active_trial:
            trial_snapshot = dict(self.active_trial)
            trial_snapshot["ended_at"] = time.time()
            trial_snapshot["end_source"] = source

        if self.trial_mode != "idle":
            stop_name = (
                f"trial_end:{trial_snapshot['label']}"
                if trial_snapshot and trial_snapshot.get("label")
                else f"stop_{self.trial_mode}"
            )
            extras = {"trial": trial_snapshot} if trial_snapshot else None
            self.mark_event(stop_name, sent_time, extra=extras)

        self.controller.stop()
        self.controller = self.closed_loop_controller
        self.controller.start()
        self.trial_mode = "idle"
        self.trial_started_at = None
        self.fsm.request("Idle")
        self._push_status(force=True)

        self.active_trial = None
        self._active_trial_definition = None

        info = {
            "state": self.state_name,
            "trial_mode": self.trial_mode,
            "controller": self.controller.info(),
        }
        if trial_snapshot:
            info["trial"] = trial_snapshot

        if self.experiment_plan:
            if self._plan_state != "paused":
                if self.experiment_plan.has_remaining():
                    self._plan_state = "loaded"
                else:
                    self._plan_state = "completed"
        else:
            if self._plan_state != "paused":
                self._plan_state = "idle"

        if (
            self.experiment_plan
            and self.experiment_plan.auto_advance
            and self.experiment_plan.has_remaining()
            and self._plan_state != "paused"
        ):
            self._schedule_next_trial(
                self.experiment_plan.inter_trial_interval,
                reason="auto_advance",
            )

        self._augment_with_plan_info(info)
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
            status = portal_protocol.build_message(
                "status",
                time=now,
                position=self.camera_position,
                velocity=self.camera_velocity,
                state=self.state_name,
                trial_mode=self.trial_mode,
                input_velocity=self.current_input_speed,
                controller=self.controller.info(),
            )
            if self.trial_started_at is not None:
                status["trial_elapsed"] = now - self.trial_started_at
            status["plan_state"] = self._plan_state
            if self._loaded_plan_id:
                status["plan_id"] = self._loaded_plan_id
            if self.experiment_plan:
                remaining = self.experiment_plan.remaining_trials()
                if remaining is not None:
                    status["remaining_trials"] = remaining
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
        if msg_type:
            ok, issues, _ = portal_protocol.validate_message(message, portal_protocol.HOST_TO_PORTAL)
            if not ok:
                print(
                    f"SOCKET WARN invalid payload for {msg_type}: {', '.join(issues)}"
                )
                sys.stdout.flush()
        if msg_type == "error":
            print(f"SOCKET ERROR {message.get('message')} raw={message.get('raw')}")
            sys.stdout.flush()
            return
        if msg_type == "ping":
            if self.socket_server:
                reply_payload = portal_protocol.build_message("pong")
                if "client_time" in message:
                    reply_payload["client_time"] = message["client_time"]
                if "request_id" in message:
                    reply_payload["request_id"] = message["request_id"]
                self.socket_server.send_message(reply_payload)
            return

        command = message.get("command") or msg_type
        if not command:
            return

        canonical_command, spec = portal_protocol.lookup_command(command)
        if canonical_command is None:
            canonical_command = command

        if spec is not None:
            ok_cmd, issues_cmd, _ = portal_protocol.validate_command_payload(canonical_command, message)
            if not ok_cmd:
                print(
                    f"SOCKET WARN command {canonical_command} missing fields: {', '.join(issues_cmd)}"
                )
                sys.stdout.flush()

        sent_time: Optional[float] = None
        for key in ("sent_time", "client_time", "timestamp"):
            value = message.get(key)
            if isinstance(value, (int, float)):
                sent_time = float(value)
                break

        try:
            success, info = self._execute_command(canonical_command, sent_time, message, source="socket")
        except Exception as exc:  # pragma: no cover - defensive
            success = False
            info = {"error": f"exception:{exc}"}
            print(f"SOCKET COMMAND ERROR {canonical_command}: {exc}")
            sys.stdout.flush()

        status = "ok" if success else "error"
        if self.socket_server:
            ack_payload = portal_protocol.build_message("ack", command=canonical_command, status=status)
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

        if cmd == "load_experiment_plan":
            plan_payload = payload.get("plan")
            plan_config: Optional[Dict[str, Any]] = None
            if isinstance(plan_payload, str):
                try:
                    decoded = json.loads(plan_payload)
                except (TypeError, ValueError):
                    return False, self._augment_with_plan_info({"error": "invalid_plan_payload"})
                if isinstance(decoded, dict):
                    plan_config = decoded
            elif isinstance(plan_payload, Mapping):
                try:
                    plan_config = json.loads(json.dumps(plan_payload))
                except (TypeError, ValueError):
                    plan_config = dict(plan_payload)
            if plan_config is None:
                return False, self._augment_with_plan_info({"error": "plan_missing"})

            previous_plan = self.experiment_plan
            previous_plan_id = self._loaded_plan_id

            if self.trial_mode != "idle":
                self._stop_trial(sent_time, source="plan_reload")
            self._cancel_next_trial_schedule()
            self._cancel_trial_timer()

            default_mode_value = payload.get("default_mode")
            if isinstance(default_mode_value, str):
                resolved_default_mode = default_mode_value.strip().lower()
            else:
                resolved_default_mode = self.default_trial_type
            if resolved_default_mode not in {"open_loop", "closed_loop"}:
                resolved_default_mode = self.default_trial_type

            plan = ExperimentPlan.from_config(plan_config, default_mode=resolved_default_mode)
            if plan is None:
                self.experiment_plan = previous_plan
                self._loaded_plan_id = previous_plan_id
                self._plan_state = "loaded" if previous_plan else "idle"
                return False, self._augment_with_plan_info({"error": "plan_invalid"})

            self.experiment_plan = plan
            self.experiment_plan.reset()
            if "auto_advance" in payload:
                self.experiment_plan.auto_advance = self._coerce_bool(
                    payload.get("auto_advance"),
                    self.experiment_plan.auto_advance,
                )
            if "auto_start" in payload:
                self.experiment_plan.auto_start = self._coerce_bool(
                    payload.get("auto_start"),
                    self.experiment_plan.auto_start,
                )
            if "inter_trial_interval" in payload:
                try:
                    self.experiment_plan.inter_trial_interval = float(payload["inter_trial_interval"])
                except (TypeError, ValueError):
                    pass

            self.active_trial = None
            self._active_trial_definition = None
            plan_id_value = payload.get("plan_id") or payload.get("id") or payload.get("name")
            self._loaded_plan_id = str(plan_id_value) if plan_id_value is not None else None
            self._plan_state = "loaded"

            if self.experiment_plan.auto_start:
                self._schedule_next_trial(
                    self.experiment_plan.inter_trial_interval,
                    reason="auto_start",
                )

            info = {
                "state": self.state_name,
                "trial_mode": self.trial_mode,
                "controller": self.controller.info(),
            }
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "run_experiment":
            if not self.experiment_plan:
                return False, self._augment_with_plan_info({"error": "no_plan"})
            if payload.get("plan_id") and self._loaded_plan_id and str(payload["plan_id"]) != self._loaded_plan_id:
                return False, self._augment_with_plan_info({"error": "plan_mismatch"})
            restart_flag = payload.get("restart")
            restart = True if restart_flag is None else self._coerce_bool(restart_flag, True)
            if restart:
                self.experiment_plan.reset()
            overrides_payload = payload.get("trial_overrides") or payload.get("overrides")
            overrides_dict = overrides_payload if isinstance(overrides_payload, dict) else None
            success, info = self._start_planned_trial(overrides_dict, sent_time, source="plan")
            self._augment_with_plan_info(info)
            return success, info

        if cmd == "pause_experiment":
            if not self.experiment_plan:
                return False, self._augment_with_plan_info({"error": "no_plan"})
            if payload.get("plan_id") and self._loaded_plan_id and str(payload["plan_id"]) != self._loaded_plan_id:
                return False, self._augment_with_plan_info({"error": "plan_mismatch"})
            if self._plan_state != "running":
                return False, self._augment_with_plan_info({"error": "not_running"})
            self._plan_state = "paused"
            info = self._stop_trial(sent_time, source="pause")
            if payload.get("reason"):
                info["reason"] = payload["reason"]
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "resume_experiment":
            if not self.experiment_plan:
                return False, self._augment_with_plan_info({"error": "no_plan"})
            if payload.get("plan_id") and self._loaded_plan_id and str(payload["plan_id"]) != self._loaded_plan_id:
                return False, self._augment_with_plan_info({"error": "plan_mismatch"})
            if self._plan_state != "paused":
                return False, self._augment_with_plan_info({"error": "not_paused"})
            overrides_payload = payload.get("trial_overrides") or payload.get("overrides")
            overrides_dict = overrides_payload if isinstance(overrides_payload, dict) else None
            success, info = self._start_planned_trial(overrides_dict, sent_time, source="resume")
            self._augment_with_plan_info(info)
            return success, info

        if cmd == "abort_experiment":
            if payload.get("plan_id") and self._loaded_plan_id and str(payload["plan_id"]) != self._loaded_plan_id:
                return False, self._augment_with_plan_info({"error": "plan_mismatch"})
            if self.trial_mode != "idle":
                info = self._stop_trial(sent_time, source="abort")
            else:
                info = {
                    "state": self.state_name,
                    "trial_mode": self.trial_mode,
                    "controller": self.controller.info(),
                }
                self._augment_with_plan_info(info)
            self._cancel_next_trial_schedule()
            self._cancel_trial_timer()
            clear_plan = self._coerce_bool(payload.get("clear_plan"), False)
            if clear_plan:
                self.experiment_plan = None
                self._loaded_plan_id = None
                self._plan_state = "idle"
            else:
                if self.experiment_plan:
                    self.experiment_plan.reset()
                    if self._plan_state != "paused":
                        self._plan_state = "loaded"
                else:
                    self._plan_state = "idle"
            self.active_trial = None
            self._active_trial_definition = None
            self._augment_with_plan_info(info)
            if payload.get("reason"):
                info["reason"] = payload["reason"]
            return True, info

        if cmd == "start_trial":
            use_plan = self.experiment_plan is not None and payload.get("use_plan", True) and not payload.get("manual")
            if use_plan:
                success, info = self._start_planned_trial(payload, sent_time, source=source)
                return success, info
            trial_type = payload.get("trial_type") or payload.get("mode") or self.default_trial_type
            info = self._start_trial(trial_type, sent_time, payload, source=source)
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "stop_trial":
            info = self._stop_trial(sent_time, source=source)
            return True, info

        if cmd == "start_open_loop":
            info = self._start_trial("open_loop", sent_time, payload, source=source)
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "start_closed_loop":
            info = self._start_trial("closed_loop", sent_time, payload, source=source)
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "set_texture":
            face = payload.get("face") or payload.get("surface")
            texture = payload.get("texture") or payload.get("path")
            if not face or not texture:
                return False, {"error": "missing_face_or_texture"}
            self.corridor.set_texture(face, texture)
            info = {"face": face, "texture": texture, "trial_mode": self.trial_mode}
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "mark_event":
            name = payload.get("name") or payload.get("event") or "event"
            extra = {k: v for k, v in payload.items() if k not in {"name", "event", "type", "command"}}
            if not extra:
                extra = None
            self.mark_event(name, sent_time, extra=extra)
            info = {"event": name, "trial_mode": self.trial_mode, "controller": self.controller.info()}
            self._augment_with_plan_info(info)
            return True, info

        if cmd in {"end", "shutdown", "quit", "exit"}:
            self._shutdown_requested = True
            info = {"state": self.state_name}
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "get_status":
            self._push_status(force=True)
            info = {
                "state": self.state_name,
                "trial_mode": self.trial_mode,
                "controller": self.controller.info(),
            }
            self._augment_with_plan_info(info)
            return True, info

        if cmd == "ping" and source == "socket":
            if self.socket_server:
                self.socket_server.send_message({"type": "pong"})
            return True, {}

        error_info = {"error": f"unknown_command:{cmd}"}
        self._augment_with_plan_info(error_info)
        return False, error_info

    def mark_event(
        self,
        name: str,
        sent_time: float | None = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        recv = time.time()
        delta = recv - sent_time if sent_time is not None else None
        record = {
            "name": name,
            "time_sent": sent_time,
            "time_received": recv,
            "position": self.camera_position,
            "delta": delta,
        }
        if self.active_trial:
            record["trial"] = dict(self.active_trial)
        if extra:
            record["extra"] = extra
        self.events.append(record)
        self.event_logger.log(sent_time, recv, self.camera_position, name)
        if self.socket_server and self.socket_server.is_client_connected():
            message = portal_protocol.build_message(
                "event",
                name=name,
                time_sent=sent_time,
                time_received=recv,
                delta=delta,
                position=self.camera_position,
                trial_mode=self.trial_mode,
                controller=self.controller.info(),
            )
            if self.active_trial:
                message["trial"] = dict(self.active_trial)
            if extra:
                for key, value in extra.items():
                    if key in {"type", "name"}:
                        continue
                    message[key] = value
            self.socket_server.send_message(message)
        print(f"EVENT {name} {recv} {self.camera_position} {delta}")
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
