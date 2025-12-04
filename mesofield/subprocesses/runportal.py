#!/usr/bin/env python3
"""MousePortal runner backed by Panda3D and ZeroMQ."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import zmq
from direct.fsm.FSM import FSM  # type: ignore[import]
from direct.gui.OnscreenText import OnscreenText  # type: ignore[import]
from direct.showbase.ShowBase import ShowBase  # type: ignore[import]
from direct.showbase.ShowBaseGlobal import globalClock  # type: ignore[import]
from direct.task import Task  # type: ignore[import]
from panda3d.core import CardMaker, Fog, NodePath, TextNode, WindowProperties, loadPrcFileData  # type: ignore[import]

TaskType = Any

loadPrcFileData("", "load-display pandagl")
loadPrcFileData("", "window-title MousePortal")

try:  # pragma: no cover - optional dependency
    from mesofield.subprocesses.experiment_engine import Engine, ExperimentSpec
except ImportError:  # pragma: no cover - optional dependency absent
    Engine = None
    ExperimentSpec = None

logger = logging.getLogger("MousePortalApp")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class DataLogger:
    """CSV logger for kinematic data."""

    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = None

    def _ensure_writer(self) -> None:
        import csv

        if self._writer is None:
            fresh_file = self._file.tell() == 0
            self._writer = csv.DictWriter(self._file, fieldnames=["timestamp", "position", "velocity"])
            if fresh_file:
                self._writer.writeheader()

    def log(self, timestamp: float, position: float, velocity: float) -> None:
        self._ensure_writer()
        assert self._writer is not None
        self._writer.writerow({"timestamp": timestamp, "position": position, "velocity": velocity})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class EventLogger:
    """Append-only JSON log."""

    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Dict[str, Any]] = []

    def record(self, payload: Dict[str, Any]) -> None:
        self._events.append(payload)

    def close(self) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._events, fh, indent=2)


class ZmqEventPublisher:
    def __init__(self, endpoint: str) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)

    def publish(self, payload: Dict[str, Any]) -> None:
        self._socket.send_multipart([b"exp-event", json.dumps(payload).encode("utf-8")])

    def close(self) -> None:
        self._socket.close(linger=0)


class ZmqCommandServer:
    def __init__(self, endpoint: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._handler = handler

    def poll(self, timeout_ms: int) -> None:
        events = dict(self._poller.poll(timeout_ms))
        if self._socket not in events:
            return
        message = self._socket.recv_json()
        try:
            reply = self._handler(message) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MousePortal command failed: %s", exc)
            reply = {"ok": False, "error": str(exc)}
        self._socket.send_json(reply)

    def close(self) -> None:
        self._poller.unregister(self._socket)
        self._socket.close(linger=0)


class EncoderData:
    def __init__(self, timestamp: float = 0.0, distance: float = 0.0, speed: float = 0.0) -> None:
        self.timestamp = timestamp
        self.distance = distance
        self.speed = speed


class SerialInputManager:
    def __init__(self, serial_port: str, baudrate: int = 57600) -> None:
        import serial  # type: ignore

        self.serial = serial.Serial(serial_port, baudrate, timeout=1)
        self.data = EncoderData()

    def _read_serial(self, task: TaskType) -> TaskType:
        try:
            raw_line = self.serial.readline()
            if raw_line:
                line = raw_line.decode("utf-8", errors="replace").strip()
                parts = line.split(",")
                if len(parts) >= 2:
                    self.data.speed = float(parts[1])
        except Exception:  # pragma: no cover - serial errors are non fatal
            pass
        return Task.cont


class DummyInputManager:
    def __init__(self) -> None:
        self.data = EncoderData()

    def _read_serial(self, task: TaskType) -> TaskType:
        return Task.cont


class ExperimentFSM(FSM):
    def __init__(self, app: "MousePortalApp") -> None:
        super().__init__("ExperimentFSM")
        self.app = app

    def enterIdle(self) -> None:
        self.app.update_state_label("Idle")
        if self.app.engine and self.app.engine.running:
            self.app.engine.stop()

    def enterRunning(self) -> None:
        self.app.update_state_label("Running")
        if self.app.engine:
            self.app.engine.start()


class Corridor:
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        self.base = base
        self.segment_length = float(config.get("segment_length", 10.0))
        self.corridor_width = float(config.get("corridor_width", 8.0))
        self.wall_height = float(config.get("wall_height", 3.0))
        self.num_segments = int(config.get("num_segments", 20))
        self.parent = base.render.attachNewNode("corridor")
        self.left_segments: list[NodePath] = []
        self.right_segments: list[NodePath] = []
        self.ceiling_segments: list[NodePath] = []
        self.floor_segments: list[NodePath] = []
        self._build_segments(config)

    def _build_segments(self, config: Dict[str, Any]) -> None:
        textures = {
            "left": config.get("left_wall_texture", ""),
            "right": config.get("right_wall_texture", ""),
            "ceiling": config.get("ceiling_texture", ""),
            "floor": config.get("floor_texture", ""),
        }
        for i in range(-2, self.num_segments):
            y_pos = i * self.segment_length
            self._apply_texture(self._make_wall(self.left_segments, y_pos, True), textures["left"])
            self._apply_texture(self._make_wall(self.right_segments, y_pos, False), textures["right"])
            self._apply_texture(self._make_ceiling(y_pos), textures["ceiling"])
            self._apply_texture(self._make_floor(y_pos), textures["floor"])

    def _make_wall(self, collection: list[NodePath], y_pos: float, left: bool) -> NodePath:
        card = CardMaker("left_wall" if left else "right_wall")
        card.setFrame(0, self.segment_length, 0, self.wall_height)
        node = self.parent.attachNewNode(card.generate())
        node.setPos((-self.corridor_width / 2 if left else self.corridor_width / 2), y_pos, 0)
        node.setHpr(90 if left else -90, 0, 0)
        collection.append(node)
        return node

    def _make_ceiling(self, y_pos: float) -> NodePath:
        card = CardMaker("ceiling")
        card.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
        node = self.parent.attachNewNode(card.generate())
        node.setPos(0, y_pos, self.wall_height)
        node.setHpr(0, 90, 0)
        self.ceiling_segments.append(node)
        return node

    def _make_floor(self, y_pos: float) -> NodePath:
        card = CardMaker("floor")
        card.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
        node = self.parent.attachNewNode(card.generate())
        node.setPos(0, y_pos, 0)
        node.setHpr(0, -90, 0)
        self.floor_segments.append(node)
        return node

    def _apply_texture(self, node: NodePath, texture_path: str) -> None:
        if texture_path:
            try:
                texture = self.base.loader.loadTexture(texture_path)
                node.setTexture(texture)
            except Exception as exc:
                logger.warning("Failed to load texture %s: %s", texture_path, exc)
        else:
            node.clearTexture()

    def recycle(self, direction: str) -> None:
        if direction == "forward":
            self._shift_forward(self.left_segments)
            self._shift_forward(self.right_segments)
            self._shift_forward(self.ceiling_segments)
            self._shift_forward(self.floor_segments)
        else:
            self._shift_backward(self.left_segments)
            self._shift_backward(self.right_segments)
            self._shift_backward(self.ceiling_segments)
            self._shift_backward(self.floor_segments)

    def _shift_forward(self, segments: list[NodePath]) -> None:
        segment = segments.pop(0)
        segment.setY(segments[-1].getY() + self.segment_length)
        segments.append(segment)

    def _shift_backward(self, segments: list[NodePath]) -> None:
        segment = segments.pop()
        segment.setY(segments[0].getY() - self.segment_length)
        segments.insert(0, segment)

    def set_texture(self, face: str, texture_path: str) -> None:
        mapping = {
            "left": self.left_segments,
            "right": self.right_segments,
            "ceiling": self.ceiling_segments,
            "floor": self.floor_segments,
        }
        nodes = mapping.get(face.lower())
        if not nodes:
            raise ValueError(f"Unknown corridor face: {face}")
        for node in nodes:
            self._apply_texture(node, texture_path)


class MousePortalApp(ShowBase):
    def __init__(self, *, cfg_path: str, command_endpoint: str, event_endpoint: str, dev: bool = False) -> None:
        ShowBase.__init__(self)
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)
        self.dev = dev or self.cfg.get("serial_port") == "dev"
        self.camera_position = 0.0
        self.camera_velocity = 0.0
        self.segment_length = float(self.cfg.get("segment_length", 10.0))
        self.distance_since_recycle = 0.0
        self.gain = float(self.cfg.get("gain", 1.0))
        self.input_reversed = bool(self.cfg.get("input_reversed", False))
        self.treadmill: SerialInputManager | DummyInputManager | None = None
        self.engine = None
        self.engine_spec = None

        self._setup_window()
        self._init_input()
        self._setup_corridor()
        self._setup_fog()
        self._setup_logging()
        self._setup_zmq(command_endpoint, event_endpoint)
        self._setup_engine()
        self._setup_tasks()

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(int(self.cfg.get("window_width", 800)), int(self.cfg.get("window_height", 600)))
        props.setTitle("MousePortal - Virtual Corridor")
        self.win.requestProperties(props)
        self.disableMouse()

    def _init_input(self) -> None:
        self.key_map = {"forward": False, "backward": False}
        self.accept("arrow_up", self._set_key, ["forward", True])
        self.accept("arrow_up-up", self._set_key, ["forward", False])
        self.accept("arrow_down", self._set_key, ["backward", True])
        self.accept("arrow_down-up", self._set_key, ["backward", False])
        self.accept("escape", self.userExit)
        if self.dev:
            self.treadmill = DummyInputManager()
        else:
            port = self.cfg.get("serial_port")
            if not port:
                raise ValueError("MousePortal config must set serial_port")
            self.treadmill = SerialInputManager(port)
            self.taskMgr.add(self.treadmill._read_serial, "readSerial")

    def _setup_corridor(self) -> None:
        self.corridor = Corridor(self, self.cfg)
        camera_height = float(self.cfg.get("camera_height", 2.0))
        self.camera.setPos(0, self.camera_position, camera_height)

    def _setup_fog(self) -> None:
        fog_density = float(self.cfg.get("fog_density", 0.06))
        fog_color = (0.5, 0.5, 0.5)
        self.setBackgroundColor(*fog_color)
        fog = Fog("fog")
        fog.setColor(*fog_color)
        fog.setExpDensity(fog_density)
        self.render.setFog(fog)

    def _setup_logging(self) -> None:
        self.data_logger = DataLogger(self.cfg.get("data_logging_file", "movement.csv"))
        self.event_logger = EventLogger(self.cfg.get("event_log_file", "events.json"))

    def _setup_zmq(self, command_endpoint: str, event_endpoint: str) -> None:
        try:
            self.event_publisher = ZmqEventPublisher(event_endpoint)
            self.command_server = ZmqCommandServer(command_endpoint, self._handle_command)
        except Exception as exc:  # pragma: no cover - initialization errors
            logger.warning("Failed to initialize ZMQ: %s", exc)
            self.event_publisher = None
            self.command_server = None

    def _setup_engine(self) -> None:
        if Engine is None or ExperimentSpec is None:
            self.engine = None
            return
        spec = self._resolve_engine_spec()
        if spec is None:
            self.engine = None
            return
        self.engine_spec = spec
        self.engine = Engine(spec, time_fn=globalClock.getFrameTime)
        self.engine.capabilities = {"set_gain": True, "set_reverse": True, "set_texture": True}
        self.engine.set_control_callback(self._control_adapter)
        self.engine.set_event_callback(lambda evt: self._emit_event(evt.to_dict()))
    def _setup_tasks(self) -> None:
        self.taskMgr.add(self._movement_task, "movement")
        if self.engine is not None:
            self.taskMgr.add(self._engine_task, "engine")
        if self.command_server is not None:
            self.taskMgr.add(self._command_task, "command")
        self.fsm = ExperimentFSM(self)
        self.fsm.request("Idle")
        if self.dev:
            self.state_text = OnscreenText(text="State: Idle", pos=(-1.3, 0.9), scale=0.07, align=TextNode.ALeft)

    def _resolve_engine_spec(self) -> Optional[ExperimentSpec]:
        if ExperimentSpec is None:
            return None
        engine_cfg = self.cfg.get("engine")
        if not engine_cfg:
            return None
        try:
            payload = engine_cfg
            if "spec_path" in engine_cfg:
                path = Path(engine_cfg["spec_path"])
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            return ExperimentSpec.from_dict(payload)
        except Exception as exc:
            logger.warning("Failed to load MousePortal spec: %s", exc)
            return None

    def update_state_label(self, state: str) -> None:
        widget = getattr(self, "state_text", None)
        if widget is not None:
            widget.setText(f"State: {state}")

    def _set_key(self, key: str, value: bool) -> None:
        self.key_map[key] = value

    def _movement_task(self, task: TaskType) -> TaskType:
        dt = globalClock.getDt()
        base_velocity = self._base_velocity()
        signed = -base_velocity if self.input_reversed else base_velocity
        self.camera_velocity = signed * self.gain
        self.camera_position += self.camera_velocity * dt
        self.camera.setY(self.camera_position)  # type: ignore[attr-defined]
        self._recycle_segments(self.camera_velocity * dt)
        self.data_logger.log(time.time(), self.camera_position, self.camera_velocity)
        return Task.cont

    def _recycle_segments(self, distance: float) -> None:
        self.distance_since_recycle += distance
        if distance > 0:
            while self.distance_since_recycle >= self.segment_length:
                self.corridor.recycle("forward")
                self.distance_since_recycle -= self.segment_length
        elif distance < 0:
            while self.distance_since_recycle <= -self.segment_length:
                self.corridor.recycle("backward")
                self.distance_since_recycle += self.segment_length

    def _engine_task(self, task: TaskType) -> TaskType:
        if self.fsm.state == "Running" and self.engine and self.engine.running:
            if not self.engine.step():
                self.fsm.request("Idle")
        return Task.cont

    def _command_task(self, task: TaskType) -> TaskType:
        if self.command_server:
            self.command_server.poll(0)
        return Task.cont

    def _base_velocity(self) -> float:
        if self.dev:
            scale = float(self.cfg.get("speed_scaling", 5.0))
            if self.key_map["forward"]:
                return scale
            if self.key_map["backward"]:
                return -scale
            return 0.0
        return self.treadmill.data.speed if self.treadmill else 0.0

    def _control_adapter(self, name: str, params: Dict[str, Any]) -> None:
        if name == "set_gain":
            self.set_gain(float(params.get("gain", self.gain)))
        elif name == "set_reverse":
            self.set_input_reversed(bool(params.get("flag", self.input_reversed)))
        elif name == "set_texture":
            target = params.get("target")
            texture = params.get("texture", params.get("slot", ""))
            if target:
                self.set_texture(str(target), str(texture))

    def set_gain(self, value: float) -> None:
        self.gain = float(value)

    def set_input_reversed(self, flag: bool) -> None:
        self.input_reversed = bool(flag)

    def set_texture(self, face: str, texture_path: str) -> None:
        try:
            self.corridor.set_texture(face, texture_path)
        except ValueError as exc:
            logger.warning("%s", exc)

    def mark_event(self, label: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "source": "mouseportal",
            "label": label,
            "time_abs": datetime.now(timezone.utc).isoformat(),
            "time_elapsed": 0.0,
            "position": self.camera_position,
        }
        if extra:
            payload["data"] = extra
        self._emit_event(payload)

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        self.event_logger.record(payload)
        if self.event_publisher:
            self.event_publisher.publish(payload)

    def _handle_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        command = (message.get("command") or "").lower()
        params = message.get("params", {}) or {}
        if command == "ping":
            return {"ok": True, "state": self.fsm.state}
        if command == "start":
            self.fsm.request("Running")
            return {"ok": True, "state": "Running"}
        if command == "stop":
            self.fsm.request("Idle")
            return {"ok": True, "state": "Idle"}
        if command == "set_control":
            control_name = params.get("name")
            if not control_name:
                return {"ok": False, "error": "missing control name"}
            self._control_adapter(control_name, params)
            return {"ok": True}
        if command == "mark_event":
            label = params.get("label", "event")
            extra = {k: v for k, v in params.items() if k != "label"}
            self.mark_event(label, extra=extra or None)
            return {"ok": True}
        if command == "load_spec":
            spec = params.get("spec")
            path = params.get("path")
            return self._load_external_spec(spec, path)
        return {"ok": False, "error": f"unknown command {command}"}

    def _load_external_spec(self, spec: Optional[Dict[str, Any]], path: Optional[str]) -> Dict[str, Any]:
        if Engine is None or ExperimentSpec is None:
            return {"ok": False, "error": "engine not available"}
        try:
            payload = spec
            if payload is None and path:
                with Path(path).open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            if payload is None:
                return {"ok": False, "error": "missing spec"}
            new_spec = ExperimentSpec.from_dict(payload)
            self.engine_spec = new_spec
            self.engine = Engine(new_spec, time_fn=globalClock.getFrameTime)
            self.engine.capabilities = {"set_gain": True, "set_reverse": True, "set_texture": True}
            self.engine.set_control_callback(self._control_adapter)
            self.engine.set_event_callback(lambda evt: self._emit_event(evt.to_dict()))
            return {"ok": True}
        except Exception as exc:
            logger.warning("Failed to load MousePortal spec: %s", exc)
            return {"ok": False, "error": str(exc)}

    def userExit(self) -> None:
        try:
            self.data_logger.close()
            self.event_logger.close()
            if self.event_publisher:
                self.event_publisher.close()
            if self.command_server:
                self.command_server.close()
        finally:
            super().userExit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MousePortal runner")
    parser.add_argument("--cfg", required=True, help="Path to configuration JSON")
    parser.add_argument("--cmd", default="tcp://*:5750", help="Command endpoint to bind")
    parser.add_argument("--pub", default="tcp://*:5751", help="Event endpoint to bind")
    parser.add_argument("--dev", action="store_true", help="Enable development mode without hardware")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    app = MousePortalApp(
        cfg_path=args.cfg,
        command_endpoint=args.cmd,
        event_endpoint=args.pub,
        dev=args.dev,
    )
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
