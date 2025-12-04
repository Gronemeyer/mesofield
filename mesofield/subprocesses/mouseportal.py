"""MousePortal controller backed by ZeroMQ sockets."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import zmq
import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mesofield.config import ExperimentConfig

from mesofield.utils._logger import get_logger

logger = get_logger("MousePortalController")


class MousePortal:
    """Thin controller that communicates with ``runportal.py`` via ZeroMQ."""

    def __init__(
        self,
        config: "ExperimentConfig",
        *,
        cfg_path: Optional[str] = None,
        python_executable: Optional[str] = None,
        script_path: Optional[str] = None,
        plugin_options: Optional[dict[str, Any]] = None,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self.config = config
        plugin_cfg = config.get_plugin_config("mouse_portal")

        plugin_cfg_path = plugin_cfg.get("config_path")
        if cfg_path:
            self.cfg_path = Path(cfg_path)
        elif plugin_cfg_path:
            self.cfg_path = Path(plugin_cfg_path)
        else:
            self.cfg_path = None
        runtime_default = plugin_cfg.get("runtime_path")
        if runtime_default:
            self.runtime_path = Path(runtime_default)
        else:
            self.runtime_path = Path(config.save_dir) / "mouseportal-runtime.json"
        self.runtime_path = self.runtime_path.resolve()

        pkg_dir = Path(__file__).resolve().parent
        self.script_path = Path(script_path) if script_path else pkg_dir / "runportal.py"
        self.python_executable = python_executable or sys.executable

        self.command_endpoint = plugin_cfg.get("command_endpoint", "tcp://127.0.0.1:5750")
        self.event_endpoint = plugin_cfg.get("event_endpoint", "tcp://127.0.0.1:5751")

        self._ctx = context or zmq.Context.instance()
        self._cmd_socket: Optional[zmq.Socket] = None
        self._event_socket: Optional[zmq.Socket] = None
        self._event_thread: Optional[threading.Thread] = None
        self._event_running = False
        self._event_callback: Optional[Callable[[dict[str, Any]], None]] = None

        self._process: Optional[subprocess.Popen] = None
        self._cfg_payload: dict[str, Any] = {}
        seq_path = plugin_cfg.get("sequence_yaml") or plugin_cfg.get("sequence_path")
        self._sequence_path = Path(seq_path).resolve() if seq_path else None

    # ------------------------------------------------------------------
    # configuration helpers
    def load_cfg(self) -> dict[str, Any]:
        if not self.cfg_path:
            raise FileNotFoundError("No configuration path supplied for MousePortal")
        with self.cfg_path.open("r", encoding="utf-8") as fh:
            if self.cfg_path.suffix.lower() in {".yaml", ".yml"}:
                self._cfg_payload = yaml.safe_load(fh) or {}
            else:
                self._cfg_payload = json.load(fh)
        self._apply_sequence_override()
        return self._cfg_payload

    def set_cfg(self, payload: dict[str, Any]) -> None:
        self._cfg_payload = dict(payload)
        self._apply_sequence_override()

    def set_cfg_path(self, path: str) -> None:
        self.cfg_path = Path(path)
        self.load_cfg()

    def save_runtime(self, payload: Optional[dict[str, Any]] = None) -> Path:
        data = payload or self._cfg_payload
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return self.runtime_path

    def remove_runtime(self) -> None:
        try:
            if self.runtime_path.exists():
                self.runtime_path.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # process lifecycle
    def launch(self, *, extra_args: Optional[list[str]] = None) -> None:
        if self._process and self._process.poll() is None:
            return
        if not self._cfg_payload and self.cfg_path:
            self.load_cfg()
        runtime = self.save_runtime()
        args = [
            self.python_executable,
            str(self.script_path),
            "--cfg",
            str(runtime),
            "--cmd",
            self.command_endpoint,
            "--pub",
            self.event_endpoint,
        ]
        if extra_args:
            args.extend(extra_args)
        logger.info("Launching MousePortal process: %s", " ".join(args))
        self._process = subprocess.Popen(args, cwd=str(self.script_path.parent))
        self._connect_command_socket()

    def terminate(self, timeout: float = 3.0) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self.remove_runtime()
        self._close_command_socket()
        self.stop_event_stream()

    # ------------------------------------------------------------------
    # zmq utilities
    def _connect_command_socket(self) -> None:
        if self._cmd_socket is not None:
            return
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.command_endpoint)
        self._cmd_socket = sock

    def _close_command_socket(self) -> None:
        if self._cmd_socket is not None:
            try:
                self._cmd_socket.close(linger=0)
            except zmq.ZMQError:
                pass
        self._cmd_socket = None

    def send_command(self, command: str, *, timeout: float | None = 5.0, **params: Any) -> dict[str, Any]:
        self._connect_command_socket()
        if self._cmd_socket is None:
            raise RuntimeError("Command socket not available")
        payload = {"command": command, "params": params, "ts": time.time()}
        sock = self._cmd_socket
        if timeout is not None:
            sock.setsockopt(zmq.RCVTIMEO, int(max(timeout, 0.0) * 1000))
        try:
            sock.send_json(payload)
            reply = sock.recv_json()
        except zmq.Again as exc:
            raise TimeoutError(f"MousePortal command '{command}' timed out") from exc
        finally:
            if timeout is not None:
                sock.setsockopt(zmq.RCVTIMEO, -1)
        if not isinstance(reply, dict):
            raise RuntimeError(f"Unexpected reply from MousePortal: {reply!r}")
        return reply

    # convenience wrappers -------------------------------------------
    def start(self, *, timeout: float | None = 5.0) -> dict[str, Any]:
        return self.send_command("start", timeout=timeout)

    def stop(self, *, timeout: float | None = 5.0) -> dict[str, Any]:
        return self.send_command("stop", timeout=timeout)

    def mark_event(self, label: str, **data: Any) -> dict[str, Any]:
        return self.send_command("mark_event", label=label, **data)

    def set_gain(self, gain: float) -> dict[str, Any]:
        return self.send_command("set_control", name="set_gain", gain=gain)

    def set_input_reversed(self, flag: bool) -> dict[str, Any]:
        return self.send_command("set_control", name="set_reverse", flag=flag)

    def set_texture(self, target: str, slot: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"target": target}
        if slot is not None:
            params["slot"] = slot
        return self.send_command("set_control", name="set_texture", **params)

    # ------------------------------------------------------------------
    # event stream
    def start_event_stream(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if self._event_running:
            self._event_callback = callback
            return
        self._event_callback = callback
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "exp-event")
        sock.connect(self.event_endpoint)
        self._event_socket = sock
        self._event_running = True
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()

    def stop_event_stream(self) -> None:
        self._event_running = False
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=1.0)
        if self._event_socket is not None:
            try:
                self._event_socket.close(linger=0)
            except zmq.ZMQError:
                pass
        self._event_socket = None
        self._event_thread = None

    # internal helpers -------------------------------------------------
    def _apply_sequence_override(self) -> None:
        if not self._sequence_path:
            return
        engine_cfg = self._cfg_payload.setdefault("engine", {})
        engine_cfg["spec_path"] = str(self._sequence_path)

    def _event_loop(self) -> None:
        assert self._event_socket is not None
        poller = zmq.Poller()
        poller.register(self._event_socket, zmq.POLLIN)
        while self._event_running:
            events = dict(poller.poll(200))
            if self._event_socket not in events:
                continue
            try:
                topic, payload = self._event_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            if topic.decode("utf-8") != "exp-event":
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if self._event_callback:
                self._event_callback(data)


__all__ = ["MousePortal"]
