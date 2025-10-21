from __future__ import annotations

"""MousePortal subprocess and socket controller.

This module orchestrates launching the external ``runportal.py`` program
while maintaining a persistent socket connection to stream trial and
status messages back into :class:`mesofield.data.manager.DataManager`.

Compared to the legacy stdin-only implementation this version mirrors
the behaviour of :mod:`mesofield.gui.portal_controller_prototype`,
providing a JSON socket protocol interface that can be consumed both by
the GUI controller widget and the data pipeline.
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from PyQt6.QtCore import QProcess

from mesofield.protocols import DataProducer
from mesofield.utils._logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from mesofield.config import ExperimentConfig
    from mesofield.data.manager import DataManager

__all__ = ["MousePortal"]


class Event:
    """Thread-safe callback container carrying (payload, device_ts)."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[Any, Any], None]] = []
        self._lock = threading.Lock()

    def connect(self, callback: Callable[[Any, Any], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def emit(self, payload: Any = None, device_ts: Any = None) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(payload, device_ts)
            except Exception:
                # Defensive: never allow downstream errors to break dispatch
                pass


class PortalSocketClient:
    """Background socket client for the MousePortal control server."""

    def __init__(self, host: str, port: int, reconnect_delay: float = 0.5) -> None:
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self._incoming: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._outgoing: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="PortalSocketClient", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._connected.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def send(self, payload: Dict[str, Any]) -> None:
        message = dict(payload)
        message.setdefault("client_time", time.time())
        self._outgoing.put(message)

    def get_message(self) -> Optional[Dict[str, Any]]:
        try:
            return self._incoming.get_nowait()
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        sock: Optional[socket.socket] = None
        buffer = ""

        while not self._stop.is_set():
            if sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=1.0)
                    sock.setblocking(False)
                    self._connected.set()
                    self._incoming.put(
                        {"type": "client_status", "status": "connected", "time": time.time()}
                    )
                except OSError as exc:
                    self._connected.clear()
                    self._incoming.put(
                        {
                            "type": "client_status",
                            "status": "connecting",
                            "error": str(exc),
                            "time": time.time(),
                        }
                    )
                    if self._stop.wait(self.reconnect_delay):
                        break
                    continue

            if sock is None:
                continue

            try:
                self._flush_outgoing(sock)
            except OSError as exc:
                self._incoming.put({"type": "client_error", "error": f"send:{exc}"})
                sock = self._drop_socket(sock)
                continue

            try:
                chunk = sock.recv(4096)
            except BlockingIOError:
                chunk = None
            except OSError as exc:
                self._incoming.put({"type": "client_error", "error": f"recv:{exc}"})
                sock = self._drop_socket(sock)
                continue

            if chunk == b"":
                sock = self._drop_socket(sock)
                continue

            if chunk:
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._incoming.put(
                            {"type": "client_error", "error": f"json:{exc}", "raw": line}
                        )
                        continue
                    self._incoming.put(message)

            if chunk is None:
                self._stop.wait(0.05)

        if sock is not None:
            self._drop_socket(sock)

    def _flush_outgoing(self, sock: socket.socket) -> None:
        while True:
            try:
                message = self._outgoing.get_nowait()
            except queue.Empty:
                break
            data = (json.dumps(message) + "\n").encode("utf-8")
            try:
                sock.sendall(data)
            except OSError:
                self._outgoing.put(message)
                raise

    def _drop_socket(self, sock: socket.socket) -> Optional[socket.socket]:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        self._connected.clear()
        self._incoming.put(
            {"type": "client_status", "status": "disconnected", "time": time.time()}
        )
        return None


class MousePortal(DataProducer):
    """Controller for the MousePortal subprocess and socket interface."""

    device_type: str
    sampling_rate: float = 0.0
    data_type: str = "json"
    file_type: str = "jsonl"
    bids_type: Optional[str] = "beh"
    output_path: str = ""
    metadata_path: Optional[str] = None

    def __init__(
        self,
        config: ExperimentConfig,
        data_manager: Optional["DataManager"] = None,
        parent=None,
        *,
        launch_process: bool = True,
    ) -> None:
        self.logger = get_logger("MousePortal")
        self._config = config
        self._parent = parent
        self._data_manager: Optional["DataManager"] = None
        self._registered_with_manager = False
        self.launch_process = launch_process

        plugin_cfg = (
            getattr(config, "plugins", {}).get("mouseportal", {}).get("config", {})
        )
        self._plugin_cfg = dict(plugin_cfg)

        self.python_executable = self._resolve_python_executable(plugin_cfg.get("env_path"))
        self.script_path = self._resolve_script_path(plugin_cfg.get("script_path", "runportal.py"))
        self.device_id = plugin_cfg.get("device_id", "mouseportal")
        self.device_type = plugin_cfg.get("device_type", "virtual")
        self.path_args = {
            "suffix": plugin_cfg.get("output_suffix", self.device_id),
            "extension": plugin_cfg.get("output_extension", self.file_type),
            "bids_type": plugin_cfg.get("bids_type", self.bids_type),
        }
        self.file_type = self.path_args["extension"]
        self.bids_type = self.path_args["bids_type"]

        self.cfg = self._prepare_runtime_config(plugin_cfg)
        self._refresh_host_port()

        base_dir = getattr(config, "save_dir", os.getcwd())
        self.runtime_path = os.path.join(base_dir, "mouseportal_runtime.json")

        self.process = QProcess(self._parent)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self.data_event = Event()
        self.message_event = Event()

        self.socket_client: Optional[PortalSocketClient] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_stop = threading.Event()
        self._ui_messages: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._message_log: list[Dict[str, Any]] = []
        self._last_status: Optional[Dict[str, Any]] = None
        self._socket_status: Dict[str, Any] = {"status": "disconnected"}
        self._last_message: Optional[Dict[str, Any]] = None
        self._request_counter = 0
        self._output_callback: Optional[Callable[[str], None]] = None
        self.is_active = False

        if data_manager is not None:
            self.attach_data_manager(data_manager)

        # Ensure the hardware registry includes this pseudo device for output paths
        hardware = getattr(config, "hardware", None)
        try:
            devices = getattr(hardware, "devices", None)
            if isinstance(devices, dict) and self.device_id not in devices:
                devices[self.device_id] = self
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _prepare_runtime_config(self, plugin_cfg: Dict[str, Any]) -> Dict[str, Any]:
        runtime = {
            k: v
            for k, v in plugin_cfg.items()
            if k not in {"env_path", "script_path", "device_id", "device_type"}
        }
        runtime.setdefault("socket_host", "127.0.0.1")
        runtime.setdefault("socket_port", 8765)

        asset_dir = runtime.get("asset_dir")
        if asset_dir:
            asset_dir_path = Path(asset_dir).expanduser().resolve()
            runtime["asset_dir"] = asset_dir_path.as_posix()
            texture_keys = (
                "left_wall_texture",
                "right_wall_texture",
                "ceiling_texture",
                "floor_texture",
            )
            for key in texture_keys:
                path_value = runtime.get(key)
                if isinstance(path_value, str) and path_value and not os.path.isabs(path_value):
                    runtime[key] = (asset_dir_path / path_value).expanduser().resolve().as_posix()
                elif isinstance(path_value, str) and path_value:
                    runtime[key] = Path(path_value).expanduser().resolve().as_posix()
        return runtime

    def _resolve_python_executable(self, env_entry: Optional[str]) -> str:
        candidate = env_entry or sys.executable
        candidate = os.path.abspath(candidate)

        if os.path.isdir(candidate):
            if os.name == "nt":
                guess = os.path.join(candidate, "python.exe")
            else:
                guess = os.path.join(candidate, "bin", "python")
            if os.path.isfile(guess):
                return guess
            self.logger.warning(
                "MousePortal env_path points to a directory without python executable: %s",
                candidate,
            )
            return sys.executable

        if not os.path.isfile(candidate):
            self.logger.warning(
                "MousePortal python executable not found at %s; falling back to current interpreter",
                candidate,
            )
            return sys.executable

        return candidate

    def _resolve_script_path(self, script_entry: str) -> str:
        script = os.path.abspath(script_entry)
        if not os.path.isfile(script):
            self.logger.warning("MousePortal script_path %s not found", script)
        return script

    def set_cfg(self, cfg: Dict[str, Any]) -> None:
        self.cfg = self._prepare_runtime_config(cfg)
        self._refresh_host_port()

    def _refresh_host_port(self) -> None:
        self.host = str(self.cfg.get("socket_host", "127.0.0.1"))
        port = self.cfg.get("socket_port", 8765)
        try:
            self.port = int(port)
        except (TypeError, ValueError):
            self.port = 8765

    def save_runtime(self) -> None:
        os.makedirs(os.path.dirname(self.runtime_path), exist_ok=True)
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

    def remove_runtime(self) -> None:
        try:
            if os.path.isfile(self.runtime_path):
                os.remove(self.runtime_path)
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    # ------------------------------------------------------------------
    # DataManager integration
    # ------------------------------------------------------------------
    def attach_data_manager(self, manager: "DataManager") -> None:
        if manager is self._data_manager and self._registered_with_manager:
            return
        self._data_manager = manager
        try:
            manager.register_hardware_device(self)
            self._registered_with_manager = True
        except Exception:
            self.logger.exception("Failed to register MousePortal with DataManager")

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        already_running = self.is_running
        if not already_running:
            self.save_runtime()
            if self.launch_process:
                args = [self.script_path, "--cfg", self.runtime_path]
                args.extend(["--socket-host", self.host, "--socket-port", str(self.port)])
                cwd = os.path.dirname(self.script_path) or None
                if cwd:
                    self.process.setWorkingDirectory(cwd)
                self.logger.info(
                    "Launching MousePortal process: %s %s", self.python_executable, args
                )
                self.process.start(self.python_executable, args)
                if not self.process.waitForStarted(3000):
                    self.logger.error(
                        "MousePortal failed to start using %s; state=%s",
                        self.python_executable,
                        self.process.state(),
                    )
        self._start_socket_client()
        self.is_active = True
        return not already_running

    def stop(self) -> bool:
        # Stop streaming without shutting down process completely
        self.send_command("stop_trial")
        return True

    def end(self, wait_ms: int = 3000) -> None:
        if self.is_running:
            self.send_command("shutdown")
            self.process.waitForFinished(wait_ms)
        self._stop_socket_client()
        self.remove_runtime()
        self.is_active = False

    def terminate(self, wait_ms: int = 3000) -> None:
        if self.is_running:
            self.process.terminate()
            self.process.waitForFinished(wait_ms)
        self._stop_socket_client()
        self.remove_runtime()
        self.is_active = False

    def shutdown(self) -> None:
        self.end()

    @property
    def is_running(self) -> bool:
        return self.process.state() == QProcess.ProcessState.Running

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------
    def _start_socket_client(self) -> None:
        self._stop_socket_client()
        self.socket_client = PortalSocketClient(self.host, self.port)
        self.socket_client.start()
        self._dispatch_stop.clear()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="MousePortalDispatch", daemon=True
        )
        self._dispatch_thread.start()

    def _stop_socket_client(self) -> None:
        self._dispatch_stop.set()
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            self._dispatch_thread.join(timeout=1.0)
        self._dispatch_thread = None
        if self.socket_client:
            self.socket_client.close()
            self.socket_client = None

    def _dispatch_loop(self) -> None:
        while not self._dispatch_stop.is_set():
            client = self.socket_client
            if client is None:
                if self._dispatch_stop.wait(0.1):
                    break
                continue
            message = client.get_message()
            if message is None:
                if self._dispatch_stop.wait(0.05):
                    break
                continue
            self._handle_socket_payload(message)

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------
    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"mp-{self._request_counter:05d}"

    def send_command(self, command: str, **payload: Any) -> None:
        message: Dict[str, Any] = {"command": command, "request_id": self._next_request_id()}
        message.update(payload)

        sent_via_socket = False
        if self.socket_client and self.socket_client.is_connected():
            self.socket_client.send(message)
            sent_via_socket = True
        elif self.is_running:
            timestamp = time.time()
            tokens = [command]
            if command == "start_trial":
                trial_type = message.get("trial_type") or message.get("mode")
                if trial_type:
                    tokens.append(str(trial_type))
            tokens.append(str(timestamp))
            self.process.write((" ".join(tokens) + "\n").encode("utf-8"))
            message.setdefault("client_time", timestamp)
        else:
            self.logger.warning("MousePortal not running; dropping command %s", command)

        self._record_message(
            {
                "type": "command",
                "command": command,
                "payload": {k: v for k, v in message.items() if k not in {"command", "request_id"}},
                "request_id": message["request_id"],
                "transport": "socket" if sent_via_socket else "stdin",
            },
            source="command",
        )

    def start_trial(self, trial_type: str | None = None) -> None:
        if trial_type is None:
            trial_type = self.cfg.get("trial_type", "closed_loop")
        self.send_command("start_trial", trial_type=trial_type, mode=trial_type)

    def stop_trial(self) -> None:
        self.send_command("stop_trial")

    def mark_event(self, label: str = "button") -> None:
        self.send_command("mark_event", name=label)

    # ------------------------------------------------------------------
    # Process output handlers
    # ------------------------------------------------------------------
    def set_output_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._output_callback = cb

    def set_finished_callback(
        self, cb: Callable[[int, QProcess.ExitStatus], None]
    ) -> None:
        self._finished_callback = cb

    def _on_ready_read(self) -> None:
        raw = self.process.readAllStandardOutput().data().decode(errors="replace")
        if not raw:
            return
        for line in raw.splitlines():
            self._emit_stdout_line(line)

    def _emit_stdout_line(self, line: str) -> None:
        if self._output_callback:
            self._output_callback(line)
        self._record_message({"type": "stdout", "text": line}, source="stdout")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.is_active = False
        self._record_message(
            {"type": "process_exit", "exit_code": exit_code, "status": exit_status.value},
            source="process",
        )
        if hasattr(self, "_finished_callback") and self._finished_callback:
            self._finished_callback(exit_code, exit_status)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        self.logger.error("MousePortal process error: %s (code=%s)", error.name, error.value)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def _handle_socket_payload(self, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "client_status":
            self._socket_status = message
        elif msg_type == "status":
            self._last_status = message
        elif msg_type == "client_error":
            self.logger.warning("MousePortal socket error: %s", message.get("error"))

        self._last_message = message
        self._record_message(message, device_ts=self._extract_device_ts(message))

    def _extract_device_ts(self, message: Dict[str, Any]) -> Optional[float]:
        for key in ("time", "time_received", "sent_time", "client_time", "timestamp"):
            value = message.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _record_message(
        self,
        message: Dict[str, Any],
        device_ts: Optional[float] = None,
        *,
        source: str = "socket",
    ) -> None:
        received = time.time()
        entry = {
            "received": received,
            "device_ts": device_ts,
            "source": source,
            "message": message,
        }
        self._message_log.append(entry)
        self._ui_messages.put(entry)
        self.message_event.emit(message, device_ts)
        self.data_event.emit(message, device_ts)

    # ------------------------------------------------------------------
    # DataProducer interface
    # ------------------------------------------------------------------
    def get_data(self) -> Optional[Dict[str, Any]]:
        return self._last_status

    def initialize(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "process_running": self.is_running,
            "socket": self._socket_status,
            "last_status": self._last_status,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._plugin_cfg)

    def save_data(self, path: Optional[str] = None) -> None:
        path = path or self.output_path
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._message_log:
                f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------
    def drain_messages(self) -> list[Dict[str, Any]]:
        messages: list[Dict[str, Any]] = []
        while True:
            try:
                messages.append(self._ui_messages.get_nowait())
            except queue.Empty:
                break
        return messages

    def get_message(self) -> Optional[Dict[str, Any]]:
        try:
            return self._ui_messages.get_nowait()
        except queue.Empty:
            return None

