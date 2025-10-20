from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QLabel,
    QComboBox,
)
from PyQt6.QtCore import QProcess, Qt, QTimer
import sys
import time
import json
import ast
import os
import socket
import threading
import queue
from typing import Optional, Dict, Any


class PortalClient:
    """Simple background connector for the portal control socket."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._incoming: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._outgoing: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="PortalSocketClient", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._connected.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

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

    # Internal helpers --------------------------------------------------------------

    def _run(self) -> None:
        sock: Optional[socket.socket] = None
        buffer = ""

        while not self._stop.is_set():
            if sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=1.0)
                    sock.setblocking(False)
                    self._connected.set()
                    self._incoming.put({"type": "client_status", "status": "connected", "time": time.time()})
                except OSError as exc:
                    self._connected.clear()
                    self._incoming.put({"type": "client_status", "status": "connecting", "error": str(exc), "time": time.time()})
                    if self._stop.wait(0.5):
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
                        self._incoming.put({"type": "client_error", "error": f"json:{exc}", "raw": line})
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
        self._incoming.put({"type": "client_status", "status": "disconnected", "time": time.time()})
        return None

class PortalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MousePortal Controller")

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.cfg_path = "cfg.json"
        self.runtime_path = "cfg_runtime.json"
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.load_cfg()
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.launch_btn = QPushButton("Launch Portal")
        self.end_btn = QPushButton("End Portal")
        self.start_btn = QPushButton("Start Trial")
        self.stop_btn = QPushButton("Stop Trial")
        self.event_btn = QPushButton("Mark Event")

        trial_row = QHBoxLayout()
        trial_row.addWidget(QLabel("Trial Type:"))
        self.trial_mode_combo = QComboBox()
        self.trial_mode_combo.addItems(["closed_loop", "open_loop"])
        trial_row.addWidget(self.trial_mode_combo)
        layout.addLayout(trial_row)

        layout.addWidget(self.launch_btn)
        layout.addWidget(self.end_btn)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.event_btn)

        self.launch_btn.clicked.connect(self.launch_process)
        self.end_btn.clicked.connect(self.end_process)
        self.start_btn.clicked.connect(self.start_trial)
        self.stop_btn.clicked.connect(self.stop_trial)
        self.event_btn.clicked.connect(self.mark_event)

        self.setCentralWidget(central)
        self.status_bar = self.statusBar()
        if self.status_bar:
            self.status_bar.showMessage("Socket: disconnected")

        self.socket_client: Optional[PortalClient] = None
        self.socket_timer = QTimer(self)
        self.socket_timer.setInterval(50)
        self.socket_timer.timeout.connect(self.poll_socket)
        self.request_counter = 0
        self.last_status: Optional[Dict[str, Any]] = None

    def load_cfg(self):
        with open(self.cfg_path, "r") as f:
            self.cfg = json.load(f)
        self.cfg.setdefault("socket_host", "127.0.0.1")
        self.cfg.setdefault("socket_port", 8765)
        self.table.setRowCount(len(self.cfg))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        for row, (key, val) in enumerate(self.cfg.items()):
            item_key = QTableWidgetItem(key)
            item_key.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, item_key)
            self.table.setItem(row, 1, QTableWidgetItem(str(val)))

    def gather_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            if key_item is None:
                continue
            key = key_item.text()
            value_item = self.table.item(row, 1)
            val_text = value_item.text() if value_item else ""
            try:
                val = ast.literal_eval(val_text)
            except Exception:
                val = val_text
            cfg[key] = val
        return cfg

    def launch_process(self):
        if self.process.state() == QProcess.ProcessState.NotRunning:
            cfg = self.gather_cfg()
            self.cfg = cfg
            with open(self.runtime_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self.process.start(sys.executable, ["runportal.py", "--dev", "--cfg", self.runtime_path])
            host = str(cfg.get("socket_host", "127.0.0.1"))
            port = cfg.get("socket_port", 8765)
            try:
                port = int(port)
            except (TypeError, ValueError):
                port = 8765
            self._launch_socket_client(host, port)
            if self.status_bar:
                self.status_bar.showMessage(f"Socket: connecting {host}:{port}")

    def start_trial(self) -> None:
        trial_type = "closed_loop"
        if hasattr(self, "trial_mode_combo") and self.trial_mode_combo is not None:
            trial_type = self.trial_mode_combo.currentText()
        self.send_command("start_trial", trial_type=trial_type, mode=trial_type)

    def stop_trial(self) -> None:
        self.send_command("stop_trial")

    def _launch_socket_client(self, host: str, port: int) -> None:
        self._close_socket_client()
        self.socket_client = PortalClient(host, port)
        self.socket_client.start()
        if not self.socket_timer.isActive():
            self.socket_timer.start()
        self._append_output(f"[socket] connecting to {host}:{port}")

    def _close_socket_client(self) -> None:
        if self.socket_client:
            self.socket_client.close()
            self.socket_client = None
        if self.socket_timer.isActive():
            self.socket_timer.stop()
        if self.status_bar:
            self.status_bar.showMessage("Socket: disconnected")

    def end_process(self):
        if self.process.state() == QProcess.ProcessState.Running:
            self.send_command("shutdown")
            self.process.waitForFinished(3000)
            if os.path.isfile(self.runtime_path):
                os.remove(self.runtime_path)
        self._close_socket_client()

    def _next_request_id(self) -> str:
        self.request_counter += 1
        return f"req-{self.request_counter:05d}"

    def send_command(self, command: str, **payload: Any) -> None:
        message: Dict[str, Any] = {"command": command, "request_id": self._next_request_id()}
        message.update(payload)
        if self.socket_client:
            self.socket_client.send(message)
            self._append_output(f"[command] {command} -> socket")
        elif self.process.state() == QProcess.ProcessState.Running:
            self._append_output(f"[command] {command} -> stdin (fallback)")
            timestamp = time.time()
            tokens = [command]
            if command == "start_trial":
                trial_type = message.get("trial_type") or message.get("mode")
                if trial_type:
                    tokens.append(str(trial_type))
            tokens.append(str(timestamp))
            self.process.write((" ".join(tokens) + "\n").encode())
        else:
            self._append_output(f"[command] {command} (portal not running)")

    def mark_event(self) -> None:
        self.send_command("mark_event", name="button")

    def poll_socket(self) -> None:
        if not self.socket_client:
            return
        while True:
            message = self.socket_client.get_message()
            if not message:
                break
            self.handle_socket_message(message)

    def handle_socket_message(self, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "client_status":
            status = message.get("status", "")
            detail = message.get("error")
            if self.status_bar:
                self.status_bar.showMessage(f"Socket: {status}")
            if detail and status != "connecting":
                self._append_output(f"[socket] {status}: {detail}")
            return
        if msg_type == "client_error":
            self._append_output(f"[socket-error] {message.get('error')}")
            return
        if msg_type == "status":
            self.last_status = message
            if self.status_bar:
                state = message.get("state", "?")
                position = message.get("position", 0.0)
                velocity = message.get("velocity", 0.0)
                trial_mode = message.get("trial_mode", "idle")
                status_text = f"State: {state} Mode: {trial_mode} Pos: {position:.2f} Vel: {velocity:.2f}"
                controller = message.get("controller")
                if isinstance(controller, dict) and controller.get("mode"):
                    status_text += f" Ctrl: {controller.get('mode')}"
                    if controller.get("mode") == "open_loop":
                        segment_label = controller.get("segment_label")
                        if segment_label is None and controller.get("segment_index") is not None:
                            segment_label = controller.get("segment_index")
                        if segment_label is not None:
                            status_text += f" Seg: {segment_label}"
                        remaining = controller.get("remaining")
                        if isinstance(remaining, (int, float)):
                            status_text += f" Rem: {remaining:.1f}s"
                        segment_count = controller.get("segment_count")
                        if isinstance(segment_count, int):
                            status_text += f" Segments: {segment_count}"
                        loop_flag = controller.get("loop")
                        if loop_flag is not None:
                            status_text += f" Loop: {loop_flag}"
                elapsed = message.get("trial_elapsed")
                if isinstance(elapsed, (int, float)):
                    status_text += f" Elapsed: {elapsed:.1f}s"
                self.status_bar.showMessage(status_text)
            return
        if msg_type == "event":
            name = message.get("name")
            position = message.get("position")
            delta = message.get("delta")
            trial_mode = message.get("trial_mode")
            controller = message.get("controller")
            controller_mode = controller.get("mode") if isinstance(controller, dict) else None
            label = None
            if isinstance(controller, dict):
                label = controller.get("segment_label") or controller.get("segment_index")
            extra = []
            if trial_mode:
                extra.append(f"mode={trial_mode}")
            if controller_mode and controller_mode != trial_mode:
                extra.append(f"ctrl={controller_mode}")
            if label is not None:
                extra.append(f"segment={label}")
            if isinstance(controller, dict) and controller.get("mode") == "open_loop":
                gain = controller.get("gain")
                bias = controller.get("bias")
                remaining = controller.get("remaining")
                if gain is not None:
                    extra.append(f"gain={gain}")
                if bias is not None:
                    extra.append(f"bias={bias}")
                if isinstance(remaining, (int, float)):
                    extra.append(f"remaining={remaining:.2f}s")
            extra_text = f" ({', '.join(extra)})" if extra else ""
            self._append_output(f"[event] {name}{extra_text} pos={position} delta={delta}")
            return
        if msg_type == "ack":
            trial_mode = message.get("trial_mode")
            controller = message.get("controller")
            controller_mode = controller.get("mode") if isinstance(controller, dict) else None
            mode_text = trial_mode or controller_mode
            text = f"[ack] {message.get('command')} status={message.get('status')} latency={self._ack_latency(message):.3f}s"
            if mode_text:
                text += f" mode={mode_text}"
                if mode_text in {"closed_loop", "open_loop"} and hasattr(self, "trial_mode_combo"):
                    index = self.trial_mode_combo.findText(mode_text)
                    if index >= 0:
                        self.trial_mode_combo.setCurrentIndex(index)
            if isinstance(controller, dict):
                segment = controller.get("segment_label")
                if segment is None and controller.get("segment_index") is not None:
                    segment = controller.get("segment_index")
                if segment is not None:
                    text += f" segment={segment}"
                remaining = controller.get("remaining")
                if isinstance(remaining, (int, float)):
                    text += f" remaining={remaining:.2f}s"
                segment_count = controller.get("segment_count")
                if isinstance(segment_count, int):
                    text += f" segments={segment_count}"
                loop_flag = controller.get("loop")
                if loop_flag is not None:
                    text += f" loop={loop_flag}"
            self._append_output(text)
            return
        if msg_type == "connected":
            self._append_output("[socket] server acknowledged connection")
            return
        self._append_output(f"[socket] {json.dumps(message)}")

    def _append_output(self, text: str) -> None:
        self.output.append(text)
        print(text)

    def _ack_latency(self, message: Dict[str, Any]) -> float:
        sent = message.get("sent_time")
        server_time = message.get("server_time")
        if isinstance(sent, (int, float)) and isinstance(server_time, (int, float)):
            return max(server_time - float(sent), 0.0)
        client_time = message.get("client_time")
        if isinstance(client_time, (int, float)) and isinstance(server_time, (int, float)):
            return max(server_time - float(client_time), 0.0)
        return 0.0

    def read_output(self) -> None:
        raw = self.process.readAllStandardOutput()
        data = raw.data().decode("utf-8", errors="ignore")
        if data:
            self.output.append(data.rstrip())

    def closeEvent(self, event):
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(3000)
        self._close_socket_client()
        if os.path.isfile(self.runtime_path):
            os.remove(self.runtime_path)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = PortalGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
