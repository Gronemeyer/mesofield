"""GUI widget for managing the MousePortal socket client."""

import ast
import time
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QComboBox,
)

from mesofield.subprocesses.mouseportal import MousePortal
from mesofield.gui import portal_protocol


class MousePortalController(QWidget):
    """Minimal GUI for launching and controlling MousePortal via sockets."""

    def __init__(
        self,
        mouseportal_process: Optional[MousePortal] = None,
        config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mouseportal = mouseportal_process
        self.config = config

        self._setup_ui()
        self._load_config()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_mouseportal)
        if self.mouseportal is not None:
            self._poll_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("MousePortal Controller")
        self.setMinimumSize(640, 720)

        layout = QVBoxLayout(self)

        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.config_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        layout.addWidget(QLabel("Configuration Parameters:"))
        layout.addWidget(self.config_table)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Trial Mode:"))
        self.trial_mode_combo = QComboBox()
        self.trial_mode_combo.addItems(["closed_loop", "open_loop"])
        status_row.addWidget(self.trial_mode_combo)
        layout.addLayout(status_row)

        self.status_label = QLabel("Socket: idle")
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Log:"))
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setFont(QFont("Consolas", 9))
        self.output_display.setMaximumHeight(220)
        layout.addWidget(self.output_display)

        self.launch_btn = QPushButton("Launch MousePortal")
        self.end_btn = QPushButton("End MousePortal")
        self.start_btn = QPushButton("Start Trial")
        self.stop_btn = QPushButton("Stop Trial")
        self.event_btn = QPushButton("Mark Event")

        for btn in (self.launch_btn, self.end_btn, self.start_btn, self.stop_btn, self.event_btn):
            layout.addWidget(btn)

        self.launch_btn.clicked.connect(self._launch_process)
        self.end_btn.clicked.connect(self._end_process)
        self.start_btn.clicked.connect(self._start_trial)
        self.stop_btn.clicked.connect(self._stop_trial)
        self.event_btn.clicked.connect(self._mark_event)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        if not self.config or not getattr(self.config, "plugins", None):
            return
        try:
            cfg = self.config.plugins["mouseportal"]["config"]
        except KeyError:
            return

        runtime_cfg = {
            k: v
            for k, v in cfg.items()
            if k not in {"env_path", "script_path", "device_id", "device_type"}
        }
        runtime_cfg.setdefault("socket_host", "127.0.0.1")
        runtime_cfg.setdefault("socket_port", 8765)

        self.config_table.setRowCount(len(runtime_cfg))
        for row, (key, value) in enumerate(sorted(runtime_cfg.items())):
            item_key = QTableWidgetItem(key)
            item_key.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.config_table.setItem(row, 0, item_key)
            self.config_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def _gather_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for row in range(self.config_table.rowCount()):
            key_item = self.config_table.item(row, 0)
            value_item = self.config_table.item(row, 1)
            if key_item is None:
                continue
            key = key_item.text()
            text = value_item.text() if value_item else ""
            try:
                value = ast.literal_eval(text)
            except Exception:
                value = text
            cfg[key] = value
        return cfg

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _launch_process(self) -> None:
        if not self.mouseportal:
            self._log_output("MousePortal handler not connected")
            return

        runtime_cfg = self._gather_config()
        self.mouseportal.set_cfg(runtime_cfg)
        started = self.mouseportal.start()
        if started:
            self._log_output("Launch command dispatched")
            if not self._poll_timer.isActive():
                self._poll_timer.start()
        else:
            self._log_output("MousePortal already running")

    def _end_process(self) -> None:
        if not self.mouseportal:
            return
        self.mouseportal.shutdown()
        self._log_output("Shutdown requested")

    def _start_trial(self) -> None:
        if not self.mouseportal:
            return
        mode = self.trial_mode_combo.currentText()
        self.mouseportal.start_trial(mode)
        self._log_output(f"start_trial → {mode}")

    def _stop_trial(self) -> None:
        if not self.mouseportal:
            return
        self.mouseportal.stop_trial()
        self._log_output("stop_trial")

    def _mark_event(self) -> None:
        if not self.mouseportal:
            return
        self.mouseportal.mark_event("button")
        self._log_output("mark_event button")

    # ------------------------------------------------------------------
    # Message polling
    # ------------------------------------------------------------------
    def _poll_mouseportal(self) -> None:
        if not self.mouseportal:
            return
        for entry in self.mouseportal.drain_messages():
            message = entry.get("message", {})
            msg_type = message.get("type")
            if msg_type == "status":
                pos = message.get("position")
                vel = message.get("velocity")
                trial = message.get("trial_mode", "idle")
                status_text = (
                    f"State: {message.get('state', '?')} "
                    f"Pos: {self._format_value(pos)} "
                    f"Vel: {self._format_value(vel)} "
                    f"Mode: {trial}"
                )
                elapsed = message.get("trial_elapsed")
                if isinstance(elapsed, (int, float)):
                    status_text += f" Elapsed: {elapsed:.1f}s"
                self.status_label.setText(status_text)
            elif msg_type == "event":
                name = message.get("name", "event")
                position = message.get("position")
                delta = message.get("delta")
                info = f"event {name} pos={self._format_value(position)}"
                if isinstance(delta, (int, float)):
                    info += f" Δ={delta:.3f}s"
                self._log_output(info)
            elif msg_type == "client_status":
                self.status_label.setText(f"Socket: {message.get('status')}")
                detail = message.get("error")
                if detail:
                    self._log_output(f"socket status: {detail}")
            elif msg_type == "client_error":
                self._log_output(f"socket error: {message.get('error')}")
            elif msg_type == "ack":
                cmd = message.get("command")
                status = message.get("status")
                self._log_output(f"ack {cmd} → {status}")
            elif msg_type == "stdout":
                self._log_output(message.get("text", ""))
            elif msg_type == "process_exit":
                self._log_output(f"process exit code={message.get('exit_code')}")
            elif msg_type == "command":
                self._log_output(f"command {message.get('command')} [{message.get('transport')}]")
            else:
                self._log_output(f"message: {message}")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_output(self, message: str) -> None:
        self.output_display.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def set_mouseportal_process(self, process: Optional[MousePortal]) -> None:
        self.mouseportal = process
        if self.mouseportal:
            self.mouseportal.set_output_callback(lambda line: self._log_output(f"stdout: {line}"))
            if not self._poll_timer.isActive():
                self._poll_timer.start()
        else:
            self._poll_timer.stop()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.mouseportal:
            self.mouseportal.set_output_callback(None)  # type: ignore[arg-type]
        self._poll_timer.stop()
        super().closeEvent(event)

    def _format_value(self, value: Any, precision: int = 2) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return str(value)
