"""
MousePortal Real-time Controller Widget

This module provides a GUI widget for real-time control of the MousePortal
subprocess. It allows users to send commands and modify parameters while
the corridor simulation is running.
"""

import os
import json
import subprocess
import time
import ast
from typing import Optional, Dict, Any

from PyQt6.QtCore import QTimer, pyqtSignal, QProcess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from mesofield.subprocesses.mouseportal import MousePortal


class MousePortalController(QWidget):
    """
    MousePortal Controller Widget - Simplified table-based interface
    
    Provides a clean interface for launching MousePortal and sending basic commands.
    Config parameters are displayed in an editable table.
    """
    
    def __init__(self, mouseportal_process: Optional[MousePortal] = None, 
                 config=None, parent=None):
        super().__init__(parent)
        self.mouseportal = mouseportal_process
        self.config = config
        self.runtime_config = {}
        
        # QProcess for subprocess management
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self.setWindowTitle("MousePortal Controller")
        self.setMinimumSize(600, 800)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Config table
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.config_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        layout.addWidget(QLabel("Configuration Parameters:"))
        layout.addWidget(self.config_table)
        
        # Output display
        layout.addWidget(QLabel("Output:"))
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMaximumHeight(200)
        self.output_display.setFont(QFont("Consolas", 9))
        layout.addWidget(self.output_display)
        
        # Control buttons
        self.launch_btn = QPushButton("Launch MousePortal")
        self.launch_btn.setStyleSheet("""
            QPushButton {
            background-color: #2E7D32; /* Green */
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            }
            QPushButton:hover {
            background-color: #388E3C;
            }
        """)
        
        self.end_btn = QPushButton("End MousePortal")
        self.end_btn.setStyleSheet("""
            QPushButton {
            background-color: #D32F2F; /* Red */
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            }
            QPushButton:hover {
            background-color: #F44336;
            }
        """)
        
        self.start_btn = QPushButton("Start Trial")
        self.stop_btn = QPushButton("Stop Trial")
        self.event_btn = QPushButton("Mark Event")
        
        # Add buttons to layout
        for btn in [self.launch_btn, self.end_btn, self.start_btn, self.stop_btn, self.event_btn]:
            layout.addWidget(btn)
            
        # Connect signals
        self.launch_btn.clicked.connect(self._launch_process)
        self.end_btn.clicked.connect(self._end_process)
        self.start_btn.clicked.connect(lambda: self._send_cmd("start_trial"))
        self.stop_btn.clicked.connect(lambda: self._send_cmd("stop_trial"))
        self.event_btn.clicked.connect(lambda: self._send_cmd("mark_event button"))
        
    def _load_config(self) -> None:
        """Load configuration into the table."""
        if not self.config:
            return
            
        try:
            cfg = self.config.plugins["mouseportal"]["config"]
            # Filter out paths that aren't runtime parameters
            runtime_cfg = {k: v for k, v in cfg.items() if k not in ["env_path", "script_path"]}
            
            self.config_table.setRowCount(len(runtime_cfg))
            for row, (key, val) in enumerate(runtime_cfg.items()):
                # Parameter name (read-only)
                item_key = QTableWidgetItem(key)
                item_key.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.config_table.setItem(row, 0, item_key)
                
                # Parameter value (editable)
                self.config_table.setItem(row, 1, QTableWidgetItem(str(val)))
                
        except Exception as e:
            self._log_output(f"Error loading config: {e}")
    
    def _gather_config(self) -> Dict[str, Any]:
        """Gather configuration from the table."""
        cfg = {}
        for row in range(self.config_table.rowCount()):
            key_item = self.config_table.item(row, 0)
            val_item = self.config_table.item(row, 1)
            
            if key_item is None or val_item is None:
                continue
                
            key = key_item.text()
            val_text = val_item.text()
            
            try:
                # Try to evaluate as Python literal (for numbers, bools, etc.)
                val = ast.literal_eval(val_text)
            except Exception:
                # Fallback to string
                val = val_text
            cfg[key] = val
        return cfg
    
    def _launch_process(self) -> None:
        """Launch MousePortal process."""
        if self.process.state() == QProcess.ProcessState.NotRunning:
            if not self.config:
                self._log_output("ERROR: No configuration available")
                return
                
            try:
                # Get paths from config
                cfg = self.config.plugins["mouseportal"]["config"]
                python_exe = cfg["env_path"]
                script_path = cfg["script_path"]
                
                # Gather current config from table
                runtime_config = self._gather_config()
                
                # Add back the missing keys that aren't in the table
                base_config = {k: v for k, v in cfg.items() if k not in ["env_path", "script_path"]}
                for key, value in base_config.items():
                    if key not in runtime_config:
                        runtime_config[key] = value
                
                # Convert Windows-style texture paths to Unix-style for Panda3D
                texture_keys = ["left_wall_texture", "right_wall_texture", "floor_texture", "ceiling_texture"]
                for key in texture_keys:
                    if key in runtime_config and isinstance(runtime_config[key], str):
                        path = runtime_config[key]
                        if path.startswith('C:/') or path.startswith('C:\\'):
                            # Convert C:/path to /c/path for Panda3D
                            unix_path = path.replace('C:/', '/c/').replace('C:\\', '/c/').replace('\\', '/')
                            runtime_config[key] = unix_path
                            self._log_output(f"Converted {key}: {path} -> {unix_path}")
                
                # Save runtime config
                runtime_path = os.path.join(os.getcwd(), "mouseportal_runtime.json")
                with open(runtime_path, "w") as f:
                    json.dump(runtime_config, f, indent=2)
                
                self._log_output(f"Runtime config saved to: {runtime_path}")
                self._log_output(f"Python exe: {python_exe}")
                self._log_output(f"Script path: {script_path}")
                
                # Set working directory for QProcess (critical for MousePortal)
                self.process.setWorkingDirectory(os.path.dirname(script_path))
                
                # Launch process
                self.process.start(python_exe, [script_path, "--cfg", runtime_path])
                self._log_output("MousePortal launch command sent")
                
            except Exception as e:
                self._log_output(f"Error launching MousePortal: {e}")
                import traceback
                traceback.print_exc()
                
    def _end_process(self) -> None:
        """End MousePortal process."""
        if self.process.state() == QProcess.ProcessState.Running:
            self._send_cmd("end")
            self.process.waitForFinished(3000)
            self._log_output("MousePortal ended")
            
            # Clean up runtime config file
            runtime_path = os.path.join(os.getcwd(), "mouseportal_runtime.json")
            if os.path.isfile(runtime_path):
                os.remove(runtime_path)
    
    def _send_cmd(self, cmd: str) -> None:
        """Send command to MousePortal process."""
        if self.process.state() == QProcess.ProcessState.Running:
            command = f"{cmd} {time.time()}\n"
            self.process.write(command.encode())
            self._log_output(f"Sent: {cmd}")
        else:
            self._log_output("ERROR: MousePortal not running")
    
    def _read_output(self) -> None:
        """Read output from MousePortal process."""
        byte_array = self.process.readAllStandardOutput()
        data = byte_array.data().decode()
        if data:
            self.output_display.append(data.rstrip())
    
    def _log_output(self, message: str) -> None:
        """Log a message to the output display."""
        self.output_display.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(3000)
            
        # Clean up runtime config file
        runtime_path = os.path.join(os.getcwd(), "mouseportal_runtime.json")
        if os.path.isfile(runtime_path):
            os.remove(runtime_path)
            
        super().closeEvent(event)
    
    # Legacy methods for compatibility
    def set_mouseportal_process(self, process: Optional[MousePortal]) -> None:
        """Set the MousePortal process reference (for compatibility)."""
        self.mouseportal = process
