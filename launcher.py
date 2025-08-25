#!/usr/bin/env python3
"""
GUI Launcher for Mesofield Application
This script creates a GUI interface to launch the Mesofield application
without requiring command-line arguments.
"""
import sys
import os
import json
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def find_config_files():
    """Find available configuration files in the current directory and subdirectories."""
    config_files = []
    current_dir = Path.cwd()
    
    # Look for JSON config files
    for pattern in ['*.json', '**/*.json']:
        for file in current_dir.glob(pattern):
            if file.is_file():
                try:
                    # Try to load the JSON to verify it's valid
                    with open(file, 'r') as f:
                        data = json.load(f)
                        # Check if it looks like a mesofield config
                        if 'Configuration' in data or 'config' in str(file).lower():
                            config_files.append(str(file))
                except (json.JSONDecodeError, Exception):
                    continue
    
    return config_files

def create_default_config():
    """Create a minimal default configuration file."""
    default_config = {
        "Configuration": {
            "hardware_config_file": "hardware.yaml",
            "experiment_directory": "./data",
            "protocol": "default_experiment",
            "experimenter": "researcher"
        },
        "DisplayKeys": ["experiment_directory", "protocol", "experimenter"]
    }
    
    config_path = "default_config.json"
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    
    return config_path

def launch_with_gui():
    """Launch the application with a GUI configuration selector."""
    try:
        from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                                   QHBoxLayout, QWidget, QPushButton, QLabel, 
                                   QComboBox, QMessageBox, QFileDialog, QTextEdit)
        from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
        from PyQt6.QtGui import QFont
        
        class ConfigSelector(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Mesofield Launcher")
                self.setFixedSize(500, 400)
                
                # Central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)
                
                # Title
                title = QLabel("Mesofield Application Launcher")
                title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
                layout.addWidget(title)
                
                # Config selection
                config_layout = QHBoxLayout()
                config_layout.addWidget(QLabel("Configuration File:"))
                
                self.config_combo = QComboBox()
                config_files = find_config_files()
                
                if config_files:
                    self.config_combo.addItems(config_files)
                else:
                    # Create a default config
                    default_config = create_default_config()
                    self.config_combo.addItem(default_config)
                
                config_layout.addWidget(self.config_combo)
                
                browse_btn = QPushButton("Browse...")
                browse_btn.clicked.connect(self.browse_config)
                config_layout.addWidget(browse_btn)
                
                layout.addLayout(config_layout)
                
                # Launch button
                launch_btn = QPushButton("Launch Mesofield")
                launch_btn.setFixedHeight(50)
                launch_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                launch_btn.clicked.connect(self.launch_application)
                layout.addWidget(launch_btn)
                
                # Status text
                self.status_text = QTextEdit()
                self.status_text.setMaximumHeight(100)
                self.status_text.setReadOnly(True)
                self.status_text.append("Ready to launch. Select a configuration file and click Launch.")
                layout.addWidget(self.status_text)
                
                # Instructions
                instructions = QLabel(
                    "Instructions:\n"
                    "1. Select or browse for a configuration JSON file\n"
                    "2. Click 'Launch Mesofield' to start the application\n"
                    "3. The main application window will open"
                )
                instructions.setWordWrap(True)
                instructions.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
                layout.addWidget(instructions)
            
            def browse_config(self):
                file_path, _ = QFileDialog.getOpenFileName(
                    self, 
                    "Select Configuration File", 
                    "", 
                    "JSON files (*.json);;All files (*.*)"
                )
                if file_path:
                    self.config_combo.addItem(file_path)
                    self.config_combo.setCurrentText(file_path)
            
            def launch_application(self):
                config_file = self.config_combo.currentText()
                if not config_file or not os.path.exists(config_file):
                    QMessageBox.warning(self, "Error", "Please select a valid configuration file.")
                    return
                
                try:
                    self.status_text.append(f"Launching with config: {config_file}")
                    
                    # Import and launch the main application
                    import click
                    from mesofield.__main__ import launch
                    
                    # Close this launcher window first
                    self.close()
                    
                    # Exit the current event loop
                    QCoreApplication.quit()
                    
                    # Launch the main application in a new process to avoid event loop conflicts
                    import subprocess
                    import sys
                    
                    # Get the Python executable
                    python_exe = sys.executable
                    
                    # Create the command to run
                    cmd = [python_exe, "-m", "mesofield", "launch", "--config", config_file]
                    
                    # Launch in new process
                    subprocess.Popen(cmd, cwd=os.getcwd())
                    
                except Exception as e:
                    self.status_text.append(f"Error launching application: {str(e)}")
                    QMessageBox.critical(self, "Launch Error", f"Failed to launch application:\n{str(e)}")
        
        # Check if QApplication already exists
        app = QCoreApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            
        selector = ConfigSelector()
        selector.show()
        
        # Only call exec if we created the application
        if app is QCoreApplication.instance():
            sys.exit(app.exec())
        
    except ImportError as e:
        # Fallback to command line if PyQt6 is not available
        print("PyQt6 not available, falling back to command line.")
        launch_command_line()

def launch_command_line():
    """Launch with command line interface."""
    print("Mesofield Application Launcher")
    print("=" * 40)
    
    config_files = find_config_files()
    
    if not config_files:
        print("No configuration files found. Creating default configuration...")
        config_file = create_default_config()
        print(f"Created default configuration: {config_file}")
    else:
        print("Available configuration files:")
        for i, config in enumerate(config_files, 1):
            print(f"{i}. {config}")
        
        while True:
            try:
                choice = input(f"\nSelect configuration file (1-{len(config_files)}) or press Enter for first: ")
                if not choice:
                    config_file = config_files[0]
                    break
                else:
                    choice = int(choice)
                    if 1 <= choice <= len(config_files):
                        config_file = config_files[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please try again.")
    
    print(f"Launching Mesofield with configuration: {config_file}")
    
    try:
        import click
        from mesofield.__main__ import launch
        ctx = click.Context(launch)
        ctx.invoke(launch, config=config_file)
    except Exception as e:
        print(f"Error launching application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    # Try GUI first, fall back to command line
    launch_with_gui()
