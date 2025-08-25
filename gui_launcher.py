#!/usr/bin/env python3
"""
Simple launcher that automatically launches the GUI with a config selector
when no arguments are provided (like when double-clicking the exe).
"""
import os
import sys
import logging

# Disable pymmcore-plus logger
package_logger = logging.getLogger('pymmcore-plus')
package_logger.setLevel(logging.CRITICAL)

# Disable debugger warning about the use of frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Disable ipykernel logger
logging.getLogger("ipykernel.inprocess.ipkernel").setLevel(logging.WARNING)

def launch_gui():
    """Launch the GUI application with automatic config detection."""
    import json
    import time
    from pathlib import Path
    
    from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox, QFileDialog
    from PyQt6.QtGui import QPixmap, QPainter, QFont
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QRadialGradient
    
    from mesofield.gui.maingui import MainWindow
    from mesofield.base import Procedure, create_procedure
    
    app = QApplication([])

    # Find config files in the current directory
    config_files = []
    current_dir = Path.cwd()
    
    for pattern in ['*.json', 'tests/*.json', 'config/*.json']:
        for file in current_dir.glob(pattern):
            if file.is_file():
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if 'Configuration' in data:
                            config_files.append(str(file))
                except:
                    continue
    
    # If no config files found, prompt user to select one
    if not config_files:
        msg = QMessageBox()
        msg.setWindowTitle("Configuration Required")
        msg.setText("No configuration files found. Please select a configuration file.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        
        if msg.exec() == QMessageBox.StandardButton.Ok:
            config_file, _ = QFileDialog.getOpenFileName(
                None, 
                "Select Configuration File", 
                "", 
                "JSON files (*.json);;All files (*.*)"
            )
            if not config_file:
                sys.exit(0)
        else:
            sys.exit(0)
    else:
        # Use the first config file found
        config_file = config_files[0]
        print(f"Using configuration file: {config_file}")

    # Create splash screen
    ascii_art = r"""
 __    __     ______     ______     ______     ______   __      ____      __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.  
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \ 
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____- 
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/ 
                                                                                  
-------------------------  Mesofield Acquisition Interface  ---------------------------------
"""

    # Create a transparent pixmap
    pixmap = QPixmap(1100, 210)
    pixmap.fill(Qt.GlobalColor.transparent)

    # Build a radial gradient: dark center that fades out at the edges
    center = pixmap.rect().center()
    radius = max(pixmap.width(), pixmap.height()) / 2
    gradient = QRadialGradient(center.x(), center.y(), radius)
    gradient.setColorAt(0.0, QColor(1, 25, 5))  # solid dark center
    gradient.setColorAt(0.7, QColor(10, 15, 0, 200))  # keep dark until 80%
    gradient.setColorAt(1.0, QColor(0, 0, 0, 0))    # fully transparent edges

    painter = QPainter(pixmap)
    # Fill entire pixmap with the gradient block
    painter.fillRect(pixmap.rect(), gradient)

    # Draw the ASCII art on top
    painter.setPen(Qt.GlobalColor.green)
    painter.setFont(QFont("Courier", 12))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, ascii_art)
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()  # ensure the splash appears

    # Load the configuration file
    with open(config_file, 'r') as f:
        cfg_json = json.load(f)

    cfg = cfg_json.get('Configuration', {})
    display_keys = cfg_json.get('DisplayKeys')
    hardware_yaml = cfg.get('hardware_config_file', 'hardware.yaml')
    data_dir = cfg.get('experiment_directory', '.')
    experiment_id = cfg.get('protocol', 'experiment')
    experimentor = cfg.get('experimenter', 'researcher')

    time.sleep(2)  # give the splash screen a moment to show :)
    
    try:
        procedure = create_procedure(
            Procedure,
            experiment_id=experiment_id,
            experimentor=experimentor,
            hardware_yaml=hardware_yaml,
            data_dir=data_dir,
            json_config=config_file
        )
        
        mesofield = MainWindow(procedure, display_keys=display_keys)
        mesofield.show()
        splash.finish(mesofield)
        app.exec()
    except Exception as e:
        splash.close()
        msg = QMessageBox()
        msg.setWindowTitle("Launch Error")
        msg.setText(f"Failed to launch application:\n{str(e)}")
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.exec()
        sys.exit(1)

if __name__ == "__main__":
    # If no command line arguments provided (like when double-clicking exe),
    # launch the GUI directly
    if len(sys.argv) == 1:
        launch_gui()
    else:
        # Otherwise, use the normal CLI
        from mesofield.__main__ import cli
        cli()
