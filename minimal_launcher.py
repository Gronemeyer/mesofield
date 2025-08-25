#!/usr/bin/env python3
"""
Minimal launcher for Mesofield that bypasses IPython console issues
"""
import os
import sys
import json
import traceback
from pathlib import Path

def find_config_files():
    """Find available JSON configuration files in common locations"""
    config_files = []
    
    # Check current directory and subdirectories
    search_paths = [
        ".",
        "tests",
        "config", 
        "configs",
        "examples"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.json'):
                        full_path = os.path.join(root, file)
                        # Try to validate it's a config file by checking for expected keys
                        try:
                            with open(full_path, 'r') as f:
                                data = json.load(f)
                                if 'Configuration' in data or 'protocol' in data or 'experimenter' in data:
                                    config_files.append(full_path)
                        except:
                            pass  # Skip invalid JSON files
    
    return config_files

def console_config_selector():
    """Simple console-based configuration selector"""
    print("Mesofield Configuration Launcher")
    print("=" * 40)
    
    config_files = find_config_files()
    
    if not config_files:
        print("No configuration files found!")
        print("Looking for JSON files with 'Configuration', 'protocol', or 'experimenter' keys")
        print("in current directory, tests/, config/, configs/, and examples/ folders")
        
        # Allow manual file selection
        manual_path = input("\nEnter path to configuration file (or press Enter to exit): ").strip()
        if manual_path and os.path.exists(manual_path):
            return manual_path
        else:
            print("Exiting...")
            return None
    
    print(f"Found {len(config_files)} configuration file(s):")
    print()
    
    for i, config_file in enumerate(config_files, 1):
        print(f"{i}. {config_file}")
        
        # Try to show some info about the config
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                cfg = data.get('Configuration', {})
                protocol = cfg.get('protocol', 'Unknown')
                experimenter = cfg.get('experimenter', 'Unknown')
                print(f"   Protocol: {protocol}, Experimenter: {experimenter}")
        except:
            print("   (Could not read config details)")
        print()
    
    while True:
        try:
            choice = input(f"Select configuration file (1-{len(config_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(config_files):
                return config_files[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(config_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def launch_mesofield_minimal(config_path):
    """Launch mesofield with minimal dependencies - no IPython console"""
    try:
        print(f"\nLaunching Mesofield with configuration: {config_path}")
        
        # Import only what we need
        import json
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor, QRadialGradient
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QSplashScreen
        
        # Import mesofield components
        from mesofield.base import Procedure, create_procedure
        
        app = QApplication([])
        
        # Show splash screen
        ascii_art = r"""
 __    __     ______     ______     ______     ______   __      ____      __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.  
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \ 
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____- 
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/ 
                                                                                  
-------------------------  Mesofield Acquisition Interface  ---------------------------------
"""
        
        # Create splash screen
        pixmap = QPixmap(1100, 210)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        center = pixmap.rect().center()
        radius = max(pixmap.width(), pixmap.height()) / 2
        gradient = QRadialGradient(center.x(), center.y(), radius)
        gradient.setColorAt(0.0, QColor(1, 25, 5))
        gradient.setColorAt(0.7, QColor(10, 15, 0, 200))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), gradient)
        painter.setPen(Qt.GlobalColor.green)
        painter.setFont(QFont("Courier", 12))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, ascii_art)
        painter.end()
        
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()
        
        # Load configuration
        with open(config_path, 'r') as f:
            cfg_json = json.load(f)
        
        cfg = cfg_json.get('Configuration', {})
        display_keys = cfg_json.get('DisplayKeys')
        hardware_yaml = cfg.get('hardware_config_file', 'hardware.yaml')
        data_dir = cfg.get('experiment_directory', '.')
        experiment_id = cfg.get('protocol', 'experiment')
        experimentor = cfg.get('experimenter', 'researcher')
        
        # Create procedure
        procedure = create_procedure(
            Procedure,
            experiment_id=experiment_id,
            experimentor=experimentor,
            hardware_yaml=hardware_yaml,
            data_dir=data_dir,
            json_config=config_path
        )
        
        # Import the minimal GUI version
        from mesofield.gui.minimal_gui import MinimalMainWindow
        
        mesofield = MinimalMainWindow(procedure, display_keys=display_keys)
        mesofield.show()
        splash.finish(mesofield)
        app.exec()
        
        return True
        
    except ImportError as e:
        if "minimal_gui" in str(e):
            # Fall back to regular GUI but with patched console
            return launch_mesofield_patched(config_path)
        else:
            print(f"Import error launching Mesofield: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Error launching Mesofield: {e}")
        traceback.print_exc()
        return False

def launch_mesofield_patched(config_path):
    """Launch mesofield with patched console initialization"""
    try:
        print(f"\nLaunching Mesofield with patched console (configuration: {config_path})")
        
        # Patch the MainWindow to skip console initialization
        import mesofield.gui.maingui as maingui_module
        
        # Store the original initialize_console method
        original_initialize_console = maingui_module.MainWindow.initialize_console
        
        # Create a dummy method that does nothing
        def dummy_initialize_console(self, procedure):
            print("Skipping IPython console initialization due to packaging issues...")
            self.console_widget = None
            self.kernel_manager = None
            self.kernel = None
            self.kernel_client = None
        
        # Replace the method
        maingui_module.MainWindow.initialize_console = dummy_initialize_console
        
        # Now import and run normally
        from mesofield.__main__ import launch
        import click
        
        # Create a click context and invoke the launch command
        ctx = click.Context(launch)
        ctx.invoke(launch, config=config_path)
        
        return True
        
    except Exception as e:
        print(f"Error launching Mesofield: {e}")
        traceback.print_exc()
        return False

def main():
    """Main launcher function"""
    try:
        # Change to the script directory to ensure relative paths work
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        print("Starting Mesofield Configuration Launcher...")
        
        # Use console selector (simpler and more reliable)
        config_path = console_config_selector()
        
        if config_path:
            # Convert to absolute path
            config_path = os.path.abspath(config_path)
            
            if os.path.exists(config_path):
                success = launch_mesofield_minimal(config_path)
                
                if not success:
                    print("Failed to launch Mesofield application.")
                    input("Press Enter to exit...")
            else:
                print(f"Configuration file not found: {config_path}")
                input("Press Enter to exit...")
        else:
            print("No configuration selected. Exiting...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
