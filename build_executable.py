#!/usr/bin/env python3
"""
Build script for creating Mesofield executable
"""
import os
import subprocess
import sys
import shutil

def clean_build():
    """Remove previous build artifacts"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)

def build_onefile():
    """Build a single-file executable"""
    print("Building single-file executable...")
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',  # Remove this if you want console output
        '--name=mesofield',
        '--icon=mesofield/gui/Mesofield_icon.png',
        '--add-data=mesofield/VERSION;mesofield',
        '--add-data=mesofield/gui/Mesofield_icon.png;mesofield/gui',
        '--add-data=LICENSE;.',
        '--add-data=tests;tests',  # Include test config files
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtGui', 
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=qtconsole',
        '--hidden-import=ipykernel',
        '--hidden-import=pymmcore_plus',
        '--hidden-import=nidaqmx',
        '--hidden-import=tkinter',
        'config_launcher.py'  # Use the config launcher instead
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        print(f"Executable created at: dist/mesofield.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def build_config_launcher():
    """Build using the spec file with config launcher"""
    print("Building with configuration launcher using spec file...")
    try:
        result = subprocess.run(['pyinstaller', 'mesofield.spec'], check=True)
        print("Build completed successfully!")
        print(f"Executable created at: dist/mesofield/mesofield.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return False

def build_onedir():
    """Build using the spec file (one-directory bundle)"""
    print("Building using spec file...")
    try:
        result = subprocess.run(['pyinstaller', 'mesofield.spec'], check=True)
        print("Build completed successfully!")
        print(f"Executable created at: dist/mesofield/mesofield.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Mesofield Executable Builder")
    print("=" * 40)
    
    build_options = {
        "onefile": "Single-file executable (slower startup, easier distribution)",
        "onedir": "Directory bundle using spec file (faster startup, more files)",
        "config": "Build with configuration launcher (recommended)"
    }
    
    if len(sys.argv) > 1:
        build_type = sys.argv[1]
    else:
        print("Available build options:")
        for key, desc in build_options.items():
            print(f"  {key}: {desc}")
        print()
        build_type = input("Select build type (onefile/onedir/config) [config]: ").strip() or "config"
    
    if build_type not in build_options:
        print(f"Unknown build type: {build_type}")
        print(f"Valid options: {', '.join(build_options.keys())}")
        sys.exit(1)
    
    print(f"Build type: {build_type} - {build_options[build_type]}")
    
    # Clean previous builds
    clean_build()
    
    # Build executable
    if build_type == "onefile":
        success = build_onefile()
    elif build_type == "config":
        success = build_config_launcher()
    else:  # onedir
        success = build_onedir()
    
    if success:
        print("\nBuild completed! You can find your executable in the 'dist' folder.")
        if build_type == "config":
            print("The executable will show a configuration file selector when launched.")
    else:
        print("\nBuild failed. Please check the error messages above.")
