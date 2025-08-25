#!/usr/bin/env python3
"""
Configuration file selector and launcher for Mesofield
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

def gui_config_selector():
    """GUI-based configuration selector using tkinter"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        
        root = tk.Tk()
        root.title("Mesofield Configuration Selector")
        root.geometry("600x400")
        
        # Center the window
        root.eval('tk::PlaceWindow . center')
        
        selected_config = None
        
        def on_select():
            nonlocal selected_config
            selection = listbox.curselection()
            if selection:
                selected_config = config_files[selection[0]]
                root.quit()
                root.destroy()
        
        def on_browse():
            nonlocal selected_config
            filename = filedialog.askopenfilename(
                title="Select Configuration File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                selected_config = filename
                root.quit()
                root.destroy()
        
        def on_cancel():
            root.quit()
            root.destroy()
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="Select Mesofield Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Config list
        config_files = find_config_files()
        
        if config_files:
            list_label = ttk.Label(main_frame, text="Available configurations:")
            list_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
            
            # Listbox with scrollbar
            list_frame = ttk.Frame(main_frame)
            list_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
            
            listbox = tk.Listbox(list_frame, height=10)
            scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
            listbox.configure(yscrollcommand=scrollbar.set)
            
            for config_file in config_files:
                # Show relative path for cleaner display
                display_path = os.path.relpath(config_file)
                listbox.insert(tk.END, display_path)
            
            listbox.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            
            list_frame.columnconfigure(0, weight=1)
            list_frame.rowconfigure(0, weight=1)
            
            # Double-click to select
            listbox.bind('<Double-1>', lambda e: on_select())
        else:
            no_config_label = ttk.Label(main_frame, text="No configuration files found in common locations.")
            no_config_label.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        if config_files:
            select_button = ttk.Button(button_frame, text="Select", command=on_select)
            select_button.grid(row=0, column=0, padx=(0, 5))
        
        browse_button = ttk.Button(button_frame, text="Browse...", command=on_browse)
        browse_button.grid(row=0, column=1, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.grid(row=0, column=2, padx=(5, 0))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Set default selection
        if config_files:
            listbox.selection_set(0)
            listbox.focus()
        
        root.mainloop()
        
        return selected_config
        
    except ImportError:
        print("tkinter not available, falling back to console selector")
        return console_config_selector()

def launch_mesofield(config_path):
    """Launch mesofield with the selected configuration"""
    try:
        print(f"\nLaunching Mesofield with configuration: {config_path}")
        
        # Import required modules
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
        
        # Try GUI first, fall back to console
        try:
            config_path = gui_config_selector()
        except:
            config_path = console_config_selector()
        
        if config_path:
            # Convert to absolute path
            config_path = os.path.abspath(config_path)
            
            if os.path.exists(config_path):
                success = launch_mesofield(config_path)
                
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
