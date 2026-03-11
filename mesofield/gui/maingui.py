import os
from typing import cast

# Necessary modules for the IPython console
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTabWidget,
    QLayout
)

from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QCoreApplication

from mesofield.gui.mdagui import MDA
from mesofield.gui.controller import ConfigController
from mesofield.gui.speedplotter import SerialWidget
from mesofield.config import ExperimentConfig
from mesofield.protocols import Procedure

class MainWindow(QMainWindow):
    def __init__(self, procedure: Procedure):
        super().__init__()
        #self.config: ExperimentConfig = cast(ExperimentConfig, procedure.config)
        self.procedure = procedure
        self.display_keys = self.procedure.config.display_keys       
        self.setWindowTitle("Mesofield")
        #============================== Widgets =============================#
        self.acquisition_gui = MDA(self.procedure.config)
        self.config_controller = ConfigController(self.procedure, display_keys=self.display_keys)
        self.encoder_widget = SerialWidget(
            cfg=self.procedure.config,
            device_attr="encoder",
            signal_name="serialSpeedUpdated",
            label="Encoder",
            value_label="Speed",
            value_units="mm/s",
        )
        self.initialize_console(self.procedure) # Initialize the IPython console
        #--------------------------------------------------------------------#

        #============================== Layout ==============================#
        central_widget = QWidget()
        # Use a vertical layout: top = acquisition/tabbed panel; bottom = encoder
        self.main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        mda_layout = QVBoxLayout()
        self.main_layout.addLayout(mda_layout)
        self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Build a tab widget with ExperimentConfig and Terminal tabs
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.config_controller, "ExperimentConfig")
        self.right_tabs.addTab(self.console_widget, "Terminal")

        # Horizontal row for acquisition GUI and the tabbed panel
        top_row = QHBoxLayout()
        top_row.addWidget(self.acquisition_gui)
        top_row.addWidget(self.right_tabs)
        mda_layout.addLayout(top_row)

        # Encoder widget below the top row
        mda_layout.addWidget(self.encoder_widget)

        #--------------------------------------------------------------------#

    #============================== Methods =================================#    
    def toggle_console(self):
        """Switch to the Terminal tab."""
        terminal_index = self.right_tabs.indexOf(self.console_widget)
        self.right_tabs.setCurrentIndex(terminal_index)
    
                
    def initialize_console(self, procedure):
        """Initialize the IPython console and embed it into the application."""
        import mesofield.data as data
        # Create an in-process kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        # suppress the kernel’s built-in banner
        self.kernel.shell.banner1 = ""
        self.kernel.shell.banner2 = ""
        self.kernel.gui = 'qt'

        # Create a kernel client and start channels
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Create the console widget
        self.console_widget = RichJupyterWidget()
        self.console_widget.kernel_manager = self.kernel_manager
        self.console_widget.kernel_client = self.kernel_client
        self.console_widget.console_width = 100
        # Expose variables to the console's namespace
        console_namespace = {
            #'mda': self.acquisition_gui.mda,
            'self': self,
            'procedure': procedure,
            'data': data
            # Optional, so you can use 'self' directly in the console
        }
        self.kernel.shell.push(console_namespace)

        # Register the what_do helper command
        def what_do():
            """Print a friendly guide on using the Mesofield terminal."""
            print(
                "\n"
                "=== Mesofield Terminal ===\n"
                "\n"
                "This is a live IPython console embedded inside Mesofield.\n"
                "You have direct access to the running experiment and can\n"
                "inspect or control it interactively.\n"
                "\n"
                "Available objects:\n"
                "  procedure  – the active Procedure (start, stop, pause)\n"
                "  procedure.config – current ExperimentConfig (paths, subjects, params)\n"
                "  data       – mesofield.data module (loading, processing, analysis)\n"
                "  self       – the MainWindow instance (GUI widgets, layout)\n"
                "\n"
                "Quick examples:\n"
                "  procedure.config.subject   # current subject ID\n"
                "  procedure.config.bids_dir  # BIDS output directory\n"
                "  data.load.sessions('path') # load session data\n"
                "\n"
                "Tips:\n"
                "  • Use tab-completion to explore objects and methods.\n"
                "  • Use '?' after a name (e.g. procedure?) for docs.\n"
                "  • Any valid Python / IPython syntax works here.\n"
            )
        self.kernel.shell.push({'what_do': what_do})

        self.console_widget.banner = (
            "Mesofield Terminal — interactive Python console\n"
            "Type what_do() for a guide on available commands and objects.\n"
        )
        dark_bg   = "#2b2b2b"
        light_txt = "#39FF14"
        self.console_widget.setStyleSheet(f"""
            /* console outer frame */
            RichJupyterWidget {{
                background-color: {dark_bg};
            }}

            /* the code / output editors */
            QPlainTextEdit, QTextEdit {{
                background-color: {dark_bg};
                color: {light_txt};
            }}

            /* the prompt numbers */
            QLabel {{
                color: {light_txt};
            }}
        """)
        #----------------------------------------------------------------------------#

    def closeEvent(self, event):
        # 1. Shut down the IPython console
        if hasattr(self, "kernel_client"):
            self.kernel_client.stop_channels()
        if hasattr(self, "kernel_manager"):
            self.kernel_manager.shutdown_kernel()
        if hasattr(self, "console_widget"):
            self.console_widget.close()

        # 2. shut down all hardware
        try:
            self.config.hardware.shutdown()
        except Exception:
            pass

        event.accept()
        QCoreApplication.quit()

    #============================== Private Methods =============================#
    def _on_end(self) -> None:
        """Called when the MDA is finished."""
        #self.config_controller.save_config()

    def _update_config(self, config):
        self.config = config
                
    def _on_pause(self, state: bool) -> None:
        """Called when the MDA is paused."""


