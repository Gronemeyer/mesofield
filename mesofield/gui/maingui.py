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
    QDockWidget,
    QSizePolicy,
    QLayout
)

from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QCoreApplication, Qt

from mesofield.gui.mdagui import MDA
from mesofield.gui.controller import ConfigController
from mesofield.gui.speedplotter import EncoderWidget
from mesofield.config import ExperimentConfig
from mesofield.protocols import Procedure

class MainWindow(QMainWindow):
    def __init__(self, procedure: Procedure):
        super().__init__()
        #self.config: ExperimentConfig = cast(ExperimentConfig, procedure.config)
        self.procedure = procedure
        self.display_keys = self.procedure.config.display_keys
        window_icon = QIcon(os.path.join(os.path.dirname(__file__), "Mesofield_icon.png"))
        self.setWindowIcon(window_icon)        
        self.setWindowTitle("Mesofield")
        #============================== Widgets =============================#
        self.acquisition_gui = MDA(self.procedure.config)
        self.config_controller = ConfigController(self.procedure, display_keys=self.display_keys)
        self.encoder_widget = EncoderWidget(self.procedure.config)
        self.initialize_console(self.procedure) # Initialize the IPython console
        #--------------------------------------------------------------------#

        #============================== Layout ==============================#
        self.toggle_console_action = self.menuBar().addAction("Toggle Console")
        self.float_action = self.menuBar().addAction("Floating Console")
        self.toggle_console_action.setCheckable(True)
        self.float_action.setCheckable(True)

        central_widget = QWidget()
        # Use a vertical layout: top = acquisition/config + encoder; bottom = console
        self.main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        mda_layout = QVBoxLayout()
        self.main_layout.addLayout(mda_layout)
        self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Horizontal row for acquisition GUI and config controller
        top_row = QHBoxLayout()
        top_row.addWidget(self.acquisition_gui)
        top_row.addWidget(self.config_controller)
        mda_layout.addLayout(top_row)

        # Encoder widget below the top row
        mda_layout.addWidget(self.encoder_widget)
        
        # embed console into a dock at the bottom
        self.console_dock = QDockWidget("Mesofield IPython Console", self)
        self.console_dock.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.console_dock.setMinimumHeight(300)
        self.console_dock.setMinimumWidth(600)
        self.console_dock.setWidget(self.console_widget)
        self.console_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        # hidden by default; toggle via View menu
        self.console_dock.hide()


        #--------------------------------------------------------------------#

        #============================== Signals =============================#
        self.toggle_console_action.toggled.connect(self.console_dock.setVisible)

        # re‐compute your “minimum size” to preserve the layout 
        self.console_dock.visibilityChanged.connect(self.adjustSize)
        self.console_dock.topLevelChanged.connect(self.adjustSize)
        
        # allow user to toggle floating (dock/undock)
        self.float_action.toggled.connect(self.console_dock.setFloating)

        #--------------------------------------------------------------------#

    #============================== Methods =================================#    
    def toggle_console(self):
        """Show or hide the docked IPython console."""
        self.console_dock.setVisible(not self.console_dock.isVisible())
    
                
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
        self.console_widget.banner = r"""
 __    __     ______     ______     ______     ______   __     ______     __         _____    
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.  
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \ 
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____- 
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/ 
                                                                                  
-------------------------  Mesofield Acquisition Interface  ---------------------------------
"""
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


