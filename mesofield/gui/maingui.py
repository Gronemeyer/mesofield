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
    QLayout,
    QToolBar,
)

from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QCoreApplication, Qt

from mesofield.gui.mdagui import MDA
from mesofield.gui.controller import ConfigController
from mesofield.gui.speedplotter import SerialWidget
from mesofield.gui.config_wizard import ConfigWizard
from mesofield.config import ExperimentConfig
from mesofield.protocols import Procedure

class MainWindow(QMainWindow):
    def __init__(self, procedure: Procedure):
        super().__init__()
        self.procedure = procedure
        self.display_keys = self.procedure.config.display_keys       
        self.setWindowTitle("Mesofield")

        #============================== Always-available widgets =============================#
        self.config_wizard = ConfigWizard(self.procedure)
        self.initialize_console(self.procedure)
        #--------------------------------------------------------------------#

        #============================== Toolbar ================================#
        self._toolbar = QToolBar("Hardware")
        self._toolbar.setMovable(False)
        self._toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._toolbar)
        self._prop_browsers: list = []  # open PropertyBrowser dialogs
        #--------------------------------------------------------------------#

        #============================== Layout ==============================#
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Build a tab widget (always present)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.config_wizard, "⚙ Setup")
        self.right_tabs.addTab(self.console_widget, "Terminal")

        # The top row layout will hold [acquisition_gui | right_tabs]
        self._top_row = QHBoxLayout()
        self._mda_layout = QVBoxLayout()
        self._top_row.addLayout(self._mda_layout)
        self._top_row.addWidget(self.right_tabs)
        self.main_layout.addLayout(self._top_row)
        #--------------------------------------------------------------------#

        # Tracking for widgets that get built after config is loaded
        self._acquisition_gui: MDA | None = None
        self._config_controller: ConfigController | None = None
        self._encoder_widget: SerialWidget | None = None

        # Connect hot-load signals
        self.config_wizard.configApplied.connect(self._on_config_applied)
        self.config_wizard.hardwareReady.connect(self._build_acquisition_ui)
        self.config_wizard.procedureChanged.connect(self._on_procedure_changed)

        # If hardware is already configured (e.g. config_path was passed),
        # build the full UI immediately.
        if self.procedure.config.hardware.is_configured:
            self._build_acquisition_ui()
            self._on_config_applied()

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
            self.procedure.config.hardware.shutdown()
        except Exception:
            pass

        event.accept()
        QCoreApplication.quit()

    #============================== Private Methods =============================#

    def _on_procedure_changed(self, new_procedure) -> None:
        """The ConfigWizard discovered a different ``Procedure`` subclass.

        Rebind the live reference so all subsequent UI rebuilds operate on
        the user's custom subclass.
        """
        self.procedure = new_procedure

    def _on_config_applied(self) -> None:
        """Rebuild config-dependent tabs after the user applies a configuration."""
        self.display_keys = self.procedure.config.display_keys

        # Rebuild the ConfigController tab
        if self._config_controller is not None:
            idx = self.right_tabs.indexOf(self._config_controller)
            self.right_tabs.removeTab(idx)
            self._config_controller.deleteLater()

        self._config_controller = ConfigController(
            self.procedure, display_keys=self.display_keys
        )
        # Insert after the Setup tab (index 1) so ordering is:
        # [Setup] [ExperimentConfig] [Terminal]
        self.right_tabs.insertTab(1, self._config_controller, "ExperimentConfig")
        self.right_tabs.setCurrentWidget(self._config_controller)

    def _build_acquisition_ui(self) -> None:
        """Build (or rebuild) hardware-dependent widgets: MDA viewer and encoder."""
        # -- MDA / acquisition GUI -------------------------------------------
        if self._acquisition_gui is not None:
            self._mda_layout.removeWidget(self._acquisition_gui)
            self._acquisition_gui.deleteLater()

        self._acquisition_gui = MDA(self.procedure.config)
        self._mda_layout.insertWidget(0, self._acquisition_gui)

        # -- Encoder widget ---------------------------------------------------
        if self.procedure.config.hardware.encoder is not None:
            if self._encoder_widget is not None:
                self.main_layout.removeWidget(self._encoder_widget)
                self._encoder_widget.deleteLater()

            self._encoder_widget = SerialWidget(
                cfg=self.procedure.config,
                device_attr="encoder",
                signal_name="serialSpeedUpdated",
                label="Encoder",
                value_label="Speed",
                value_units="mm/s",
            )
            self.main_layout.addWidget(self._encoder_widget)

        # -- Refresh the MM config section in the wizard ---------------------
        self.config_wizard.refresh_mm_section()

        # -- Property browser toolbar buttons --------------------------------
        self._build_property_browsers()

    def _build_property_browsers(self) -> None:
        """Add a toolbar button per MicroManager camera that opens a PropertyBrowser."""
        # Close any existing browsers and clear toolbar actions
        for dlg in self._prop_browsers:
            dlg.close()
            dlg.deleteLater()
        self._prop_browsers.clear()
        self._toolbar.clear()

        cameras = self.procedure.config.hardware.cameras
        mm_cams = [
            cam for cam in cameras
            if cam.backend == "micromanager" and hasattr(cam, "core")
        ]
        if not mm_cams:
            return

        try:
            from pymmcore_widgets import PropertyBrowser
        except ImportError:
            # pymmcore-widgets not installed – skip toolbar
            return

        for cam in mm_cams:
            browser = PropertyBrowser(mmcore=cam.core, parent=self)
            browser.setWindowTitle(f"Properties — {cam.name}")
            browser.resize(900, 600)
            self._prop_browsers.append(browser)

            action = QAction(f"🔬 {cam.name} Properties", self)
            action.setToolTip(
                f"Open the device property browser for {cam.name}"
            )
            # Use a default-argument closure to capture the correct browser
            action.triggered.connect(
                lambda checked, b=browser: self._show_property_browser(b)
            )
            self._toolbar.addAction(action)

    @staticmethod
    def _show_property_browser(browser) -> None:
        """Show (or raise) a PropertyBrowser dialog."""
        browser.show()
        browser.raise_()
        browser.activateWindow()

    def _on_end(self) -> None:
        """Called when the MDA is finished."""

    def _update_config(self, config):
        self.procedure.config = config
                
    def _on_pause(self, state: bool) -> None:
        """Called when the MDA is paused."""


