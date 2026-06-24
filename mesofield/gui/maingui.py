# Necessary modules for the IPython console
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTabWidget,
    QLayout,
    QToolBar,
    QSizePolicy,
)

from PyQt6.QtGui import QAction
from PyQt6.QtCore import QCoreApplication, Qt

from mesofield.gui.mdagui import MDA
from mesofield.gui import theme
from mesofield.gui.controller import ConfigController
from mesofield.gui.speedplotter import SerialWidget
from mesofield.gui.config_wizard import ConfigWizard
from mesofield.config import ExperimentConfig
from mesofield.protocols import Procedure

class MainWindow(QMainWindow):
    def __init__(self, procedure: Procedure, *, force_wizard: bool = False):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app)

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
        # Disable the toolbar's context menu so users cannot hide it via
        # QMainWindow's default "Toolbars" toggle popup.
        self._toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._toolbar)
        self._prop_browsers: list = []  # open PropertyBrowser dialogs
        self._prop_actions: list = []  # QAction objects for property browsers
        #--------------------------------------------------------------------#

        #=========================== Toolbar action ==========================#
        # Place frequently used tools on the main hardware toolbar.
        self._act_tiff_viewer = QAction("TIFF Viewer\u2026", self)
        self._act_tiff_viewer.setToolTip(
            "Open the TIFF ROI viewer (read-only; refuses files in the active recording)."
        )
        self._act_tiff_viewer.triggered.connect(self._open_tiff_viewer)
        self._toolbar.addAction(self._act_tiff_viewer)
        self._tiff_viewer = None  # keep a reference so the window isn't GC'd
        #--------------------------------------------------------------------#

        #============================== Layout ==============================#
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Build a tab widget (always present). Pin its horizontal size so the
        # ConfigController tab (fixed width) doesn't get stretched when the
        # main window is enlarged — extra horizontal space goes to the MDA
        # acquisition view on the left instead.
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.config_wizard, "⚙ Setup")
        self.right_tabs.addTab(self.console_widget, "Terminal")
        self.right_tabs.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        # The top row layout will hold [acquisition_gui | right_tabs]
        self._top_row = QHBoxLayout()
        self._mda_layout = QVBoxLayout()
        # Give all extra width to the MDA column; right_tabs stays at sizeHint.
        self._top_row.addLayout(self._mda_layout, 1)
        self._top_row.addWidget(self.right_tabs, 0)
        self.main_layout.addLayout(self._top_row)

        # Row that holds every live SerialWidget (device + processor plots),
        # laid out left-to-right so multiple serial traces sit side-by-side on
        # a single row rather than stacked vertically.
        self._plots_row = QHBoxLayout()
        self.main_layout.addLayout(self._plots_row)
        #--------------------------------------------------------------------#

        # Tracking for widgets that get built after config is loaded
        self._acquisition_gui: MDA | None = None
        self._config_controller: ConfigController | None = None
        # One live SerialWidget per streaming (non-camera) data producer,
        # keyed by device_id. Tracked so we can tear them down on rebuild.
        self._device_widgets: dict[str, SerialWidget] = {}
        self._mouseportal_controller: QWidget | None = None
        self._psychopy_controller: QWidget | None = None
        # Widgets built from `procedure.processors` -- one SerialWidget per
        # FrameProcessor whose `plot_enabled` is True. Tracked here so we
        # can tear them down on a `_build_acquisition_ui` rebuild.
        self._processor_widgets: list[SerialWidget] = []

        # Connect hot-load signals
        self.config_wizard.hardwareAboutToChange.connect(self._on_hardware_about_to_change)
        self.config_wizard.configApplied.connect(self._on_config_applied)
        self.config_wizard.hardwareReady.connect(self._build_acquisition_ui)
        self.config_wizard.procedureChanged.connect(self._on_procedure_changed)

        # A configured rig boots straight into acquisition; otherwise the
        # Setup tab is the entry point. ``--wizard`` forces it to the front
        # even when already configured.
        configured = self.procedure.config.hardware.is_configured
        if configured:
            self._build_acquisition_ui()
            self._on_config_applied()
        if force_wizard or not configured:
            self.right_tabs.setCurrentWidget(self.config_wizard)

    #============================== Methods =================================#    
    def toggle_console(self):
        """Switch to the Terminal tab."""
        terminal_index = self.right_tabs.indexOf(self.console_widget)
        self.right_tabs.setCurrentIndex(terminal_index)

    def _open_tiff_viewer(self):
        """Launch the TIFF ROI viewer pre-pointed at the current experiment dir.

        The viewer is given a reference to the running ``Procedure`` so it can
        refuse to open any file inside the active recording's output directory
        while a camera is acquiring.
        """
        from mesofield.gui.tiff_viewer import TiffViewer

        cfg = self.procedure.config
        initial_dir = (
            getattr(cfg, "bids_dir", None)
            or getattr(cfg, "save_dir", None)
            or ""
        )

        # Re-use existing window if still open; otherwise create a new one.
        if self._tiff_viewer is not None and self._tiff_viewer.isVisible():
            self._tiff_viewer.raise_()
            self._tiff_viewer.activateWindow()
            return

        viewer = TiffViewer(initial_dir=initial_dir, procedure=self.procedure)
        viewer.setWindowFlag(Qt.WindowType.Window, True)
        viewer.resize(1100, 800)
        viewer.show()
        self._tiff_viewer = viewer
    
                
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
        from mesofield.gui import theme
        self.console_widget.setStyleSheet(theme.terminal_qss())
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
        # Refresh the embedded IPython console's namespace so that typing
        # ``procedure`` in the Terminal returns the live, hardware-initialized
        # instance instead of the empty default created at launch.
        if getattr(self, "kernel", None) is not None:
            self.kernel.shell.push({"procedure": new_procedure})
        # Keep the Setup tab's displayed/persisted paths pinned to the file the
        # swapped-in procedure actually loaded, so it can't drift from the
        # ExperimentConfig tab built from the same procedure.
        self.config_wizard.sync_from_procedure()

    def _on_config_applied(self) -> None:
        """Rebuild config-dependent tabs after the user applies a configuration."""
        self.display_keys = self.procedure.config.display_keys

        # Rebuild the ConfigController tab
        if self._config_controller is not None:
            idx = self.right_tabs.indexOf(self._config_controller)
            self.right_tabs.removeTab(idx)
            try:
                self._config_controller.cleanup()
            except Exception:
                pass
            self._config_controller.deleteLater()

        self._config_controller = ConfigController(
            self.procedure, display_keys=self.display_keys
        )
        self._config_controller.set_stop_live_hook(
            lambda: self._acquisition_gui.stop_all_live_previews()
            if self._acquisition_gui is not None
            else None
        )
        # Insert after the Setup tab (index 1) so ordering is:
        # [Setup] [ExperimentConfig] [Terminal]
        self.right_tabs.insertTab(1, self._config_controller, "ExperimentConfig")
        self.right_tabs.setCurrentWidget(self._config_controller)

        # MousePortal tab (only when a mouseportal device is loaded). Editable
        # corridor + gain-trial parameters, persisted via update_mouseportal().
        if self._mouseportal_controller is not None:
            idx = self.right_tabs.indexOf(self._mouseportal_controller)
            if idx >= 0:
                self.right_tabs.removeTab(idx)
            # Sever its connections to the (persistent) Procedure events before
            # deletion, or the dead controller keeps reacting to procedure_started
            # and crashes on its deleted widgets.
            try:
                self._mouseportal_controller.cleanup()
            except Exception:
                pass
            self._mouseportal_controller.deleteLater()
            self._mouseportal_controller = None
        if self.procedure.config.hardware.devices.get("mouseportal") is not None:
            from mesofield.gui.mouseportal_controller import MousePortalController
            self._mouseportal_controller = MousePortalController(self.procedure)
            # Saving a task binding re-derives the ExperimentConfig task choices.
            self._mouseportal_controller.tasksChanged.connect(
                lambda: self._config_controller.set_display_keys(
                    self.procedure.config.display_keys
                )
            )
            # Insert right after ExperimentConfig: [Setup][ExperimentConfig][MousePortal][Terminal]
            self.right_tabs.insertTab(2, self._mouseportal_controller, "MousePortal")

        # PsychoPy tab (only when a psychopy device is loaded). Editable
        # task->script map persisted via update_psychopy(); saving re-derives the
        # ExperimentConfig task dropdown via the tasksChanged hook below.
        if self._psychopy_controller is not None:
            idx = self.right_tabs.indexOf(self._psychopy_controller)
            if idx >= 0:
                self.right_tabs.removeTab(idx)
            # Sever its connections to the (persistent) Procedure events before
            # deletion, or the dead controller keeps reacting to procedure_started
            # and crashes on its deleted widgets.
            try:
                self._psychopy_controller.cleanup()
            except Exception:
                pass
            self._psychopy_controller.deleteLater()
            self._psychopy_controller = None
        if self.procedure.config.hardware.devices.get("psychopy") is not None:
            from mesofield.gui.psychopy_controller import PsychoPyController
            self._psychopy_controller = PsychoPyController(self.procedure)
            # Rebuilding the ConfigController re-reads the task choices that Save
            # just re-derived from the script map.
            self._psychopy_controller.tasksChanged.connect(
                lambda: self._config_controller.set_display_keys(
                    self.procedure.config.display_keys
                )
            )
            # Insert right after ExperimentConfig (before any MousePortal tab).
            self.right_tabs.insertTab(2, self._psychopy_controller, "PsychoPy")

        # Pin the right column width to the ConfigController's fixed width
        # (plus a small allowance for tab frame/margins) so the tab area is
        # not stretched by wider tabs (e.g. the Terminal/Setup wizard).
        cc_width = self._config_controller.width() or self._config_controller.sizeHint().width()
        self.right_tabs.setFixedWidth(cc_width + 12)

    def _on_hardware_about_to_change(self) -> None:
        """Sever live viewers from the outgoing cameras before they're torn down.

        Fires on ``hardwareAboutToChange`` (emitted by the wizard before
        ``load_config`` deinitializes the old hardware), so frames in flight
        from a still-running camera can't reach a viewer that's about to be
        rebuilt. The full rebuild still happens later on ``hardwareReady``.
        """
        if self._acquisition_gui is not None:
            try:
                self._acquisition_gui.cleanup()
            except Exception:
                pass

    def _build_acquisition_ui(self) -> None:
        """Build (or rebuild) hardware-dependent widgets: MDA viewer and encoder."""
        # -- MDA / acquisition GUI -------------------------------------------
        if self._acquisition_gui is not None:
            # Disconnect previews from their (longer-lived) cameras BEFORE the
            # async deleteLater(), so a still-streaming camera can't fire a frame
            # into a half-deleted viewer. Mirrors the _build_device_plots idiom.
            try:
                self._acquisition_gui.cleanup()
            except Exception:
                pass
            self._mda_layout.removeWidget(self._acquisition_gui)
            self._acquisition_gui.deleteLater()
            self._acquisition_gui = None

        cameras = tuple(getattr(self.procedure.config.hardware, "cameras", ()) or ())
        if cameras:
            self._acquisition_gui = MDA(self.procedure)
            self._mda_layout.insertWidget(0, self._acquisition_gui)
            self._top_row.setStretch(0, 1)
        else:
            # No cameras: keep the left column truly collapsed.
            self._top_row.setStretch(0, 0)

        # -- Live plots for every streaming (non-camera) data producer -------
        self._build_device_plots()

        # -- Plots for procedure-authored FrameProcessors --------------------
        self._build_processor_plots()

        # -- Refresh the MM config section in the wizard ---------------------
        self.config_wizard.refresh_mm_section()

        # -- Property browser toolbar buttons --------------------------------
        self._build_property_browsers()

    def _build_device_plots(self) -> None:
        """Add a live :class:`SerialWidget` for every streaming data producer.

        A device qualifies when it (a) exposes the standard ``signals.data``
        bundle, (b) is not a camera (cameras get the MDA viewer instead), and
        (c) is not a frame *processor* (those are handled by
        :meth:`_build_processor_plots`). This is what makes a freshly authored
        device — e.g. a lick detector subclassing ``BaseSerialDevice`` — show
        up in the GUI with zero GUI code: the bridge to Qt is attached here,
        lazily, rather than hand-wired in each device's ``__init__``.
        """
        from mesofield.gui.qt_device_adapter import QtDeviceAdapter

        # Tear down the previous pass.
        for widget in self._device_widgets.values():
            try:
                widget.cleanup()
                self._plots_row.removeWidget(widget)
                widget.deleteLater()
            except Exception:
                pass
        self._device_widgets.clear()

        cfg = self.procedure.config
        hardware = cfg.hardware
        cameras = set(getattr(hardware, "cameras", ()) or ())

        for dev_id, device in getattr(hardware, "devices", {}).items():
            dev_type = getattr(device, "device_type", None)
            if device in cameras or dev_type == "camera":
                continue
            if dev_type == "stimulus":
                continue
            if dev_type == "nidaq":
                continue
            # Must speak the standard signal contract to be plottable.
            signals = getattr(device, "signals", None)
            if signals is None or not hasattr(signals, "data"):
                continue

            # Multi-channel devices declare a `plot_config` mapping
            # channel -> styling and expose one `{channel}Updated` pyqtSignal per channel
            plot_cfg = getattr(device, "plot_config", None)
            if isinstance(plot_cfg, dict) and plot_cfg:
                self._build_multichannel_device_plots(cfg, dev_id, device, plot_cfg)
                continue

            # Lazily attach the psygnal->Qt bridge if the device didn't build
            # one itself, then expose the live signal under the conventional
            # attribute name SerialWidget looks for.
            if getattr(device, "serialSpeedUpdated", None) is None:
                try:
                    adapter = QtDeviceAdapter(device)
                except Exception as exc:
                    self._log_exception(f"attach Qt adapter to {dev_id}", exc)
                    continue
                # Keep a ref on the device so the adapter (a QObject) outlives
                # this method and the psygnal connection stays alive.
                device._gui_qt_adapter = adapter
                device.serialDataReceived = adapter.serialDataReceived
                device.serialSpeedUpdated = adapter.serialSpeedUpdated

            # Styling defaults give a usable plot with zero device config; a
            # device may refine them by declaring a `gui_plot_config` dict
            # (e.g. the wheel encoder labels its axis "Speed / mm/s").
            styling = {
                "label": str(dev_id).replace("_", " ").title(),
                "value_label": "Value",
                "value_units": "",
                "value_scale": 1.0,
            }
            override = getattr(device, "gui_plot_config", None)
            if isinstance(override, dict):
                styling.update(override)
            try:
                widget = SerialWidget(
                    cfg=cfg,
                    device_attr=dev_id,
                    signal_name="serialSpeedUpdated",
                    **styling,
                )
            except Exception as exc:
                self._log_exception(f"build SerialWidget for {dev_id}", exc)
                continue
            self._plots_row.addWidget(widget)
            self._device_widgets[dev_id] = widget

    def _build_multichannel_device_plots(self, cfg, dev_id, device, plot_cfg) -> None:
        """Build one :class:`SerialWidget` per channel for a multi-channel device.

        The device declares a ``plot_config`` mapping channel -> styling; each
        entry may carry a ``"source"`` naming the payload-dict key (default: the
        channel name) that yields that channel's scalar. A single
        :class:`DeviceChannelSampler` subscribes to the device's ``signals.data``
        and maintains a bounded ring buffer per channel; each :class:`SerialWidget`
        *pulls* a snapshot on its own redraw timer. This is pull-based by design:
        no per-sample Qt signal crosses threads, so a fast device (~1 kHz) cannot
        flood the GUI event queue. A device gets multi-channel plots from
        ``plot_config`` alone, with no GUI or Qt code of its own.

        Widgets are tracked in ``self._device_widgets`` under
        ``"{dev_id}:{channel}"`` keys so the normal teardown in
        :meth:`_build_device_plots` cleans them up.
        """
        from mesofield.gui.qt_device_adapter import DeviceChannelSampler

        channels = getattr(device, "channels", None) or tuple(plot_cfg.keys())
        channels = [ch for ch in channels if ch in plot_cfg]
        if not channels:
            return

        # One sampler bridges psygnal -> ring buffers for every channel. The
        # channel's payload source is its `plot_config["source"]`, defaulting to
        # the channel name. Size the buffers to the largest channel's window.
        channel_sources = {ch: plot_cfg[ch].get("source", ch) for ch in channels}
        max_points = max(
            int(plot_cfg[ch].get("max_points", 100)) for ch in channels
        )
        try:
            sampler = DeviceChannelSampler(device, channel_sources, max_points=max_points)
        except Exception as exc:
            self._log_exception(f"attach channel sampler to {dev_id}", exc)
            return
        # Keep a ref on the device so the sampler outlives this method and the
        # psygnal connection stays alive.
        device._gui_channel_sampler = sampler

        for channel in channels:
            # `source` is dispatch metadata, not SerialWidget styling.
            styling = {k: v for k, v in plot_cfg[channel].items() if k != "source"}
            styling.setdefault(
                "label", f"{str(dev_id).replace('_', ' ').title()} — {channel}"
            )
            styling.setdefault("value_label", "Value")
            styling.setdefault("value_units", "")
            styling.setdefault("value_scale", 1.0)
            try:
                widget = SerialWidget(
                    cfg=cfg,
                    device_attr=dev_id,
                    signal_name=f"{channel}Updated",
                    data_provider=sampler.provider(channel),
                    **styling,
                )
            except Exception as exc:
                self._log_exception(
                    f"build SerialWidget for {dev_id}:{channel}", exc
                )
                continue
            self._plots_row.addWidget(widget)
            self._device_widgets[f"{dev_id}:{channel}"] = widget

    def _build_processor_plots(self) -> None:
        """Add one SerialWidget per (processor, channel) where the channel
        has an entry in ``processor.plot_config``.

        The procedure (or the ``@processor`` decorator) is responsible for
        constructing the FrameProcessor and declaring per-channel plot
        styling.  We just discover ``procedure.processors`` and render
        anything the user opted in to plot.
        """
        # Tear down anything we built on the previous pass.
        for widget in self._processor_widgets:
            try:
                widget.cleanup()
                self._plots_row.removeWidget(widget)
                widget.deleteLater()
            except Exception:
                pass
        self._processor_widgets.clear()

        cfg = self.procedure.config
        for proc in getattr(self.procedure, "processors", []):
            plot_cfg = getattr(proc, "plot_config", None) or {}
            if not plot_cfg:
                continue
            attr = proc.device_id
            # Expose on cfg.hardware once so SerialWidget's ``device_attr``
            # lookup resolves to this processor for every channel.
            setattr(cfg.hardware, attr, proc)
            for channel in getattr(proc, "channels", ("value",)):
                if channel not in plot_cfg:
                    continue
                styling = dict(plot_cfg[channel])
                styling.setdefault(
                    "label",
                    f"{attr.replace('_', ' ').title()} — {channel}"
                    if len(plot_cfg) > 1 else attr.replace("_", " ").title(),
                )
                styling.setdefault("value_label", "Value")
                styling.setdefault("value_units", "")
                styling.setdefault("y_range", (0, 4096))
                styling.setdefault("value_scale", 1.0)
                try:
                    widget = SerialWidget(
                        cfg=cfg,
                        device_attr=attr,
                        signal_name=f"{channel}Updated",
                        **styling,
                    )
                except Exception as exc:
                    # Don't let a bad plot config kill the whole UI.
                    self._log_exception(
                        f"build SerialWidget for {attr}:{channel}", exc
                    )
                    continue
                self._plots_row.addWidget(widget)
                self._processor_widgets.append(widget)

    def _log_exception(self, ctx: str, exc: Exception) -> None:
        # Defensive logger — MainWindow has no central logger today.
        try:
            print(f"[MainWindow] {ctx} failed: {exc}")
        except Exception:
            pass

    def _build_property_browsers(self) -> None:
        """Add a toolbar button per MicroManager camera that opens a PropertyBrowser."""
        # Close any existing browsers and remove only property-browser actions
        for dlg in self._prop_browsers:
            dlg.close()
            dlg.deleteLater()
        self._prop_browsers.clear()
        # Remove previously-added property-browser actions but keep other toolbar actions
        for act in getattr(self, "_prop_actions", []):
            try:
                self._toolbar.removeAction(act)
            except Exception:
                pass
        self._prop_actions.clear()

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
            self._prop_actions.append(action)

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


# ======================================================================
# Programmatic GUI entry point
# ======================================================================

# Font: Sub-Zero; character width: Full, Character Height: Fitted
# https://patorjk.com/software/taag/#p=display&h=0&v=1&f=Sub-Zero&t=Mesofield
_SPLASH_ASCII = r"""
 __    __     ______     ______     ______     ______   __      ____      __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____-
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/

-------------------------  Mesofield Acquisition Interface  ---------------------------------
"""


def _make_splash():
    """Build the green-on-dark ASCII splash screen used while the GUI loads."""
    from PyQt6.QtWidgets import QSplashScreen
    from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor, QRadialGradient

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
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, _SPLASH_ASCII)
    painter.end()

    return QSplashScreen(pixmap)


def run_gui(procedure: Procedure, *, splash: bool = True,
            force_wizard: bool = False) -> int:
    """Open the Mesofield GUI for an already-built ``procedure`` and block.

    Creates (or reuses) the :class:`QApplication`, applies the theme, optionally
    shows the splash screen, builds a :class:`MainWindow`, and runs the Qt event
    loop until the window closes. Returns the app exit code.

    This is the shared entry point behind both the ``mesofield launch`` CLI
    command and :meth:`mesofield.base.Procedure.launch`, so a user can either
    launch from the command line or call ``proc.launch()`` from an ordinary
    Python script.
    """
    import os
    import time

    from PyQt6.QtGui import QIcon

    app = QApplication.instance() or QApplication([])
    # Drain psygnal `thread="main"` emissions on the GUI thread. mmcore/MDA
    # signals fire on the acquisition worker thread; widgets subscribe with
    # thread="main" so their slots (timer start, widget mutation) run here.
    from psygnal.qt import start_emitting_from_queue
    start_emitting_from_queue()
    theme.apply_theme(app)
    icon_path = os.path.join(os.path.dirname(__file__), "Mesofield_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash_screen = _make_splash() if splash else None
    if splash_screen is not None:
        splash_screen.show()
        app.processEvents()
        time.sleep(0.5)  # give the splash a moment to show

    window = MainWindow(procedure, force_wizard=force_wizard)
    window.show()
    if splash_screen is not None:
        splash_screen.finish(window)
    return app.exec()


