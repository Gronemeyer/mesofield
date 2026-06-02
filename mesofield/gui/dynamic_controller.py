from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

class DynamicController(QWidget):
    # TODO: For now, widget attribute name constants for easier refactoring
    # Should be moved to a separate file or class or made global
    LED_TEST_BTN = 'led_test_button'
    STOP_BTN = 'stop_btn'
    SNAP_BTN = 'snap_btn'
    PSYCHOPY_BTN = 'psychopy_btn'
    NIDAQ_BTN = 'nidaq_btn'
    MOUSEPORTAL_PANEL = 'mouseportal_panel'
    """
    A dynamic controller that loads and arranges GUI components based on hardware features.
    """
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.main_layout = QVBoxLayout(self)

        # Initialize device-specific component registry:
        # maps feature key -> (factory method, layout section)
        self._component_registry = {
            'led_control': (self._create_led_controls, 'buttons'),
            'camera_snap': (self._create_snap_control, 'buttons'),
            'psychopy': (self._create_psychopy_controls, 'buttons'),
            'nidaq_test': (self._create_nidaq_controls, 'buttons'),
            'mouseportal': (self._create_mouseportal_panel, 'panels'),
            # more mappings as needed are added here
        }

        # 'panels' stacks status widgets above the row of action 'buttons'.
        self._sections = {
            'panels': QVBoxLayout(),
            'buttons': QHBoxLayout(),
            # examples:
            # 'dropdowns': QVBoxLayout(),
            # 'tables': QVBoxLayout(), etc.
        }
        for section_layout in self._sections.values():
            self.main_layout.addLayout(section_layout)

        self._load_components()

    def _load_components(self):
        """
        Instantiate and place components based on the aggregated widget list
        provided by HardwareManager.
        """
        # Flattened canonical list of widget keys
        for widget in getattr(self.config.hardware, 'widgets', []):
            if widget in self._component_registry:
                fn, section = self._component_registry[widget]
                fn(self._sections[section])

    def _create_led_controls(self, layout):
        """Create LED test and stop buttons."""
        test_btn = QPushButton("Test LED")
        stop_btn = QPushButton("Stop LED")
        # Expose buttons via constants
        setattr(self, self.LED_TEST_BTN, test_btn)
        setattr(self, self.STOP_BTN, stop_btn)
        layout.addWidget(test_btn)
        layout.addWidget(stop_btn)

    def _create_snap_control(self, layout):
        """Create camera snap button."""
        snap_btn = QPushButton("Snap Image")
        setattr(self, self.SNAP_BTN, snap_btn)
        layout.addWidget(snap_btn)

    def _create_psychopy_controls(self, layout):
        """Create PsychoPy launch button."""
        psychopy_btn = QPushButton("Launch PsychoPy")
        setattr(self, self.PSYCHOPY_BTN, psychopy_btn)
        layout.addWidget(psychopy_btn)
    
    def _create_nidaq_controls(self, layout):
        """Create NIDAQ control button."""
        nidaq_btn = QPushButton("NIDAQ Digital Pulse")
        setattr(self, self.NIDAQ_BTN, nidaq_btn)
        layout.addWidget(nidaq_btn)

    def _create_mouseportal_panel(self, layout):
        """Create the MousePortal load/readiness status panel."""
        device = self.config.hardware.devices.get('mouseportal')
        if device is None:
            return
        from mesofield.gui.mouseportal_panel import MousePortalPanel
        panel = MousePortalPanel(self.config, device, parent=self)
        setattr(self, self.MOUSEPORTAL_PANEL, panel)
        layout.addWidget(panel)
    # Additional factory methods can be added here
