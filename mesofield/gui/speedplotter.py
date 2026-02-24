from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
import pyqtgraph as pg


class SerialWidget(QWidget):
    """
    A live-plotting widget for any serial device that emits a 
    pyqtSignal(float, float) representing (time, value).
    
    Parameters
    ----------
    cfg : Configurator
        The experiment configuration object.  The device is resolved from
        ``cfg.hardware.<device_attr>``.  If that attribute is ``None`` or
        missing the widget renders in a safe *disconnected* state.
    device_attr : str
        Attribute name on ``cfg.hardware`` to look up (default ``"encoder"``).
    signal_name : str
        Name of the pyqtSignal attribute on the device to connect to.
    label : str
        Human-readable name shown in the info label.
    value_label : str
        Y-axis label for the plot.
    value_units : str
        Y-axis units.
    y_range : tuple[float, float]
        Initial Y-axis range.
    value_scale : float
        Multiplicative factor applied to incoming values before plotting.
    max_points : int
        Number of data points to keep visible.
    """

    def __init__(
        self,
        cfg,
        device_attr: str = "encoder",
        signal_name: str = "serialSpeedUpdated",
        label: str = "Serial Device",
        value_label: str = "Value",
        value_units: str = "",
        y_range: tuple = (-1, 1),
        value_scale: float = 0.01,
        max_points: int = 100,
    ):
        super().__init__()
        self.device = getattr(cfg.hardware, device_attr, None)
        self.signal_name = signal_name
        self.label = label
        self.value_label = value_label
        self.value_units = value_units
        self.y_range = y_range
        self.value_scale = value_scale
        self.max_points = max_points
        self.connected = self.device is not None

        self._signal = getattr(self.device, self.signal_name, None) if self.connected else None

        self.init_ui()
        self.init_data()
        self.setFixedHeight(300)

    # ---- UI ----------------------------------------------------------------

    def init_ui(self):
        self.layout = QVBoxLayout()

        if self.connected:
            port = getattr(self.device, 'serial_port', '?')
            baud = getattr(self.device, 'baud_rate', '?')
            status_text = "Click 'Start Live View' to begin."
            info_text = f'{self.label} on Port: {port} | Baud: {baud}'
        else:
            status_text = "No device connected."
            info_text = f"{self.label}: not available"

        self.status_label = QLabel(status_text)
        self.info_label = QLabel(info_text)
        self.start_button = QPushButton("Start Live View")
        self.start_button.setCheckable(True)
        self.start_button.setEnabled(self.connected)
        self.plot_widget = pg.PlotWidget()

        self.start_button.clicked.connect(self.toggle_serial_thread)

        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        units_str = self.value_units or None
        self.plot_widget.setTitle('Serial Trace')
        self.plot_widget.setLabel('left', self.value_label, units=units_str)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.data_curve = self.plot_widget.plot(pen='y')

        self.plot_widget.setYRange(*self.y_range)
        self.plot_widget.showGrid(x=True, y=True)

        if self._signal is not None:
            self._signal.connect(self.receive_data)

    # ---- Data --------------------------------------------------------------

    def init_data(self):
        self.times = []
        self.values = []
        self.start_time = None
        self.timer = None
        self.previous_time = 0

    def toggle_serial_thread(self):
        if not self.connected:
            return
        if self.start_button.isChecked():
            self._signal.connect(self.receive_data)
            self.device.start()
            self.status_label.setText("Serial thread started.")
        else:
            self.stop_serial_thread()
            self.status_label.setText("Serial thread stopped.")

    def stop_serial_thread(self):
        if self._signal is not None:
            self._signal.disconnect()

    def receive_data(self, time, value):
        self.times.append(time)
        self.values.append(value * self.value_scale)
        self.times = self.times[-self.max_points:]
        self.values = self.values[-self.max_points:]
        self.update_plot()
        unit_suffix = f" {self.value_units}" if self.value_units else ""
        self.status_label.setText(
            f"{self.value_label}: {value:.2f}{unit_suffix}"
        )

    def update_plot(self):
        try:
            if self.times and self.values:
                self.data_curve.setData(self.times, self.values)
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0)
            else:
                self.plot_widget.clear()
                self.plot_widget.setTitle('No data received.')
        except Exception as e:
            print(f"Exception in update_plot: {e}")


# Backwards-compatible alias
EncoderWidget = SerialWidget
