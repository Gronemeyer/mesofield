from collections import deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
import pyqtgraph as pg


class SerialWidget(QWidget):
    """
    A live-plotting widget for any serial device that emits a 
    pyqtSignal(float, float) representing (time, value).
    
    Parameters
    ----------
    cfg : ExperimentConfig
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
        data_provider=None,
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

        # Two ways to feed the plot:
        #   * pull mode  -- `data_provider()` returns ``(times, values)``; the
        #     redraw timer pulls a snapshot. No per-sample Qt signal, so a fast
        #     device cannot flood the GUI event queue. Preferred for high rates.
        #   * push mode  -- connect to a ``pyqtSignal(float, float)`` named
        #     ``signal_name`` and buffer each emission in `receive_data`.
        self.data_provider = data_provider
        self._signal = (
            getattr(self.device, self.signal_name, None)
            if (self.connected and data_provider is None)
            else None
        )
        self._slot_connected = False

        self.init_ui()
        self.init_data()
        self.setFixedHeight(300)

    # ---- Signal connection (idempotent) ------------------------------------

    def _connect_signal(self):
        if self._signal is not None and not self._slot_connected:
            self._signal.connect(self.receive_data)
            self._slot_connected = True

    def _disconnect_signal(self):
        if self._signal is not None and self._slot_connected:
            try:
                self._signal.disconnect(self.receive_data)
            except (TypeError, RuntimeError):
                # Already disconnected, or the slot was never connected.
                pass
            self._slot_connected = False

    # ---- UI ----------------------------------------------------------------

    def init_ui(self):
        self.layout = QVBoxLayout()

        if self.connected:
            port = getattr(self.device, 'port', '?')
            baud = getattr(self.device, 'baudrate', '?')
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

        # Status label, info label, and button share a single row above the plot.
        self.controls_row = QHBoxLayout()
        self.controls_row.addWidget(self.status_label)
        self.controls_row.addWidget(self.info_label)
        self.controls_row.addStretch(1)
        self.controls_row.addWidget(self.start_button)

        self.layout.addLayout(self.controls_row)
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        units_str = self.value_units or None
        self.plot_widget.setTitle('Serial Trace')
        self.plot_widget.setLabel('left', self.value_label, units=units_str)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.data_curve = self.plot_widget.plot(pen='y')

        # Auto-scale the Y axis to the incoming data rather than pinning it to
        # a fixed initial range.
        self.plot_widget.enableAutoRange(axis='y', enable=True)
        self.plot_widget.showGrid(x=True, y=True)

        self._connect_signal()

    # ---- Data --------------------------------------------------------------

    # Plot redraws are decoupled from data arrival
    redraw_interval_ms = 33  # ~30 FPS

    def init_data(self):
        self.times = deque(maxlen=self.max_points)
        self.values = deque(maxlen=self.max_points)
        self._latest_value = None
        self._dirty = False
        self._last_render_n = -1  # pull mode: skip redraw when nothing changed

        # The render timer runs whenever the widget is alive; when no new data
        # has arrived (`_dirty` is False) the tick is a no-op
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(self.redraw_interval_ms)
        self._render_timer.timeout.connect(self._render)
        if self.connected:
            self._render_timer.start()

    def toggle_serial_thread(self):
        if not self.connected:
            return
        if self.start_button.isChecked():
            self._connect_signal()
            self.device.start()
            self._render_timer.start()
            self.status_label.setText("Serial thread started.")
        else:
            self.stop_serial_thread()
            self.status_label.setText("Serial thread stopped.")

    def stop_serial_thread(self):
        self._disconnect_signal()
        self._render_timer.stop()

    def cleanup(self):
        """Sever the inbound signal connection before this widget is destroyed."""
        if getattr(self, "_render_timer", None) is not None:
            self._render_timer.stop()
        self._disconnect_signal()

    def receive_data(self, time, value):
        """Slot for every incoming sample -- buffer only, never redraw here."""
        self.times.append(time)
        self.values.append(value * self.value_scale)
        self._latest_value = value
        self._dirty = True

    def _render(self):
        """Timer-driven repaint; runs at most ``redraw_interval_ms`` apart."""
        if self.data_provider is not None:
            # Pull mode: snapshot the device-side ring buffer (raw values).
            # `count` is monotonic, so this still detects new data after the
            # buffer saturates (when len(xs) is pinned at max_points).
            xs, ys_raw, count = self.data_provider()
            if count == self._last_render_n:
                return  # no new samples since last tick
            self._last_render_n = count
            if not xs:
                return
            ys = [y * self.value_scale for y in ys_raw] if self.value_scale != 1.0 else ys_raw
            self._latest_value = ys_raw[-1]
            self.update_plot(xs, ys)
        else:
            # Push mode: redraw only if a sample arrived since the last tick.
            if not self._dirty:
                return
            self._dirty = False
            self.update_plot(list(self.times), list(self.values))
        if self._latest_value is not None:
            unit_suffix = f" {self.value_units}" if self.value_units else ""
            self.status_label.setText(
                f"{self.value_label}: {self._latest_value:.2f}{unit_suffix}"
            )

    def update_plot(self, xs=None, ys=None):
        if xs is None or ys is None:
            xs = list(self.times)
            ys = list(self.values)
        try:
            if xs and ys:
                self.data_curve.setData(xs, ys)
                self.plot_widget.setXRange(xs[0], xs[-1], padding=0)
            else:
                self.plot_widget.clear()
                self.plot_widget.setTitle('No data received.')
        except Exception as e:
            print(f"Exception in update_plot: {e}")


# Backwards-compatible alias
EncoderWidget = SerialWidget
