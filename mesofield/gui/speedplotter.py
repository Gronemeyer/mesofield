from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
import pyqtgraph as pg

from mesofield.protocols import Configurator
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from mesofield.config import ExperimentConfig

class EncoderWidget(QWidget):
    def __init__(self, cfg: 'Configurator'):
        super().__init__()
        self.config = cfg
        self.init_ui()
        self.init_data()
        self.setFixedHeight(300)

    def init_ui(self):
        self.layout = QVBoxLayout()

        # Status label to show connection status
        self.status_label = QLabel("Click 'Start Live View' to begin.")
        self.info_label = QLabel(f'Viewing data from {self.config.hardware.encoder} at Port: {self.config.hardware.encoder.serial_port} | Baud: {self.config.hardware.encoder.baud_rate}')
        self.start_button = QPushButton("Start Live View")
        self.start_button.setCheckable(True)
        self.plot_widget = pg.PlotWidget()

        self.start_button.clicked.connect(self.toggle_serial_thread)
        self.start_button.setEnabled(True)

        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.plot_widget.setTitle('Serial Trace')
        self.plot_widget.setLabel('left', 'Speed', units='mm/s')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.speed_curve = self.plot_widget.plot(pen='y')

        # Limit the range of the y-axis to +/- 2
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.showGrid(x=True, y=True)
        
        #================================= SerialWorker Signals ================================#
        # self.encoder.serialStreamStarted.connect(self.start_live_view)
        # self.encoder.serialDataReceived.connect(self.process_data)
        # self.encoder.serialStreamStopped.connect(self.stop_timer)
        self.config.hardware.encoder.serialSpeedUpdated.connect(self.receive_speed_data) 
        #========================================================================================#

    def init_data(self):
        self.times = []
        self.speeds = []
        self.start_time = None
        self.timer = None
        self.previous_time = 0

    def toggle_serial_thread(self):
        if self.start_button.isChecked():
            # Create a new encoder instance each time before starting the thread
            self.config.hardware.encoder.serialSpeedUpdated.connect(self.receive_speed_data)
            self.config.hardware.encoder.start()
            self.status_label.setText("Serial thread started.")
        else:
            self.stop_serial_thread()
            self.status_label.setText("Serial thread stopped.")

    def stop_serial_thread(self):
        if self.config.hardware.encoder is not None:
            self.config.hardware.encoder.serialSpeedUpdated.disconnect()

    def receive_speed_data(self, time, speed):
        self.times.append(time)
        self.speeds.append(speed/100)
        # Keep only the last 100 data points
        self.times = self.times[-100:]
        self.speeds = self.speeds[-100:]
        self.update_plot()
        self.status_label.setText(f"Speed: {speed:.2f} mm/s")

    def update_plot(self):
        try:
            if self.times and self.speeds:
                # Update the curve with the last 100 data points
                self.speed_curve.setData(self.times, self.speeds)
                # Adjust x-axis range to show the recent data points
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0)
            else:
                self.plot_widget.clear()
                self.plot_widget.setTitle('No data received.')
        except Exception as e:
            print(f"Exception in update_plot: {e}")
