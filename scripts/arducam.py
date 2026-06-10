"""
Adapted from https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1

"""

from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
import sys
#import cv2
from PyQt6.QtCore import pyqtSignal, QThread
import numpy as np
#from mesofield.gui.viewer import ImagePreview

class VideoThread(QThread):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.core = None  # Placeholder for the camera core

    def run(self):
        # capture from web cam
        capture = cv2.VideoCapture(0)
        while self._run_flag:
            ret, img = capture.read()
            if ret:
                self.image_ready.emit(img)
        # shut down capture system
        capture.release()

    def shutdown(self):
        self.stop()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")

        # Create a vertical layout
        vbox = QVBoxLayout(self)
        self.setLayout(vbox)

        # Create the video capture thread
        self.thread = VideoThread()

        # Create an ImagePreview (PlotWidget) and pass the external signal
        self.image_preview = ImagePreview(
            parent=self,
            mmcore=None,  # Set appropriately or leave None
            image_payload=self.thread.image_ready
        )
        vbox.addWidget(self.image_preview)

        # Start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())