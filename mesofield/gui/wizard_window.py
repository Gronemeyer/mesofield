"""Standalone launcher window that hosts the :class:`ConfigWizard`.

Configuration is a *pre-flight* step, not one tab among many: the user picks a
rig, optionally an experiment, presses Apply, and watches hardware come up. This
window gives that step its own frame -- wizard on top, a live console beneath --
and closes itself once the configuration has been applied, handing off to the
main acquisition window.

The console is what makes Apply legible. ``Procedure.load_config`` initialises
hardware synchronously on the GUI thread, so without help the window would
freeze for the several seconds a camera or serial device takes to come up. The
:class:`LogConsole` sink pumps the event loop as each line arrives, so the
window keeps painting and the user can watch the rig boot instead of staring at
a hung dialog.
"""

from __future__ import annotations

import sys
from typing import TextIO

from PyQt6.QtCore import QEventLoop, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mesofield.gui import theme
from mesofield.gui.config_wizard import ConfigWizard, UI
from mesofield.utils import sysinfo
from mesofield.utils._logger import log_file


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
# Printed in the console at launch, above everything else. Put your own lines
# in CUSTOM_STARTUP_LINES: each entry is a plain string, or a zero-argument
# callable returning a string or a list of them, evaluated once when the window
# opens. Keep callables cheap -- they run before the window paints. One that
# raises is reported inline rather than taking the GUI down with it.
#
#     CUSTOM_STARTUP_LINES = [
#         "rig: two-photon bay 3",
#         lambda: f"disk free: {shutil.disk_usage('D:/').free // 2**30} GB",
#     ]

CUSTOM_STARTUP_LINES: list = []


def _resolve_custom_lines() -> list[str]:
    """Flatten :data:`CUSTOM_STARTUP_LINES` into printable strings."""
    out: list[str] = []
    for entry in CUSTOM_STARTUP_LINES:
        try:
            value = entry() if callable(entry) else entry
        except Exception as exc:
            out.append(f"!  custom startup line failed: {exc!r}")
            continue
        if isinstance(value, (list, tuple)):
            out.extend(str(v) for v in value)
        elif value is not None:
            out.append(str(value))
    return out


# Lines are dropped on the floor past this many, so a chatty device driver
# can't grow the console without bound during a long initialisation.
_MAX_BLOCKS = 5000

# Delay between "configuration applied" and the window closing, so the last
# few log lines are actually readable rather than flashing past.
_CLOSE_DELAY_MS = 800


class _StreamTee:
    """Mirror writes to *original* into a callback.

    Devices and third-party libraries print rather than log, so capturing the
    loguru sink alone would leave the console half-empty during hardware init.
    """

    def __init__(self, original: TextIO | None, emit) -> None:
        self._original = original
        self._emit = emit

    def write(self, text: str) -> int:
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        if text and text.strip():
            self._emit(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):  # delegate fileno(), encoding, ...
        return getattr(self._original, name)


class LogConsole(QPlainTextEdit):
    """Read-only pane showing loguru records and anything printed to stdout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_BLOCKS)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {theme.BG}; color: {theme.TEXT_DIM}; "
            f"border: 1px solid {theme.PANEL_HI}; }}"
        )

        self._sink_id: int | None = None
        self._saved_stdout: TextIO | None = None
        self._saved_stderr: TextIO | None = None
        # Guards the processEvents() below: a slot invoked from that call can
        # itself print, and re-entering the event loop from there risks running
        # the same handler on a half-built widget tree.
        self._pumping = False

    # -- capture lifecycle ---------------------------------------------------

    def start_capture(self) -> None:
        """Route loguru records and stdout/stderr writes into this pane.

        Everything logged before this window existed -- config parsing, hardware
        discovery, the whole launch preamble that used to scroll past in the
        terminal -- is replayed first, so the console shows the full session
        rather than starting mid-story.
        """
        if self._sink_id is None:
            from loguru import logger
            from mesofield.utils._logger import GUI_FORMAT, buffered_records

            for line in buffered_records():
                self.appendPlainText(line)
            self._scroll_to_tail()

            self._sink_id = logger.add(
                lambda message: self.append_line(str(message).rstrip("\n")),
                format=GUI_FORMAT,
                level="INFO",
                colorize=False,
            )

        if self._saved_stdout is None:
            self._saved_stdout = sys.stdout
            self._saved_stderr = sys.stderr
            sys.stdout = _StreamTee(self._saved_stdout, self.append_line)
            sys.stderr = _StreamTee(self._saved_stderr, self.append_line)

    def stop_capture(self) -> None:
        """Undo :meth:`start_capture`. Safe to call twice."""
        if self._sink_id is not None:
            from loguru import logger

            try:
                logger.remove(self._sink_id)
            except ValueError:
                pass  # sink already gone (e.g. logger reconfigured)
            self._sink_id = None

        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
            sys.stderr = self._saved_stderr
            self._saved_stdout = None
            self._saved_stderr = None

    # -- output --------------------------------------------------------------

    def _scroll_to_tail(self) -> None:
        """Follow the newest line, but stay pinned to the left margin.

        Appending leaves the cursor at the end of the text, and Qt scrolls
        horizontally to keep *that* visible -- on a long path or traceback the
        view ends up parked mid-line. Setting the scrollbar back to 0 is not
        enough on its own (before the first layout its range is still empty, so
        the write is discarded), so the cursor is moved to the start of the last
        line and Qt's own follow-the-cursor logic does the right thing.
        """
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        self.setTextCursor(cursor)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.horizontalScrollBar().setValue(0)

    def append_line(self, text: str) -> None:
        """Append *text* and repaint, even while the GUI thread is blocked.

        ``load_config`` never returns to the event loop, so the usual
        "append and let Qt repaint" contract does not hold. Pumping here is
        what keeps the window alive through hardware initialisation.
        """
        self.appendPlainText(text)
        self._scroll_to_tail()
        if self._pumping:
            return
        self._pumping = True
        try:
            # Exclude user input: Apply is mid-flight, and re-delivering clicks
            # here would let the user press it again from inside load_config.
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )
        finally:
            self._pumping = False


class WizardWindow(QDialog):
    """Frame around a :class:`ConfigWizard` plus the live console.

    The wizard widget is *borrowed*, not owned: the caller keeps the same
    instance (and its signal connections) across open/close cycles, so the
    window re-parents it back out on close.
    """

    def __init__(self, wizard: ConfigWizard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wizard = wizard
        self.setWindowTitle(UI.WINDOW_TITLE)
        self.setSizeGripEnabled(True)
        self.resize(620, 560)

        self.console = LogConsole()
        self._tiff_viewer = None  # keep a reference so the viewer isn't GC'd

        tools = QHBoxLayout()
        tiff_btn = QPushButton("TIFF Viewer…")
        tiff_btn.setToolTip(
            "Open the TIFF ROI viewer (read-only; refuses files in the active recording)."
        )
        tiff_btn.clicked.connect(self._open_tiff_viewer)
        tools.addWidget(tiff_btn)
        tools.addStretch()

        # The console and its header travel together in the splitter, so
        # dragging the handle moves the label with the pane it names.
        console_pane = QWidget()
        console_box = QVBoxLayout(console_pane)
        console_box.setContentsMargins(0, 0, 0, 0)
        console_box.setSpacing(3)
        console_box.addLayout(self._console_header())
        console_box.addWidget(self.console, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(wizard)
        splitter.addWidget(console_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        # The wizard's own content is ~330px tall; give it that and let the
        # console take the rest, so nothing is scrolled at the default size.
        splitter.setSizes([330, 190])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(tools)
        layout.addWidget(splitter)

        # Banner last: start_capture() replays the pre-GUI records, and the
        # console follows its tail, so anything printed before the replay would
        # be scrolled out of sight the moment the window opens.
        self.console.start_capture()
        self._print_banner()
        wizard.configApplied.connect(self._on_applied)

    # -- console header ------------------------------------------------------

    def _console_header(self) -> QHBoxLayout:
        """``Console`` label, with a link to the log file this session writes."""
        row = QHBoxLayout()
        label = QLabel("Console")
        label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(label)

        path = log_file()
        if path is None:
            # setup_logging() hasn't run, so there is no file sink to point at.
            hint = QLabel("· not logging to file")
            hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
            row.addWidget(hint)
        else:
            link = QLabel(f'· <a href="{QUrl.fromLocalFile(str(path)).toString()}">'
                          f'{path.name}</a>')
            link.setToolTip(str(path))
            # Open through QDesktopServices rather than setOpenExternalLinks so
            # the click is routed by us (and can be logged/blocked later).
            link.linkActivated.connect(
                lambda url: QDesktopServices.openUrl(QUrl(url))
            )
            row.addWidget(link)

        row.addStretch()
        return row

    # -- startup banner ------------------------------------------------------

    def _print_banner(self) -> None:
        """Write the version/machine banner, plus any custom lines.

        Appended directly rather than through ``append_line`` -- pumping the
        event loop once per line here would just slow the window's first paint.
        """
        self.console.appendPlainText("-" * 64)
        for line in sysinfo.startup_lines() + _resolve_custom_lines():
            self.console.appendPlainText(line)
        self.console.appendPlainText("-" * 64)
        self.console._scroll_to_tail()

    # -- tools ---------------------------------------------------------------

    def _open_tiff_viewer(self) -> None:
        """Launch the TIFF ROI viewer pre-pointed at the current experiment dir.

        The viewer gets a reference to the live ``Procedure`` so it can refuse
        to open any file inside the active recording's output directory while a
        camera is acquiring.
        """
        from mesofield.gui.tiff_viewer import TiffViewer

        if self._tiff_viewer is not None and self._tiff_viewer.isVisible():
            self._tiff_viewer.raise_()
            self._tiff_viewer.activateWindow()
            return

        procedure = self._wizard.procedure
        cfg = procedure.config
        initial_dir = (
            getattr(cfg, "bids_dir", None) or getattr(cfg, "save_dir", None) or ""
        )

        viewer = TiffViewer(initial_dir=initial_dir, procedure=procedure)
        viewer.setWindowFlag(Qt.WindowType.Window, True)
        viewer.resize(1100, 800)
        viewer.show()
        self._tiff_viewer = viewer

    # -- lifecycle -----------------------------------------------------------

    def _on_applied(self) -> None:
        """Configuration succeeded: let the tail of the log settle, then close."""
        self.console.append_line("--- configuration applied ---")
        QTimer.singleShot(_CLOSE_DELAY_MS, self.accept)

    def closeEvent(self, event) -> None:
        self._detach()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        self._detach()
        super().done(result)

    def _detach(self) -> None:
        """Release the borrowed wizard and stop capturing output."""
        self.console.stop_capture()
        try:
            self._wizard.configApplied.disconnect(self._on_applied)
        except TypeError:
            pass  # already disconnected
        if self._wizard.parent() is not None:
            self._wizard.setParent(None)
