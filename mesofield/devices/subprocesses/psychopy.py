"""Windows helpers for launching PsychoPy as a subprocess.

The PsychoPy stimulus *lifecycle* now lives on the shared
:class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice` engine (see
:mod:`mesofield.devices.psychopy_device`); this module retains only the two
Windows-specific helpers that the device and the GUI start gate still need:

- :func:`get_psychopy_python_exe` -- locate the standalone PsychoPy interpreter
  via the registry, so the stimulus script runs in PsychoPy's own environment.
- :func:`force_foreground` -- raise a Qt widget over PsychoPy's full-screen
  window so an operator keypress lands on our dialog, not the stimulus.
"""

import os
import winreg


def get_psychopy_python_exe():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\PsychoPy", 0, winreg.KEY_READ)
        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        python_exe = os.path.join(install_path, "python.exe")
        if os.path.exists(python_exe):
            return python_exe
    except OSError:
        pass
    return r"C:\Program Files\PsychoPy\python.exe"


def force_foreground(widget) -> None:
    """Force *widget* to the foreground with keyboard focus.

    PsychoPy runs its stimulus in a separate process whose full-screen window
    owns the foreground, and Windows blocks a background process from simply
    calling ``SetForegroundWindow``. The reliable workaround is to momentarily
    attach our input thread to the current foreground window's thread, raise our
    window, then detach. Without this the operator has to click the dialog by
    hand before a keypress reaches it. Best-effort and Windows-specific; any
    failure is swallowed (the dialog still works, it just may not auto-focus).
    """
    from PyQt6.QtCore import Qt

    widget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    widget.show()
    widget.raise_()
    widget.activateWindow()
    # Best-effort Win32 foreground nudge. We intentionally avoid
    # AttachThreadInput here: coupling our GUI thread's input queue to
    # PsychoPy's full-screen-window thread can wedge input handling, and
    # WindowStaysOnTopHint + activateWindow is enough in practice.
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(widget.winId())
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
