"""Locate the standalone PsychoPy interpreter, so stimulus scripts run in
PsychoPy's own environment rather than mesofield's.
"""

import os

_DEFAULT = r"C:\Program Files\PsychoPy\python.exe"


def get_psychopy_python_exe():
    if os.name != "nt":
        return _DEFAULT
    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\PsychoPy", 0, winreg.KEY_READ)
        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        python_exe = os.path.join(install_path, "python.exe")
        if os.path.exists(python_exe):
            return python_exe
    except OSError:
        pass
    return _DEFAULT
