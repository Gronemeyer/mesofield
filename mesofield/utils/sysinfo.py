"""Short, human-readable descriptions of the machine and the software stack.

Used for the banner the Mesofield Wizard prints before anything is configured,
so a rig's console log carries enough provenance to interpret it later: which
mesofield/pymmcore-plus built the data, and what hardware it ran on.

Every probe here is best-effort. A rig that can't answer one of these questions
is not a broken rig, so each lookup degrades to ``"unknown"`` rather than
raising -- nothing in this module should be able to stop the GUI from opening.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache

# Windows shells would flash a console window for each probe below.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

_UNKNOWN = "unknown"


def _run(args: list[str], timeout: float = 2.0) -> str:
    """Run *args* and return stripped stdout, or ``""`` on any failure."""
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------

def package_version(name: str) -> str:
    """Installed version of *name*, or ``"unknown"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return _UNKNOWN
    except Exception:
        return _UNKNOWN


def python_version() -> str:
    return platform.python_version()


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def os_name() -> str:
    """e.g. ``"Windows 11 Pro (10.0.26100)"``, ``"macOS 14.5"``, ``"Linux 6.8"``."""
    system = platform.system()

    if system == "Windows":
        # platform.release() still reports "10" on Windows 11 -- the build
        # number is the only reliable discriminator (11 starts at 22000).
        build_str = platform.version()          # "10.0.26100"
        try:
            build = int(build_str.split(".")[-1])
        except ValueError:
            build = 0
        release = "11" if build >= 22000 else platform.release()
        # The edition ("Pro", "Enterprise") is only available via a CIM query
        # that costs ~0.5s of frozen GUI at launch -- not worth it for a banner.
        return f"Windows {release} ({build_str})"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0] or platform.release()}"

    if system == "Linux":
        # /etc/os-release is the distro's own name for itself.
        try:
            with open("/etc/os-release", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PRETTY_NAME="):
                        pretty = line.split("=", 1)[1].strip().strip('"')
                        return f"{pretty} (kernel {platform.release()})"
        except OSError:
            pass
        return f"Linux {platform.release()}"

    return platform.platform() or _UNKNOWN


@lru_cache(maxsize=1)
def cpu_name() -> str:
    """Marketing name of the CPU, e.g. ``"11th Gen Intel Core i9-11900K"``.

    ``platform.processor()`` is a family/model/stepping string on Windows and
    empty on many Linux builds, so it is only the last resort.
    """
    system = platform.system()

    if system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if name:
                return " ".join(str(name).split())
        except Exception:
            pass

    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    return platform.processor() or _UNKNOWN


@lru_cache(maxsize=1)
def gpu_name() -> str:
    """Marketing name of the primary GPU, or ``"unknown"``.

    NVIDIA is asked first (``nvidia-smi`` is cheap and these rigs usually have
    one); otherwise fall back to whatever the OS lists as a display adapter.
    Multiple adapters are joined, so an integrated + discrete pair is visible.
    """
    smi = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if smi:
        return ", ".join(dict.fromkeys(smi.splitlines()))

    system = platform.system()
    if system == "Windows":
        out = _run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            timeout=4.0,
        )
        if out:
            names = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return ", ".join(dict.fromkeys(names)) or _UNKNOWN

    elif system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"], timeout=6.0)
        for line in out.splitlines():
            if "Chipset Model:" in line:
                return line.split(":", 1)[1].strip()

    elif system == "Linux":
        out = _run(["lspci"], timeout=4.0)
        for line in out.splitlines():
            if "VGA compatible controller" in line or "3D controller" in line:
                return line.split(":", 2)[-1].strip()

    return _UNKNOWN


@lru_cache(maxsize=1)
def ram_gb() -> str:
    """Total physical memory as e.g. ``"64 GB"``, or ``"unknown"``."""
    total = 0
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:
        try:  # POSIX without psutil
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            total = 0

    if total <= 0:
        return _UNKNOWN
    return f"{total / (1024 ** 3):.0f} GB"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def startup_lines() -> list[str]:
    """The ``label: value`` lines shown in the wizard console at launch."""
    rows = [
        ("mesofield", package_version("mesofield")),
        ("pymmcore-plus", package_version("pymmcore-plus")),
        ("python", f"{python_version()} ({sys.executable})"),
        ("os", os_name()),
        ("cpu", cpu_name()),
        ("gpu", gpu_name()),
        ("ram", ram_gb()),
    ]
    width = max(len(label) for label, _ in rows)
    return [f"{label:<{width}}  {value}" for label, value in rows]
