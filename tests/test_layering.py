"""Layering rules that are cheaper to enforce than to remember.

Devices may use Qt core primitives (threads, signals) but must not build UI:
operator interaction goes through :mod:`mesofield.ui`, whose Qt implementation
owns the GUI-thread rule. A widget built on a device's own thread cannot be
parented and its modal never returns.
"""

from __future__ import annotations

from pathlib import Path

DEVICES = Path(__file__).resolve().parents[1] / "mesofield" / "devices"


def test_devices_do_not_build_widgets():
    offenders = [
        f"{path.relative_to(DEVICES)}:{n}: {line.strip()}"
        for path in DEVICES.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "QtWidgets" in line or "QMessageBox" in line
    ]
    assert not offenders, "device code must ask via mesofield.ui:\n" + "\n".join(offenders)
