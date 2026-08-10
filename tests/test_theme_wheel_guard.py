"""The scroll wheel must never edit a spin box.

Scrolling a config tab past a spin box used to silently change acquisition
parameters. The guard lives in :mod:`mesofield.gui.theme` and is installed by
``apply_theme``.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QApplication, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from mesofield.gui import theme


def _wheel() -> QWheelEvent:
    return QWheelEvent(
        QPointF(5, 5), QPointF(5, 5), QPoint(0, -120), QPoint(0, -120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )


def _themed_app(qtbot) -> QApplication:
    app = QApplication.instance()
    theme.apply_theme(app)
    return app


def test_wheel_over_spinbox_scrolls_the_page_instead_of_changing_the_value(qtbot):
    app = _themed_app(qtbot)

    area = QScrollArea()
    area.setWidgetResizable(True)
    body = QWidget()
    layout = QVBoxLayout(body)
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    layout.addWidget(spin)
    for _ in range(50):  # make the body taller than the viewport
        layout.addWidget(QSpinBox())
    area.setWidget(body)
    area.resize(300, 200)
    qtbot.addWidget(area)
    area.show()

    before = area.verticalScrollBar().value()
    app.sendEvent(spin, _wheel())

    assert spin.value() == 50, "wheel must not edit the spin box"
    assert area.verticalScrollBar().value() > before, (
        "the spin box must not be a dead zone -- the wheel goes to the scroll area"
    )


def test_spinbox_outside_a_scroll_area_just_ignores_the_wheel(qtbot):
    app = _themed_app(qtbot)

    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(7)
    qtbot.addWidget(spin)
    spin.show()

    app.sendEvent(spin, _wheel())
    assert spin.value() == 7


def test_increment_buttons_and_keys_still_step(qtbot):
    _themed_app(qtbot)

    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(7)
    qtbot.addWidget(spin)
    spin.show()

    spin.stepUp()
    assert spin.value() == 8
    qtbot.keyClick(spin, Qt.Key.Key_Down)
    assert spin.value() == 7


def test_theme_does_not_style_spinbox_increment_buttons(qtbot):
    """Styling ``::up-button``/``::down-button`` drops the native arrow glyphs."""
    assert "up-button" not in theme.STYLESHEET
    assert "down-button" not in theme.STYLESHEET
