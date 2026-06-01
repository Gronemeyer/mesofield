"""Mesofield GUI theme — "Control Panel / Net-Runner" design language.

A single source of truth for the application's dark-mode look. The aesthetic is
a mechanical control-system panel with restrained net-runner cues: deep gunmetal
surfaces, hairline borders, square-ish corners, and a muted phosphor-green accent
(with a brighter phosphor reserved for active/recording states).

Usage (call once, right after creating the QApplication)::

    from mesofield.gui import theme
    app = QApplication([])
    theme.apply_theme(app)

The module is intentionally dependency-light (only PyQt6) and cross-platform:
it relies solely on QSS + QPalette + web-safe font *stacks*, so it renders
consistently on macOS (Cocoa) and Windows 11.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    """Canonical color tokens for the Mesofield theme."""

    BG: str = "#0e1216"        # app background (near-black blue-grey)
    PANEL: str = "#161b22"     # cards / group boxes / inputs
    PANEL_HI: str = "#1d242d"  # hover / alternating rows
    BORDER: str = "#2b333d"    # hairline mechanical borders
    BORDER_HI: str = "#3a4450"  # focused / hovered borders
    TEXT: str = "#c8d0d8"      # primary text
    TEXT_DIM: str = "#7d8893"  # secondary / disabled text
    ACCENT: str = "#4fd6a0"    # muted phosphor (primary accent)
    ACCENT_HI: str = "#39FF14"  # bright phosphor (active / recording only)
    WARN: str = "#e0a23c"      # amber (status warnings)
    DANGER: str = "#e0574f"    # red (abort / record)


PALETTE = Palette()

# Convenience module-level aliases (so callers can do ``theme.ACCENT``).
BG = PALETTE.BG
PANEL = PALETTE.PANEL
PANEL_HI = PALETTE.PANEL_HI
BORDER = PALETTE.BORDER
BORDER_HI = PALETTE.BORDER_HI
TEXT = PALETTE.TEXT
TEXT_DIM = PALETTE.TEXT_DIM
ACCENT = PALETTE.ACCENT
ACCENT_HI = PALETTE.ACCENT_HI
WARN = PALETTE.WARN
DANGER = PALETTE.DANGER

# Monospace stack for data / numeric / terminal surfaces only.
MONO_FONT = '"JetBrains Mono", "Cascadia Mono", "SF Mono", "Menlo", "Consolas", monospace'


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
STYLESHEET = f"""
/* ===================== Base surfaces ===================== */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QWidget:disabled {{
    color: {TEXT_DIM};
}}

/* ===================== Toolbar ===================== */
QToolBar {{
    background-color: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px;
}}
QToolBar QToolButton {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
}}
QToolBar QToolButton:hover {{
    background-color: {PANEL_HI};
    border: 1px solid {BORDER_HI};
}}
QToolBar QToolButton:pressed {{
    background-color: {BG};
}}

/* ===================== Menus ===================== */
QMenuBar {{
    background-color: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
}}
QMenuBar::item:selected {{
    background-color: {PANEL_HI};
}}
QMenu {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QMenu::item {{
    padding: 5px 22px;
}}
QMenu::item:selected {{
    background-color: {PANEL_HI};
    color: {ACCENT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 6px;
}}

/* ===================== Tabs ===================== */
QTabWidget::pane {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {BG};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 1px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background-color: {PANEL_HI};
}}
QTabBar::tab:selected {{
    background-color: {PANEL};
    color: {ACCENT};
    border-top: 2px solid {ACCENT};
}}

/* ===================== Group boxes ===================== */
QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 5px;
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ===================== Buttons ===================== */
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background-color: {PANEL_HI};
    border: 1px solid {ACCENT};
}}
QPushButton:pressed {{
    background-color: {BG};
}}
QPushButton:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton:disabled {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
}}
QPushButton:checked {{
    border: 1px solid {ACCENT};
    color: {ACCENT};
}}

/* ===================== Inputs ===================== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QAbstractSpinBox {{
    background-color: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QAbstractSpinBox:hover {{
    border: 1px solid {BORDER_HI};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus, QAbstractSpinBox:focus {{
    border: 1px solid {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {TEXT_DIM};
    background-color: {PANEL};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {PANEL_HI};
    selection-color: {ACCENT};
    outline: none;
}}

/* ===================== Check / radio ===================== */
QCheckBox, QRadioButton {{
    background: transparent;
    color: {TEXT};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    background-color: {BG};
    border: 1px solid {BORDER_HI};
}}
QCheckBox::indicator {{
    border-radius: 2px;
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border: 1px solid {ACCENT};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}

/* ===================== Tables / trees ===================== */
QTableView, QTreeView, QTableWidget, QListView {{
    background-color: {PANEL};
    alternate-background-color: {PANEL_HI};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    selection-background-color: {PANEL_HI};
    selection-color: {ACCENT};
    outline: none;
}}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {{
    background-color: {PANEL_HI};
    color: {ACCENT};
}}
QHeaderView::section {{
    background-color: {BG};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QTableCornerButton::section {{
    background-color: {BG};
    border: 1px solid {BORDER};
}}

/* ===================== Scrollbars (slim, mechanical) ===================== */
QScrollBar:vertical {{
    background: {BG};
    width: 11px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: {BG};
    height: 11px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 2px;
    min-height: 24px;
    min-width: 24px;
    margin: 2px;
}}
QScrollBar::handle:hover {{
    background: {BORDER_HI};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ===================== Splitter ===================== */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:hover {{
    background-color: {ACCENT};
}}

/* ===================== Misc ===================== */
QToolTip {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {ACCENT};
    padding: 4px 6px;
}}
QProgressBar {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
}}
QLabel {{
    background: transparent;
}}
"""


# ---------------------------------------------------------------------------
# Helpers for special widget states
# ---------------------------------------------------------------------------
def record_button_qss() -> str:
    """QSS for the record button: idle gunmetal → armed bright phosphor.

    Use ``setProperty("armed", True)`` + ``style().polish()`` to switch the
    button into its active (bright phosphor) recording state.
    """
    return f"""
        QPushButton {{
            background-color: {PANEL};
            color: {TEXT};
            border: 1px solid {DANGER};
            border-radius: 3px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: {PANEL_HI};
            border: 1px solid {ACCENT_HI};
        }}
        QPushButton:pressed {{
            background-color: {BG};
        }}
        QPushButton[armed="true"] {{
            background-color: {DANGER};
            color: {BG};
            border: 1px solid {ACCENT_HI};
        }}
    """


def terminal_qss() -> str:
    """QSS for the embedded IPython console (monospace phosphor-on-panel)."""
    return f"""
        RichJupyterWidget {{
            background-color: {PANEL};
        }}
        QPlainTextEdit, QTextEdit {{
            background-color: {PANEL};
            color: {ACCENT_HI};
            font-family: {MONO_FONT};
        }}
        QLabel {{
            color: {ACCENT_HI};
        }}
    """


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
def _build_palette() -> QPalette:
    """Construct a dark QPalette so native / unstyled widgets render dark.

    This also guards against the known Windows 11 dark-mode issue where Qt
    paints dark backgrounds on native controls but leaves text dark.
    """
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG))
    p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.Base, QColor(BG))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL_HI))
    p.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_DIM))
    p.setColor(QPalette.ColorRole.Button, QColor(PANEL))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.BrightText, QColor(ACCENT_HI))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(BG))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.Link, QColor(ACCENT))

    # Disabled-state roles.
    disabled = QColor(TEXT_DIM)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def apply_theme(app: QApplication) -> None:
    """Apply the Mesofield dark theme to *app* (palette first, then QSS)."""
    app.setPalette(_build_palette())
    app.setStyleSheet(STYLESHEET)
