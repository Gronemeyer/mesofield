"""Mesofield Qt GUI package.

Defines the desktop application widgets used by ``mesofield launch``:
the main window, acquisition view, dynamic device controls, config
wizard, image preview, and serial / encoder plotters.

This top-level module exposes :class:`ConfigTableModel`, a thin
``QAbstractTableModel`` that presents a :class:`ConfigRegister` as a
table of (key, value) rows so the GUI can edit experiment parameters
live.
"""

from typing import Any, List

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant

class ConfigTableModel(QAbstractTableModel):
    """A table model that presents ConfigRegister as rows of (key, value)."""
    def __init__(self, registry):
        super().__init__()
        self._registry = registry
        self._keys: List[str] = self._registry.keys()
        # Listen to external changes
        for key in self._keys:
            self._registry.register_callback(key, self._on_config_changed)

    def rowCount(self, parent=QModelIndex()):
        return len(self._keys)

    def columnCount(self, parent=QModelIndex()):
        return 2  # "Parameter" and "Value"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        key = self._keys[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return key
            elif index.column() == 1:
                return str(self._registry.get(key))
        return QVariant()

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ["Parameter", "Value"][section]
        return super().headerData(section, orientation, role)

    def flags(self, index: QModelIndex):
        base = super().flags(index)
        if index.column() == 1:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value: Any, role=Qt.ItemDataRole.EditRole):
        if index.isValid() and index.column() == 1 and role == Qt.ItemDataRole.EditRole:
            key = self._keys[index.row()]
            try:
                # Attempt to set the new value (will auto-convert via ConfigRegister)
                self._registry.set(key, value)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
                return True
            except TypeError as e:
                # you could pop up a QMessageBox here
                print(f"Type error: {e}")
        return False

    def _on_config_changed(self, key: str, new_val: Any):
        """Called by ConfigRegister whenever a key changes externally."""
        try:
            row = self._keys.index(key)
        except ValueError:
            return
        idx = self.index(row, 1)
        self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DisplayRole])
