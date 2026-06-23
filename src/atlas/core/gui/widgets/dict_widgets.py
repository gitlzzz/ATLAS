"""Custom widgets for dict-type schema fields (kspacing, INCAR)."""

from __future__ import annotations

import contextlib

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _ResizeHandle(QFrame):
    """Thin draggable bar below a widget to resize it vertically."""

    def __init__(self, target: QWidget, parent=None):
        super().__init__(parent)
        self._target = target
        self._dragging = False
        self._start_y = 0
        self._start_height = 0
        self.setFixedHeight(6)
        self.setCursor(QCursor(Qt.SplitVCursor))
        self.setStyleSheet('QFrame { background: palette(mid); border-radius: 2px; }')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_y = event.globalPosition().y()
            self._start_height = self._target.height()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = int(event.globalPosition().y() - self._start_y)
            new_height = max(80, self._start_height + delta)
            self._target.setFixedHeight(new_height)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()


class KspacingWidget(QWidget):
    """Editor for kspacing dict: ATL_DEFAULT float + per-phase overrides."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        default_row = QHBoxLayout()
        default_row.addWidget(QLabel('Default (all phases):'))
        self._default_spin = QDoubleSpinBox()
        self._default_spin.setRange(0.001, 10.0)
        self._default_spin.setDecimals(4)
        self._default_spin.setValue(0.125)
        self._default_spin.valueChanged.connect(self.value_changed)
        default_row.addWidget(self._default_spin)
        default_row.addStretch()
        layout.addLayout(default_row)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(['Phase', 'K-spacing (Å⁻¹)'])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._table.setFixedHeight(150)
        self._table.cellChanged.connect(lambda: self.value_changed.emit())
        layout.addWidget(self._table)
        layout.addWidget(_ResizeHandle(self._table))

        btn_row = QHBoxLayout()
        add_btn = QPushButton('Add phase override')
        add_btn.clicked.connect(self._add_phase)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton('Remove selected')
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _add_phase(self):
        dlg = _AddPhaseDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name, val = dlg.result()
            if name:
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(name))
                self._table.setItem(row, 1, QTableWidgetItem(str(val)))
                self.value_changed.emit()

    def _remove_selected(self):
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._table.removeRow(row)
        self.value_changed.emit()

    def get_value(self) -> dict:
        result = {'ATL_DEFAULT': self._default_spin.value()}
        for row in range(self._table.rowCount()):
            phase_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            if phase_item and val_item:
                with contextlib.suppress(ValueError):
                    result[phase_item.text()] = float(val_item.text())
        return result

    def set_value(self, data: dict):
        self._table.blockSignals(True)
        self._default_spin.blockSignals(True)
        self._table.setRowCount(0)

        if isinstance(data, (int, float)):
            self._default_spin.setValue(float(data))
            self._table.blockSignals(False)
            self._default_spin.blockSignals(False)
            return

        if not isinstance(data, dict):
            self._table.blockSignals(False)
            self._default_spin.blockSignals(False)
            return

        default_val = data.get('ATL_DEFAULT', 0.125)
        self._default_spin.setValue(float(default_val))

        for phase, val in data.items():
            if phase == 'ATL_DEFAULT':
                continue
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(phase))
            self._table.setItem(row, 1, QTableWidgetItem(str(val)))

        self._table.blockSignals(False)
        self._default_spin.blockSignals(False)


class _AddPhaseDialog(QDialog):
    """Small dialog to add a phase k-spacing override."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Phase Override')
        layout = QFormLayout(self)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText('e.g. fcc, bcc, rutile...')
        layout.addRow('Phase name:', self._name_edit)

        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.001, 10.0)
        self._val_spin.setDecimals(4)
        self._val_spin.setValue(0.125)
        layout.addRow('K-spacing (Å⁻¹):', self._val_spin)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton('Add')
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

    def result(self) -> tuple[str, float]:
        return self._name_edit.text().strip(), self._val_spin.value()


class IncarWidget(QWidget):
    """Editor for INCAR dict fields with key-value table and INCAR import."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(['INCAR Tag', 'Value'])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setFixedHeight(280)
        self._table.cellChanged.connect(lambda: self.value_changed.emit())
        layout.addWidget(self._table)
        layout.addWidget(_ResizeHandle(self._table))

        btn_row = QHBoxLayout()
        add_btn = QPushButton('Add tag')
        add_btn.clicked.connect(self._add_tag)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton('Remove selected')
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        import_btn = QPushButton('Import from INCAR text...')
        import_btn.clicked.connect(self._import_incar)
        btn_row.addWidget(import_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _add_tag(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(''))
        self._table.setItem(row, 1, QTableWidgetItem(''))
        self._table.editItem(self._table.item(row, 0))

    def _remove_selected(self):
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._table.removeRow(row)
        self.value_changed.emit()

    def _import_incar(self):
        dlg = _IncarImportDialog(self)
        if dlg.exec() == QDialog.Accepted:
            parsed = dlg.parsed_dict()
            if parsed:
                self.set_value(parsed)
                self.value_changed.emit()

    def get_value(self) -> dict:
        result = {}
        for row in range(self._table.rowCount()):
            key_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            if key_item and val_item:
                key = key_item.text().strip().lower()
                if key:
                    result[key] = _parse_incar_value(val_item.text().strip())
        return result

    def set_value(self, data: dict):
        self._table.blockSignals(True)
        self._table.setRowCount(0)

        if not isinstance(data, dict):
            self._table.blockSignals(False)
            return

        for key, val in data.items():
            if key.startswith('@'):
                continue
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(key).upper()))
            self._table.setItem(row, 1, QTableWidgetItem(_format_value(val)))

        self._table.blockSignals(False)


class _IncarImportDialog(QDialog):
    """Dialog with a text area to paste INCAR content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Import INCAR')
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel('Paste INCAR file contents below:'))

        self._text = QPlainTextEdit()
        self._text.setPlaceholderText(
            'ENCUT = 450\nEDIFF = 1E-06\nIBRION = 2\nNSW = 20\n...'
        )
        layout.addWidget(self._text)

        self._status = QLabel('')
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        parse_btn = QPushButton('Parse and Import')
        parse_btn.clicked.connect(self._do_parse)
        btn_row.addWidget(parse_btn)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._parsed: dict = {}

    def _do_parse(self):
        text = self._text.toPlainText()
        if not text.strip():
            self._status.setText('Nothing to parse.')
            return

        parsed = self._parse_incar_text(text)
        if parsed:
            self._parsed = parsed
            self._status.setText(f'Parsed {len(parsed)} tags. Importing...')
            self.accept()
        else:
            self._status.setText('Could not parse any INCAR tags.')

    def _parse_incar_text(self, text: str) -> dict:
        try:
            from pymatgen.io.vasp.inputs import Incar

            incar = Incar.from_str(text)
            return {
                k.lower(): v
                for k, v in incar.as_dict().items()
                if not k.startswith('@')
            }
        except Exception:
            pass

        # Fallback: manual parsing
        result = {}
        for line in text.splitlines():
            line = line.split('!')[0].split('#')[0].strip()
            if not line or line.startswith('*'):
                continue
            for part in line.split(';'):
                part = part.strip()
                if '=' not in part:
                    continue
                key, _, val = part.partition('=')
                key = key.strip().lower()
                val = val.strip()
                if key:
                    result[key] = _parse_incar_value(val)
        return result

    def parsed_dict(self) -> dict:
        return self._parsed


def _parse_incar_value(text: str):
    """Try to convert an INCAR value string to the appropriate Python type."""
    if not text:
        return text
    low = text.lower().strip()
    if low in ('.true.', 'true'):
        return True
    if low in ('.false.', 'false'):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # Check for list-like values (space-separated numbers)
    parts = text.split()
    if len(parts) > 1:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    return text


def _format_value(val) -> str:
    """Format a Python value for display in the table."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, list):
        return ' '.join(str(v) for v in val)
    return str(val)
