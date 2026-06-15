"""Card-based phase editor for the Phase Diagram config section.

Instead of rendering each phase as a nested inline form, phases are
displayed as compact summary cards in a flow grid.  Adding or editing a
phase opens a dialog with all the fields.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.app_params import pretty_label

CARD_STYLE = (
    'QFrame#phaseCard {'
    '  border: 1px solid palette(mid);'
    '  border-radius: 8px;'
    '}'
    'QFrame#phaseCard:hover {'
    '  border: 1px solid palette(highlight);'
    '}'
    ' QFrame#phaseCard QLabel { border: none; background: transparent; }'
    ' QFrame#phaseCard QPushButton { border: 1px solid palette(mid);'
    '   border-radius: 4px; padding: 3px 10px; }'
)

CARD_MIN_W = 230
CARD_MAX_W = 300
GRID_COLS = 3


class PhaseCardPanel(QWidget):
    """Grid of phase summary cards with add / edit / remove actions."""

    data_changed = Signal()

    def __init__(
        self,
        template: dict,
        path: tuple,
        field_font: QFont,
        sibling_widgets: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._template = template
        self._path = path
        self._field_font = field_font
        self._sibling_widgets = sibling_widgets or {}
        self._phases: list[dict] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(10)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._grid_container)

        add_btn = QPushButton('+ Add Phase')
        add_btn.setFont(field_font)
        add_btn.setMinimumHeight(36)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self._on_add)
        outer.addWidget(add_btn)

    # --------------------------------------------------------------- public

    def phase_count(self) -> int:
        return len(self._phases)

    def collect_data(self) -> dict:
        result = {}
        for phase in self._phases:
            result[phase['_key']] = {k: v for k, v in phase.items() if k != '_key'}
        return result

    def populate(self, data: dict) -> None:
        self._phases.clear()
        for key, values in data.items():
            entry = dict(values)
            entry['_key'] = key
            self._phases.append(entry)
        self._rebuild_grid()

    # ---------------------------------------------------------- element list

    def _get_element_list(self) -> list[str]:
        widget = self._sibling_widgets.get('element_list')
        if widget is None:
            return []
        text = widget.text().strip() if hasattr(widget, 'text') else ''
        if not text:
            return []
        return [e.strip() for e in text.split(',') if e.strip()]

    # ---------------------------------------------------------------- slots

    def _on_add(self) -> None:
        existing_keys = {p['_key'] for p in self._phases}
        idx = 1
        while f'phase_{idx}' in existing_keys:
            idx += 1
        default_key = f'phase_{idx}'

        dlg = PhaseEditDialog(
            self._template,
            default_key,
            {},
            self._field_font,
            self._get_element_list(),
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            entry = dlg.collected_data()
            self._phases.append(entry)
            self._rebuild_grid()
            self.data_changed.emit()

    def _on_edit(self, index: int) -> None:
        phase = self._phases[index]
        key = phase['_key']
        values = {k: v for k, v in phase.items() if k != '_key'}

        dlg = PhaseEditDialog(
            self._template,
            key,
            values,
            self._field_font,
            self._get_element_list(),
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            self._phases[index] = dlg.collected_data()
            self._rebuild_grid()
            self.data_changed.emit()

    def _on_remove(self, index: int) -> None:
        phase = self._phases[index]
        answer = QMessageBox.question(
            self,
            'Remove Phase',
            f'Remove phase "{phase["_key"]}"?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._phases.pop(index)
            self._rebuild_grid()
            self.data_changed.emit()

    # --------------------------------------------------------- grid rebuild

    def _rebuild_grid(self) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, phase in enumerate(self._phases):
            card = self._build_card(i, phase)
            row, col = divmod(i, GRID_COLS)
            self._grid_layout.addWidget(card, row, col)

        remainder = len(self._phases) % GRID_COLS
        if remainder and self._phases:
            row = len(self._phases) // GRID_COLS
            for col in range(remainder, GRID_COLS):
                spacer = QWidget()
                spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                self._grid_layout.addWidget(spacer, row, col)

    def _build_card(self, index: int, phase: dict) -> QFrame:
        card = QFrame()
        card.setObjectName('phaseCard')
        card.setStyleSheet(CARD_STYLE)
        card.setMinimumWidth(CARD_MIN_W)
        card.setMaximumWidth(CARD_MAX_W)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)

        key = phase.get('_key', '?')
        name = phase.get('name', key)

        title = QLabel(f'<b style="font-size:13px;">{name}</b>')
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        prototype = phase.get('prototype', '—')
        proto_label = QLabel(f'<b>Prototype:</b> {prototype}')
        proto_label.setTextFormat(Qt.RichText)
        layout.addWidget(proto_label)

        comp = phase.get('composition', {})
        if comp:
            parts = []
            for elem, ranges in comp.items():
                if isinstance(ranges, dict):
                    lo = ranges.get('min', '?')
                    hi = ranges.get('max', '?')
                    parts.append(f'{elem}: {lo}–{hi}')
                else:
                    parts.append(f'{elem}: {ranges}')
            comp_text = ', '.join(parts) if parts else '—'
        else:
            comp_text = '—'
        comp_label = QLabel(f'<b>Composition:</b> {comp_text}')
        comp_label.setTextFormat(Qt.RichText)
        comp_label.setWordWrap(True)
        layout.addWidget(comp_label)

        max_structs = phase.get('limit_max_num_structures', '—')
        offset = phase.get('offset', '—')
        detail = QLabel(
            f'<b>Max structs:</b> {max_structs}&nbsp;&nbsp;<b>Offset:</b> {offset}'
        )
        detail.setTextFormat(Qt.RichText)
        layout.addWidget(detail)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        edit_btn = QPushButton('Edit')
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.setStyleSheet(
            'QPushButton { border: 1px solid palette(mid); border-radius: 4px;'
            ' padding: 3px 12px; }'
        )
        edit_btn.clicked.connect(lambda _=False, idx=index: self._on_edit(idx))

        remove_btn = QPushButton('Remove')
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setStyleSheet(
            'QPushButton { border: 1px solid palette(mid); border-radius: 4px;'
            ' padding: 3px 12px; color: #c0392b; }'
        )
        remove_btn.clicked.connect(lambda _=False, idx=index: self._on_remove(idx))

        btn_row.addWidget(edit_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return card


class PhaseEditDialog(QDialog):
    """Modal dialog with all fields for one phase."""

    def __init__(
        self,
        template: dict,
        key: str,
        data: dict,
        field_font: QFont,
        element_list: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f'Phase — {key}')
        self.setMinimumWidth(520)
        self._field_font = field_font
        self._key = key
        self._widgets: dict = {}
        self._composition_widgets: list[dict] = []
        self._template = template
        self._element_list = element_list or []

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(8)

        # Key field
        key_edit = QLineEdit(key)
        key_edit.setFont(field_font)
        key_edit.setPlaceholderText('Unique phase identifier (used in TOML)')
        key_edit.textChanged.connect(lambda t: self.setWindowTitle(f'Phase — {t}'))
        self._key_edit = key_edit
        lbl = QLabel('Phase Key *')
        lbl.setStyleSheet('font-weight: bold;')
        form.addRow(lbl, key_edit)

        # Build fields from template
        for field_key, field_def in template.items():
            if not isinstance(field_def, dict):
                continue

            if field_key == 'composition':
                self._build_composition_section(form, data.get('composition', {}))
                continue

            if field_key == 'replacements':
                self._build_replacements_section(
                    form, data.get('replacements', {}), field_def
                )
                continue

            if 'type' not in field_def:
                continue

            widget = self._make_field(field_def)
            if field_key in data:
                self._set_value(widget, data[field_key], field_def.get('type', 'str'))

            is_mandatory = field_def.get('mandatory', False)
            label_text = field_def.get('name', pretty_label(field_key))
            label = QLabel(f'{label_text} *' if is_mandatory else label_text)
            if is_mandatory:
                label.setStyleSheet('font-weight: bold;')
            form.addRow(label, widget)
            self._widgets[field_key] = (widget, field_def)

        scroll.setWidget(form_container)
        outer.addWidget(scroll, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton('Save')
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        outer.addLayout(btn_row)

    def collected_data(self) -> dict:
        result: dict = {'_key': self._key_edit.text().strip()}
        for field_key, (widget, field_def) in self._widgets.items():
            val = self._get_value(widget, field_def.get('type', 'str'))
            if val is not None and val != '':
                result[field_key] = val

        comp = self._collect_composition()
        if comp:
            result['composition'] = comp

        repl = self._collect_replacements()
        if repl:
            result['replacements'] = repl

        return result

    def _on_save(self) -> None:
        key = self._key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, 'Missing Key', 'Phase key is required.')
            return

        problems = self._validate_composition()
        if problems:
            QMessageBox.warning(
                self,
                'Composition Error',
                '\n'.join(problems),
            )
            return

        self.accept()

    # --------------------------------------------------- composition section

    def _build_composition_section(self, form: QFormLayout, data: dict) -> None:
        group = QGroupBox('Composition Ranges')
        group.setStyleSheet(
            'QGroupBox { border: 1px solid palette(mid); border-radius: 4px;'
            ' margin-top: 8px; padding-top: 14px; }'
            'QGroupBox::title { subcontrol-origin: margin;'
            ' left: 8px; padding: 0 4px; }'
        )
        self._comp_layout = QVBoxLayout(group)
        self._comp_layout.setSpacing(6)

        for elem, ranges in data.items():
            self._add_composition_row(elem, ranges)

        add_btn = QPushButton('+ Add Element')
        add_btn.clicked.connect(lambda: self._add_composition_row('', {}))
        self._comp_layout.addWidget(add_btn)
        self._comp_add_btn = add_btn

        form.addRow(group)

    def _other_rows_range(self) -> tuple[float, float]:
        """Sum of (min, max) across all existing composition rows."""
        total_min = sum(e['min'].value() for e in self._composition_widgets)
        total_max = sum(e['max'].value() for e in self._composition_widgets)
        return total_min, total_max

    def _add_composition_row(self, elem: str, ranges: dict) -> None:
        is_new = not ranges

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        if self._element_list:
            elem_combo = QComboBox()
            elem_combo.addItem('', '')
            for e in self._element_list:
                elem_combo.addItem(e, e)
            if elem:
                idx = elem_combo.findData(elem)
                if idx >= 0:
                    elem_combo.setCurrentIndex(idx)
                else:
                    elem_combo.addItem(elem, elem)
                    elem_combo.setCurrentIndex(elem_combo.count() - 1)
            elem_combo.setMinimumWidth(70)
            elem_widget = elem_combo
        else:
            elem_widget = QLineEdit(elem)
            elem_widget.setPlaceholderText('Element')
            elem_widget.setMaximumWidth(90)

        min_spin = QDoubleSpinBox()
        min_spin.setRange(0.0, 1.0)
        min_spin.setDecimals(3)
        min_spin.setSingleStep(0.05)

        max_spin = QDoubleSpinBox()
        max_spin.setRange(0.0, 1.0)
        max_spin.setDecimals(3)
        max_spin.setSingleStep(0.05)

        if is_new:
            other_min, other_max = self._other_rows_range()
            suggested_min = round(max(0.0, 1.0 - other_max), 3)
            suggested_max = round(max(0.0, 1.0 - other_min), 3)
            min_spin.setValue(suggested_min)
            max_spin.setValue(suggested_max)
        else:
            if 'min' in ranges:
                min_spin.setValue(float(ranges['min']))
            if 'max' in ranges:
                max_spin.setValue(float(ranges['max']))

        remove_btn = QPushButton('x')
        remove_btn.setFixedWidth(28)

        row_layout.addWidget(elem_widget)
        row_layout.addWidget(QLabel('min:'))
        row_layout.addWidget(min_spin)
        row_layout.addWidget(QLabel('max:'))
        row_layout.addWidget(max_spin)
        row_layout.addWidget(remove_btn)

        entry = {
            'widget': row_widget,
            'elem': elem_widget,
            'min': min_spin,
            'max': max_spin,
        }
        self._composition_widgets.append(entry)

        remove_btn.clicked.connect(lambda: self._remove_composition_row(entry))

        idx = self._comp_layout.count() - 1
        self._comp_layout.insertWidget(idx, row_widget)

    def _remove_composition_row(self, entry: dict) -> None:
        entry['widget'].deleteLater()
        self._composition_widgets.remove(entry)

    def _collect_composition(self) -> dict:
        result = {}
        for entry in self._composition_widgets:
            w = entry['elem']
            if isinstance(w, QComboBox):
                elem = w.currentData() or w.currentText()
            else:
                elem = w.text().strip()
            if not elem:
                continue
            result[elem] = {
                'min': entry['min'].value(),
                'max': entry['max'].value(),
            }
        return result

    def _validate_composition(self) -> list[str]:
        problems: list[str] = []
        for entry in self._composition_widgets:
            w = entry['elem']
            if isinstance(w, QComboBox):
                elem = w.currentData() or w.currentText()
            else:
                elem = w.text().strip()
            if not elem:
                continue
            lo = entry['min'].value()
            hi = entry['max'].value()
            if lo < 0 or hi < 0:
                problems.append(f'{elem}: composition values must be ≥ 0.')
            if lo > 1 or hi > 1:
                problems.append(f'{elem}: composition values must be ≤ 1.')
            if lo > hi:
                problems.append(f'{elem}: min ({lo}) must be ≤ max ({hi}).')
        return problems

    # ------------------------------------------------- replacements section

    def _build_replacements_section(
        self,
        form: QFormLayout,
        data: dict,
        field_def: dict,
    ) -> None:
        group = QGroupBox('Replacements')
        group.setStyleSheet(
            'QGroupBox { border: 1px solid palette(mid); border-radius: 4px;'
            ' margin-top: 8px; padding-top: 14px; }'
            'QGroupBox::title { subcontrol-origin: margin;'
            ' left: 8px; padding: 0 4px; }'
        )
        repl_form = QFormLayout(group)

        replace_cb = QCheckBox()
        replace_cb.setChecked(bool(data.get('replace', False)))
        repl_form.addRow('Enable replacements', replace_cb)
        self._repl_replace = replace_cb

        elem_list = QLineEdit(
            ', '.join(data.get('element_list', []))
            if isinstance(data.get('element_list'), list)
            else ''
        )
        elem_list.setPlaceholderText('e.g. Ti, V')
        repl_form.addRow('Elements to replace', elem_list)
        self._repl_element_list = elem_list

        replace_with = QLineEdit(data.get('replace_with', ''))
        replace_with.setPlaceholderText('e.g. Ir')
        repl_form.addRow('Replace with', replace_with)
        self._repl_replace_with = replace_with

        form.addRow(group)

    def _collect_replacements(self) -> dict:
        if not hasattr(self, '_repl_replace'):
            return {}
        result: dict = {}
        if self._repl_replace.isChecked():
            result['replace'] = True
        elems = self._repl_element_list.text().strip()
        if elems:
            result['element_list'] = [e.strip() for e in elems.split(',') if e.strip()]
        rw = self._repl_replace_with.text().strip()
        if rw:
            result['replace_with'] = rw
        return result

    # -------------------------------------------------------- field helpers

    def _make_field(self, field_def: dict) -> QWidget:
        ftype = field_def.get('type', 'str')
        default = field_def.get('default')

        if 'bool' in ftype:
            w = QCheckBox()
            if default is not None:
                w.setChecked(bool(default))
            return w
        if 'int' in ftype:
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000_000)
            if default is not None:
                w.setValue(int(default))
            return w
        if 'float' in ftype:
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(5)
            if default is not None:
                w.setValue(float(default))
            return w

        w = QLineEdit(str(default) if default and default != 'None' else '')
        return w

    @staticmethod
    def _get_value(widget: QWidget, ftype: str):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if not text:
                return None
            if 'list' in ftype:
                return [t.strip() for t in text.split(',') if t.strip()]
            try:
                if 'int' in ftype:
                    return int(text)
                if 'float' in ftype:
                    return float(text)
            except (ValueError, TypeError):
                pass
            return text
        return None

    @staticmethod
    def _set_value(widget: QWidget, value, ftype: str) -> None:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QLineEdit):
            if isinstance(value, list):
                widget.setText(', '.join(map(str, value)))
            else:
                widget.setText(str(value))
