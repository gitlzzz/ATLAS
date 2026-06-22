"""Interactive 2D structure viewer using ASE's plot_atoms."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

ROTATION_PRESETS: dict[str, str] = {
    'Default': '',
    'Top (z)': '0x,0y,0z',
    'Front (y)': '270x,0y,0z',
    'Side (x)': '270x,0y,90z',
    'Perspective': '10x,10y,0z',
}


class StructureViewer(QWidget):
    """Renders an ASE ``Atoms`` object via ``plot_atoms``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._atoms = None

        self._fig = Figure(figsize=(5, 5), dpi=100)
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvasQTAgg(self._fig)

        self._rotation_combo = QComboBox()
        self._rotation_combo.addItems(list(ROTATION_PRESETS.keys()))
        self._rotation_combo.currentTextChanged.connect(self._on_rotation_changed)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('View:'))
        controls.addWidget(self._rotation_combo, stretch=1)

        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._info_label.setCursor(Qt.IBeamCursor)
        self._info_label.setStyleSheet(
            'padding: 6px; border: 1px solid palette(mid); border-radius: 4px;'
        )
        self._info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._placeholder = QLabel('Select a structure in the table to view it.')
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet('padding: 20px;')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(controls)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._placeholder, stretch=1)
        layout.addWidget(self._info_label)

        self._show_placeholder(True)

    def set_atoms(self, atoms, info: dict | None = None) -> None:
        self._atoms = atoms
        self._info = info or {}
        self._show_placeholder(atoms is None)
        if atoms is not None:
            self._update_info()
            self._render()

    def clear(self) -> None:
        self._atoms = None
        self._info = {}
        self._show_placeholder(True)

    def _show_placeholder(self, show: bool) -> None:
        self._placeholder.setVisible(show)
        self._canvas.setVisible(not show)
        self._info_label.setVisible(not show)
        self._rotation_combo.setEnabled(not show)

    def _on_rotation_changed(self, _text: str) -> None:
        if self._atoms is not None:
            self._render()

    def _render(self) -> None:
        from ase.visualize.plot import plot_atoms

        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._ax.clear()
        self._ax.set_axis_off()

        preset_key = self._rotation_combo.currentText()
        rotation = ROTATION_PRESETS.get(preset_key, '')

        kwargs = {'show_unit_cell': 2, 'radii': 0.8, 'scale': 0.7}
        if rotation:
            kwargs['rotation'] = rotation

        plot_atoms(self._atoms, self._ax, **kwargs)
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw_idle()

    def _update_info(self) -> None:
        atoms = self._atoms
        if atoms is None:
            self._info_label.setText('')
            return

        lines = [f'<b>{atoms.get_chemical_formula()}</b>']
        lines.append(f'{len(atoms)} atoms')

        if self._info.get('phase'):
            lines.append(f'Phase: {self._info["phase"]}')
        if self._info.get('struct_type'):
            lines.append(f'Type: {self._info["struct_type"]}')
        if self._info.get('atl_id'):
            lines.append(f'ID: {self._info["atl_id"]}')
        if self._info.get('calc_energy') is not None:
            lines.append(f'Energy: {self._info["calc_energy"]:.4f} eV')

        cell = atoms.get_cell()
        if cell is not None and cell.any():
            a, b, c = cell.lengths()
            lines.append(f'Cell: {a:.2f} x {b:.2f} x {c:.2f} Å')

        self._info_label.setText('<br>'.join(lines))
