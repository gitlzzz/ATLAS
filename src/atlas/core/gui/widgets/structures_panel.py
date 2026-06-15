"""Structures overview panel for the Init DB Outputs tab.

Layout (3 rows, 2 columns):
  Row 1 left : "By Phase" + "By Type" charts
  Row 1 right: Structure viewer (spans rows 1-2)
  Row 2 left : "Atom Count" + "Modifications" charts
  Row 2 right: (continued viewer)
  Row 3      : Full-width sortable table
"""

from __future__ import annotations

import contextlib
import json

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

VISIBLE_COLUMNS = [
    ('atl_id', 'ID'),
    ('formula', 'Formula'),
    ('n_atoms', 'Atoms'),
    ('phase', 'Phase'),
    ('struct_type', 'Type'),
    ('modifications', 'Modifications'),
    ('calc_performed', 'DFT Done'),
    ('calc_energy', 'Energy (eV)'),
]

TYPE_COLORS = {
    'bulk': '#458588',
    'surface': '#fe8019',
    'cluster': '#d3869b',
    'isolated_atom': '#365c54',
    'unknown': '#ee0000',
}
MOD_COLORS = {
    'perturb': '#d79921',
    'deformation': '#b16286',
    'vacancy': '#689d6a',
    'replacement': '#076678',
}
PHASE_CMAP = 'viridis'


class _StructuresTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._columns = VISIBLE_COLUMNS

    def load(self, rows: list[dict]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent=None):
        return len(self._rows)

    def columnCount(self, parent=None):
        return len(self._columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._columns[section][1]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row = self._rows[index.row()]
        key = self._columns[index.column()][0]
        value = row.get(key)
        if key == 'calc_performed':
            return 'Yes' if value else 'No'
        if key == 'calc_energy' and value is not None:
            return f'{value:.4f}'
        if key == 'modifications':
            if not value or value == '[]':
                return '—'
            try:
                return ', '.join(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                return str(value)
        if value is None:
            return '—'
        return str(value)


class _ChartPair(QWidget):
    """Two charts side-by-side in a single Figure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(6, 2.5), dpi=100, tight_layout=True)
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def draw(self, left_func, right_func) -> None:
        self._fig.clear()
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        axes = self._fig.subplots(1, 2)
        left_func(axes[0])
        right_func(axes[1])
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()


def _viridis_colors(n: int) -> list:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(PHASE_CMAP)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _draw_phase_bar(ax, data: list[tuple[str, int]], total: int) -> None:
    if not data:
        ax.set_visible(False)
        return
    labels, counts = zip(*data, strict=True)
    n = len(labels)
    colors = _viridis_colors(n)
    y = np.arange(n)
    bars = ax.barh(y, counts, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('By Phase', fontsize=10, fontweight='bold')
    for bar, c in zip(bars, counts, strict=True):
        pct = c / total * 100 if total else 0
        ax.text(
            bar.get_width() + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{c} ({pct:.0f}%)',
            va='center',
            fontsize=8,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_type_bar(ax, data: list[tuple[str, int]]) -> None:
    if not data:
        ax.set_visible(False)
        return
    labels, counts = zip(*data, strict=True)
    colors = [TYPE_COLORS.get(lb, '#757779') for lb in labels]
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('By Type', fontsize=10, fontweight='bold')
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.02, i, str(c), va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_atoms_hist(ax, n_atoms: list[int]) -> None:
    if not n_atoms:
        ax.set_visible(False)
        return
    arr = np.array(n_atoms)
    n_bins = min(max(10, len(set(arr)) // 2), 50)
    ax.hist(arr, bins=n_bins, color='#458588', alpha=0.7, edgecolor='white')
    ax.axvline(
        np.median(arr),
        color='#cc241d',
        ls='--',
        lw=1.2,
        label=f'median = {int(np.median(arr))}',
    )
    ax.set_xlabel('Atoms per structure')
    ax.set_ylabel('Count')
    ax.set_title('Atom Count Distribution', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_mods_bar(
    ax,
    data: list[tuple[str, int]],
    total: int,
    labelled: int,
) -> None:
    if not data:
        ax.text(
            0.5,
            0.5,
            'No modifications',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=10,
            color='#6c757d',
        )
        ax.set_title('Modifications', fontsize=10, fontweight='bold')
        ax.set_axis_off()
        return
    labels, counts = zip(*data, strict=True)
    colors = [MOD_COLORS.get(lb, '#757779') for lb in labels]
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Structures')
    ax.set_title('Modifications', fontsize=10, fontweight='bold')
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.02, i, str(c), va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if total:
        ax.text(
            0.98,
            0.02,
            f'DFT: {labelled}/{total}',
            ha='right',
            va='bottom',
            transform=ax.transAxes,
            fontsize=8,
            color='#6c757d',
        )


class StructuresPanel(QWidget):
    """Chart grid + sortable table of indexed structures, with structure viewer.

    Layout: charts on the left, structure viewer on the right (spanning both
    chart rows), full-width table at the bottom.
    """

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._init_db_cache = None

        self._charts_top = _ChartPair()
        self._charts_bottom = _ChartPair()

        self._model = _StructuresTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)
        self._table.selectionModel().selectionChanged.connect(
            self._on_row_selected,
        )

        from atlas.core.gui.widgets.structure_viewer import StructureViewer

        self._viewer = StructureViewer()

        self._empty_label = QLabel(
            'No structures indexed yet.\n'
            'Generate a database in the Config tab, then come back here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('padding: 40px;')

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        # --- Layout: 3 rows x 2 cols ---
        # Left column: charts stacked vertically
        charts_col = QSplitter(Qt.Vertical)
        charts_col.addWidget(self._charts_top)
        charts_col.addWidget(self._charts_bottom)
        charts_col.setStretchFactor(0, 1)
        charts_col.setStretchFactor(1, 1)

        # Top area: charts (left) + viewer (right)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(charts_col)
        top_splitter.addWidget(self._viewer)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        # Main vertical split: top area + full-width table
        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(top_splitter)
        self._main_splitter.addWidget(self._table)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 2)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(refresh_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(btn_row)
        layout.addWidget(self._main_splitter, stretch=1)
        layout.addWidget(self._empty_label)

        self.refresh()

    def refresh(self) -> None:
        self._init_db_cache = None
        self._viewer.clear()

        # Re-index from disk so SQLite data stays fresh
        with contextlib.suppress(Exception):
            self._project.refresh_structures_index()

        counts = self._project.structure_counts()
        total = counts['total']

        if total == 0:
            self._main_splitter.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._main_splitter.setVisible(True)
        self._empty_label.setVisible(False)

        phase_data = self._project.structure_breakdown('phase')
        type_data = self._project.structure_breakdown('struct_type')
        n_atoms = self._project.n_atoms_list()
        mods_data = self._project.modifications_breakdown()
        labelled = counts['labelled']

        self._charts_top.draw(
            lambda ax: _draw_phase_bar(ax, phase_data, total),
            lambda ax: _draw_type_bar(ax, type_data),
        )
        self._charts_bottom.draw(
            lambda ax: _draw_atoms_hist(ax, n_atoms),
            lambda ax: _draw_mods_bar(ax, mods_data, total, labelled),
        )

        rows = self._project.list_structures()
        self._model.load(rows)

    def _on_row_selected(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return
        source_idx = self._proxy.mapToSource(indexes[0])
        row = self._model._rows[source_idx.row()]
        atl_id = row.get('atl_id')
        if not atl_id:
            return
        self._load_and_show_structure(atl_id, row)

    def _load_and_show_structure(self, atl_id: str, row: dict) -> None:
        try:
            df = self._get_init_db()
            if df is None:
                return
            id_col = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
            match = df[df[id_col] == atl_id]
            if match.empty:
                return
            struct = match.iloc[0].get('structure')
            if struct is None:
                return

            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(struct)
            struct_type = row.get('struct_type')
            if not struct_type:
                record = match.iloc[0]
                for flag in ('bulk', 'surface', 'cluster', 'isolated_atom'):
                    if bool(record.get(flag)):
                        struct_type = flag
                        break
            info = {
                'phase': row.get('phase'),
                'struct_type': struct_type,
                'atl_id': atl_id,
                'calc_energy': row.get('calc_energy'),
            }
            self._viewer.set_atoms(atoms, info)
        except Exception:
            self._viewer.clear()

    def _get_init_db(self):
        if self._init_db_cache is not None:
            return self._init_db_cache
        path = self._project.init_db_path()
        if not path.exists():
            return None
        from atlas.core.initial_db import InitialDatabase

        loaded = InitialDatabase.load_database(path)
        df = getattr(loaded, 'df', loaded)
        if df is None or getattr(df, 'empty', True):
            return None
        self._init_db_cache = df
        return df


class StructuresTablePanel(QWidget):
    """Full-viewport sortable table of indexed structures."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        self._model = _StructuresTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setFilterKeyColumn(-1)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)

        self._summary = QLabel()
        self._summary.setStyleSheet('padding: 0 4px;')

        from PySide6.QtWidgets import QLineEdit

        self._filter = QLineEdit()
        self._filter.setPlaceholderText('Filter rows…')
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._proxy.setFilterFixedString)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._summary, 1)
        top.addWidget(self._filter)
        top.addWidget(refresh_btn)

        self._empty_label = QLabel(
            'No structures indexed yet.\n'
            'Generate a database in the Config tab, then come back here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('padding: 40px;')

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._empty_label)

        self.refresh()

    def refresh(self) -> None:
        with contextlib.suppress(Exception):
            self._project.refresh_structures_index()

        rows = self._project.list_structures()
        self._model.load(rows)

        if rows:
            self._table.setVisible(True)
            self._empty_label.setVisible(False)
            self._summary.setText(f'{len(rows):,} structures')
        else:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            self._summary.setText('')
