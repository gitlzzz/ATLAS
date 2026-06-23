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
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtCore import QThread as _QThread
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
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


# ---------------------------------------------------------------------------
# Reusable filter widgets
# ---------------------------------------------------------------------------


class _CheckableComboBox(QComboBox):
    """Combobox with checkable items for multi-select filtering."""

    selection_changed = Signal(object)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setFocusPolicy(Qt.NoFocus)
        self.view().viewport().installEventFilter(self)
        self.model().dataChanged.connect(self._on_data_changed)
        self._update_display()

    def mousePressEvent(self, event):
        self.showPopup()
        event.accept()

    def eventFilter(self, obj, event):
        if (
            obj is self.view().viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
        ):
            index = self.view().indexAt(event.pos())
            if index.isValid():
                item = self.model().itemFromIndex(index)
                new = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new)
                return True
        return super().eventFilter(obj, event)

    def set_items(self, items: list[str], preserve_selection: bool = True) -> None:
        old_checked = self.checked_items() if preserve_selection else set()
        self.model().blockSignals(True)
        self.model().clear()
        for text in items:
            item = QStandardItem(text)
            item.setCheckable(True)
            checked = not old_checked or text in old_checked
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.model().appendRow(item)
        self.model().blockSignals(False)
        self._update_display()

    def checked_items(self) -> set[str]:
        result = set()
        mdl = self.model()
        for i in range(mdl.rowCount()):
            item = mdl.item(i)
            if item and item.checkState() == Qt.Checked:
                result.add(item.text())
        return result

    def check_all(self) -> None:
        mdl = self.model()
        mdl.blockSignals(True)
        for i in range(mdl.rowCount()):
            mdl.item(i).setCheckState(Qt.Checked)
        mdl.blockSignals(False)
        self._update_display()
        self.selection_changed.emit(None)

    def _on_data_changed(self):
        self._update_display()
        checked = self.checked_items()
        total = self.model().rowCount()
        self.selection_changed.emit(None if len(checked) == total else checked)

    def _update_display(self):
        checked = self.checked_items()
        total = self.model().rowCount()
        if total == 0 or len(checked) == total:
            text = f'{self._label}: All'
        elif len(checked) == 0:
            text = f'{self._label}: None'
        elif len(checked) <= 2:
            text = f'{self._label}: {", ".join(sorted(checked))}'
        else:
            text = f'{self._label}: {len(checked)} selected'
        self.lineEdit().setText(text)


class _StructuresFilterProxy(QSortFilterProxyModel):
    """Multi-attribute filter proxy for the structures table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text_filter: str = ''
        self._phase_filter: set[str] | None = None
        self._type_filter: set[str] | None = None
        self._mod_filter: set[str] | None = None
        self._n_atoms_min: int | None = None
        self._n_atoms_max: int | None = None

    def set_text_filter(self, text: str) -> None:
        self._text_filter = text.lower()
        self.invalidateFilter()

    def set_phase_filter(self, values: set[str] | None) -> None:
        self._phase_filter = values
        self.invalidateFilter()

    def set_type_filter(self, values: set[str] | None) -> None:
        self._type_filter = values
        self.invalidateFilter()

    def set_mod_filter(self, values: set[str] | None) -> None:
        self._mod_filter = values
        self.invalidateFilter()

    def set_n_atoms_range(self, min_val: int | None, max_val: int | None) -> None:
        self._n_atoms_min = min_val
        self._n_atoms_max = max_val
        self.invalidateFilter()

    def has_active_filters(self) -> bool:
        return bool(
            self._text_filter
            or self._phase_filter is not None
            or self._type_filter is not None
            or self._mod_filter is not None
            or self._n_atoms_min is not None
            or self._n_atoms_max is not None
        )

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if source_row >= len(model._rows):
            return False
        row = model._rows[source_row]

        if self._text_filter and not any(
            self._text_filter in str(row.get(k) or '').lower()
            for k, _ in VISIBLE_COLUMNS
        ):
            return False

        if (
            self._phase_filter is not None
            and (row.get('phase') or 'unknown') not in self._phase_filter
        ):
            return False

        if (
            self._type_filter is not None
            and (row.get('struct_type') or 'unknown') not in self._type_filter
        ):
            return False

        if self._mod_filter is not None:
            mods_raw = row.get('modifications')
            row_mods = set()
            if mods_raw and mods_raw != '[]':
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    row_mods = set(json.loads(mods_raw))

            want_none = 'none' in self._mod_filter
            want_specific = self._mod_filter - {'none'}
            if want_none and not want_specific:
                if row_mods:
                    return False
            elif want_specific and not want_none:
                if not row_mods & want_specific:
                    return False
            elif (
                want_specific
                and want_none
                and row_mods
                and not (row_mods & want_specific)
            ):
                return False

        n_atoms = row.get('n_atoms')
        if n_atoms is not None:
            if self._n_atoms_min is not None and n_atoms < self._n_atoms_min:
                return False
            if self._n_atoms_max is not None and n_atoms > self._n_atoms_max:
                return False

        return True


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
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole):
            return None
        row = self._rows[index.row()]
        key = self._columns[index.column()][0]
        value = row.get(key)

        if role == Qt.UserRole:
            if key == 'n_atoms':
                return int(value) if value is not None else -1
            if key == 'calc_energy':
                return float(value) if value is not None else float('inf')
            if key == 'calc_performed':
                return 1 if value else 0
            return str(value or '')

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
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
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
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
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
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
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
        self._proxy.setSortRole(Qt.UserRole)

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
    """Full-viewport sortable table with filtering, selection, and import/delete."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._atoms_range_initialized = False
        self._worker = None

        # -- Model / proxy --------------------------------------------------
        self._model = _StructuresTableModel(self)
        self._proxy = _StructuresFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.UserRole)

        # -- Table -----------------------------------------------------------
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.ExtendedSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed,
        )

        # -- Top row: summary + text filter + refresh -----------------------
        self._summary = QLabel()
        self._summary.setStyleSheet('padding: 0 4px;')

        self._filter = QLineEdit()
        self._filter.setPlaceholderText('Filter rows…')
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._proxy.set_text_filter)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._summary, 1)
        top.addWidget(self._filter)
        top.addWidget(refresh_btn)

        # -- Filter bar ------------------------------------------------------
        self._phase_combo = _CheckableComboBox('Phase')
        self._type_combo = _CheckableComboBox('Type')
        self._mods_combo = _CheckableComboBox('Modifications')

        self._atoms_min = QSpinBox()
        self._atoms_min.setPrefix('min ')
        self._atoms_min.setRange(0, 99999)

        self._atoms_max = QSpinBox()
        self._atoms_max.setPrefix('max ')
        self._atoms_max.setRange(0, 99999)

        clear_btn = QPushButton('Clear Filters')
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(self._clear_filters)

        self._phase_combo.selection_changed.connect(self._proxy.set_phase_filter)
        self._type_combo.selection_changed.connect(self._proxy.set_type_filter)
        self._mods_combo.selection_changed.connect(self._proxy.set_mod_filter)
        self._atoms_min.valueChanged.connect(self._on_atoms_range_changed)
        self._atoms_max.valueChanged.connect(self._on_atoms_range_changed)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self._phase_combo)
        filter_row.addWidget(self._type_combo)
        filter_row.addWidget(self._mods_combo)
        filter_row.addWidget(QLabel('Atoms:'))
        filter_row.addWidget(self._atoms_min)
        filter_row.addWidget(self._atoms_max)
        filter_row.addWidget(clear_btn)

        # -- Bottom action bar -----------------------------------------------
        self._selection_label = QLabel()
        self._selection_label.setStyleSheet('padding: 0 4px;')

        self._import_btn = QPushButton('Import Structures…')
        self._import_btn.clicked.connect(self._on_import)

        self._delete_btn = QPushButton('Delete Selected')
        self._delete_btn.setEnabled(False)
        self._delete_btn.setStyleSheet(
            'QPushButton { color: #ef5350; }QPushButton:disabled { color: #888; }'
        )
        self._delete_btn.clicked.connect(self._on_delete)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        bottom = QHBoxLayout()
        bottom.addWidget(self._selection_label, 1)
        bottom.addWidget(self._import_btn)
        bottom.addWidget(self._delete_btn)

        # -- Empty label -----------------------------------------------------
        self._empty_label = QLabel(
            'No structures indexed yet.\n'
            'Generate a database in the Config tab, then come back here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('padding: 40px;')

        # -- Layout ----------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(filter_row)
        layout.addWidget(self._table, 1)
        layout.addLayout(bottom)
        layout.addWidget(self._progress)
        layout.addWidget(self._empty_label)

        self._proxy.layoutChanged.connect(self._update_summary)
        self.refresh()

    # ---------------------------------------------------------------- refresh

    def refresh(self) -> None:
        with contextlib.suppress(Exception):
            self._project.refresh_structures_index()

        rows = self._project.list_structures()
        self._model.load(rows)

        if not rows:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            self._summary.setText('')
            return

        self._table.setVisible(True)
        self._empty_label.setVisible(False)

        # Populate filter widgets (block signals to avoid repeated invalidations)
        self._phase_combo.blockSignals(True)
        self._type_combo.blockSignals(True)
        self._mods_combo.blockSignals(True)
        self._atoms_min.blockSignals(True)
        self._atoms_max.blockSignals(True)

        phases = sorted({r.get('phase') or 'unknown' for r in rows})
        types = sorted({r.get('struct_type') or 'unknown' for r in rows})

        all_mods: set[str] = set()
        for r in rows:
            raw = r.get('modifications')
            if raw and raw != '[]':
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    all_mods.update(json.loads(raw))
        mod_items = ['none'] + sorted(all_mods)

        self._phase_combo.set_items(phases)
        self._type_combo.set_items(types)
        self._mods_combo.set_items(mod_items)

        n_atoms_vals = [r['n_atoms'] for r in rows if r.get('n_atoms') is not None]
        if n_atoms_vals:
            data_min, data_max = min(n_atoms_vals), max(n_atoms_vals)
            self._atoms_min.setRange(0, data_max)
            self._atoms_max.setRange(0, data_max)
            if not self._atoms_range_initialized:
                self._atoms_min.setValue(data_min)
                self._atoms_max.setValue(data_max)
                self._atoms_range_initialized = True

        self._phase_combo.blockSignals(False)
        self._type_combo.blockSignals(False)
        self._mods_combo.blockSignals(False)
        self._atoms_min.blockSignals(False)
        self._atoms_max.blockSignals(False)

        self._proxy.invalidateFilter()
        self._update_summary()

    # ---------------------------------------------------------- filter helpers

    def _on_atoms_range_changed(self) -> None:
        self._proxy.set_n_atoms_range(
            self._atoms_min.value(),
            self._atoms_max.value(),
        )

    def _clear_filters(self) -> None:
        self._filter.clear()
        self._phase_combo.check_all()
        self._type_combo.check_all()
        self._mods_combo.check_all()
        n_atoms_vals = [
            r['n_atoms'] for r in self._model._rows if r.get('n_atoms') is not None
        ]
        if n_atoms_vals:
            self._atoms_min.setValue(min(n_atoms_vals))
            self._atoms_max.setValue(max(n_atoms_vals))
        self._proxy.set_n_atoms_range(
            self._atoms_min.value(),
            self._atoms_max.value(),
        )

    def _update_summary(self) -> None:
        visible = self._proxy.rowCount()
        total = self._model.rowCount()
        if total == 0:
            self._summary.setText('')
        elif visible < total:
            self._summary.setText(f'{visible:,} of {total:,} structures (filtered)')
        else:
            self._summary.setText(f'{total:,} structures')

    # ------------------------------------------------------- selection / delete

    def _on_selection_changed(self) -> None:
        count = len(self._table.selectionModel().selectedRows())
        self._selection_label.setText(f'{count} selected' if count else '')
        self._delete_btn.setEnabled(count > 0)

    def _selected_atl_ids(self) -> list[str]:
        ids: list[str] = []
        for proxy_idx in self._table.selectionModel().selectedRows():
            source_idx = self._proxy.mapToSource(proxy_idx)
            row = self._model._rows[source_idx.row()]
            aid = row.get('atl_id')
            if aid:
                ids.append(aid)
        return ids

    def _on_delete(self) -> None:
        atl_ids = self._selected_atl_ids()
        if not atl_ids:
            return
        answer = QMessageBox.warning(
            self,
            'Delete Structures',
            f'Permanently delete {len(atl_ids)} structure(s)?\n\n'
            'This modifies the on-disk database and cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._set_busy(True)
        db_path = self._project.init_db_path()
        self._worker = _DeleteWorker(db_path, atl_ids)
        self._worker.finished_signal.connect(
            lambda err: self._on_delete_done(err, atl_ids),
        )
        self._worker.start()

    def _on_delete_done(self, error: str, atl_ids: list[str]) -> None:
        self._set_busy(False)
        if error:
            QMessageBox.critical(self, 'Delete Failed', error)
            return
        self._project.remove_structures_from_index(atl_ids)
        self._atoms_range_initialized = False
        self.refresh()

    # ----------------------------------------------------------------- import

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import structures',
            '',
            'ATLAS Database (*.xz);;Extended XYZ (*.xyz *.extxyz);;All Files (*)',
        )
        if not path:
            return
        current_db = self._project.init_db_path()
        if not current_db.exists():
            QMessageBox.warning(
                self,
                'Import',
                'No database exists yet. Generate one first.',
            )
            return
        self._set_busy(True)
        self._worker = _ImportWorker(current_db, Path(path))
        self._worker.finished_signal.connect(self._on_import_done)
        self._worker.start()

    def _on_import_done(self, count: int, error: str) -> None:
        self._set_busy(False)
        if error:
            QMessageBox.critical(self, 'Import Failed', error)
            return
        self._project._structures_index_mtime = None
        self._atoms_range_initialized = False
        self.refresh()
        QMessageBox.information(
            self,
            'Import Complete',
            f'{count} new structure(s) imported.',
        )

    # ---------------------------------------------------------------- helpers

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self._import_btn.setEnabled(not busy)
        self._delete_btn.setEnabled(not busy)


# ---------------------------------------------------------------------------
# Background workers for heavy I/O
# ---------------------------------------------------------------------------


class _DeleteWorker(_QThread):
    finished_signal = Signal(str)

    def __init__(self, db_path: Path, atl_ids: list[str], parent=None):
        super().__init__(parent)
        self._db_path = db_path
        self._atl_ids = atl_ids

    def run(self):
        try:
            from atlas.core.initial_db import InitialDatabase

            db = InitialDatabase.load_database(self._db_path)
            id_col = 'atl_id' if 'atl_id' in db.df.columns else 'unique_id'
            ids_set = set(self._atl_ids)
            db.df = db.df[~db.df[id_col].astype(str).isin(ids_set)]
            db.database_name = self._db_path.stem
            db.save_database(path=str(self._db_path.parent))
            self.finished_signal.emit('')
        except Exception as exc:
            self.finished_signal.emit(str(exc))


class _ImportWorker(_QThread):
    finished_signal = Signal(int, str)

    def __init__(self, current_db_path: Path, source_path: Path, parent=None):
        super().__init__(parent)
        self._current_path = current_db_path
        self._source_path = source_path

    def run(self):
        try:
            import pandas as pd

            from atlas.core.initial_db import InitialDatabase

            source = InitialDatabase.load_database(self._source_path)
            source_df = getattr(source, 'df', source)
            if source_df is None or getattr(source_df, 'empty', True):
                self.finished_signal.emit(0, '')
                return

            current = InitialDatabase.load_database(self._current_path)
            id_col_src = 'atl_id' if 'atl_id' in source_df.columns else 'unique_id'
            id_col_cur = 'atl_id' if 'atl_id' in current.df.columns else 'unique_id'

            existing_ids = set(current.df[id_col_cur].astype(str))
            new_rows = source_df[~source_df[id_col_src].astype(str).isin(existing_ids)]
            if new_rows.empty:
                self.finished_signal.emit(0, '')
                return

            current.df = pd.concat([current.df, new_rows], ignore_index=True)
            current.database_name = self._current_path.stem
            current.save_database(path=str(self._current_path.parent))
            self.finished_signal.emit(len(new_rows), '')
        except Exception as exc:
            self.finished_signal.emit(0, str(exc))
