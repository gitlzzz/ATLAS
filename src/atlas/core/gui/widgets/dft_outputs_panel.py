"""DFT outputs panel — structures table with 3D viewer.

Combines the sortable/filterable structures table with a structure
viewer on the right.  Clicking a row loads the 3D view and metadata,
identical to the boundary panel's detail pane.
"""

from __future__ import annotations

import contextlib

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project
from atlas.core.gui.widgets.structures_panel import (
    _StructuresTableModel,
)


class DftOutputsPanel(QWidget):
    """Structures table (left 2/3) + structure viewer (right 1/3)."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._init_db_cache = None

        # --- table ---
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
        self._table.selectionModel().selectionChanged.connect(
            self._on_row_selected,
        )

        self._summary = QLabel()
        self._summary.setStyleSheet('color: #555; padding: 0 4px;')

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

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addLayout(top)
        table_layout.addWidget(self._table, 1)

        # --- structure viewer ---
        from atlas.core.gui.widgets.structure_viewer import StructureViewer

        self._viewer = StructureViewer()
        self._viewer.setMinimumWidth(280)

        self._detail_placeholder = QLabel(
            'Select a structure in the table\nto view it here.'
        )
        self._detail_placeholder.setAlignment(Qt.AlignCenter)
        self._detail_placeholder.setWordWrap(True)
        self._detail_placeholder.setStyleSheet(
            'color: #6c757d; padding: 30px; font-size: 13px;'
        )

        detail_container = QWidget()
        detail_layout = QVBoxLayout(detail_container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self._viewer)
        detail_layout.addWidget(self._detail_placeholder)
        self._viewer.setVisible(False)

        # --- splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(table_container)
        splitter.addWidget(detail_container)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # --- empty state ---
        self._empty_label = QLabel(
            'No structures indexed yet.\n'
            'Generate a database and run DFT labelling to see results here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('color: #6c757d; padding: 40px;')

        # --- main layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._empty_label)

        self._splitter = splitter
        self.refresh()

    # ---------------------------------------------------------- public API

    def refresh(self) -> None:
        self._init_db_cache = None
        self._viewer.clear()
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)

        with contextlib.suppress(Exception):
            self._project.refresh_structures_index()

        rows = self._project.list_structures()
        self._model.load(rows)

        if rows:
            self._splitter.setVisible(True)
            self._empty_label.setVisible(False)
            self._summary.setText(f'{len(rows):,} structures')
        else:
            self._splitter.setVisible(False)
            self._empty_label.setVisible(True)
            self._summary.setText('')

    # --------------------------------------------------------- selection

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
                self._show_fallback(atl_id)
                return
            id_col = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
            match = df[df[id_col] == atl_id]
            if match.empty:
                self._show_fallback(atl_id)
                return
            struct = match.iloc[0].get('structure')
            if struct is None:
                self._show_fallback(atl_id)
                return

            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(struct)
            info = {
                'phase': row.get('phase'),
                'struct_type': row.get('struct_type'),
                'atl_id': atl_id,
                'calc_energy': row.get('calc_energy'),
            }
            self._detail_placeholder.setVisible(False)
            self._viewer.setVisible(True)
            self._viewer.set_atoms(atoms, info)
        except Exception:
            self._show_fallback(atl_id)

    def _show_fallback(self, atl_id: str) -> None:
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)
        self._detail_placeholder.setText(
            f'Structure: {atl_id}\n\n'
            f'Could not load 3D structure.\n'
            f'The database may not be available.'
        )

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
