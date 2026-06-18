"""Database management panel — delete and export the initial database."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

_BOUNDARY_PATTERNS = (
    'concave_hull*.png',
    'comp_plot*.png',
    'descriptors_concave_hull*.png',
)


def _remove_boundary_files(directory: Path) -> None:
    """Remove latent-space boundary artifacts from a single directory."""
    npz = directory / 'latent_space.npz'
    if npz.is_file():
        npz.unlink()
    for pattern in _BOUNDARY_PATTERNS:
        for f in directory.glob(pattern):
            f.unlink()


class DbManagePanel(QWidget):
    """Delete or export the project's initial database."""

    database_deleted = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self._project = project

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(16)

        # ── Info ───────────────────────────────────────────────────
        info_group = QGroupBox('Database Info')
        info_layout = QVBoxLayout(info_group)
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addWidget(self._info_label)
        outer.addWidget(info_group)

        # ── Export ─────────────────────────────────────────────────
        export_group = QGroupBox('Export')
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(
            QLabel('Export the initial database as an extended XYZ file.')
        )
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton('Export as .extxyz')
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        export_layout.addLayout(btn_row)
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 0)
        self._export_progress.setVisible(False)
        export_layout.addWidget(self._export_progress)
        outer.addWidget(export_group)

        # ── Delete ─────────────────────────────────────────────────
        delete_group = QGroupBox('Danger Zone')
        delete_group.setStyleSheet(
            'QGroupBox { border: 1px solid #ef5350; border-radius: 4px; }'
            'QGroupBox::title { color: #ef5350; }'
        )
        delete_layout = QVBoxLayout(delete_group)
        delete_layout.addWidget(
            QLabel('Permanently delete the generated database. This cannot be undone.')
        )
        btn_row2 = QHBoxLayout()
        self._delete_btn = QPushButton('Delete Database')
        self._delete_btn.setStyleSheet(
            'QPushButton { color: white; background-color: #ef5350;'
            ' border: none; padding: 6px 16px; border-radius: 4px; }'
            'QPushButton:hover { background-color: #e53935; }'
        )
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row2.addWidget(self._delete_btn)
        btn_row2.addStretch()
        delete_layout.addLayout(btn_row2)
        outer.addWidget(delete_group)

        outer.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        db_path = self._project.init_db_path()
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            counts = self._project.structure_counts()
            self._info_label.setText(
                f'<b>Path:</b> {db_path}<br>'
                f'<b>Size:</b> {size_mb:.1f} MB<br>'
                f'<b>Structures:</b> {counts["total"]}'
            )
            self._export_btn.setEnabled(True)
            self._delete_btn.setEnabled(True)
        else:
            self._info_label.setText('No database has been generated yet.')
            self._export_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)

    # ── Export ──────────────────────────────────────────────────────

    def _on_export(self) -> None:
        db_path = self._project.init_db_path()
        if not db_path.exists():
            QMessageBox.warning(self, 'Export', 'No database found to export.')
            return

        default_name = db_path.stem.replace('.xz', '') + '.extxyz'
        dest, _ = QFileDialog.getSaveFileName(
            self,
            'Export database as extxyz',
            str(db_path.parent / default_name),
            'Extended XYZ (*.extxyz);;All Files (*)',
        )
        if not dest:
            return

        self._export_btn.setEnabled(False)
        self._export_progress.setVisible(True)
        self._worker = _ExportWorker(db_path, dest)
        self._worker.finished_signal.connect(self._on_export_done)
        self._worker.start()

    def _on_export_done(self, error: str) -> None:
        self._export_btn.setEnabled(True)
        self._export_progress.setVisible(False)
        if error:
            QMessageBox.critical(self, 'Export Failed', error)
        else:
            QMessageBox.information(
                self, 'Export Complete', 'Database exported successfully.'
            )

    # ── Delete ─────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        db_path = self._project.init_db_path()
        if not db_path.exists():
            return

        answer = QMessageBox.warning(
            self,
            'Delete Database',
            f'Are you sure you want to permanently delete:\n\n'
            f'{db_path}\n\n'
            'This action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        confirm = QMessageBox.warning(
            self,
            'Confirm Deletion',
            'This will permanently remove the database file. Proceed?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            db_path.unlink()
            self._project.clear_structures_index()
            db_dir = db_path.parent
            log_dir = db_dir / 'logs'
            if log_dir.is_dir():
                shutil.rmtree(log_dir)

            # Clean up boundary output files (latent_space.npz and hull plots)
            # Search both databases/ and project root to mirror BoundaryPanel.
            _remove_boundary_files(db_dir)
            _remove_boundary_files(self._project.dir)
        except OSError as exc:
            QMessageBox.critical(
                self, 'Delete Failed', f'Could not delete database: {exc}'
            )
            return

        self.refresh()
        self.database_deleted.emit()


class _ExportWorker(QThread):
    finished_signal = Signal(str)

    def __init__(self, db_path: Path, dest: str, parent=None):
        super().__init__(parent)
        self._db_path = db_path
        self._dest = dest

    def run(self):
        try:
            from atlas.core.initial_db import InitialDatabase

            db = InitialDatabase.load_database(self._db_path)
            dest_path = Path(self._dest)
            db.export_db(
                out_format='extxyz',
                file_name=dest_path.stem,
                file_path=dest_path.parent,
            )
            self.finished_signal.emit('')
        except Exception as exc:
            self.finished_signal.emit(str(exc))
