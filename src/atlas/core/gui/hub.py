"""Hub dialog — entry point for `atl_gui`.

The Hub lets the user create, open, or pick a recently-used project.  On
success it sets ``self.selected_project`` to an open ``Project`` instance
and accepts; otherwise the dialog is rejected and the application exits.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from atlas import __version__
from atlas.core.gui.app_params import ApplicationParameters, pretty_label
from atlas.core.gui.project import Project, ProjectError, RecentProjects

ASSETS_DIR = Path(__file__).resolve().parent / 'assets'


class _VersionCheckWorker(QObject):
    """Fetch latest ATLAS version in a background thread."""

    finished = Signal(str, str, str)  # current, latest, hash

    def run(self) -> None:
        try:
            from atlas.core.code_utils import get_atl_version_info

            curr, latest, hash_str = get_atl_version_info()
            self.finished.emit(str(curr), str(latest), str(hash_str))
        except Exception:
            self.finished.emit(__version__, 'unknown', 'unknown')


class HubDialog(QDialog):
    """Entry-point dialog: create / open / pick recent project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ATLAS Hub')
        self.setMinimumSize(720, 480)
        self.setModal(True)

        from atlas.core.gui.icons import app_icon

        self.setWindowIcon(app_icon())

        self.recents = RecentProjects()
        self.recents.prune_missing()
        self.selected_project: Project | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)

        outer.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(20)
        body.addLayout(self._build_actions(), 1)
        body.addWidget(self._build_recents(), 2)
        outer.addLayout(body, 1)

        outer.addWidget(self._build_version_bar())
        self._start_version_check()

    # ------------------------------------------------------------ logo

    def _update_logo(self) -> None:
        from atlas.core.gui.themes import saved_global_theme, theme_variant

        variant = theme_variant(saved_global_theme())
        suffix = 'dark' if variant == 'dark' else 'light'
        logo_path = ASSETS_DIR / f'atlas_logo_{suffix}.png'
        if not logo_path.exists():
            logo_path = ASSETS_DIR / 'atlas_logo_light.png'
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            self._logo_label.setPixmap(
                pixmap.scaledToWidth(280, Qt.SmoothTransformation),
            )
        else:
            self._logo_label.setText('ATLAS')

    # ----------------------------------------------------------- layout

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(6)
        header.setAlignment(Qt.AlignCenter)

        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignCenter)
        self._update_logo()
        header.addWidget(self._logo_label)

        title_font = QFont()
        title_font.setFamilies(ApplicationParameters.FONT_FAMILIES_REGULAR)
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label = QLabel('ATLAS Hub')
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        header.addWidget(title_label)

        subtitle = QLabel(
            'Create a new project, open an existing one, or pick a recent.'
        )
        subtitle.setAlignment(Qt.AlignCenter)
        header.addWidget(subtitle)
        return header

    def _build_actions(self) -> QVBoxLayout:
        actions = QVBoxLayout()
        actions.setSpacing(8)

        button_font = QFont()
        button_font.setFamilies(ApplicationParameters.FONT_FAMILIES_REGULAR)
        button_font.setPointSize(12)

        new_btn = QPushButton('New Project…')
        new_btn.setFont(button_font)
        new_btn.setMinimumHeight(48)
        new_btn.clicked.connect(self._on_new)
        actions.addWidget(new_btn)

        open_btn = QPushButton('Open Project…')
        open_btn.setFont(button_font)
        open_btn.setMinimumHeight(48)
        open_btn.clicked.connect(self._on_open)
        actions.addWidget(open_btn)

        actions.addSpacing(4)

        self.load_recent_btn = QPushButton('Select a recent project')
        self.load_recent_btn.setFont(button_font)
        self.load_recent_btn.setMinimumHeight(48)
        self.load_recent_btn.setEnabled(False)
        self.load_recent_btn.clicked.connect(self._on_load_recent)
        actions.addWidget(self.load_recent_btn)

        actions.addStretch()

        quit_btn = QPushButton('Quit')
        quit_btn.setFont(button_font)
        quit_btn.clicked.connect(self.reject)
        actions.addWidget(quit_btn)
        return actions

    def _build_recents(self) -> QGroupBox:
        box = QGroupBox('Recent Projects')
        layout = QVBoxLayout(box)
        self.recents_list = QListWidget()
        self.recents_list.itemActivated.connect(self._on_recent_activated)
        self.recents_list.itemDoubleClicked.connect(self._on_recent_activated)
        self.recents_list.currentItemChanged.connect(self._on_recents_selection_changed)
        layout.addWidget(self.recents_list)

        if not self.recents.entries:
            self.recents_list.addItem('No recent projects yet.')
            self.recents_list.setEnabled(False)
        else:
            for entry in self.recents.entries:
                overall, status_line = self._load_project_status(entry.path)
                color = self._status_color(overall)
                icon = self._make_circle_icon(color)
                display = (
                    f'{entry.name}\n{entry.path}\n{status_line} '
                    f' •  {entry.last_opened_when}'
                )
                item = QListWidgetItem(icon, display)
                item.setData(Qt.UserRole, entry.path)
                self.recents_list.addItem(item)
        return box

    # --------------------------------------------------------- version bar

    def _build_version_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet('QWidget { border-radius: 6px; }')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)

        self._version_label = QLabel(f'ATLAS v{__version__}')
        self._version_label.setStyleSheet('font-size: 12px;')

        self._update_label = QLabel('Checking for updates…')
        self._update_label.setStyleSheet('font-size: 12px;')
        self._update_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._version_label)
        layout.addStretch()
        layout.addWidget(self._update_label)
        return bar

    def _start_version_check(self) -> None:
        self._vc_thread = QThread(self)
        self._vc_worker = _VersionCheckWorker()
        self._vc_worker.moveToThread(self._vc_thread)
        self._vc_worker.finished.connect(self._on_version_result)
        self._vc_worker.finished.connect(self._vc_thread.quit)
        self._vc_thread.started.connect(self._vc_worker.run)
        self._vc_thread.start()

    def _on_version_result(self, current: str, latest: str, hash_str: str) -> None:
        short_hash = hash_str[:7] if hash_str != 'unknown' else ''
        if latest == 'unknown':
            self._update_label.setText('Could not check for updates')
            self._update_label.setStyleSheet('font-size: 12px;')
            return

        from packaging.version import Version

        curr_v = Version(current)
        latest_v = Version(latest)

        if curr_v < latest_v:
            self._update_label.setText(f'Update available: v{latest}  ({short_hash})')
            self._update_label.setStyleSheet(
                'color: #d97706; font-size: 12px; font-weight: bold;'
            )
        elif curr_v > latest_v:
            self._version_label.setText(f'ATLAS v{current} (dev, {short_hash})')
            self._update_label.setText('Unreleased version')
            self._update_label.setStyleSheet('font-size: 12px;')
        else:
            self._version_label.setText(f'ATLAS v{current} ({short_hash})')
            self._update_label.setText('Up to date ✓')
            self._update_label.setStyleSheet('color: #16a34a; font-size: 12px;')

    # ----------------------------------------------------------- actions

    def _on_new(self) -> None:
        dlg = NewProjectDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        parent_dir = dlg.parent_dir
        name = dlg.project_name
        try:
            project = Project.create(parent_dir, name)
        except ProjectError as exc:
            QMessageBox.critical(self, 'Create failed', str(exc))
            return
        self._accept_with(project)

    def _on_open(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Open ATLAS Project', '', 'ATLAS Project (*.atlasproj)'
        )
        if not filepath:
            return
        try:
            project = Project.open(filepath)
        except ProjectError as exc:
            QMessageBox.critical(self, 'Open failed', str(exc))
            return
        self._accept_with(project)

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if not path:
            return
        try:
            project = Project.open(path)
        except ProjectError as exc:
            self.recents.remove(Path(path))
            QMessageBox.critical(self, 'Open failed', str(exc))
            return
        self._accept_with(project)

    def _on_recents_selection_changed(
        self, current: QListWidgetItem, _previous: QListWidgetItem
    ) -> None:
        if current is not None:
            self.load_recent_btn.setText('Load Project')
            self.load_recent_btn.setEnabled(True)
        else:
            self.load_recent_btn.setText('Select a recent project')
            self.load_recent_btn.setEnabled(False)

    def _on_load_recent(self) -> None:
        item = self.recents_list.currentItem()
        if item is not None:
            self._on_recent_activated(item)

    # ============================================================== status

    @staticmethod
    def _status_color(overall: str) -> QColor:
        return {
            'running': QColor('#3b82f6'),
            'completed': QColor('#22c55e'),
            'stopped': QColor('#f59e0b'),
            'error': QColor('#ef4444'),
        }.get(overall, QColor('#9ca3af'))

    @staticmethod
    def _make_circle_icon(color: QColor, size: int = 16) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(1, 1, size - 2, size - 2)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _load_project_status(path: str) -> tuple[str, str]:
        """Return (overall_status, human-readable status line)."""
        try:
            project = Project.open(path)
            state = project.workflow_state()
            stages = state['stages']
            next_rec = state.get('next_recommended')
        except ProjectError:
            return ('error', 'Could not load project')

        stage_statuses = {s['status'] for s in stages.values()}

        if 'running' in stage_statuses:
            overall = 'running'
        elif stage_statuses == {'done'}:
            overall = 'completed'
        else:
            overall = 'stopped'

        stage_label = 'All done' if next_rec is None else pretty_label(next_rec)

        status_line = f'Stage: {stage_label}  •  Status: {overall.title()}'
        project.close()
        return (overall, status_line)

    def _accept_with(self, project: Project) -> None:
        self.recents.touch(project.path, project.name)
        self.selected_project = project
        self.accept()


class NewProjectDialog(QDialog):
    """Small modal to pick a parent directory and project name."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New ATLAS Project')
        self.setMinimumWidth(480)

        self.parent_dir: str = os.getcwd()
        self.project_name: str = ''

        form = QFormLayout(self)

        parent_row = QHBoxLayout()
        self.parent_edit = QLineEdit(self.parent_dir)
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._browse_parent)
        parent_row.addWidget(self.parent_edit, 1)
        parent_row.addWidget(browse_btn)
        form.addRow('Parent directory:', parent_row)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('my_atlas_project')
        form.addRow('Project name:', self.name_edit)

        hint = QLabel(
            'A project bundle will be created as '
            '<i>&lt;parent&gt;/&lt;name&gt;.atlasproj</i> with a sibling '
            '<i>&lt;name&gt;.atlas/</i> directory holding configs, databases, '
            'and outputs.'
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        create = QPushButton('Create')
        create.setDefault(True)
        create.clicked.connect(self._on_create)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        form.addRow(buttons)

    def _browse_parent(self) -> None:
        start = self.parent_edit.text() or os.path.expanduser('~')
        chosen = QFileDialog.getExistingDirectory(self, 'Pick parent directory', start)
        if chosen:
            self.parent_edit.setText(chosen)

    def _on_create(self) -> None:
        parent = self.parent_edit.text().strip()
        name = self.name_edit.text().strip()
        if not parent or not Path(parent).is_dir():
            QMessageBox.warning(self, 'Invalid parent', 'Parent directory must exist.')
            return
        if not name:
            QMessageBox.warning(self, 'Invalid name', 'Project name is required.')
            return
        self.parent_dir = parent
        self.project_name = name
        self.accept()
