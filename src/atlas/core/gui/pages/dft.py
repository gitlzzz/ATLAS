"""DFT batch run page (``atl_run_dft_database``)."""

from __future__ import annotations

import os
import uuid as _uuid
from datetime import UTC, datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.process.runner import (
    DetachedProcessMonitor,
    ProcessRunner,
    find_detached_process,
)
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.preflight_panel import Check, PreflightPanel
from atlas.core.gui.widgets.prereq_banner import PrereqBanner

CANONICAL_CONFIG = 'dft_settings.toml'

DFT_TABS: list[tuple[str, list[str]]] = [
    ('General', ['general']),
    ('Calculation & K-points', ['calculation', 'kpoints']),
    ('Queue & HPC', ['queue']),
    ('AiiDA-VASP', ['aiida_vasp']),
    ('INCAR', ['incar']),
]


class DftPage(WorkflowPage):
    """Configure and launch a DFT labelling run on an ATLAS database."""

    DISPLAY_NAME = 'DFT Labelling'
    NAVIGATION_KEY = 'dft'

    def __init__(
        self,
        project: Project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        parent=None,
    ):
        super().__init__(
            project,
            schema_data,
            application_font,
            log,
            navigate,
            notification,
            parent,
        )

        self.config_panel = self._build_config_panel()
        self.run_button = QPushButton(
            self._themed_icon('play_arrow'), ' Submit DFT Run'
        )
        self.run_button.clicked.connect(self.run)
        self.cancel_button = self._make_cancel_button()
        self.config_panel.add_action_button(self.run_button)
        self.config_panel.add_action_button(self.cancel_button)

        self._preflight = PreflightPanel(
            'Pre-flight checks',
            self._build_preflight_checks(),
            focus_field=self.config_panel.focus_field,
        )
        self.config_panel.data_changed.connect(self._preflight.run_checks)

        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self._build_db_picker())
        config_layout.addWidget(self._preflight)
        config_layout.addWidget(self.config_panel, 1)

        self._queue_panel = _DftQueuePanel(project)

        from atlas.core.gui.widgets.dft_outputs_panel import DftOutputsPanel

        self._outputs_panel = DftOutputsPanel(project)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('pageTab')
        self.tabs.addTab(config_tab, 'Config')
        self.tabs.addTab(self._outputs_panel, 'Outputs')
        self.tabs.addTab(self._queue_panel, 'Queue')
        self._mark_output_tabs(1)

        self._setup_banner = PrereqBanner(
            'ATLAS initial setup is incomplete.',
            'Go to Settings',
            lambda: self._navigate('settings'),
        )
        self._setup_banner.setVisible(False)

        self.prereq_banner = PrereqBanner(
            message=(
                'DFT labelling needs an initial database first. '
                'Generate one in the previous step before submitting.'
            ),
            action_label='Go to Initial DB',
            on_action=lambda: self._navigate('init_db'),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._setup_banner)
        layout.addWidget(self.prereq_banner)
        layout.addWidget(self.tabs)

        self._update_prereq_banner()
        self._preflight.run_checks()

    def on_shown(self) -> None:
        self._update_prereq_banner()
        self._refresh_setup_banner()
        init_db = self.project.init_db_path()
        if init_db.exists() and not self.db_path_edit.text().strip():
            self.db_path_edit.setText(str(init_db))
        self._try_reconnect_detached()
        self._refresh_suggestions()
        self._preflight.run_checks()
        self._queue_panel.refresh()
        self._outputs_panel.refresh()

    def _try_reconnect_detached(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        logs_dir = self.project.dir / 'logs'
        info = find_detached_process(logs_dir, 'atl_run_dft_database')
        if info is None:
            return
        self._log(f'🔄 Found running DFT process (PID {info["pid"]}), reconnecting…')
        self._set_running(self.run_button, self.cancel_button)
        self.worker = DetachedProcessMonitor(
            pid=info['pid'],
            log_file=info['log_file'],
            pid_file=info['pid_file'],
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(
            lambda rc: self._on_finished(rc, None),
        )
        self.worker.start()

    def _update_prereq_banner(self) -> None:
        state = self.project.workflow_state()
        init_done = state['stages']['init_db']['status'] in ('done', 'partial')
        self.prereq_banner.setVisible(not init_done)

    def _refresh_setup_banner(self) -> None:
        from atlas.core.gui.widgets.setup_wizard import check_setup_problems

        problems = check_setup_problems()
        if problems:
            self._setup_banner.set_message(
                'ATLAS initial setup is incomplete: '
                + '; '.join(problems)
                + '. Open Settings to run the setup wizard.'
            )
            self._setup_banner.setVisible(True)
        else:
            self._setup_banner.setVisible(False)

    def _build_config_panel(self):
        from atlas.core.gui.widgets.config_panel import ConfigPanel

        return ConfigPanel(
            schema_data=self._schema_data,
            section_key='dft',
            sub_section_tabs=DFT_TABS,
            project=self.project,
            application_font=self._application_font,
        )

    def _refresh_suggestions(self) -> None:
        from atlas.core.gui.widgets.aiida_suggestions import (
            fetch_aiida_suggestions_async,
        )

        self.config_panel.set_suggestions_loading()
        fetch_aiida_suggestions_async(self.config_panel.populate_suggestions)

    def _build_preflight_checks(self) -> list[Check]:
        def _check_aiida() -> str | None:
            if not _aiida_available():
                return 'No AiiDA profile loaded, set AIIDA_PROFILE or load one first.'
            return None

        def _check_code_string() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (parsed or {}).get('queue', {}).get('code_string', '')
            if not val:
                return (
                    'Missing code_string in Queue & HPC tab (e.g. "vasp@my_cluster").'
                )
            try:
                from aiida import orm

                orm.load_code(val)
            except Exception:
                return f'Code "{val}" not found in AiiDA, check the label.'
            return None

        def _check_potential_family() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (
                (parsed or {})
                .get('calculation', {})
                .get(
                    'aiida_potential_family',
                    '',
                )
            )
            if not val:
                return 'Missing aiida_potential_family in Calculation & K-points tab.'
            try:
                from aiida import orm

                orm.Group.collection.get(label=val)
            except Exception:
                return f'Potential family "{val}" not found in AiiDA.'
            return None

        def _check_db_file() -> str | None:
            path = self.db_path_edit.text().strip()
            if not path:
                return 'Select a database file in the picker above the config panel.'
            if not os.path.exists(path):
                return f'File not found: {path}'
            return None

        return [
            Check('AiiDA profile', 'AiiDA profile loaded', _check_aiida),
            Check(
                'VASP code',
                'Code label exists in AiiDA',
                _check_code_string,
                requires_aiida=True,
                field_key='code_string',
            ),
            Check(
                'Potential family',
                'Potential family exists in AiiDA',
                _check_potential_family,
                requires_aiida=True,
                field_key='aiida_potential_family',
            ),
            Check('Database file', 'Database file selected and exists', _check_db_file),
        ]

    def _build_db_picker(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Database file:'))
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText('Path to an ATLAS .xyz or .xz database')
        init_db = self.project.init_db_path()
        if init_db.exists():
            self.db_path_edit.setText(str(init_db))
        layout.addWidget(self.db_path_edit, 1)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self._browse_db)
        layout.addWidget(browse)
        return bar

    def _browse_db(self) -> None:
        start = str(self.project.dir / 'databases')
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            'Select ATLAS database',
            start,
            'ATLAS database (*.xyz *.xz);;All files (*)',
        )
        if filepath:
            self.db_path_edit.setText(filepath)

    # ============================================================ run

    def run(self) -> None:
        db_path = self.db_path_edit.text().strip()
        if not db_path or not os.path.exists(db_path):
            self._log('❌ Provide an existing database file before running DFT.')
            return

        if not self.config_panel.save_to_project(label='run'):
            self._log('❌ Cannot run: configuration could not be saved.')
            return

        parsed, err = self.config_panel.parsed_config()
        if err or not parsed:
            self._log(f'❌ Cannot run: {err or "empty configuration"}')
            return

        problems = _preflight_check(parsed)
        if problems:
            for p in problems:
                self._log(f'❌ {p}')
            return

        self._preflight.run_checks()
        preflight_warn = self._preflight.failing_summary()
        if preflight_warn:
            answer = QMessageBox.warning(
                self,
                'Pre-flight checks failed',
                'Some pre-flight checks are not passing:\n\n'
                f'{preflight_warn}\n\n'
                'Do you want to proceed anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        if (
            QMessageBox.question(
                self,
                'Confirm DFT Run',
                'Submit DFT labelling with the current configuration?',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        # Record in project DB
        calc_uuid = str(_uuid.uuid4())
        try:
            self.project.record_dft_submission(
                calc_uuid=calc_uuid,
                aiida_pk=None,
                atl_id=None,
                queue=parsed.get('queue', {}).get('code_string'),
                status='submitted',
            )
            self._log(f'📒 Recorded DFT run {calc_uuid[:8]}… in the project.')
        except Exception as exc:
            self._log(f'⚠ Could not record DFT run: {exc}')

        command = [
            'atl_run_dft_database',
            '--db_file',
            db_path,
            '-c',
            CANONICAL_CONFIG,
        ]
        self._set_running(self.run_button, self.cancel_button)
        self._pending_calc_uuid = calc_uuid
        self.worker = ProcessRunner(
            command,
            cwd=self.project.cwd(),
            log_file=self.project.log_path('atl_run_dft_database'),
            detached=True,
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(
            lambda rc: self._on_finished(rc, calc_uuid),
        )
        self.worker.start()

    def _on_finished(self, return_code: int, calc_uuid: str | None) -> None:
        self._log(
            f'\n✅ atl_run_dft_database finished with exit code: {return_code}\n',
        )
        self._set_idle(self.run_button, self.cancel_button, 'Submit DFT Run')
        self._notification('DFT Labelling', return_code == 0)

        if calc_uuid is None:
            calc_uuid = getattr(self, '_pending_calc_uuid', None)

        if calc_uuid is not None:
            new_status = 'finished' if return_code == 0 else 'failed'
            try:
                now = datetime.now(tz=UTC).isoformat()
                with self.project.conn:
                    self.project.conn.execute(
                        'UPDATE dft_runs SET status = ?, finished_at = ? '
                        'WHERE calc_uuid = ?',
                        (new_status, now, calc_uuid),
                    )
            except Exception as exc:
                self._log(f'⚠ Could not update DFT run status: {exc}')

        if return_code == 0:
            try:
                count = self.project.refresh_structures_index()
                if count:
                    self._log(f'📦 Re-indexed {count} structures.')
            except Exception as exc:
                self._log(f'⚠ Failed to re-index structures: {exc}')

        self._queue_panel.refresh()
        self._outputs_panel.refresh()
        self.workflow_state_changed.emit()


# ================================================================ helpers


def _preflight_check(parsed: dict) -> list[str]:
    """Return a list of problems; empty if ready to submit."""
    problems: list[str] = []

    calc = parsed.get('calculation', {})
    if not calc.get('aiida_potential_family'):
        problems.append('Missing calculation.aiida_potential_family.')

    queue = parsed.get('queue', {})
    if not queue.get('code_string'):
        problems.append(
            'Missing queue.code_string, set the AiiDA code label '
            '(e.g. "vasp@my_cluster").'
        )

    if not _aiida_available():
        problems.append(
            'AiiDA is not loaded or not available in this environment. '
            'DFT submission requires a running AiiDA daemon.'
        )

    return problems


def _aiida_available() -> bool:
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        return profile is not None
    except Exception:
        return False


# ============================================================= queue panel


class _DftQueuePanel(QWidget):
    """Shows DFT run records from the project database."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        self._summary = QLabel()
        self._summary.setStyleSheet('padding: 8px; color: #495057;')

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            [
                'UUID',
                'Status',
                'Submitted',
                'Finished',
                'Notes',
            ]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)

        self._sync_btn = QPushButton('Sync from AiiDA')
        self._sync_btn.setFixedWidth(130)
        self._sync_btn.setToolTip('Query AiiDA for current DFT calculation statuses')
        self._sync_btn.clicked.connect(self._on_sync_aiida)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._summary, stretch=1)
        top.addWidget(self._sync_btn)
        top.addWidget(refresh_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._table, stretch=1)

        self.refresh()

    def refresh(self) -> None:
        counts = self._project.dft_run_counts()
        parts = []
        for status in ('submitted', 'running', 'finished', 'failed'):
            n = counts.get(status, 0)
            if n:
                parts.append(f'{n} {status}')
        total = sum(counts.values())
        if total == 0:
            self._summary.setText('No DFT runs recorded yet.')
        else:
            self._summary.setText(f'DFT runs: {", ".join(parts)} ({total} total)')

        rows = self._project.list_dft_runs()
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._table.setItem(i, 0, _item(row.get('calc_uuid', '')[:12]))
            status = row.get('status', '')
            status_item = _item(status)
            if status == 'finished':
                status_item.setForeground(Qt.darkGreen)
            elif status == 'failed':
                status_item.setForeground(Qt.red)
            self._table.setItem(i, 1, status_item)
            self._table.setItem(i, 2, _item(row.get('submitted_at', '')))
            self._table.setItem(i, 3, _item(row.get('finished_at', '') or '—'))
            self._table.setItem(i, 4, _item(row.get('notes', '') or ''))

    def _on_sync_aiida(self) -> None:
        if hasattr(self, '_sync_worker') and self._sync_worker.isRunning():
            return
        self._sync_btn.setEnabled(False)
        self._sync_btn.setText('Syncing…')
        self._sync_worker = _DftSyncWorker(self._project)
        self._sync_worker.finished_signal.connect(self._on_sync_done)
        self._sync_worker.start()

    def _on_sync_done(self, summary: str, errors: str) -> None:
        self._sync_btn.setEnabled(True)
        self._sync_btn.setText('Sync from AiiDA')
        if errors:
            QMessageBox.warning(self, 'AiiDA sync', errors)
        self.refresh()


class _DftSyncWorker(QThread):
    finished_signal = Signal(str, str)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            from atlas.core.gui.project.aiida_sync import sync_dft_runs

            result = sync_dft_runs(self._project)
            errors = '\n'.join(result.errors) if result.errors else ''
            self.finished_signal.emit(result.summary, errors)
        except Exception as exc:
            self.finished_signal.emit('', str(exc))


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
