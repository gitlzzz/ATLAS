"""Initial database generation page (``atl_gen_init_db``)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.process.runner import (
    DetachedProcessMonitor,
    ProcessRunner,
    find_detached_process,
)
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.boundary_panel import BoundaryPanel
from atlas.core.gui.widgets.db_manage_panel import DbManagePanel
from atlas.core.gui.widgets.prereq_banner import SuccessBanner
from atlas.core.gui.widgets.structures_panel import (
    StructuresPanel,
    StructuresTablePanel,
)

CANONICAL_CONFIG = 'database_generation_settings.toml'

DB_GEN_TABS: list[tuple[str, list[str]]] = [
    ('Database', ['database']),
    ('Phase Diagram', ['phase_diagram']),
    ('Generation', ['generation']),
    (
        'Modifications',
        [
            'deformation',
            'perturbation',
            'vacancies',
            'adsorbates',
            'targeted_modification',
        ],
    ),
    ('Filters', ['struct_filters']),
    ('Hull', ['concave_hull']),
]


class InitDbPage(WorkflowPage):
    """Configure and launch initial database generation."""

    DISPLAY_NAME = 'Initial DB'
    NAVIGATION_KEY = 'init_db'

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
            self._themed_icon('play_arrow'), ' Run Database Generation'
        )
        self.run_button.clicked.connect(self.run)
        self.cancel_button = self._make_cancel_button()
        self.config_panel.add_action_button(self.run_button)
        self.config_panel.add_action_button(self.cancel_button)

        self.structures_panel = StructuresPanel(project)
        self.table_panel = StructuresTablePanel(project)
        self.boundary_panel = BoundaryPanel(project)
        self._manage_panel = DbManagePanel(project)
        self._manage_panel.database_deleted.connect(self._on_db_deleted)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.config_panel, 'Config')
        self.tabs.addTab(self.structures_panel, 'Outputs — Database')
        self.tabs.addTab(self.table_panel, 'Table View')
        self.tabs.addTab(self.boundary_panel, 'Outputs — Boundary')
        self.tabs.addTab(self._manage_panel, 'Manage')

        self.success_banner = SuccessBanner(
            message='Database generated successfully!',
            actions=[
                ('View Outputs', self._go_to_outputs),
                ('Go to DFT Labelling →', lambda: self._navigate('dft')),
            ],
        )
        self.success_banner.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.success_banner)
        layout.addWidget(self.tabs)

    def _build_config_panel(self):
        from atlas.core.gui.widgets.config_panel import ConfigPanel

        return ConfigPanel(
            schema_data=self._schema_data,
            section_key='database_generation',
            sub_section_tabs=DB_GEN_TABS,
            project=self.project,
            application_font=self._application_font,
        )

    def run(self) -> None:
        if not self.config_panel.save_to_project(label='run'):
            self._log('❌ Cannot run: configuration could not be saved.')
            return

        parsed, err = self.config_panel.parsed_config()
        if err or not parsed:
            self._log(f'❌ Cannot run: {err or "empty configuration"}')
            return
        phases = (parsed.get('phase_diagram') or {}).get('phase')
        if not phases:
            self._log(
                '❌ Cannot run: no phases defined in the phase diagram.\n'
                '   Add at least one phase (with a prototype and composition) '
                'before generating the database.'
            )
            return

        success, errors, warnings = self.config_panel._run_validator(parsed)
        if not success:
            self._log('❌ Cannot run: configuration validation failed:')
            for e in errors:
                self._log(f'   • {e}')
            self._log('Fix the errors above before generating the database.')
            return
        if warnings:
            for w in warnings:
                self._log(f'⚠ {w}')

        if (
            QMessageBox.question(
                self,
                'Confirm Run',
                'Generate the initial database with the current configuration?',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        self.success_banner.hide()
        command = ['atl_gen_init_db', 'generate', '-c', CANONICAL_CONFIG]
        self._set_running(self.run_button, self.cancel_button)
        self.worker = ProcessRunner(
            command,
            cwd=self.project.cwd(),
            log_file=self.project.log_path('atl_gen_init_db'),
            detached=True,
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

    def on_shown(self) -> None:
        self._try_reconnect_detached()
        self.structures_panel.refresh()
        self.table_panel.refresh()
        self.boundary_panel.refresh()
        self._manage_panel.refresh()
        self._update_success_banner()

    def _try_reconnect_detached(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        logs_dir = self.project.dir / 'logs'
        info = find_detached_process(logs_dir, 'atl_gen_init_db')
        if info is None:
            return
        self._log(f'🔄 Found running process (PID {info["pid"]}), reconnecting…')
        self._set_running(self.run_button, self.cancel_button)
        self.worker = DetachedProcessMonitor(
            pid=info['pid'],
            log_file=info['log_file'],
            pid_file=info['pid_file'],
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

    def _update_success_banner(self) -> None:
        state = self.project.workflow_state()
        init_done = state['stages']['init_db']['status'] in ('done', 'partial')
        if init_done and not self.success_banner.isVisible():
            pass

    def _go_to_outputs(self) -> None:
        self.tabs.setCurrentWidget(self.structures_panel)

    def _on_db_deleted(self) -> None:
        self._log('🗑 Database deleted.')
        self.structures_panel.refresh()
        self.table_panel.refresh()
        self.boundary_panel.refresh()
        self.success_banner.hide()
        self.workflow_state_changed.emit()

    def _on_finished(self, return_code: int) -> None:
        self._log(f'\n✅ atl_gen_init_db finished with exit code: {return_code}\n')
        self._set_idle(self.run_button, self.cancel_button, 'Run Database Generation')
        self._notification('Initial Database Generation', return_code == 0)
        if return_code == 0:
            try:
                count = self.project.refresh_structures_index()
            except Exception as exc:
                self._log(f'⚠ Failed to index structures: {exc}')
                return
            self._log(f'📦 Indexed {count} structures into the project.')
            self.structures_panel.refresh()
            self.table_panel.refresh()
            self.boundary_panel.refresh()
            self.success_banner.show()
            self.workflow_state_changed.emit()
