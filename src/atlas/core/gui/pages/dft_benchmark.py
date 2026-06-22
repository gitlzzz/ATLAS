"""DFT parameter benchmark page (``atl_dft_benchmark``)."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
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

CANONICAL_CONFIG = 'dft_benchmark_settings.toml'

BENCHMARK_TABS: list[tuple[str, list[str]]] = [
    ('General & Database', ['general', 'database']),
    ('Calculation & K-points', ['calculation', 'kpoints']),
    ('Queue & HPC', ['queue']),
    ('INCAR', ['incar']),
    ('Benchmarks', ['benchmark']),
]


class DftBenchmarkPage(WorkflowPage):
    """Configure and launch DFT parameter benchmarks."""

    DISPLAY_NAME = 'DFT Benchmark'
    NAVIGATION_KEY = 'dft_benchmark'

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
            self._themed_icon('play_arrow'),
            ' Run Benchmark',
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

        self._results_panel = _BenchmarkResultsPanel(project)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('pageTab')
        self.tabs.addTab(config_tab, 'Config')
        self.tabs.addTab(self._results_panel, 'Results')
        self._mark_output_tabs(1)

        self._setup_banner = PrereqBanner(
            'ATLAS initial setup is incomplete.',
            'Go to Settings',
            lambda: self._navigate('settings'),
        )
        self._setup_banner.setVisible(False)

        self.prereq_banner = PrereqBanner(
            message=(
                'DFT benchmarking needs an initial database first. '
                'Generate one in the Initial DB step before running.'
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
        self._refresh_suggestions()
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
        self._results_panel.refresh()

    def _try_reconnect_detached(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        logs_dir = self.project.dir / 'logs'
        info = find_detached_process(logs_dir, 'atl_dft_benchmark')
        if info is None:
            return
        self._log(f'🔄 Found running benchmark (PID {info["pid"]}), reconnecting…')
        self._set_running(self.run_button, self.cancel_button)
        self.worker = DetachedProcessMonitor(
            pid=info['pid'],
            log_file=info['log_file'],
            pid_file=info['pid_file'],
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
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
                + '. Open Settings to run the setup wizard.',
            )
            self._setup_banner.setVisible(True)
        else:
            self._setup_banner.setVisible(False)

    def _build_config_panel(self):
        from atlas.core.gui.widgets.config_panel import ConfigPanel

        return ConfigPanel(
            schema_data=self._schema_data,
            section_key='dft_benchmark',
            sub_section_tabs=BENCHMARK_TABS,
            project=self.project,
            application_font=self._application_font,
        )

    def _refresh_suggestions(self) -> None:
        from atlas.core.gui.widgets.aiida_suggestions import (
            fetch_aiida_suggestions,
        )

        self.config_panel.populate_suggestions(fetch_aiida_suggestions())

    def _build_preflight_checks(self) -> list[Check]:
        def _check_aiida() -> str | None:
            if not _aiida_available():
                return 'No AiiDA profile loaded, set AIIDA_PROFILE or load one first.'
            return None

        def _check_code_string() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (parsed or {}).get('queue', {}).get('code_string', '')
            if not val:
                return 'Missing code_string in Queue & HPC tab.'
            try:
                from aiida import orm

                orm.load_code(val)
            except Exception:
                return f'Code "{val}" not found in AiiDA.'
            return None

        def _check_potential_family() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (
                (parsed or {}).get('calculation', {}).get('aiida_potential_family', '')
            )
            if not val:
                return 'Missing aiida_potential_family in Calculation tab.'
            try:
                from aiida import orm

                orm.Group.collection.get(label=val)
            except Exception:
                return f'Potential family "{val}" not found in AiiDA.'
            return None

        def _check_db_file() -> str | None:
            path = self.db_path_edit.text().strip()
            if not path:
                return 'Select a database file in the picker above.'
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
            self._log('❌ Provide an existing database file before running.')
            return

        if not self.config_panel.save_to_project(label='run'):
            self._log('❌ Cannot run: configuration could not be saved.')
            return

        parsed, err = self.config_panel.parsed_config()
        if err or not parsed:
            self._log(f'❌ Cannot run: {err or "empty configuration"}')
            return

        self._preflight.run_checks()
        preflight_warn = self._preflight.failing_summary()
        if preflight_warn:
            answer = QMessageBox.warning(
                self,
                'Pre-flight checks failed',
                f'Some pre-flight checks are not passing:\n\n{preflight_warn}'
                '\n\nDo you want to proceed anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        if (
            QMessageBox.question(
                self,
                'Confirm Benchmark',
                'Run DFT parameter benchmark with the current configuration?',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        # Inject db_path into the saved config
        config_path = self.project.config_path('dft_benchmark')
        self._inject_db_path(config_path, db_path)

        command = ['atl_dft_benchmark', '-c', CANONICAL_CONFIG]
        self._set_running(self.run_button, self.cancel_button)
        self.worker = ProcessRunner(
            command,
            cwd=self.project.cwd(),
            log_file=self.project.log_path('atl_dft_benchmark'),
            detached=True,
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

    def _inject_db_path(self, config_path: Path, db_path: str) -> None:
        """Write the selected db_path into the saved TOML config."""
        try:
            import tomli
            import tomli_w

            with open(config_path, 'rb') as f:
                data = tomli.load(f)
            data.setdefault('database', {})['db_path'] = db_path
            with open(config_path, 'wb') as f:
                tomli_w.dump(data, f)
        except Exception:
            pass

    def _on_finished(self, return_code: int) -> None:
        self._log(
            f'\n✅ atl_dft_benchmark finished with exit code: {return_code}\n',
        )
        self._set_idle(self.run_button, self.cancel_button, 'Run Benchmark')
        self._notification('DFT Benchmark', return_code == 0)
        self._results_panel.refresh()
        self.workflow_state_changed.emit()


# ============================================================= results panel


class _BenchmarkResultsPanel(QWidget):
    """Displays convergence plots and recommended settings from a benchmark run."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        self._summary = QLabel('No benchmark results yet.')
        self._summary.setStyleSheet('padding: 8px; color: #495057;')

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._summary, stretch=1)
        top.addWidget(refresh_btn)

        # Scrollable area for plots
        self._plots_container = QWidget()
        self._plots_layout = QVBoxLayout(self._plots_container)
        self._plots_layout.setContentsMargins(8, 8, 8, 8)
        self._plots_layout.setSpacing(16)
        self._plots_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._plots_container)

        # Recommended settings text area
        self._toml_label = QLabel('Recommended settings:')
        self._toml_label.setStyleSheet('font-weight: bold; padding: 4px;')
        self._toml_label.hide()

        self._toml_view = QTextEdit()
        self._toml_view.setReadOnly(True)
        self._toml_view.setMaximumHeight(200)
        self._toml_view.setStyleSheet('font-family: monospace; font-size: 12px;')
        self._toml_view.hide()

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._toml_label)
        layout.addWidget(self._toml_view)

    def refresh(self) -> None:
        # Clear previous plots
        while self._plots_layout.count() > 1:
            item = self._plots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        output_dir = self._find_output_dir()
        if output_dir is None:
            self._summary.setText('No benchmark results yet.')
            self._toml_label.hide()
            self._toml_view.hide()
            return

        # Load convergence plots
        png_files = sorted(output_dir.glob('convergence_*.png'))
        if not png_files:
            self._summary.setText(f'Output dir found ({output_dir}) but no plots yet.')
            return

        self._summary.setText(
            f'{len(png_files)} convergence plot(s) from {output_dir.name}',
        )

        for png in png_files:
            pix = QPixmap(str(png))
            if pix.isNull():
                continue
            label = QLabel()
            scaled = pix.scaledToWidth(
                min(800, pix.width()),
                Qt.SmoothTransformation,
            )
            label.setPixmap(scaled)
            label.setAlignment(Qt.AlignCenter)

            title = QLabel(png.stem.replace('convergence_', '').upper())
            title.setStyleSheet('font-weight: bold; font-size: 13px; padding: 2px;')
            title.setAlignment(Qt.AlignCenter)

            idx = self._plots_layout.count() - 1
            self._plots_layout.insertWidget(idx, title)
            self._plots_layout.insertWidget(idx + 1, label)

        # Load recommended settings
        toml_path = output_dir / 'recommended_settings.toml'
        if toml_path.exists():
            self._toml_label.show()
            self._toml_view.show()
            self._toml_view.setPlainText(toml_path.read_text(encoding='utf-8'))
        else:
            self._toml_label.hide()
            self._toml_view.hide()

    def _find_output_dir(self) -> Path | None:
        """Find the benchmark output directory from the active config."""
        config_path = self._project.config_path('dft_benchmark')
        if config_path.exists():
            try:
                import tomli

                with open(config_path, 'rb') as f:
                    data = tomli.load(f)
                rel = data.get('benchmark', {}).get(
                    'output_dir',
                    './dft_benchmark_results',
                )
                candidate = (self._project.cwd() / rel).resolve()
                if candidate.is_dir():
                    return candidate
            except Exception:
                pass

        # Fallback: check default name
        default = self._project.cwd() / 'dft_benchmark_results'
        if default.is_dir():
            return default
        return None


# ============================================================= helpers


def _aiida_available() -> bool:
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        return profile is not None
    except Exception:
        return False
