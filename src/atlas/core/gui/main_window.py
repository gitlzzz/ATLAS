"""Main ATLAS GUI window with sidebar navigation and log dock.

Layout
------
* Left: ``QListWidget`` sidebar listing each workflow page.
* Centre: ``QStackedWidget`` holding one page per sidebar entry.
* Bottom: ``QDockWidget`` log viewer shared by all pages.
* Top: standard menu bar (File / Edit / Format / Help).
"""

from __future__ import annotations

import contextlib
import os
import time
from functools import partial

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QFont, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QProgressBar,
    QSplitter,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QWidget,
)

from atlas.core.gui.pages.active_learning import ActiveLearningPage
from atlas.core.gui.pages.dft import DftPage
from atlas.core.gui.pages.dft_benchmark import DftBenchmarkPage
from atlas.core.gui.pages.init_db import InitDbPage
from atlas.core.gui.pages.overview import OverviewPage
from atlas.core.gui.pages.reports import ReportsPage
from atlas.core.gui.pages.settings import SettingsPage
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.toml_editor import TomlHighlighter

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'config_schema.yaml'
)

PAGE_ICON_NAMES = {
    'overview': 'dashboard',
    'init_db': 'database',
    'dft': 'science',
    'dft_benchmark': 'tune',
    'al': 'sync',
    'reports': 'assessment',
    'settings': 'settings',
}

SIDEBAR_STYLE = """
QListWidget {
    background-color: #f4f5f7;
    border: none;
    border-right: 1px solid #d0d4db;
    padding: 4px;
    font-size: 13px;
}
QListWidget::item {
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 4px;
    color: #333;
}
QListWidget::item:selected {
    background-color: #d0e0ff;
    color: #1a3a6b;
    font-weight: bold;
}
QListWidget::item:hover:!selected {
    background-color: #e4e8ee;
}
"""


class MainWindow(QMainWindow):
    """Sidebar-navigated main window for the ATLAS GUI."""

    def __init__(
        self,
        project: Project,
        application_font: QFont | None = None,
        progress_callback=None,
        verbose: bool = False,
    ):
        super().__init__()
        self._progress = progress_callback or (lambda *_: None)
        self._verbose = verbose
        self._t0 = time.perf_counter()

        self.setGeometry(100, 100, 1600, 900)
        self.project = project
        self.setWindowTitle(f'ATLAS — {project.name}')
        self.setWindowIcon(self._app_icon())

        self._application_font = application_font or QFont()
        self._schema_data: dict = {}

        self._log_timing('Loading configuration schema...')
        self._progress(10, 'Loading configuration schema...')
        self._load_schema(SCHEMA_PATH)

        self._force_quit = False
        self._current_theme = project.meta('app_theme', 'Default (Light)')
        self._update_matplotlib_style(self._current_theme)

        self._log_timing('Building interface...')
        self._progress(20, 'Building interface...')
        self._build_log_dock()
        self._build_central_widget()

        self._log_timing('Applying theme...')
        self._progress(25, 'Applying theme...')
        self._apply_app_theme(self._current_theme)

        self._log_timing('Loading pages...')
        self._progress(30, 'Loading pages...')
        self._build_pages()

        self._log_timing('Creating menus and shortcuts...')
        self._progress(85, 'Creating menus and shortcuts...')
        self._create_menu_bar()
        self._create_shortcuts()
        self._build_status_bar()
        self._build_tray_icon()

        self.sidebar.setCurrentRow(0)

        self._log_timing('Ready')
        self._progress(100, 'Ready')
        if self._verbose:
            elapsed = time.perf_counter() - self._t0
            print(f'[ATLAS] Project loaded in {elapsed:.2f}s')

    def _log_timing(self, stage: str) -> None:
        if not self._verbose:
            return
        elapsed = time.perf_counter() - self._t0
        print(f'[ATLAS] {elapsed:6.3f}s — {stage}')

    # ============================================================ schema

    def _load_schema(self, filepath: str) -> None:
        try:
            with open(filepath, encoding='utf-8') as f:
                self._schema_data = yaml.safe_load(f) or {}
        except Exception as exc:
            self._schema_data = {}
            self.statusBar().showMessage(f'Error loading schema: {exc}', 10000)

    # ========================================================== layout

    def _build_log_dock(self) -> None:
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setPlaceholderText('Process logs and status messages.')

        dock = QDockWidget('Logs & Output', self)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        dock.setWidget(self.log_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.log_dock = dock

    def _build_central_widget(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        self.sidebar = QListWidget()
        self.sidebar.setMaximumWidth(200)
        self.sidebar.setStyleSheet(SIDEBAR_STYLE)
        splitter.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        splitter.addWidget(self.stack)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 1420])

        self.sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        self.setCentralWidget(splitter)

    def _build_pages(self) -> None:
        page_classes = (
            OverviewPage,
            InitDbPage,
            DftPage,
            DftBenchmarkPage,
            ActiveLearningPage,
            ReportsPage,
            SettingsPage,
        )
        page_names = (
            'Overview',
            'Initial DB',
            'DFT Labelling',
            'DFT Benchmark',
            'Active Learning',
            'Reports',
            'Settings',
        )
        self.pages: list = []
        self._page_index_by_key: dict[str, int] = {}
        for idx, page_cls in enumerate(page_classes):
            pct = 30 + int(50 * idx / len(page_classes))
            self._log_timing(f'  Loading page: {page_names[idx]}...')
            self._progress(pct, f'Loading page: {page_names[idx]}...')
            page = page_cls(
                project=self.project,
                schema_data=self._schema_data,
                application_font=self._application_font,
                log=self.log,
                navigate=self.navigate_to,
                notification=self._show_tray_notification,
            )
            self.pages.append(page)
            self._page_index_by_key[page.NAVIGATION_KEY] = idx
            self.stack.addWidget(page)

            item = QListWidgetItem(page.DISPLAY_NAME)
            icon_name = PAGE_ICON_NAMES.get(page.NAVIGATION_KEY)
            if icon_name:
                item.setIcon(self._themed_sidebar_icon(icon_name))
            self.sidebar.addItem(item)

            page.workflow_state_changed.connect(self._broadcast_workflow_refresh)
            page.process_running_changed.connect(self._on_process_running_changed)
            if page.config_panel is not None:
                page.config_panel.save_succeeded.connect(
                    lambda: self.statusBar().showMessage(
                        '✓ Configuration saved.',
                        3000,
                    )
                )

        # Settings page broadcasts theme changes; route them to every page.
        settings_page = self.pages[-1]
        settings_page.theme_changed.connect(self._apply_theme)
        settings_page.app_theme_changed.connect(self._apply_app_theme)

    def _broadcast_workflow_refresh(self) -> None:
        """Ask every page (including the sender) to recompute pipeline UI."""
        for page in self.pages:
            page.on_shown()
        self._update_status_bar()

    def navigate_to(self, key: str) -> None:
        """Switch the active page by ``NAVIGATION_KEY``."""
        idx = self._page_index_by_key.get(key)
        if idx is not None:
            self.sidebar.setCurrentRow(idx)

    def _on_sidebar_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        if 0 <= idx < len(self.pages):
            self.pages[idx].on_shown()
        self._update_status_bar()

    # ====================================================== status bar

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._status_project = QLabel(f'  {self.project.name}')
        self._status_project.setStyleSheet('font-weight: bold; padding: 0 8px;')
        self._status_stage = QLabel()
        self._status_stage.setStyleSheet('padding: 0 8px;')

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(160)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            'QProgressBar { border: 1px solid palette(mid); border-radius: 4px; }'
            'QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }'
        )
        self._progress_bar.hide()

        bar.addPermanentWidget(self._progress_bar)
        bar.addPermanentWidget(self._status_project)
        bar.addPermanentWidget(self._status_stage)
        self._update_status_bar()

    def _on_process_running_changed(self, running: bool) -> None:
        if running:
            self._progress_bar.show()
        else:
            any_running = any(
                p.worker is not None and p.worker.isRunning() for p in self.pages
            )
            if not any_running:
                self._progress_bar.hide()

    def _running_task_name(self) -> str | None:
        for p in self.pages:
            if p.worker is not None and p.worker.isRunning():
                return p.DISPLAY_NAME
        return None

    def _update_status_bar(self) -> None:
        try:
            state = self.project.workflow_state()
            next_key = state.get('next_recommended', '')
            stage_names = {
                'init_db': 'Initial DB',
                'dft': 'DFT Labelling',
                'al': 'Active Learning',
                'reports': 'Reports',
            }
            if next_key:
                name = stage_names.get(next_key, next_key)
                self._status_stage.setText(f'Next: {name}')
            else:
                self._status_stage.setText('All stages complete')
        except Exception:
            self._status_stage.setText('')

    # ======================================================= shortcuts

    def _create_shortcuts(self) -> None:
        save_shortcut = QAction('Save', self)
        save_shortcut.setShortcut(QKeySequence.Save)
        save_shortcut.triggered.connect(self._current_save_snapshot)
        self.addAction(save_shortcut)

        run_shortcut = QAction('Run', self)
        run_shortcut.setShortcut(QKeySequence('Ctrl+R'))
        run_shortcut.triggered.connect(self._current_run)
        self.addAction(run_shortcut)

        for i in range(min(9, len(self.pages))):
            action = QAction(f'Page {i + 1}', self)
            action.setShortcut(QKeySequence(f'Ctrl+{i + 1}'))
            action.triggered.connect(partial(self.sidebar.setCurrentRow, i))
            self.addAction(action)

    # ========================================================== menu bar

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('&File')
        load_action = QAction('&Import TOML…', self)
        load_action.triggered.connect(self._current_load_toml)
        file_menu.addAction(load_action)
        save_action = QAction('&Save Snapshot', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._current_save_snapshot)
        file_menu.addAction(save_action)
        export_action = QAction('&Export TOML…', self)
        export_action.triggered.connect(self._current_export_toml)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        close_action = QAction('&Close to Tray', self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        exit_action = QAction('E&xit', self)
        exit_action.triggered.connect(self._tray_quit)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu('&Edit')
        undo_action = QAction('&Undo', self)
        undo_action.setEnabled(False)
        edit_menu.addAction(undo_action)
        redo_action = QAction('&Redo', self)
        redo_action.setEnabled(False)
        edit_menu.addAction(redo_action)

        format_menu = menu_bar.addMenu('F&ormat')
        theme_menu = format_menu.addMenu('Syntax Highlighting Theme')
        theme_group = QActionGroup(self)
        for theme_name in TomlHighlighter.THEMES:
            action = QAction(theme_name, self, checkable=True)
            if theme_name == 'Default':
                action.setChecked(True)
            action.triggered.connect(partial(self._apply_theme, theme_name))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction(self.log_dock.toggleViewAction())

    # =========================================================== helpers

    def _current_page(self) -> QWidget | None:
        idx = self.sidebar.currentRow()
        if idx < 0 or idx >= len(self.pages):
            return None
        return self.pages[idx]

    def _current_load_toml(self) -> None:
        page = self._current_page()
        if page is not None and page.config_panel is not None:
            page.config_panel.load_toml_from_dialog()

    def _current_save_snapshot(self) -> None:
        page = self._current_page()
        if (
            page is not None
            and page.config_panel is not None
            and page.config_panel.save_to_project()
        ):
            self.statusBar().showMessage('Configuration saved.', 3000)

    def _current_export_toml(self) -> None:
        page = self._current_page()
        if page is not None and page.config_panel is not None:
            page.config_panel.save_to_file()

    def _current_run(self) -> None:
        page = self._current_page()
        if page is not None and hasattr(page, 'run'):
            page.run()

    def _apply_theme(self, theme_name: str) -> None:
        for page in self.pages:
            page.set_theme(theme_name)

    def _apply_app_theme(self, theme_name: str) -> None:
        from atlas.core.gui.themes import DEFAULT_THEME, apply_theme_to_app

        self._current_theme = theme_name
        apply_theme_to_app(theme_name)
        if theme_name == DEFAULT_THEME:
            self.sidebar.setStyleSheet(SIDEBAR_STYLE)
        else:
            self.sidebar.setStyleSheet('')
        self._update_matplotlib_style(theme_name)
        self._refresh_sidebar_icons()

    @staticmethod
    def _update_matplotlib_style(theme_name: str) -> None:
        """Set matplotlib rcParams to match the current app theme."""
        import matplotlib as mpl

        from atlas.core.gui.themes import theme_colors

        c = theme_colors(theme_name)
        bg, fg = c['bg'], c['fg']
        surface = c['surface']
        border = c['border']
        axes_bg = c['axes_bg']

        mpl.rcParams.update(
            {
                'figure.facecolor': bg,
                'axes.facecolor': axes_bg,
                'axes.edgecolor': border,
                'axes.labelcolor': fg,
                'text.color': fg,
                'xtick.color': fg,
                'ytick.color': fg,
                'grid.color': border,
                'legend.facecolor': surface,
                'legend.edgecolor': border,
            }
        )

    def log(self, message: str) -> None:
        self.log_viewer.append(message)

    # ================================================== task notification

    def _show_tray_notification(self, task_name: str, success: bool) -> None:
        if self._tray is None or not self._tray.isVisible():
            return

        if success:
            self._tray.showMessage(
                task_name,
                'Completed successfully.',
                QSystemTrayIcon.Information,
                3000,
            )
        else:
            self._tray.showMessage(
                f'{task_name} — Failed',
                'The process did not complete successfully.',
                QSystemTrayIcon.Warning,
                3000,
            )

    # ======================================================= system tray

    def _build_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(self._app_icon())
        self._tray.setToolTip(f'ATLAS — {self.project.name}')

        menu = QMenu()
        show_action = menu.addAction('Show ATLAS')
        show_action.triggered.connect(self._tray_show)
        menu.addSeparator()
        quit_action = menu.addAction('Quit')
        quit_action.triggered.connect(self._tray_quit)
        self._tray.setContextMenu(menu)

        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _themed_sidebar_icon(self, name: str) -> QIcon:
        from atlas.core.gui.icons import themed_icon
        from atlas.core.gui.themes import theme_colors

        color = theme_colors(self._current_theme)['fg']
        return themed_icon(name, color, size=20)

    def _refresh_sidebar_icons(self) -> None:
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if i < len(self.pages):
                icon_name = PAGE_ICON_NAMES.get(self.pages[i].NAVIGATION_KEY)
                if icon_name:
                    item.setIcon(self._themed_sidebar_icon(icon_name))

    @staticmethod
    def _app_icon() -> QIcon:
        from atlas.core.gui.icons import app_icon

        return app_icon()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._tray_show()

    def _tray_show(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _tray_quit(self) -> None:
        self._force_quit = True
        self.close()
        from PySide6.QtWidgets import QApplication

        QApplication.instance().quit()

    # ================================================== setup wizard

    def _check_first_run(self) -> None:
        from atlas.core.gui.widgets.setup_wizard import (
            SetupWizard,
            needs_first_run_wizard,
        )

        if needs_first_run_wizard():
            wizard = SetupWizard(self)
            wizard.exec()
            self._broadcast_workflow_refresh()

    # ======================================================= close

    def closeEvent(self, event):
        if not self._force_quit and self._tray is not None and self._tray.isVisible():
            self.hide()
            task = self._running_task_name()
            if task:
                detail = f'{task} is running and will continue in the background.'
            else:
                detail = 'Still running in the background.'
            self._tray.showMessage(
                'ATLAS',
                detail + ' Right-click the tray icon to quit.',
                QSystemTrayIcon.Information,
                2000,
            )
            event.ignore()
            return

        if self._tray is not None:
            self._tray.hide()
        with contextlib.suppress(Exception):
            self.project.close()
        super().closeEvent(event)
