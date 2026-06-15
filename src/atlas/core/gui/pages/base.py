"""Base class for ATLAS GUI workflow pages."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from atlas.core.gui.process.runner import DetachedProcessMonitor, ProcessRunner
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.config_panel import ConfigPanel


class WorkflowPage(QWidget):
    """Common interface every sidebar-navigated workflow page provides.

    ``NAVIGATION_KEY`` is the stable identifier passed to ``navigate()`` so
    other pages can jump here without hard-coding sidebar indices.

    Pages emit ``workflow_state_changed`` after actions that may affect the
    project's pipeline status (e.g. a successful Run, a recorded submission).
    ``MainWindow`` listens on every page and refreshes peers so the Overview
    tracker and downstream banners stay current without manual navigation.
    """

    DISPLAY_NAME: str = 'Page'
    NAVIGATION_KEY: str = 'page'

    workflow_state_changed = Signal()
    process_running_changed = Signal(bool)

    def __init__(
        self,
        project: Project,
        schema_data: dict | None,
        application_font: QFont | None,
        log: Callable[[str], None],
        navigate: Callable[[str], None],
        notification: Callable[[str, bool], None] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.project = project
        self._schema_data = schema_data or {}
        self._application_font = application_font or QFont()
        self._log = log
        self._navigate = navigate
        self._notification = notification or (lambda *_: None)
        self.config_panel: ConfigPanel | None = None
        self.worker: ProcessRunner | None = None

    @staticmethod
    def _themed_icon(name: str) -> QIcon:
        from atlas.core.gui.icons import themed_icon
        from atlas.core.gui.themes import saved_global_theme, theme_colors

        color = theme_colors(saved_global_theme())['fg']
        return themed_icon(name, color, size=18)

    def _make_cancel_button(self) -> QPushButton:
        btn = QPushButton(self._themed_icon('stop'), ' Cancel')
        btn.setStyleSheet(
            'QPushButton { background-color: #e74c3c; color: white;'
            ' border-radius: 4px; padding: 4px 12px; }'
            'QPushButton:hover { background-color: #c0392b; }'
        )
        btn.hide()
        btn.clicked.connect(self._cancel_run)
        return btn

    def _set_running(
        self, run_btn: QPushButton, cancel_btn: QPushButton, label: str = 'Running…'
    ) -> None:
        run_btn.setEnabled(False)
        run_btn.setIcon(QIcon())
        run_btn.setText(label)
        cancel_btn.show()
        self.process_running_changed.emit(True)

    def _set_idle(
        self, run_btn: QPushButton, cancel_btn: QPushButton, label: str = 'Run'
    ) -> None:
        run_btn.setEnabled(True)
        run_btn.setIcon(self._themed_icon('play_arrow'))
        run_btn.setText(label)
        cancel_btn.hide()
        self.process_running_changed.emit(False)

    def _cancel_run(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return

        if isinstance(self.worker, DetachedProcessMonitor):
            choice = QMessageBox.question(
                self,
                'Detached Process',
                'This process runs independently of the GUI.\n\n'
                'Disconnect — stop watching but let it keep running.\n'
                'Terminate — kill the process.\n\n'
                'Disconnect from the process?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if choice == QMessageBox.Yes:
                self.worker.stop()
            elif choice == QMessageBox.No:
                self.worker.terminate_process()
            return

        if (
            QMessageBox.question(
                self,
                'Cancel Process',
                'Are you sure you want to cancel the running process?',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.worker.stop()

    def set_theme(self, theme_name: str) -> None:
        if self.config_panel is not None:
            self.config_panel.set_theme(theme_name)

    def on_shown(self) -> None:
        """Hook called when the sidebar switches to this page. No-op by default."""
