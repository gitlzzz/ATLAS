"""Full-view logs page with a shared text document."""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout

from atlas.core.gui.pages.base import WorkflowPage


class LogsPage(WorkflowPage):
    """Full-height log viewer accessible from the sidebar."""

    DISPLAY_NAME = 'Logs'
    NAVIGATION_KEY = 'logs'

    def __init__(
        self,
        project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        shared_doc=None,
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._viewer = QTextEdit()
        if shared_doc is not None:
            self._viewer.setDocument(shared_doc)
        self._viewer.setReadOnly(True)
        self._viewer.setPlaceholderText('Process logs and status messages.')
        self._viewer.setFont(_monospace_font())

        outer.addWidget(self._viewer, 1)

        clear_row = QHBoxLayout()
        clear_row.setContentsMargins(8, 4, 8, 4)
        clear_row.addStretch()
        self._clear_btn = QPushButton('Clear')
        self._clear_btn.setFixedWidth(80)
        self._clear_btn.clicked.connect(self._viewer.clear)
        clear_row.addWidget(self._clear_btn)
        outer.addLayout(clear_row)


def _monospace_font() -> QFont:
    font = QFont()
    font.setFamily('Fira Code')
    font.setStyleHint(QFont.TypeWriter)
    font.setPixelSize(12)
    return font
