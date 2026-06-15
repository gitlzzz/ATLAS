"""Validation error dialog with clickable field links."""

from __future__ import annotations

import re
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_PARAM_PATH_RE = re.compile(
    r'(?:parameter[:\s]+|Parameter\s+)'
    r'([a-z][a-z0-9_.]+(?:\.[a-z][a-z0-9_]+)+)',
    re.IGNORECASE,
)

_DOTTED_PATH_RE = re.compile(
    r'\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+){1,})\b',
    re.IGNORECASE,
)


def _extract_field_key(error: str) -> str | None:
    m = _PARAM_PATH_RE.search(error)
    if m:
        return m.group(1)
    m = _DOTTED_PATH_RE.search(error)
    if m:
        return m.group(1)
    return None


def _error_to_html(error: str, field_key: str | None) -> str:
    from html import escape

    escaped = escape(error)
    if field_key:
        escaped_key = escape(field_key)
        escaped = escaped + (
            f'&nbsp;&nbsp;'
            f'<a href="{escaped_key}" '
            f'style="color: #42a5f5; text-decoration: underline;">'
            f'Go to field</a>'
        )
    return escaped


class ValidationDialog(QDialog):
    """Dialog to display validation errors with clickable links to fields."""

    def __init__(
        self,
        errors: list[str],
        section_key: str,
        focus_field: Callable[[str], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Configuration Invalid')
        self.setMinimumWidth(500)
        self._focus_field = focus_field

        layout = QVBoxLayout(self)

        header = QLabel(
            f'<b>Configuration validation failed for section "{section_key}".</b>'
        )
        header.setTextFormat(Qt.RichText)
        header.setWordWrap(True)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMaximumHeight(350)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)

        for error in errors:
            field_key = _extract_field_key(error)
            html = _error_to_html(error, field_key)
            lbl = QLabel(f'• {html}')
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setOpenExternalLinks(False)
            if field_key and focus_field:
                lbl.linkActivated.connect(self._on_link)
            content_layout.addWidget(lbl)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _on_link(self, field_key: str) -> None:
        if self._focus_field:
            self.accept()
            self._focus_field(field_key)
