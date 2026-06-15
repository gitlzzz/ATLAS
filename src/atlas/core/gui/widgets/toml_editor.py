"""TOML preview editor and syntax highlighter for the ATLAS GUI.

`TomlHighlighter` colours TOML text using one of several palettes.
`TomlEditor` wraps a `QTextEdit` with the highlighter, theme switching, and a
schema-path-aware line highlight helper.
"""

from __future__ import annotations

import string

from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class TomlHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for TOML with multiple colour themes."""

    THEMES = {
        'Default': {
            'background': None,
            'section': '#107c10',
            'comment': '#808080',
            'keyword': '#0033cc',
            'string': '#A31515',
            'number': '#c500c5',
        },
        'Gruvbox Light': {
            'background': None,
            'section': '#79740e',
            'comment': '#928374',
            'keyword': '#427b58',
            'string': '#9d0006',
            'number': '#8f3f71',
        },
        'Catppuccin Latte': {
            'background': None,
            'section': '#40a02b',
            'comment': '#9ca0b0',
            'keyword': '#8839ef',
            'string': '#d20f39',
            'number': '#e64553',
        },
        'Solarized Light': {
            'background': '#fdf6e3',
            'section': '#859900',
            'comment': '#93a1a1',
            'keyword': '#d33682',
            'string': '#dc322f',
            'number': '#cb4b16',
        },
    }

    def __init__(self, parent=None, theme='Default'):
        super().__init__(parent)
        self.highlighting_rules = []

        palette = self.THEMES.get(theme, self.THEMES['Default'])

        section_format = QTextCharFormat()
        section_format.setForeground(QColor(palette['section']))
        section_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append(
            (QRegularExpression(r'\[([^]]*)\]'), section_format)
        )

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(palette['comment']))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression(r'#[^\n]*'), comment_format))

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(palette['keyword']))
        keyword_format.setFontWeight(QFont.Bold)
        for word in ('true', 'false'):
            pattern = QRegularExpression(rf'\b{word}\b')
            self.highlighting_rules.append((pattern, keyword_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor(palette['string']))
        self.highlighting_rules.append((QRegularExpression(r'"[^"]*"'), string_format))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor(palette['number']))
        self.highlighting_rules.append(
            (QRegularExpression(r'\b-?\d+(\.\d+)?\b'), number_format)
        )

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        for pattern, fmt in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class TomlEditor(QWidget):
    """Editable TOML preview with syntax highlighting and path-based line highlight.

    Signals
    -------
    user_edited
        Emitted when the user (not programmatic code) modifies the text.
    """

    user_edited = Signal()

    def __init__(self, parent=None, theme='Default', monospace_families=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        font = QFont('monospace')
        if monospace_families:
            font.setFamilies(monospace_families)
        self.text_edit.setFont(font)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.text_edit)

        self.highlighter = TomlHighlighter(self.text_edit.document(), theme)

        self._highlight_format = QTextCharFormat()
        self._highlight_format.setBackground(QColor('#ffe8a3'))
        self._highlight_format.setForeground(QColor('#1b1b1b'))
        self._highlight_format.setProperty(QTextFormat.FullWidthSelection, True)

        self._updating = False
        self._last_highlight_path = None

        self.text_edit.textChanged.connect(self._on_text_changed)

    # ------------------------------------------------------------------ theme

    def set_theme(self, theme_name):
        """Switch the syntax highlighting theme."""
        palette = TomlHighlighter.THEMES.get(theme_name, {})
        background = palette.get('background')
        if background:
            self.text_edit.setStyleSheet(f'background-color: {background};')
        else:
            self.text_edit.setStyleSheet('')
        self.highlighter = TomlHighlighter(self.text_edit.document(), theme_name)
        self.highlighter.rehighlight()

    # ----------------------------------------------------------------- text

    def set_text(self, text):
        """Set the editor contents without emitting `user_edited`."""
        self._updating = True
        self.text_edit.blockSignals(True)
        self.text_edit.setPlainText(text)
        self.text_edit.blockSignals(False)
        self._updating = False
        self._refresh_highlight()

    def plain_text(self):
        return self.text_edit.toPlainText()

    def _on_text_changed(self):
        if self._updating:
            return
        self._refresh_highlight()
        self.user_edited.emit()

    # ----------------------------------------------------------- highlight

    def highlight_path(self, path):
        """Highlight the line whose dotted key matches ``path``.

        ``path`` is a sequence of section/key segments (as stored in the YAML
        schema).  Missing entries are silently ignored.
        """
        if not path:
            self.clear_highlight()
            return

        self._last_highlight_path = tuple(path)
        text = self.text_edit.toPlainText()
        index = self._locate_toml_index(text, list(path))
        if index is None:
            self._last_highlight_path = None
            self.clear_highlight()
            return

        doc = self.text_edit.document()
        block = doc.findBlock(index)
        if not block.isValid():
            self._last_highlight_path = None
            self.clear_highlight()
            return

        cursor = QTextCursor(block)
        cursor.select(QTextCursor.LineUnderCursor)

        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        selection.format = QTextCharFormat(self._highlight_format)

        self.text_edit.setExtraSelections([selection])
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
        # Scroll so the target line is at the top of the viewport
        cr = self.text_edit.cursorRect()
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() + cr.top() - 4)
        self.text_edit.viewport().update()

    def clear_highlight(self):
        self._last_highlight_path = None
        self.text_edit.setExtraSelections([])
        self.text_edit.viewport().update()

    def _refresh_highlight(self):
        if self._last_highlight_path:
            # Re-resolve, since text changed.
            path = list(self._last_highlight_path)
            self._last_highlight_path = None  # avoid infinite recursion
            self.highlight_path(path)
        else:
            self.text_edit.setExtraSelections([])
            self.text_edit.viewport().update()

    # ----------------------------------------------------- path resolution

    def _locate_toml_index(self, text, path):
        if not text or not path:
            return None

        # Single-element path → treat as TOML section header ([key])
        if len(path) == 1:
            header = f'[{path[0]}]'
            idx = text.find(header)
            if idx != -1:
                return idx
            # Section may not have direct keys; look for sub-section header
            sub_header = f'[{path[0]}.'
            idx = text.find(sub_header)
            return idx if idx != -1 else None

        section_start = 0
        split_index = 0

        for i in range(1, len(path)):
            header = f'[{".".join(path[:i])}]'
            header_idx = text.find(header, section_start)
            if header_idx == -1:
                break
            split_index = i
            newline_idx = text.find('\n', header_idx)
            if newline_idx == -1:
                section_start = len(text)
                break
            section_start = newline_idx + 1
        else:
            split_index = len(path) - 1

        section_end = text.find('\n[', section_start)
        if section_end == -1:
            section_end = len(text)
        segment = text[section_start:section_end]

        key_segments = path[split_index:] or [path[-1]]
        key_expression = self._build_toml_dotted_key(key_segments)

        for pattern in (f'{key_expression} =', f'{key_expression}='):
            rel_idx = segment.find(pattern)
            if rel_idx != -1:
                return section_start + rel_idx

        for pattern in (f'{key_expression} =', f'{key_expression}='):
            rel_idx = text.find(pattern)
            if rel_idx != -1:
                return rel_idx

        return None

    def _build_toml_dotted_key(self, segments):
        parts = []
        for segment in segments:
            escaped = segment.replace('"', '\\"')
            if self._requires_toml_quotes(segment):
                parts.append(f'"{escaped}"')
            else:
                parts.append(segment)
        return '.'.join(parts)

    @staticmethod
    def _requires_toml_quotes(segment):
        if not segment:
            return True
        allowed = set(string.ascii_letters + string.digits + '_-')
        if segment[0].isdigit():
            return True
        return any(char not in allowed for char in segment)
