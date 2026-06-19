"""Foldable log panel with a toggle bar, unread badge, and smooth animation."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QFont, QTextDocument
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget


class CollapsibleLogPanel(QWidget):
    """Log panel that collapses to a toggle bar with an unread-line badge.

    Signals
    -------
    toggled : Signal(bool)
        Emitted on expand/collapse with the new ``expanded`` state.
    """

    toggled = Signal(bool)
    unread_count_changed = Signal(int)

    def __init__(
        self,
        shared_doc: QTextDocument | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = False
        self._unread_count = 0
        self._animation: QPropertyAnimation | None = None
        self._target_h = 200

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- toggle bar ----
        self._bar = QWidget()
        self._bar.setObjectName('logToggleBar')
        self._bar.setFixedHeight(26)
        self._bar.setMouseTracking(True)
        self._bar.installEventFilter(self)
        self._bar.setCursor(Qt.CursorShape.PointingHandCursor)

        bar_layout = QHBoxLayout(self._bar)
        bar_layout.setContentsMargins(8, 2, 8, 2)
        bar_layout.setSpacing(6)

        self._chevron = QLabel()
        self._chevron.setFixedSize(16, 16)
        self._chevron.setAlignment(Qt.AlignCenter)

        self._label = QLabel('Show Logs')
        self._label.setStyleSheet('font-weight: bold;')

        self._badge = QLabel()
        self._badge.setFixedSize(20, 18)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            'background-color: #e74c3c; color: #ffffff;'
            ' border-radius: 9px; font-weight: bold; font-size: 10px; padding: 0 4px;'
        )
        self._badge.hide()

        bar_layout.addWidget(self._chevron)
        bar_layout.addWidget(self._label)
        bar_layout.addStretch()
        bar_layout.addWidget(self._badge)

        outer.addWidget(self._bar)

        # ---- expandable container ----
        container = QVBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.setSpacing(0)

        self._viewer = QTextEdit()
        if shared_doc is not None:
            self._viewer.setDocument(shared_doc)
        self._viewer.setReadOnly(True)
        self._viewer.setPlaceholderText('Process logs and status messages.')
        self._viewer.setFont(self._monospace_font())

        # Clear button row
        clear_row = QHBoxLayout()
        clear_row.setContentsMargins(4, 0, 4, 2)
        clear_row.addStretch()

        from PySide6.QtWidgets import QPushButton

        self._clear_btn = QPushButton('Clear')
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self._viewer.clear)
        clear_row.addWidget(self._clear_btn)

        container.addLayout(clear_row)
        container.addWidget(self._viewer, 1)

        container_widget = QWidget()
        container_widget.setLayout(container)
        container_widget.hide()

        self._container = container_widget
        outer.addWidget(self._container, 1)

    # ---------------------------------------------------------- public API

    def viewer(self) -> QTextEdit:
        """Return the internal ``QTextEdit``."""
        return self._viewer

    def append(self, text: str) -> None:
        """Append *text* to the log viewer."""
        if not self._expanded:
            self._unread_count += 1
            self._badge.setText(str(min(self._unread_count, 999)))
            self._badge.show()
            self.unread_count_changed.emit(self._unread_count)
        self._viewer.append(text)

    def clear(self) -> None:
        """Clear the log viewer and reset the unread count."""
        self._viewer.clear()
        self._unread_count = 0
        self._badge.hide()
        self.unread_count_changed.emit(0)

    def toggle(self) -> None:
        """Toggle the expanded/collapsed state with animation."""
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def set_theme(self, fg_color: str) -> None:
        """Refresh the chevron icon for the current theme foreground colour."""
        from atlas.core.gui.icons import themed_icon

        icon_name = 'chevron_up' if self._expanded else 'chevron_down'
        pixmap = themed_icon(icon_name, fg_color, size=16).pixmap(16, 16)
        self._chevron.setPixmap(pixmap)

    # ---------------------------------------------------------- expand / collapse

    def _expand(self) -> None:
        self._target_h = self._target_height()
        self._container.show()
        self._container.setMaximumHeight(0)

        self._animation = QPropertyAnimation(
            self._container,
            b'maximumHeight',
            parent=self,
        )
        self._animation.setDuration(100)
        self._animation.setStartValue(0)
        self._animation.setEndValue(self._target_h)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_expand_finished)
        self._animation.start()

        self._expanded = True
        self._unread_count = 0
        self._badge.hide()
        self.unread_count_changed.emit(0)
        self._update_bar_text()
        self._update_chevron()
        self.toggled.emit(True)

    def _collapse(self) -> None:
        self._animation = QPropertyAnimation(
            self._container,
            b'maximumHeight',
            parent=self,
        )
        self._animation.setDuration(100)
        self._animation.setStartValue(self._target_h)
        self._animation.setEndValue(0)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_collapse_finished)
        self._animation.start()

        self._expanded = False
        self._update_bar_text()
        self._update_chevron()
        self.toggled.emit(False)

    def _on_expand_finished(self) -> None:
        self._container.setMinimumHeight(self._target_h)
        self._container.setMaximumHeight(16777215)

    def _on_collapse_finished(self) -> None:
        self._container.setMinimumHeight(0)
        self._container.hide()
        self._container.setMaximumHeight(16777215)

    def _target_height(self) -> int:
        parent = self.window()
        parent_height = parent.height() if parent else 900
        return min(400, max(150, parent_height // 4))

    def _update_bar_text(self) -> None:
        self._label.setText('Hide Logs' if self._expanded else 'Show Logs')

    def _update_chevron(self) -> None:
        from atlas.core.gui.icons import themed_icon

        theme_name = _current_theme_name()
        fg = _theme_fg(theme_name)
        icon_name = 'chevron_up' if self._expanded else 'chevron_down'
        self._chevron.setPixmap(themed_icon(icon_name, fg, size=16).pixmap(16, 16))

    # ---------------------------------------------------------- event filter

    def eventFilter(self, obj, event) -> bool:
        if obj is self._bar and event.type() == event.Type.MouseButtonRelease:
            self.toggle()
            return True
        return super().eventFilter(obj, event)

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _monospace_font() -> QFont:
        font = QFont()
        font.setFamily('Fira Code')
        font.setStyleHint(QFont.TypeWriter)
        font.setPixelSize(12)
        return font


# ---------------------------------------------------------- theme helpers


def _current_theme_name() -> str:
    from atlas.core.gui.themes import DEFAULT_THEME, saved_global_theme

    try:
        return saved_global_theme()
    except Exception:
        return DEFAULT_THEME


def _theme_fg(theme_name: str) -> str:
    from atlas.core.gui.themes import theme_colors

    return theme_colors(theme_name)['fg']
