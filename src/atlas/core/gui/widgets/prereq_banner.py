"""Just-in-time prerequisite / success banners.

``PrereqBanner`` (amber) is shown when an upstream pipeline stage hasn't
produced what the current page needs.

``SuccessBanner`` (green) is shown after a stage completes — it offers
navigation buttons to the outputs tab or the next workflow stage, and a
dismiss button (×) to reclaim screen space.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


def _banner_colors(accent: str) -> tuple[str, str, str, str]:
    """Return (bg, border, fg, btn_bg) for a banner accent colour."""
    from atlas.core.gui.themes import THEMES, _mix, saved_global_theme, theme_variant

    name = saved_global_theme()
    t = THEMES.get(name)
    is_dark = theme_variant(name) == 'dark'
    if t is not None:
        bg = _mix(t.background, accent, 0.12)
        fg = t.foreground
        btn_bg = (
            _mix(t.background, accent, 0.06)
            if is_dark
            else _mix('#ffffff', accent, 0.05)
        )
    else:
        bg = _mix('#ffffff', accent, 0.10)
        fg = '#1a1a2e'
        btn_bg = '#ffffff'
    return bg, accent, fg, btn_bg


class PrereqBanner(QFrame):
    """Amber notice + action button shown above a workflow tab area."""

    def __init__(
        self,
        message: str,
        action_label: str,
        on_action: Callable[[], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)

        bg, border, fg, btn_bg = _banner_colors('#b8860b')
        self.setStyleSheet(
            f'QFrame {{'
            f' background-color: {bg};'
            f' border: 1px solid {border};'
            f' border-radius: 4px;'
            f'}}'
            f' QLabel {{ border: none; background-color: transparent; color: {fg}; }}'
            f' QPushButton {{ border: 1px solid {border}; border-radius: 4px;'
            f'   padding: 4px 12px; background-color: {btn_bg}; color: {fg}; }}'
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        icon = QLabel('ⓘ')
        icon.setStyleSheet(
            f'color: {border}; font-size: 16px; font-weight: bold;'
            ' border: none; background-color: transparent;'
        )
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label, 1)

        self.action_button = QPushButton(action_label)
        self.action_button.clicked.connect(on_action)
        layout.addWidget(self.action_button)

    def set_message(self, message: str) -> None:
        self.message_label.setText(message)


class SuccessBanner(QFrame):
    """Green banner shown after a successful run, with dismiss button."""

    def __init__(
        self,
        message: str,
        actions: list[tuple[str, Callable[[], None]]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)

        bg, border, fg, btn_bg = _banner_colors('#4caf50')
        self.setStyleSheet(
            f'QFrame {{'
            f' background-color: {bg};'
            f' border: 1px solid {border};'
            f' border-radius: 4px;'
            f'}}'
            f' QLabel {{ border: none; background-color: transparent; color: {fg}; }}'
            f' QPushButton {{ border: 1px solid {border}; border-radius: 4px;'
            f'   padding: 4px 12px; background-color: {btn_bg}; color: {fg}; }}'
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        icon = QLabel('✓')
        icon.setStyleSheet(
            f'color: {border}; font-size: 16px; font-weight: bold;'
            ' border: none; background-color: transparent;'
        )
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label, 1)

        for label, callback in actions or []:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            layout.addWidget(btn)

        close_btn = QPushButton('✕')
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            f'QPushButton {{ border: none; color: {fg}; font-weight: bold;'
            f' background-color: transparent; padding: 0; }}'
        )
        close_btn.clicked.connect(self.hide)
        layout.addWidget(close_btn)

    def set_message(self, message: str) -> None:
        self.message_label.setText(message)
