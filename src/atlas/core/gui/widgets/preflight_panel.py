"""Pre-flight check panel for workflow pages.

Shows a compact list of readiness checks (AiiDA profile, required config
fields, etc.) that update live as the user edits the configuration.
Each check is a single row with a pass/fail/skipped icon and a description.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class Check:
    """A single pre-flight check."""

    label: str
    description: str
    run: Callable[[], str | None]
    requires_aiida: bool = field(default=False)
    field_key: str = field(default='')


def _theme_colors():
    from atlas.core.gui.themes import THEMES, _mix, saved_global_theme, theme_variant

    name = saved_global_theme()
    t = THEMES.get(name)
    is_dark = theme_variant(name) == 'dark'
    if t is not None:
        bg = t.background
        fg = t.foreground
    else:
        bg = '#ffffff'
        fg = '#1a1a2e'
    return bg, fg, is_dark, _mix


class PreflightPanel(QFrame):
    """Collapsible pre-flight checklist.

    Parameters
    ----------
    title
        Header text (e.g. "Pre-flight checks").
    checks
        List of ``Check`` objects.  Each ``run`` callable returns ``None``
        on success or an error string on failure.  Checks with
        ``requires_aiida=True`` are skipped (shown greyed) when the first
        check (AiiDA profile) fails.
    """

    def __init__(
        self,
        title: str,
        checks: list[Check],
        parent: QWidget | None = None,
        focus_field: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self._checks = checks
        self._rows: list[tuple[QLabel, QLabel, QLabel]] = []
        self._expanded = False
        self._title = title
        self._has_errors = False
        self._focus_field = focus_field

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 8, 4)
        self._toggle_btn = QPushButton('▸ ' + title)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            'QPushButton { text-align: left; border: none; padding: 2px 0;'
            ' background: transparent; }'
        )
        self._toggle_btn.clicked.connect(self._toggle)
        header.addWidget(self._toggle_btn, 1)

        self._summary_label = QLabel()
        self._summary_label.setContentsMargins(0, 0, 8, 0)
        header.addWidget(self._summary_label)

        self._recheck_btn = QPushButton('Re-check')
        self._recheck_btn.setFixedHeight(24)
        self._recheck_btn.setStyleSheet('QPushButton { padding: 2px 8px; }')
        self._recheck_btn.clicked.connect(self.run_checks)
        header.addWidget(self._recheck_btn)
        outer.addLayout(header)

        self._body = QWidget()
        self._body.setStyleSheet('background: transparent;')
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 0, 8, 8)
        body_layout.setSpacing(2)

        self._grid = QGridLayout()
        self._grid.setColumnStretch(2, 1)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(2)
        for i, check in enumerate(checks):
            icon_lbl = QLabel()
            icon_lbl.setFixedWidth(20)
            icon_lbl.setAlignment(Qt.AlignCenter)
            name_lbl = QLabel(check.label)
            detail_lbl = QLabel(check.description)
            detail_lbl.setWordWrap(True)
            self._grid.addWidget(icon_lbl, i, 0)
            self._grid.addWidget(name_lbl, i, 1)
            self._grid.addWidget(detail_lbl, i, 2)
            self._rows.append((icon_lbl, name_lbl, detail_lbl))

        body_layout.addLayout(self._grid)
        outer.addWidget(self._body)
        self._body.setVisible(False)

        self._apply_panel_style()

    @property
    def has_errors(self) -> bool:
        return self._has_errors

    def _apply_panel_style(self) -> None:
        bg, _fg, _is_dark, _mix = _theme_colors()
        if self._has_errors:
            accent = '#ef5350'
            panel_bg = _mix(bg, accent, 0.08)
            border = _mix(bg, accent, 0.25)
        else:
            accent = '#5c6bc0'
            panel_bg = _mix(bg, accent, 0.06)
            border = _mix(bg, accent, 0.18)
        self.setStyleSheet(
            f'PreflightPanel {{'
            f'  background-color: {panel_bg};'
            f'  border: 1px solid {border};'
            f'  border-radius: 4px;'
            f'}}'
        )

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        arrow = '▾' if self._expanded else '▸'
        self._toggle_btn.setText(f'{arrow} {self._title}')

    def _set_row_pass(self, icon_lbl, name_lbl, detail_lbl, check):
        icon_lbl.setText('✓')
        icon_lbl.setStyleSheet('color: #4caf50; border: none; background: transparent;')
        name_lbl.setEnabled(True)
        detail_lbl.setText(check.description)
        detail_lbl.setStyleSheet(
            'color: palette(mid); border: none; background: transparent;'
        )

    def _set_row_fail(self, icon_lbl, name_lbl, detail_lbl, err, check=None):
        icon_lbl.setText('✕')
        icon_lbl.setStyleSheet('color: #ef5350; border: none; background: transparent;')
        name_lbl.setEnabled(True)

        if check and check.field_key and self._focus_field:
            fk = check.field_key
            detail_lbl.setText(
                f'{err}  <a href="{fk}" style="color: #42a5f5;">Go to field</a>'
            )
            detail_lbl.setTextFormat(Qt.RichText)
            detail_lbl.setOpenExternalLinks(False)
            if not detail_lbl.property('_link_connected'):
                detail_lbl.linkActivated.connect(self._on_link_clicked)
                detail_lbl.setProperty('_link_connected', True)
        else:
            detail_lbl.setText(err)

        detail_lbl.setStyleSheet(
            'color: #ef5350; border: none; background: transparent;'
        )

    def _on_link_clicked(self, field_key: str) -> None:
        if self._focus_field:
            self._focus_field(field_key)

    def _set_row_skipped(self, icon_lbl, name_lbl, detail_lbl):
        icon_lbl.setText('—')
        icon_lbl.setStyleSheet(
            'color: palette(mid); border: none; background: transparent;'
        )
        name_lbl.setEnabled(False)
        detail_lbl.setText('Requires AiiDA profile')
        detail_lbl.setStyleSheet(
            'color: palette(mid); border: none; background: transparent;'
        )

    def run_checks(self) -> list[str]:
        """Execute all checks and update the UI.  Returns error strings."""
        errors: list[str] = []
        aiida_ok = True

        for check, (icon_lbl, name_lbl, detail_lbl) in zip(
            self._checks,
            self._rows,
            strict=True,
        ):
            if check.requires_aiida and not aiida_ok:
                self._set_row_skipped(icon_lbl, name_lbl, detail_lbl)
                continue

            err = check.run()
            if err is None:
                self._set_row_pass(icon_lbl, name_lbl, detail_lbl, check)
            else:
                self._set_row_fail(icon_lbl, name_lbl, detail_lbl, err, check)
                errors.append(err)
                if not check.requires_aiida and check is self._checks[0]:
                    aiida_ok = False

        n_pass = len(self._checks) - len(errors)
        n_skipped = sum(1 for c in self._checks if c.requires_aiida and not aiida_ok)
        n_total = len(self._checks)

        self._has_errors = bool(errors)
        self._apply_panel_style()

        if errors:
            parts = [f'{n_pass}/{n_total} passed']
            if n_skipped:
                parts.append(f'{n_skipped} skipped')
            self._summary_label.setText(', '.join(parts))
            self._summary_label.setStyleSheet(
                'color: #ef5350; border: none; background: transparent;'
            )
        else:
            self._summary_label.setText(f'{n_total}/{n_total} passed')
            self._summary_label.setStyleSheet(
                'color: #4caf50; border: none; background: transparent;'
            )
        return errors

    def failing_summary(self) -> str | None:
        """Return a human-readable summary of failing checks, or None."""
        if not self._has_errors:
            return None
        lines: list[str] = []
        for check, (icon_lbl, _name_lbl, detail_lbl) in zip(
            self._checks,
            self._rows,
            strict=True,
        ):
            if icon_lbl.text() == '✕':
                lines.append(f'  • {check.label}: {detail_lbl.text()}')
        return '\n'.join(lines)
