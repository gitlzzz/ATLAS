"""Project overview page — first sidebar entry.

Layout:
* Project metadata at the top.
* A horizontal pipeline tracker (Initial DB → DFT → AL → Reports) showing
  each stage's status and metric; the next recommended stage is visually
  highlighted with a "Start here" badge.  Clicking a card jumps to the
  matching workflow page.
* Indexed data counts.
* Refresh actions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.prereq_banner import PrereqBanner

# Stage key -> (number, display name, navigation key, prereq description).
PIPELINE = [
    (
        'init_db',
        1,
        'Initial Database',
        'init_db',
        'Configure and generate the structure database.',
    ),
    (
        'dft',
        2,
        'DFT Labelling',
        'dft',
        'Label the database with reference DFT calculations.',
    ),
    (
        'al',
        3,
        'Active Learning',
        'al',
        'Train the MLIP and iteratively grow the labelled set.',
    ),
    (
        'reports',
        4,
        'Reports',
        'reports',
        'Inspect training curves, latent space, and benchmarks.',
    ),
]


def _card_style(
    border: str,
    bg: str = 'transparent',
    btn_bg: str = 'palette(mid)',
    btn_border: str = 'palette(mid)',
    btn_fg: str = '',
) -> str:
    btn_color = f' color: {btn_fg};' if btn_fg else ''
    return (
        f'QFrame {{ border: {border}; border-radius: 6px;'
        f' background-color: {bg}; }}'
        ' QLabel { border: none; background-color: transparent; }'
        f' QPushButton {{ border: 1px solid {btn_border}; border-radius: 4px;'
        f'   padding: 4px 12px; background-color: {btn_bg};{btn_color} }}'
    )


def _build_card_styles() -> dict[str, str]:
    from atlas.core.gui.themes import (
        THEMES,
        _mix,
        saved_global_theme,
        theme_variant,
    )

    name = saved_global_theme()
    t = THEMES.get(name)
    is_dark = theme_variant(name) == 'dark'

    if t is not None:
        success = t.c11 if is_dark else t.c03
        warning = t.c12 if is_dark else t.c04
        primary = t.c13 if is_dark else t.c05
        bg = t.background
        fg = t.foreground
        done_bg = _mix(bg, success, 0.10)
        running_bg = _mix(bg, warning, 0.10)
        next_bg = _mix(bg, primary, 0.10)
        btn_surface = _mix(bg, fg, 0.10) if is_dark else _mix(bg, fg, 0.08)
        border = _mix(bg, fg, 0.15) if is_dark else _mix(bg, fg, 0.16)
    else:
        success, warning, primary = '#2e8b57', '#b8860b', '#2962ff'
        done_bg = '#eaf5ef'
        running_bg = '#f5f0e3'
        next_bg = '#e8eeff'
        btn_surface = '#ececec'
        border = '#d0d4db'

    if t is not None:
        next_btn_bg = (
            _mix(bg, primary, 0.25) if is_dark else _mix(primary, '#ffffff', 0.85)
        )
    else:
        next_btn_bg = '#dce6ff'

    return {
        'done': _card_style(
            f'1px solid {success}', done_bg, btn_bg=btn_surface, btn_border=border
        ),
        'running': _card_style(
            f'1px solid {warning}', running_bg, btn_bg=btn_surface, btn_border=border
        ),
        'partial': _card_style(
            f'1px solid {warning}', running_bg, btn_bg=btn_surface, btn_border=border
        ),
        'empty': _card_style(
            '1px solid palette(mid)', btn_bg=btn_surface, btn_border=border
        ),
        'next': _card_style(
            f'2px solid {primary}', next_bg, btn_bg=next_btn_bg, btn_border=primary
        ),
    }


STATUS_ICON = {
    'done': '✓',
    'running': '⚙',
    'partial': '◐',
    'empty': '○',
}

STATUS_LABEL = {
    'done': 'Done',
    'running': 'Running',
    'partial': 'In progress',
    'empty': 'Not started',
}


class OverviewPage(WorkflowPage):
    """Project metadata, milestone tracker, and refresh actions."""

    DISPLAY_NAME = 'Overview'
    NAVIGATION_KEY = 'overview'

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

        self._cards: dict[str, dict] = {}

        outer = QVBoxLayout(self)

        self._setup_banner = PrereqBanner(
            'ATLAS initial setup is incomplete.',
            'Go to Settings',
            lambda: self._navigate('settings'),
        )
        self._setup_banner.setVisible(False)
        outer.addWidget(self._setup_banner)

        outer.addWidget(self._build_meta_box())
        outer.addWidget(self._build_pipeline_box())
        outer.addWidget(self._build_counts_box())
        outer.addWidget(self._build_actions_box())
        outer.addStretch(1)

        self.refresh()

    def on_shown(self) -> None:
        self.refresh()

    # ============================================================ widgets

    def _build_meta_box(self) -> QGroupBox:
        box = QGroupBox('Project')
        form = QFormLayout(box)
        form.addRow('Name', QLabel(self.project.name))
        form.addRow('Bundle', _path_label(self.project.path))
        form.addRow('Directory', _path_label(self.project.dir))
        form.addRow(
            'Created',
            QLabel(self.project.meta('created_at', 'unknown')),
        )
        form.addRow(
            'Last opened',
            QLabel(self.project.meta('last_opened_at', 'unknown')),
        )
        return box

    def _build_pipeline_box(self) -> QGroupBox:
        box = QGroupBox('Workflow Progress')
        layout = QHBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        for idx, (key, number, name, nav_key, hint) in enumerate(PIPELINE):
            card = self._build_card(key, number, name, nav_key, hint)
            layout.addWidget(card, 1)
            if idx < len(PIPELINE) - 1:
                layout.addWidget(_arrow_label(), 0)
        return box

    def _build_card(
        self,
        key: str,
        number: int,
        name: str,
        nav_key: str,
        hint: str,
    ) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        frame.setMinimumHeight(150)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QLabel(f'<b>{number}. {name}</b>')
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)

        status_label = QLabel('—')
        status_label.setTextFormat(Qt.RichText)
        layout.addWidget(status_label)

        metric_label = QLabel('—')
        metric_label.setWordWrap(True)
        layout.addWidget(metric_label)

        hint_label = QLabel(hint)
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        layout.addStretch(1)

        button = QPushButton('Open')
        button.clicked.connect(lambda _checked=False, nk=nav_key: self._navigate(nk))
        layout.addWidget(button)

        self._cards[key] = {
            'frame': frame,
            'status_label': status_label,
            'metric_label': metric_label,
            'button': button,
        }
        return frame

    def _build_counts_box(self) -> QGroupBox:
        box = QGroupBox('Indexed Data')
        layout = QVBoxLayout(box)

        self.structures_label = QLabel('Structures: …')
        layout.addWidget(self.structures_label)

        self.dft_label = QLabel('DFT runs: …')
        layout.addWidget(self.dft_label)

        self.al_label = QLabel('AL runs: …')
        layout.addWidget(self.al_label)

        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox('Refresh')
        layout = QHBoxLayout(box)

        struct_btn = QPushButton('Refresh structures from disk')
        struct_btn.setToolTip(
            'Re-scan the project’s initial database (.xz) and update the '
            'structures index.'
        )
        struct_btn.clicked.connect(self._on_refresh_structures)
        layout.addWidget(struct_btn)

        self._aiida_btn = QPushButton('Refresh AL / DFT from AiiDA')
        self._aiida_btn.setToolTip(
            'Query the active AiiDA profile for the workchains tracked in this '
            'project and refresh their cached summaries.'
        )
        self._aiida_btn.clicked.connect(self._on_refresh_aiida)
        layout.addWidget(self._aiida_btn)

        layout.addStretch()
        return box

    # =========================================================== refresh

    def refresh(self) -> None:
        """Recompute counts and pipeline tracker. Safe to call repeatedly."""
        self._refresh_setup_banner()
        self._refresh_counts()
        self._refresh_pipeline()

    def _refresh_setup_banner(self) -> None:
        from atlas.core.gui.widgets.setup_wizard import check_setup_problems

        problems = check_setup_problems()
        if problems:
            self._setup_banner.set_message(
                'ATLAS initial setup is incomplete: '
                + '; '.join(problems)
                + '. Open Settings to run the setup wizard.'
            )
            self._setup_banner.setVisible(True)
        else:
            self._setup_banner.setVisible(False)

    def _refresh_counts(self) -> None:
        counts = self.project.structure_counts()
        self.structures_label.setText(
            f'Structures: {counts["total"]} total · '
            f'{counts["labelled"]} labelled · '
            f'{counts["unlabelled"]} unlabelled'
        )

        dft_counts = self.project.dft_run_counts()
        if dft_counts:
            parts = ' · '.join(f'{k}: {v}' for k, v in sorted(dft_counts.items()))
        else:
            parts = 'none recorded'
        self.dft_label.setText(f'DFT runs: {parts}')

        al_runs = self.project.list_al_runs()
        active_states = ('submitted', 'running')
        running = sum(1 for r in al_runs if (r.get('status') or '') in active_states)
        self.al_label.setText(f'AL runs: {len(al_runs)} total · {running} active')

    def _refresh_pipeline(self) -> None:
        state = self.project.workflow_state()
        next_key = state['next_recommended']
        card_styles = _build_card_styles()

        for key, _number, _name, _nav, _hint in PIPELINE:
            stage = state['stages'][key]
            card = self._cards[key]

            is_next = key == next_key
            style_key = 'next' if is_next else stage['status']
            card['frame'].setStyleSheet(card_styles[style_key])

            icon = STATUS_ICON[stage['status']]
            label = STATUS_LABEL[stage['status']]
            if is_next:
                status_text = f'<b>→ Start here</b> ({label})'
                card['button'].setText('Start →')
            else:
                status_text = f'<b>{icon} {label}</b>'
                card['button'].setText('Open')

            card['status_label'].setText(status_text)
            card['metric_label'].setText(stage['metric'])

    # =========================================================== actions

    def _on_refresh_structures(self) -> None:
        try:
            count = self.project.refresh_structures_index()
        except Exception as exc:
            QMessageBox.critical(self, 'Refresh failed', str(exc))
            return
        self._log(f'🔁 Refreshed structures index: {count} entries.')
        self.refresh()

    def _on_refresh_aiida(self) -> None:
        if hasattr(self, '_aiida_worker') and self._aiida_worker.isRunning():
            return
        self._aiida_btn.setEnabled(False)
        self._aiida_btn.setText('Syncing with AiiDA…')
        self._aiida_worker = _AiidaSyncWorker(self.project)
        self._aiida_worker.finished_signal.connect(self._on_aiida_sync_done)
        self._aiida_worker.start()

    def _on_aiida_sync_done(self, summary: str, errors: str) -> None:
        self._aiida_btn.setEnabled(True)
        self._aiida_btn.setText('Refresh AL / DFT from AiiDA')
        if errors:
            self._log(f'⚠ AiiDA sync: {errors}')
        if summary:
            self._log(f'🔁 AiiDA sync: {summary}')
        self.refresh()
        self.workflow_state_changed.emit()


class _AiidaSyncWorker(QThread):
    """Run AiiDA queries off the main thread."""

    finished_signal = Signal(str, str)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            from atlas.core.gui.project.aiida_sync import sync_all

            result = sync_all(self._project)
            errors = '\n'.join(result.errors) if result.errors else ''
            self.finished_signal.emit(result.summary, errors)
        except Exception as exc:
            self.finished_signal.emit('', str(exc))


def _path_label(path) -> QLabel:
    label = QLabel(str(path))
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setWordWrap(True)
    return label


def _arrow_label() -> QLabel:
    arrow = QLabel('→')
    arrow.setAlignment(Qt.AlignCenter)
    arrow.setStyleSheet('font-size: 20px;')
    arrow.setMinimumWidth(24)
    arrow.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    return arrow
