"""ATLAS GUI entry point.

Flow: ``main()`` opens the Hub; on a successful project pick it opens the
main window scoped to that project; closing the main window returns to the
Hub so users can switch projects without restarting.

Run via the ``atl_gui`` script (see ``[project.scripts]``) or
``python -m atlas.core.gui.app``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from atlas.core.gui.app_params import ApplicationParameters
from atlas.core.gui.hub import HubDialog
from atlas.core.gui.main_window import MainWindow
from atlas.core.gui.themes import apply_theme_to_app, saved_global_theme


def _isolate_pyside6_qt():
    """Re-exec with PySide6's bundled Qt libs to avoid system Qt conflicts."""
    if os.environ.get('_ATLAS_QT_ISOLATED'):
        return
    try:
        import PySide6

        qt_lib = os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'lib')
        if not os.path.isdir(qt_lib):
            return
    except ImportError:
        return
    os.environ['LD_LIBRARY_PATH'] = qt_lib
    os.environ['_ATLAS_QT_ISOLATED'] = '1'
    os.execvp(sys.executable, [sys.executable] + sys.argv)


_isolate_pyside6_qt()

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QSystemTrayIcon,
    QVBoxLayout,
)

ASSETS_DIR = Path(__file__).resolve().parent / 'assets'


class _LoadingDialog(QDialog):
    """Splash-style loading screen shown while opening a project."""

    def __init__(self, project_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ATLAS')
        self.setFixedSize(400, 200)
        self.setWindowFlags(
            Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        logo_path = ASSETS_DIR / 'atlas_logo_light.png'
        if logo_path.exists():
            logo_label = QLabel()
            pix = QPixmap(str(logo_path)).scaledToHeight(48, Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        title = QLabel(f'Opening project: {project_name}')
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self._status = QLabel('Initializing...')
        self._status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        layout.addStretch()

    def update_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._status.setText(message)
        QApplication.processEvents()


def _detach() -> None:
    """Fork and let the parent exit so the GUI detaches from the terminal."""
    if os.fork() != 0:
        os._exit(0)
    os.setsid()


def _parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(
        prog='atl_gui',
        description='ATLAS graphical user interface for AL workflows.',
    )
    parser.add_argument(
        '--console',
        action='store_true',
        help='Keep attached to the terminal (do not fork/detach). '
        'Useful for debugging, stdout/stderr remain visible.',
    )
    parser.add_argument(
        '--project',
        '-p',
        metavar='PATH',
        help='Open an ATLAS project (.atlasproj) directly, skipping the Hub.',
    )
    parser.add_argument(
        '--theme',
        '-t',
        metavar='NAME',
        help='Override the application theme on startup '
        '(e.g. "Default (Light)", "Nord (Dark)").',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.console and sys.platform != 'win32':
        _detach()

    qt_argv = [sys.argv[0]]
    app = QApplication.instance() or QApplication(qt_argv)
    application_font = app.font()
    application_font.setFamilies(ApplicationParameters.FONT_FAMILIES_REGULAR)
    app.setFont(application_font)

    from atlas.core.gui.icons import app_icon

    app.setWindowIcon(app_icon())

    if QSystemTrayIcon.isSystemTrayAvailable():
        app.setQuitOnLastWindowClosed(False)

    theme = args.theme or saved_global_theme()
    apply_theme_to_app(theme)

    # If --project given, open directly without the Hub
    initial_project = None
    if args.project:
        from atlas.core.gui.project import Project, ProjectError

        try:
            initial_project = Project.open(args.project)
        except ProjectError as exc:
            print(f'Error opening project: {exc}', file=sys.stderr)
            return 1

    while True:
        if initial_project is not None:
            project = initial_project
            initial_project = None
        else:
            hub = HubDialog()
            if hub.exec() != QDialog.Accepted or hub.selected_project is None:
                break
            project = hub.selected_project

        loading = _LoadingDialog(project.name)
        loading.show()
        QApplication.processEvents()

        main_window = MainWindow(
            project=project,
            application_font=application_font,
            progress_callback=loading.update_progress,
            verbose=args.console,
        )

        loading.close()
        main_window.show()
        main_window._check_first_run()
        app.exec()
        del main_window

    return 0


if __name__ == '__main__':
    sys.exit(main())
