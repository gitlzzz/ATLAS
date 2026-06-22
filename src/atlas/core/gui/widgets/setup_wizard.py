"""GUI setup wizard, mirrors atl_init_setup in a QWizard."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import pathlib
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

# ── helpers ────────────────────────────────────────────────────────


def _config_dir() -> pathlib.Path:
    from atlas.core.code_utils import get_config_path

    return get_config_path() / 'atl'


def _secrets_path() -> pathlib.Path:
    return _config_dir() / 'secrets.json'


def _read_mp_key() -> str:
    path = _secrets_path()
    if path.exists():
        try:
            return json.loads(path.read_text()).get('API_KEY', '')
        except Exception:
            pass
    return os.environ.get('MP_API_KEY', '')


def _write_mp_key(key: str) -> None:
    path = _secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        with contextlib.suppress(Exception):
            data = json.loads(path.read_text())
    data['API_KEY'] = key
    path.write_text(json.dumps(data, indent=2) + '\n')
    os.chmod(path, 0o600)


def _aiida_available() -> bool:
    try:
        import aiida  # noqa: F401

        return True
    except ImportError:
        return False


def _list_profiles() -> tuple[list[str], str | None]:
    try:
        from aiida.manage.configuration import get_config

        config = get_config()
        return list(config.profile_names), config.default_profile_name
    except Exception:
        return [], None


def _current_profile() -> str | None:
    try:
        from aiida.manage.configuration import get_profile

        p = get_profile()
        if p is not None:
            return p.name
        # No profile loaded yet, try loading the default.
        profiles, default = _list_profiles()
        if default:
            from aiida import load_profile

            load_profile(default)
            p = get_profile()
            return p.name if p else None
        return None
    except Exception:
        return None


def _broker_configured() -> bool | None:
    """True if broker is set, False if not, None if can't determine."""
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        if profile is None:
            return None
        return profile.process_control_backend is not None
    except Exception:
        return None


def _list_computers() -> list[str]:
    try:
        from aiida import orm

        qb = orm.QueryBuilder()
        qb.append(orm.Computer, project=['label'])
        qb.distinct()
        return sorted(row[0] for row in qb.all())
    except Exception:
        return []


def _list_codes() -> list[str]:
    try:
        from aiida import orm

        qb = orm.QueryBuilder()
        qb.append(orm.Code, project=['label'])
        qb.distinct()
        return sorted(row[0] for row in qb.all())
    except Exception:
        return []


def _list_potential_families() -> list[str]:
    try:
        from aiida import orm

        qb = orm.QueryBuilder()
        qb.append(orm.Group, project=['label'])
        qb.distinct()
        return sorted(row[0] for row in qb.all())
    except Exception:
        return []


def _parse_ssh_config() -> dict[str, dict[str, str]]:
    """Parse ~/.ssh/config and return {host_alias: {key: value}}."""
    config_path = pathlib.Path.home() / '.ssh' / 'config'
    if not config_path.exists():
        return {}
    hosts: dict[str, dict[str, str]] = {}
    current_host: str | None = None
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        match = re.match(r'^(\S+)\s+(.+)$', stripped)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        if key == 'host':
            if '*' in value or '?' in value:
                current_host = None
            else:
                current_host = value
                hosts[current_host] = {}
        elif current_host is not None:
            hosts[current_host][key] = value
    return hosts


def needs_first_run_wizard() -> bool:
    """Return True if ATLAS has never been set up (no secrets.json)."""
    return not _secrets_path().exists()


def check_setup_problems() -> list[str]:
    """Return a list of human-readable setup issues (empty = all OK)."""
    problems: list[str] = []
    if not _read_mp_key():
        problems.append('Materials Project API key is not configured')
    if not _aiida_available():
        problems.append('AiiDA is not installed')
    elif not _current_profile():
        problems.append('No AiiDA profile is loaded')
    else:
        if _broker_configured() is False:
            problems.append(
                'No message broker (RabbitMQ) configured, workflow submission will fail'
            )
        if not _list_computers():
            problems.append('No compute resources configured in AiiDA')
        if not _list_codes():
            problems.append('No simulation codes configured in AiiDA')
    return problems


# ── Helpers ────────────────────────────────────────────────────────


class _BashHighlighter(QSyntaxHighlighter):
    """Minimal bash syntax highlighter for prepend/append text editors."""

    _KEYWORDS = (
        'module',
        'export',
        'source',
        'load',
        'unload',
        'if',
        'then',
        'else',
        'fi',
        'for',
        'do',
        'done',
        'echo',
        'cd',
        'set',
    )

    def highlightBlock(self, text: str) -> None:
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(Qt.darkGreen)
        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(Qt.darkBlue)
        keyword_fmt.setFontWeight(700)

        stripped = text.lstrip()
        if stripped.startswith('#'):
            self.setFormat(0, len(text), comment_fmt)
            return

        for word in self._KEYWORDS:
            idx = 0
            while True:
                idx = text.find(word, idx)
                if idx < 0:
                    break
                end = idx + len(word)
                before_ok = idx == 0 or not text[idx - 1].isalnum()
                after_ok = end >= len(text) or not text[end].isalnum()
                if before_ok and after_ok:
                    self.setFormat(idx, len(word), keyword_fmt)
                idx = end


# ── Wizard pages ───────────────────────────────────────────────────


class _WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Welcome to ATLAS')
        self.setSubTitle(
            'This wizard will help you configure the essential components '
            'for running ATLAS workflows.\n\n'
            'You can re-run this wizard at any time from the Settings page.'
        )
        layout = QVBoxLayout(self)
        info = QLabel(
            'The wizard will guide you through:\n\n'
            '  1. Materials Project API key\n'
            '  2. AiiDA profile selection\n'
            '  3. Message broker (RabbitMQ) check\n'
            '  4. AiiDA computer setup\n'
            '  5. AiiDA code registration\n'
            '  6. VASP potential family upload\n\n'
            'Steps 2–6 require AiiDA and can be skipped if you only\n'
            'need database generation.\n\n'
            'Note: AiiDA settings (profiles, computers, codes, and potentials)\n'
            'are stored in your AiiDA installation and persist across all\n'
            'ATLAS projects and sessions. You only need to configure them once.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()


class _MpKeyPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Materials Project API Key')
        self.setSubTitle('Enter your MP API key for querying crystal structures.')
        layout = QFormLayout(self)

        link = QLabel(
            'Get one at: '
            '<a href="https://next-gen.materialsproject.org/api">'
            'next-gen.materialsproject.org/api</a>'
        )
        link.setOpenExternalLinks(True)
        link.setTextFormat(Qt.RichText)
        layout.addRow('', link)

        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.Password)
        self._key_edit.setMinimumWidth(350)
        existing = _read_mp_key()
        if existing:
            self._key_edit.setText(existing)
        layout.addRow('API key', self._key_edit)

        vis_row = QHBoxLayout()
        self._toggle_btn = QPushButton('Show')
        self._toggle_btn.setFixedWidth(60)
        self._toggle_btn.clicked.connect(self._toggle_visibility)
        vis_row.addWidget(self._toggle_btn)
        vis_row.addStretch()
        layout.addRow('', vis_row)

        self._status = QLabel()
        layout.addRow('', self._status)
        self._update_status()

    def _toggle_visibility(self) -> None:
        if self._key_edit.echoMode() == QLineEdit.Password:
            self._key_edit.setEchoMode(QLineEdit.Normal)
            self._toggle_btn.setText('Hide')
        else:
            self._key_edit.setEchoMode(QLineEdit.Password)
            self._toggle_btn.setText('Show')

    def _update_status(self) -> None:
        if _secrets_path().exists():
            self._status.setText('Key already configured.')
            self._status.setStyleSheet('color: #4caf50;')
        else:
            self._status.setText('')

    def validatePage(self) -> bool:
        key = self._key_edit.text().strip()
        if not key:
            QMessageBox.warning(
                self, 'API Key', 'Please enter an API key or press Skip.'
            )
            return False
        try:
            _write_mp_key(key)
        except Exception as exc:
            QMessageBox.critical(self, 'Save Failed', str(exc))
            return False
        return True


class _AiidaProfilePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('AiiDA Profile')
        self.setSubTitle(
            'Select an existing AiiDA profile or create one.\n'
            'Skip this step if you only need database generation.'
        )
        layout = QVBoxLayout(self)

        if not _aiida_available():
            layout.addWidget(
                QLabel(
                    'AiiDA is not installed in this environment.\n'
                    'Skip this step or install aiida-core first.'
                )
            )
            self._combo = None
            layout.addStretch()
            return

        self._combo = QComboBox()
        self._combo.setMinimumWidth(300)

        self._status = QLabel()

        form = QFormLayout()
        form.addRow('Profile', self._combo)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(refresh_btn)

        presto_btn = QPushButton('Quick Setup (verdi presto)')
        presto_btn.setToolTip(
            'Create a basic AiiDA profile using "verdi presto".\n'
            'Good for getting started quickly with a local setup.'
        )
        presto_btn.clicked.connect(self._run_verdi_presto)
        btn_row.addWidget(presto_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        docs_link = QLabel(
            'For advanced setups, see the '
            '<a href="https://aiida.readthedocs.io/projects/aiida-core/en/latest/'
            'installation/guide_quick.html">AiiDA documentation</a>.'
        )
        docs_link.setOpenExternalLinks(True)
        docs_link.setTextFormat(Qt.RichText)
        layout.addWidget(docs_link)

        layout.addWidget(self._status)
        self._refresh()
        layout.addStretch()

    def _refresh(self) -> None:
        if self._combo is None:
            return
        self._combo.clear()
        profiles, default = _list_profiles()
        if profiles:
            self._combo.addItems(profiles)
            current = _current_profile()
            if current and current in profiles:
                self._combo.setCurrentText(current)
            elif default:
                self._combo.setCurrentText(default)
        else:
            self._combo.addItem('(no profiles found)')
        self._update_status()

    def _update_status(self) -> None:
        current = _current_profile()
        if current:
            self._status.setText(f'Active profile: {current}')
            self._status.setStyleSheet('color: #4caf50;')
        else:
            self._status.setText('No profile loaded')
            self._status.setStyleSheet('color: palette(mid);')

    def _run_verdi_presto(self) -> None:
        import subprocess

        self._status.setText('Running "verdi presto"...')
        self._status.setStyleSheet('color: palette(mid);')
        self._status.repaint()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            result = subprocess.run(
                ['verdi', 'presto'],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                self._status.setText('Profile created successfully.')
                self._status.setStyleSheet('color: #4caf50;')
                self._refresh()
            else:
                msg = result.stderr.strip() or result.stdout.strip()
                QMessageBox.critical(self, 'verdi presto', msg or 'Unknown error.')
                self._status.setText('Profile creation failed.')
                self._status.setStyleSheet('color: #ef5350;')
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                'verdi presto',
                '"verdi" command not found.\nIs aiida-core installed?',
            )
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, 'verdi presto', 'Command timed out (2 min).')

    def validatePage(self) -> bool:
        if self._combo is None:
            return True
        name = self._combo.currentText()
        if name.startswith('('):
            return True
        try:
            from aiida import load_profile

            load_profile(name, allow_switch=True)
        except Exception as exc:
            QMessageBox.warning(
                self,
                'Profile Error',
                f'Could not load profile "{name}": {exc}',
            )
            return False
        self._update_status()
        return True


class _BrokerPage(QWizardPage):
    """Check and configure the RabbitMQ message broker for AiiDA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Message Broker (RabbitMQ)')
        self.setSubTitle(
            'AiiDA uses RabbitMQ as a message broker to manage workflow\n'
            'submissions to remote computers. Without it, calculations\n'
            'cannot be submitted.\n'
            'You can skip this step and configure it later.'
        )
        layout = QVBoxLayout(self)

        if not _aiida_available():
            layout.addWidget(
                QLabel(
                    'AiiDA is not installed in this environment.\n'
                    'Skip this step or install aiida-core first.'
                )
            )
            self._status = None
            layout.addStretch()
            return

        info = QLabel(
            'If you created your profile with "verdi presto", RabbitMQ\n'
            'is likely not configured. To install RabbitMQ:\n\n'
            '  Ubuntu/Debian:  sudo apt install rabbitmq-server\n'
            '  macOS:          brew install rabbitmq\n'
            '  conda:          conda install -c conda-forge rabbitmq-server'
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._status = QLabel()
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        configure_btn = QPushButton('Configure RabbitMQ')
        configure_btn.setToolTip(
            'Run "verdi profile configure-rabbitmq" to connect\n'
            'the current profile to a running RabbitMQ server.'
        )
        configure_btn.clicked.connect(self._run_configure_rabbitmq)
        btn_row.addWidget(configure_btn)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._check_broker)
        btn_row.addWidget(refresh_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        docs_link = QLabel(
            'For details, see the '
            '<a href="https://aiida.readthedocs.io/projects/aiida-core/en/latest/'
            'installation/guide_quick.html#quick-install-limitations">'
            'AiiDA broker documentation</a>.'
        )
        docs_link.setOpenExternalLinks(True)
        docs_link.setTextFormat(Qt.RichText)
        layout.addWidget(docs_link)

        layout.addStretch()

    def initializePage(self) -> None:
        self._check_broker()

    def _check_broker(self) -> None:
        if self._status is None:
            return
        result = _broker_configured()
        if result is True:
            self._status.setText('Message broker is configured.')
            self._status.setStyleSheet('color: #4caf50;')
        elif result is False:
            self._status.setText(
                'No message broker configured for this profile.\n'
                'Click "Configure RabbitMQ" if RabbitMQ is installed and running.'
            )
            self._status.setStyleSheet('color: #ef5350;')
        else:
            self._status.setText('No AiiDA profile loaded.')
            self._status.setStyleSheet('color: palette(mid);')

    def _run_configure_rabbitmq(self) -> None:
        import subprocess

        self._status.setText('Configuring RabbitMQ...')
        self._status.setStyleSheet('color: palette(mid);')
        self._status.repaint()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            result = subprocess.run(
                ['verdi', 'profile', 'configure-rabbitmq'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                self._status.setText('RabbitMQ configured successfully.')
                self._status.setStyleSheet('color: #4caf50;')
            else:
                msg = result.stderr.strip() or result.stdout.strip()
                QMessageBox.critical(
                    self,
                    'RabbitMQ Configuration',
                    msg or 'Unknown error. Is RabbitMQ installed and running?',
                )
                self._status.setText('Configuration failed.')
                self._status.setStyleSheet('color: #ef5350;')
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                'RabbitMQ Configuration',
                '"verdi" command not found.\nIs aiida-core installed?',
            )
        except subprocess.TimeoutExpired:
            QMessageBox.critical(
                self, 'RabbitMQ Configuration', 'Command timed out (30 seconds).'
            )

    def validatePage(self) -> bool:
        return True


class _ComputerPage(QWizardPage):
    """Add an AiiDA computer (optionally pre-filled from SSH config)."""

    TRANSPORT_CHOICES = ['core.ssh', 'core.local']
    SCHEDULER_CHOICES = ['core.slurm', 'core.sge', 'core.pbspro', 'core.direct']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('AiiDA Computer')
        self.setSubTitle(
            'Configure a remote or local computer for running calculations.\n'
            'This step is optional, skip if not needed now.'
        )
        layout = QVBoxLayout(self)

        if not _aiida_available():
            layout.addWidget(QLabel('AiiDA is not installed. Skipping.'))
            self._form = None
            layout.addStretch()
            return

        # Skip checkbox at top
        self._skip_check = QCheckBox('Skip this step')
        self._skip_check.setChecked(True)
        self._skip_check.toggled.connect(self._on_skip_toggled)
        layout.addWidget(self._skip_check)

        # Content container
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 8, 0, 0)

        # SSH config import
        self._ssh_hosts = _parse_ssh_config()
        self._ssh_combo = None
        if self._ssh_hosts:
            ssh_group = QGroupBox('Import from SSH config')
            ssh_layout = QHBoxLayout(ssh_group)
            self._ssh_combo = QComboBox()
            self._ssh_combo.addItem('(manual entry)')
            self._ssh_combo.addItems(list(self._ssh_hosts.keys()))
            self._ssh_combo.currentTextChanged.connect(self._on_ssh_selected)
            ssh_layout.addWidget(QLabel('SSH host:'))
            ssh_layout.addWidget(self._ssh_combo, 1)
            content_layout.addWidget(ssh_group)

        # Form
        self._form = QFormLayout()
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText('e.g. my_cluster')
        self._label_edit.setToolTip('A unique name to identify this computer in AiiDA.')
        self._form.addRow('Label', self._label_edit)

        self._hostname_edit = QLineEdit()
        self._hostname_edit.setPlaceholderText('e.g. login.cluster.edu')
        self._hostname_edit.setToolTip(
            'The hostname or IP address of the remote machine.'
        )
        self._form.addRow('Hostname', self._hostname_edit)

        self._transport_combo = QComboBox()
        self._transport_combo.addItems(self.TRANSPORT_CHOICES)
        self._transport_combo.setToolTip(
            'How AiiDA connects to the computer.\n'
            'Use "core.ssh" for remote machines, "core.local" for localhost.'
        )
        self._form.addRow('Transport', self._transport_combo)

        self._scheduler_combo = QComboBox()
        self._scheduler_combo.addItems(self.SCHEDULER_CHOICES)
        self._scheduler_combo.setToolTip(
            'The job scheduler on the remote machine.\n'
            'Common choices: core.slurm (Slurm), core.sge (SGE), core.pbspro (PBS).'
        )
        self._form.addRow('Scheduler', self._scheduler_combo)

        self._workdir_edit = QLineEdit(f'/scratch/{getpass.getuser()}/aiida/')
        self._workdir_edit.setToolTip(
            'Remote directory where AiiDA will run calculations.\n'
            'Must be writable by your user on the remote machine.'
        )
        self._form.addRow('Work directory', self._workdir_edit)

        self._mpirun_edit = QLineEdit('mpirun -np {tot_num_mpiprocs}')
        self._mpirun_edit.setToolTip(
            'Command to launch MPI processes.\n'
            '{tot_num_mpiprocs} is replaced by the number of MPI processes at runtime.'
        )
        self._form.addRow('mpirun command', self._mpirun_edit)

        self._prepend_text = ''
        self._prepend_btn = QPushButton('Edit Prepend Text...')
        self._prepend_btn.setToolTip(
            'Shell commands run before every job on this computer.\n'
            'Typically used for "module load" commands.'
        )
        self._prepend_btn.clicked.connect(
            lambda: self._open_text_editor('Prepend Text', '_prepend_text')
        )
        self._form.addRow('Prepend text', self._prepend_btn)

        self._append_text = ''
        self._append_btn = QPushButton('Edit Append Text...')
        self._append_btn.setToolTip(
            'Shell commands run after every job on this computer.\n'
            'Use for cleanup commands or environment teardown.'
        )
        self._append_btn.clicked.connect(
            lambda: self._open_text_editor('Append Text', '_append_text')
        )
        self._form.addRow('Append text', self._append_btn)

        content_layout.addLayout(self._form)

        self._status = QLabel()
        content_layout.addWidget(self._status)

        layout.addWidget(self._content)
        layout.addStretch()

        # Show existing
        existing = _list_computers()
        if existing:
            self._status.setText(f'Existing computers: {", ".join(existing)}')
            self._status.setStyleSheet('color: palette(mid);')

        from PySide6.QtWidgets import QGraphicsOpacityEffect

        self._opacity = QGraphicsOpacityEffect(self._content)
        self._opacity.setOpacity(0.35)
        self._content.setGraphicsEffect(self._opacity)
        self._content.setEnabled(False)

    def _on_skip_toggled(self, checked: bool) -> None:
        self._content.setEnabled(not checked)
        self._opacity.setOpacity(0.35 if checked else 1.0)

    def _open_text_editor(self, title: str, attr: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumSize(500, 300)
        layout = QVBoxLayout(dlg)

        editor = QTextEdit()
        editor.setPlainText(getattr(self, attr, ''))
        editor.setStyleSheet('font-family: monospace; font-size: 12px;')
        _BashHighlighter(editor.document())
        layout.addWidget(editor)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.Accepted:
            setattr(self, attr, editor.toPlainText())
            text = getattr(self, attr)
            btn = self._prepend_btn if attr == '_prepend_text' else self._append_btn
            if text.strip():
                first_line = text.strip().splitlines()[0]
                btn.setText(
                    f'{title}: {first_line[:40]}...'
                    if len(first_line) > 40
                    else f'{title}: {first_line}'
                )
            else:
                btn.setText(f'Edit {title}...')

    def _on_ssh_selected(self, alias: str) -> None:
        if alias == '(manual entry)' or alias not in self._ssh_hosts:
            return
        opts = self._ssh_hosts[alias]
        self._label_edit.setText(alias)
        self._hostname_edit.setText(opts.get('hostname', ''))
        user = opts.get('user', '')
        if user:
            self._workdir_edit.setText(f'/scratch/{user}/aiida/')
        self._skip_check.setChecked(False)

    def validatePage(self) -> bool:
        if self._form is None or self._skip_check.isChecked():
            return True

        label = self._label_edit.text().strip()
        hostname = self._hostname_edit.text().strip()
        if not label:
            QMessageBox.warning(self, 'Computer', 'Label cannot be empty.')
            return False
        if not hostname:
            QMessageBox.warning(self, 'Computer', 'Hostname cannot be empty.')
            return False

        existing = _list_computers()
        if label in existing:
            QMessageBox.information(
                self,
                'Computer',
                f'Computer "{label}" already exists.',
            )
            return True

        try:
            from aiida import orm

            computer = orm.Computer(
                label=label,
                hostname=hostname,
                transport_type=self._transport_combo.currentText(),
                scheduler_type=self._scheduler_combo.currentText(),
                workdir=self._workdir_edit.text().strip(),
            )
            computer.set_mpirun_command(self._mpirun_edit.text().split())
            if self._prepend_text.strip():
                computer.set_prepend_text(self._prepend_text.strip())
            if self._append_text.strip():
                computer.set_append_text(self._append_text.strip())
            computer.store()

            # Configure transport
            ssh_preset = {}
            if self._ssh_combo and self._ssh_combo.currentText() in self._ssh_hosts:
                ssh_preset = self._ssh_hosts[self._ssh_combo.currentText()]

            if self._transport_combo.currentText() == 'core.ssh':
                config = {}
                user = ssh_preset.get('user', '')
                if user:
                    config['username'] = user
                key = ssh_preset.get('identityfile', '')
                if key:
                    config['key_filename'] = str(pathlib.Path(key).expanduser())
                port = ssh_preset.get('port', '')
                if port:
                    config['port'] = int(port)
                proxy_jump = ssh_preset.get('proxyjump', '')
                proxy_command = ssh_preset.get('proxycommand', '')
                if proxy_jump:
                    config['proxy_jump'] = proxy_jump
                elif proxy_command:
                    config['proxy_command'] = proxy_command
                computer.configure(**config)
            else:
                computer.configure()

            self._status.setText(f'Computer "{label}" created and configured.')
            self._status.setStyleSheet('color: #4caf50;')
        except Exception as exc:
            QMessageBox.critical(self, 'Computer Error', str(exc))
            return False
        return True


class _CodePage(QWizardPage):
    """Add an AiiDA code."""

    PRESETS = {
        'VASP': ('vasp', 'vasp.vasp', '/opt/vasp/bin/vasp_std'),
        'LAMMPS': ('lammps', 'lammps.raw', '/usr/bin/lmp'),
        'MACE': ('mace', 'mace.train', 'mace_run_train'),
        'Custom': ('', '', ''),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('AiiDA Code')
        self.setSubTitle(
            'Register a code (executable) on a configured computer.\n'
            'This step is optional, all codes can be added later.'
        )
        layout = QVBoxLayout(self)

        if not _aiida_available():
            layout.addWidget(QLabel('AiiDA is not installed. Skipping.'))
            self._form = None
            layout.addStretch()
            return

        # Skip checkbox at top
        self._skip_check = QCheckBox('Skip this step')
        self._skip_check.setChecked(True)
        self._skip_check.toggled.connect(self._on_skip_toggled)
        layout.addWidget(self._skip_check)

        # Content container
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 8, 0, 0)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel('Preset:'))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(self.PRESETS.keys()))
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch()
        content_layout.addLayout(preset_row)

        # Form
        self._form = QFormLayout()

        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText('e.g. vasp')
        self._label_edit.setToolTip('A unique name for this code in AiiDA.')
        self._form.addRow('Code label', self._label_edit)

        self._computer_combo = QComboBox()
        self._computer_combo.setEditable(True)
        self._computer_combo.setToolTip(
            'The computer where this code is installed.\n'
            'Must be configured in the previous step.'
        )
        self._form.addRow('Computer', self._computer_combo)

        self._plugin_edit = QLineEdit()
        self._plugin_edit.setPlaceholderText('e.g. vasp.vasp')
        self._plugin_edit.setToolTip(
            'The AiiDA CalcJob plugin that knows how to run this code.\n'
            'For VASP use "vasp.vasp", for LAMMPS use "lammps.raw".'
        )
        self._form.addRow('CalcJob plugin', self._plugin_edit)

        self._exec_edit = QLineEdit()
        self._exec_edit.setPlaceholderText('e.g. /opt/vasp/bin/vasp_std')
        self._exec_edit.setToolTip(
            'Absolute path to the executable on the remote machine.\n'
            'You can find it by running "which vasp_std" on the cluster.'
        )
        self._form.addRow('Executable path', self._exec_edit)

        self._prepend_text = ''
        self._prepend_btn = QPushButton('Edit Prepend Text...')
        self._prepend_btn.setToolTip(
            'Shell commands to run before the executable.\n'
            'Typically used for "module load" commands.'
        )
        self._prepend_btn.clicked.connect(
            lambda: self._open_text_editor('Prepend Text', '_prepend_text')
        )
        self._form.addRow('Prepend text', self._prepend_btn)

        self._append_text = ''
        self._append_btn = QPushButton('Edit Append Text...')
        self._append_btn.setToolTip(
            'Shell commands to run after the executable.\n'
            'Use for cleanup commands or environment teardown.'
        )
        self._append_btn.clicked.connect(
            lambda: self._open_text_editor('Append Text', '_append_text')
        )
        self._form.addRow('Append text', self._append_btn)

        content_layout.addLayout(self._form)

        self._status = QLabel()
        content_layout.addWidget(self._status)

        layout.addWidget(self._content)
        layout.addStretch()

        # Apply first preset
        self._on_preset_changed('VASP')
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        self._opacity = QGraphicsOpacityEffect(self._content)
        self._opacity.setOpacity(0.35)
        self._content.setGraphicsEffect(self._opacity)
        self._content.setEnabled(False)

    def _on_skip_toggled(self, checked: bool) -> None:
        self._content.setEnabled(not checked)
        self._opacity.setOpacity(0.35 if checked else 1.0)

    def initializePage(self) -> None:
        if self._form is None:
            return
        # Refresh computer list
        self._computer_combo.clear()
        computers = _list_computers()
        if computers:
            self._computer_combo.addItems(computers)

        existing = _list_codes()
        if existing:
            self._status.setText(f'Existing codes: {", ".join(existing)}')
            self._status.setStyleSheet('color: palette(mid);')

    def _open_text_editor(self, title: str, attr: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumSize(500, 300)
        layout = QVBoxLayout(dlg)

        editor = QTextEdit()
        editor.setPlainText(getattr(self, attr, ''))
        editor.setStyleSheet('font-family: monospace; font-size: 12px;')
        _BashHighlighter(editor.document())
        layout.addWidget(editor)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.Accepted:
            setattr(self, attr, editor.toPlainText())
            text = getattr(self, attr)
            btn = self._prepend_btn if attr == '_prepend_text' else self._append_btn
            if text.strip():
                first_line = text.strip().splitlines()[0]
                btn.setText(
                    f'{title}: {first_line[:40]}...'
                    if len(first_line) > 40
                    else f'{title}: {first_line}'
                )
            else:
                btn.setText(f'Edit {title}...')

    def _on_preset_changed(self, name: str) -> None:
        if name not in self.PRESETS:
            return
        label_hint, plugin, exec_hint = self.PRESETS[name]
        self._label_edit.setText(label_hint)
        self._plugin_edit.setText(plugin)
        self._exec_edit.setText(exec_hint)

    def validatePage(self) -> bool:
        if self._form is None or self._skip_check.isChecked():
            return True

        label = self._label_edit.text().strip()
        computer_label = self._computer_combo.currentText().strip()
        plugin = self._plugin_edit.text().strip()
        filepath = self._exec_edit.text().strip()

        if not label:
            QMessageBox.warning(self, 'Code', 'Label cannot be empty.')
            return False
        if not computer_label:
            QMessageBox.warning(self, 'Code', 'Select a computer.')
            return False
        if not filepath:
            QMessageBox.warning(self, 'Code', 'Executable path cannot be empty.')
            return False

        try:
            from aiida import orm

            computer = orm.load_computer(computer_label)
            code = orm.InstalledCode(
                label=label,
                computer=computer,
                filepath_executable=filepath,
                default_calc_job_plugin=plugin or None,
                prepend_text=self._prepend_text.strip(),
                append_text=self._append_text.strip(),
            ).store()
            self._status.setText(f'Code created: {code.full_label}')
            self._status.setStyleSheet('color: #4caf50;')
        except Exception as exc:
            QMessageBox.critical(self, 'Code Error', str(exc))
            return False
        return True


class _PotcarPage(QWizardPage):
    """Upload a VASP POTCAR potential family."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('VASP Potential Family (POTCAR)')
        self.setSubTitle(
            'Upload VASP PAW potentials as an AiiDA potential family.\n'
            'Requires aiida-vasp installed and a POTCAR archive from the VASP portal.\n'
            'This step is optional, skip if not using VASP.'
        )
        layout = QVBoxLayout(self)

        if not _aiida_available():
            layout.addWidget(QLabel('AiiDA is not installed. Skipping.'))
            self._form = None
            layout.addStretch()
            return

        # Skip checkbox at top
        self._skip_check = QCheckBox('Skip this step')
        self._skip_check.setChecked(True)
        self._skip_check.toggled.connect(self._on_skip_toggled)
        layout.addWidget(self._skip_check)

        # Content container
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 8, 0, 0)

        self._form = QFormLayout()

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            'e.g. ~/potpaw_PBE.54.tar or ~/potpaw_PBE.54/'
        )
        self._path_edit.setMinimumWidth(350)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit)
        browse_file_btn = QPushButton('Browse File...')
        browse_file_btn.setFixedWidth(100)
        browse_file_btn.clicked.connect(self._browse_file)
        path_row.addWidget(browse_file_btn)
        browse_dir_btn = QPushButton('Browse Folder...')
        browse_dir_btn.setFixedWidth(110)
        browse_dir_btn.clicked.connect(self._browse_folder)
        path_row.addWidget(browse_dir_btn)
        self._form.addRow('POTCAR path', path_row)

        self._name_edit = QLineEdit('PBE.54')
        self._name_edit.setPlaceholderText('e.g. PBE.54, LDA.54')
        self._form.addRow('Family name', self._name_edit)

        self._desc_edit = QLineEdit('PBE.54 PAW potentials')
        self._form.addRow('Description', self._desc_edit)

        content_layout.addLayout(self._form)

        self._status = QLabel()
        content_layout.addWidget(self._status)

        layout.addWidget(self._content)
        layout.addStretch()

        existing = _list_potential_families()
        if existing:
            self._status.setText(f'Existing families: {", ".join(existing)}')
            self._status.setStyleSheet('color: palette(mid);')

        from PySide6.QtWidgets import QGraphicsOpacityEffect

        self._opacity = QGraphicsOpacityEffect(self._content)
        self._opacity.setOpacity(0.35)
        self._content.setGraphicsEffect(self._opacity)
        self._content.setEnabled(False)

    def _on_skip_toggled(self, checked: bool) -> None:
        self._content.setEnabled(not checked)
        self._opacity.setOpacity(0.35 if checked else 1.0)

    def _browse_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select POTCAR archive',
            str(pathlib.Path.home()),
            'Archives (*.tar *.tar.gz *.tgz *.zip);;All files (*)',
        )
        if path:
            self._path_edit.setText(path)

    def _browse_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self,
            'Select POTCAR folder',
            str(pathlib.Path.home()),
        )
        if path:
            self._path_edit.setText(path)
            self._auto_detect_family(pathlib.Path(path))

    _ELEMENT_SYMBOLS = {
        'H',
        'He',
        'Li',
        'Be',
        'B',
        'C',
        'N',
        'O',
        'F',
        'Ne',
        'Na',
        'Mg',
        'Al',
        'Si',
        'P',
        'S',
        'Cl',
        'Ar',
        'K',
        'Ca',
        'Sc',
        'Ti',
        'V',
        'Cr',
        'Mn',
        'Fe',
        'Co',
        'Ni',
        'Cu',
        'Zn',
        'Ga',
        'Ge',
        'As',
        'Se',
        'Br',
        'Kr',
        'Rb',
        'Sr',
        'Y',
        'Zr',
        'Nb',
        'Mo',
        'Tc',
        'Ru',
        'Rh',
        'Pd',
        'Ag',
        'Cd',
        'In',
        'Sn',
        'Sb',
        'Te',
        'I',
        'Xe',
        'Cs',
        'Ba',
        'La',
        'Ce',
        'Pr',
        'Nd',
        'Pm',
        'Sm',
        'Eu',
        'Gd',
        'Tb',
        'Dy',
        'Ho',
        'Er',
        'Tm',
        'Yb',
        'Lu',
        'Hf',
        'Ta',
        'W',
        'Re',
        'Os',
        'Ir',
        'Pt',
        'Au',
        'Hg',
        'Tl',
        'Pb',
        'Bi',
        'Po',
        'At',
        'Rn',
        'Fr',
        'Ra',
        'Ac',
        'Th',
        'Pa',
        'U',
        'Np',
        'Pu',
        'Am',
        'Cm',
        'Bk',
        'Cf',
    }

    def _auto_detect_family(self, folder: pathlib.Path) -> None:
        """Try to detect potential family name from folder structure."""
        element_dirs = set()
        for child in folder.iterdir():
            if not child.is_dir():
                continue
            name_base = child.name.split('_')[0]
            if name_base in self._ELEMENT_SYMBOLS:
                element_dirs.add(name_base)
                continue
            for sub in child.iterdir():
                if sub.is_dir():
                    sub_base = sub.name.split('_')[0]
                    if sub_base in self._ELEMENT_SYMBOLS:
                        element_dirs.add(sub_base)

        if element_dirs:
            family_name = folder.name
            for prefix in ('potpaw_', 'potcar_', 'pot_'):
                if family_name.lower().startswith(prefix):
                    family_name = family_name[len(prefix) :]
                    break
            self._name_edit.setText(family_name)
            self._desc_edit.setText(f'{family_name} PAW potentials')
            self._status.setText(f'Detected {len(element_dirs)} element(s) in folder.')
            self._status.setStyleSheet('color: #4caf50;')

    def validatePage(self) -> bool:
        if self._form is None or self._skip_check.isChecked():
            return True

        path = self._path_edit.text().strip()
        name = self._name_edit.text().strip()
        desc = self._desc_edit.text().strip()

        if not path:
            QMessageBox.warning(self, 'POTCAR', 'Path cannot be empty.')
            return False
        if not name:
            QMessageBox.warning(self, 'POTCAR', 'Family name cannot be empty.')
            return False

        import subprocess

        self._status.setText('Uploading... this may take a minute.')
        self._status.setStyleSheet('color: palette(mid);')
        self._status.repaint()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            result = subprocess.run(
                [
                    'verdi',
                    'data',
                    'vasp.potcar',
                    'uploadfamily',
                    f'--path={path}',
                    f'--name={name}',
                    f'--description={desc or name}',
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                self._status.setText(f'Uploaded: {result.stdout.strip()}')
                self._status.setStyleSheet('color: #4caf50;')
            else:
                msg = result.stderr.strip() or result.stdout.strip()
                QMessageBox.critical(self, 'Upload Failed', msg)
                self._status.setText('Upload failed.')
                self._status.setStyleSheet('color: #ef5350;')
                return False
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                'POTCAR',
                '"verdi" command not found.\nIs aiida-core installed?',
            )
            return False
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, 'POTCAR', 'Upload timed out (5 min).')
            return False
        return True


class _StatusPage(QWizardPage):
    """Final page showing a summary of what's configured."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Setup Complete')
        self.setSubTitle(
            'Here is a summary of your current ATLAS configuration.\n'
            'You can change these settings at any time in the Settings page.'
        )
        layout = QVBoxLayout(self)
        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(['Item', 'Status'])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

    def initializePage(self) -> None:
        rows = []
        # MP key
        key = _read_mp_key()
        if key:
            masked = key[:4] + '...' + key[-4:] if len(key) > 8 else '***'
            rows.append(('MP API key', f'Configured ({masked})'))
        else:
            rows.append(('MP API key', 'Not configured'))

        rows.append(('AiiDA installed', 'Yes' if _aiida_available() else 'No'))

        # Profile
        current = _current_profile()
        rows.append(('AiiDA profile', current or 'None'))

        # Broker
        broker = _broker_configured()
        if broker is True:
            rows.append(('Message broker', 'Configured'))
        elif broker is False:
            rows.append(('Message broker', 'Not configured'))
        else:
            rows.append(('Message broker', 'N/A'))

        # Computers
        computers = _list_computers()
        rows.append(('Computers', ', '.join(computers) if computers else 'None'))

        # Codes
        codes = _list_codes()
        rows.append(('Codes', ', '.join(codes) if codes else 'None'))

        # Potential families
        families = _list_potential_families()
        rows.append(('Potential families', ', '.join(families) if families else 'None'))

        self._table.setRowCount(len(rows))
        for i, (item, status) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(item))
            self._table.setItem(i, 1, QTableWidgetItem(status))
        self._table.resizeColumnsToContents()


# ── Main wizard ────────────────────────────────────────────────────


class SetupWizard(QWizard):
    """ATLAS setup wizard, GUI equivalent of ``atl_init_setup``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ATLAS Setup')
        self.setMinimumSize(650, 500)
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(_WelcomePage())
        self.addPage(_MpKeyPage())
        self.addPage(_AiidaProfilePage())
        self.addPage(_BrokerPage())
        self.addPage(_ComputerPage())
        self.addPage(_CodePage())
        self.addPage(_PotcarPage())
        self.addPage(_StatusPage())


# ── Status panel for Settings page ─────────────────────────────────


class SetupStatusPanel(QGroupBox):
    """Compact panel showing current ATLAS setup status with a re-run button."""

    from PySide6.QtCore import Signal as _Signal

    setup_completed = _Signal()

    def __init__(self, parent=None):
        super().__init__('Setup Status', parent)
        layout = QVBoxLayout(self)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(['Component', 'Status'])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._wizard_btn = QPushButton('Run Setup Wizard...')
        self._wizard_btn.setFixedWidth(160)
        self._wizard_btn.clicked.connect(self._run_wizard)
        btn_row.addWidget(self._wizard_btn)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def refresh(self) -> None:
        rows = []
        key = _read_mp_key()
        if key:
            masked = key[:4] + '...' + key[-4:] if len(key) > 8 else '***'
            rows.append(('MP API key', f'Configured ({masked})'))
        else:
            rows.append(('MP API key', 'Not configured'))

        rows.append(('AiiDA installed', 'Yes' if _aiida_available() else 'No'))

        current = _current_profile()
        rows.append(('AiiDA profile', current or 'None'))

        broker = _broker_configured()
        if broker is True:
            rows.append(('Message broker', 'Configured'))
        elif broker is False:
            rows.append(('Message broker', 'Not configured'))
        else:
            rows.append(('Message broker', 'N/A'))

        computers = _list_computers()
        rows.append(('Computers', ', '.join(computers) if computers else 'None'))

        codes = _list_codes()
        rows.append(('Codes', ', '.join(codes) if codes else 'None'))

        families = _list_potential_families()
        rows.append(('Potential families', ', '.join(families) if families else 'None'))

        self._table.setRowCount(len(rows))
        for i, (item, status) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(item))
            self._table.setItem(i, 1, QTableWidgetItem(status))
        self._table.resizeColumnToContents(0)

    def _run_wizard(self) -> None:
        wizard = SetupWizard(self)
        wizard.exec()
        self.refresh()
        self.setup_completed.emit()
