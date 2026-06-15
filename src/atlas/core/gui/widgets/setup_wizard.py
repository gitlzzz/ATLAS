"""GUI setup wizard — mirrors atl_init_setup in a QWizard."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import pathlib
import re

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
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
        return p.name if p else None
    except Exception:
        return None


def _list_computers() -> list[str]:
    try:
        from aiida import orm

        return sorted(
            c.label for c in orm.QueryBuilder().append(orm.Computer).all(flat=True)
        )
    except Exception:
        return []


def _list_codes() -> list[str]:
    try:
        from aiida import orm

        return sorted(
            c.full_label for c in orm.QueryBuilder().append(orm.Code).all(flat=True)
        )
    except Exception:
        return []


def _list_potential_families() -> list[str]:
    try:
        from aiida import orm

        return sorted(
            g.label for g in orm.QueryBuilder().append(orm.Group).all(flat=True)
        )
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
        if not _list_computers():
            problems.append('No compute resources configured in AiiDA')
        if not _list_codes():
            problems.append('No simulation codes configured in AiiDA')
    return problems


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
            '  3. AiiDA computer setup\n'
            '  4. AiiDA code registration\n'
            '  5. VASP potential family upload\n\n'
            'Steps 2–5 require AiiDA and can be skipped if you only\n'
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
        self.setSubTitle(
            'Enter your MP API key for querying crystal structures.\n'
            'Get one at: https://next-gen.materialsproject.org/api'
        )
        layout = QFormLayout(self)

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
        btn_row.addStretch()
        layout.addLayout(btn_row)

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


class _ComputerPage(QWizardPage):
    """Add an AiiDA computer (optionally pre-filled from SSH config)."""

    TRANSPORT_CHOICES = ['core.ssh', 'core.local']
    SCHEDULER_CHOICES = ['core.slurm', 'core.sge', 'core.pbspro', 'core.direct']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('AiiDA Computer')
        self.setSubTitle(
            'Configure a remote or local computer for running calculations.\n'
            'This step is optional — skip if not needed now.'
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
        self._form.addRow('Label', self._label_edit)

        self._hostname_edit = QLineEdit()
        self._hostname_edit.setPlaceholderText('e.g. login.cluster.edu')
        self._form.addRow('Hostname', self._hostname_edit)

        self._transport_combo = QComboBox()
        self._transport_combo.addItems(self.TRANSPORT_CHOICES)
        self._form.addRow('Transport', self._transport_combo)

        self._scheduler_combo = QComboBox()
        self._scheduler_combo.addItems(self.SCHEDULER_CHOICES)
        self._form.addRow('Scheduler', self._scheduler_combo)

        self._workdir_edit = QLineEdit(f'/scratch/{getpass.getuser()}/aiida/')
        self._form.addRow('Work directory', self._workdir_edit)

        self._mpirun_edit = QLineEdit('mpirun -np {tot_num_mpiprocs}')
        self._form.addRow('mpirun command', self._mpirun_edit)

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
            'This step is optional — all codes can be added later.'
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
        self._form.addRow('Code label', self._label_edit)

        self._computer_combo = QComboBox()
        self._computer_combo.setEditable(True)
        self._form.addRow('Computer', self._computer_combo)

        self._plugin_edit = QLineEdit()
        self._plugin_edit.setPlaceholderText('e.g. vasp.vasp')
        self._form.addRow('CalcJob plugin', self._plugin_edit)

        self._exec_edit = QLineEdit()
        self._exec_edit.setPlaceholderText('e.g. /opt/vasp/bin/vasp_std')
        self._form.addRow('Executable path', self._exec_edit)

        self._prepend_edit = QLineEdit()
        self._prepend_edit.setPlaceholderText('e.g. module load vasp/6.4.1')
        self._form.addRow('Prepend text', self._prepend_edit)

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
                prepend_text=self._prepend_edit.text().strip(),
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
            'This step is optional — skip if not using VASP.'
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
        browse_btn = QPushButton('Browse...')
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
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

    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select POTCAR archive',
            str(pathlib.Path.home()),
            'Archives (*.tar *.tar.gz *.tgz *.zip);;All files (*)',
        )
        if path:
            self._path_edit.setText(path)

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
    """ATLAS setup wizard — GUI equivalent of ``atl_init_setup``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ATLAS Setup')
        self.setMinimumSize(650, 500)
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(_WelcomePage())
        self.addPage(_MpKeyPage())
        self.addPage(_AiidaProfilePage())
        self.addPage(_ComputerPage())
        self.addPage(_CodePage())
        self.addPage(_PotcarPage())
        self.addPage(_StatusPage())


# ── Status panel for Settings page ─────────────────────────────────


class SetupStatusPanel(QGroupBox):
    """Compact panel showing current ATLAS setup status with a re-run button."""

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

        self.refresh()

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
