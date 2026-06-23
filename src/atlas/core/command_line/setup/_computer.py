"""AiiDA computer setup."""

from __future__ import annotations

import getpass
import pathlib
import re
import subprocess

import rich.prompt as rp
from rich import print as rprint
from rich.table import Table

from atlas.core.code_utils import custom_print
from atlas.core.command_line.setup._common import (
    heading,
    list_computers,
    require_profile,
)

TRANSPORT_CHOICES = ['core.ssh', 'core.local']
SCHEDULER_CHOICES = ['core.slurm', 'core.sge', 'core.pbspro', 'core.direct']


# ── SSH config parsing ─────────────────────────────────────────────


def _parse_ssh_config() -> dict[str, dict[str, str]]:
    """Parse ~/.ssh/config and return a dict of {host_alias: {key: value}}."""
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
            # Skip wildcard entries
            if '*' in value or '?' in value:
                current_host = None
            else:
                current_host = value
                hosts[current_host] = {}
        elif current_host is not None:
            hosts[current_host][key] = value

    return hosts


def _show_ssh_hosts(hosts: dict[str, dict[str, str]]) -> None:
    """Display a table of SSH config hosts."""
    table = Table(title='SSH config hosts (~/.ssh/config)', show_lines=False)
    table.add_column('Host alias', style='cyan')
    table.add_column('HostName')
    table.add_column('User')
    table.add_column('Port')
    for alias, opts in hosts.items():
        table.add_row(
            alias,
            opts.get('hostname', '—'),
            opts.get('user', '—'),
            opts.get('port', '—'),
        )
    rprint(table)
    rprint()


# ── Main setup ─────────────────────────────────────────────────────


def setup_computer() -> None:
    """Create and configure an AiiDA computer interactively."""
    if not require_profile():
        return

    heading(
        'AiiDA Computer Setup',
        'A computer represents a remote (or local) machine where calculations run.\n'
        'You will configure the hostname, scheduler, and transport (SSH or local).\n\n'
        'Docs: https://aiida.readthedocs.io/projects/aiida-core/en/stable/howto/run_codes.html#how-to-set-up-a-computer',
    )

    existing = list_computers()
    if existing:
        table = Table(title='Existing computers', show_lines=False)
        table.add_column('Label', style='cyan')
        for c in existing:
            table.add_row(c)
        rprint(table)
        rprint()

    # Check for SSH config hosts and offer to import
    ssh_hosts = _parse_ssh_config()
    ssh_preset: dict[str, str] = {}

    if ssh_hosts:
        _show_ssh_hosts(ssh_hosts)
        use_ssh = rp.Confirm.ask(
            'Import settings from an SSH config host?',
            default=True,
        )
        if use_ssh:
            host_alias = rp.Prompt.ask(
                'Which host?',
                choices=list(ssh_hosts.keys()),
            )
            ssh_preset = ssh_hosts[host_alias]
            ssh_preset['_alias'] = host_alias

    # ── Gather parameters (pre-filled from SSH config if available) ──

    default_label = ssh_preset.get('_alias', '')
    label = rp.Prompt.ask(
        'Computer label',
        default=default_label or None,
    )
    if not label or not label.strip():
        custom_print('Label cannot be empty.', 'error')
        return
    label = label.strip()

    if label in existing:
        custom_print(f'Computer "{label}" already exists. Skipping.', 'warn')
        return

    default_hostname = ssh_preset.get('hostname', '')
    hostname = rp.Prompt.ask(
        'Hostname',
        default=default_hostname or None,
    )
    if not hostname or not hostname.strip():
        custom_print('Hostname cannot be empty.', 'error')
        return
    hostname = hostname.strip()

    transport = rp.Prompt.ask(
        'Transport plugin',
        choices=TRANSPORT_CHOICES,
        default='core.ssh' if ssh_preset else 'core.ssh',
    )

    scheduler = rp.Prompt.ask(
        'Scheduler plugin',
        choices=SCHEDULER_CHOICES,
        default='core.slurm',
    )

    ssh_user = ssh_preset.get('user', getpass.getuser())
    work_dir = rp.Prompt.ask(
        'Remote work directory',
        default=f'/scratch/{ssh_user}/aiida/',
    )

    mpirun = rp.Prompt.ask(
        'mpirun command',
        default='mpirun -np {tot_num_mpiprocs}',
    )

    try:
        from aiida import orm

        computer = orm.Computer(
            label=label,
            hostname=hostname,
            transport_type=transport,
            scheduler_type=scheduler,
            workdir=work_dir,
        )
        computer.set_mpirun_command(mpirun.split())
        computer.store()

        custom_print(f'Computer "{label}" created.', 'done')
    except Exception as exc:
        custom_print(f'Failed to create computer: {exc}', 'error')
        return

    # Configure transport
    if transport == 'core.ssh':
        _configure_ssh(computer, ssh_preset)
    else:
        try:
            computer.configure()
            custom_print('Local transport configured.', 'done')
        except Exception as exc:
            custom_print(f'Failed to configure transport: {exc}', 'error')

    # Test
    run_test = rp.Confirm.ask(
        f'Test computer "{label}" now?',
        default=True,
    )
    if run_test:
        _test_computer(label)


def _configure_ssh(
    computer,
    ssh_preset: dict[str, str],
) -> None:
    """Configure SSH transport, pre-filling from SSH config when available."""
    default_user = ssh_preset.get('user', getpass.getuser())
    ssh_user = rp.Prompt.ask('SSH username', default=default_user)

    ssh_dir = pathlib.Path.home() / '.ssh'

    # Resolve key from SSH config or scan ~/.ssh/
    default_key = ssh_preset.get('identityfile', '')
    if default_key:
        default_key = str(pathlib.Path(default_key).expanduser())

    if not default_key:
        key_files = (
            sorted(p.name for p in ssh_dir.glob('id_*') if not p.name.endswith('.pub'))
            if ssh_dir.is_dir()
            else []
        )
        if key_files:
            custom_print(f'Found SSH keys: {", ".join(key_files)}', 'info')

    key_filename = rp.Prompt.ask(
        'SSH key file (empty to use SSH config default)',
        default=default_key or '',
    )

    config: dict[str, str | int] = {'username': ssh_user}
    if key_filename.strip():
        config['key_filename'] = str(pathlib.Path(key_filename.strip()).expanduser())

    # Port from SSH config
    port = ssh_preset.get('port', '')
    if port:
        config['port'] = int(port)

    # ProxyJump / ProxyCommand
    proxy_jump = ssh_preset.get('proxyjump', '')
    proxy_command = ssh_preset.get('proxycommand', '')
    if proxy_jump:
        config['proxy_jump'] = proxy_jump
        custom_print(f'Using ProxyJump: {proxy_jump}', 'info')
    elif proxy_command:
        config['proxy_command'] = proxy_command
        custom_print(f'Using ProxyCommand: {proxy_command}', 'info')

    try:
        computer.configure(**config)
        custom_print('SSH transport configured.', 'done')
    except Exception as exc:
        custom_print(f'Failed to configure SSH: {exc}', 'error')


def _test_computer(label: str) -> None:
    """Run verdi computer test."""
    custom_print(f'Testing computer "{label}"...', 'info')
    try:
        result = subprocess.run(
            ['verdi', 'computer', 'test', label],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            custom_print(result.stdout.strip(), 'done')
        else:
            custom_print(
                f'Test failed:\n{result.stderr.strip()}',
                'error',
            )
    except FileNotFoundError:
        custom_print('"verdi" command not found.', 'error')
    except subprocess.TimeoutExpired:
        custom_print('Computer test timed out after 60s.', 'error')
