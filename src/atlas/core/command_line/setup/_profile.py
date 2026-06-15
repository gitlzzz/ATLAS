"""AiiDA profile setup."""

from __future__ import annotations

import subprocess

import rich.prompt as rp
from rich import print as rprint
from rich.table import Table

from atlas.core.code_utils import custom_print
from atlas.core.command_line.setup._common import (
    check_aiida,
    heading,
    list_profiles,
)


def setup_profile() -> None:
    """Create or select an AiiDA profile interactively."""
    if not check_aiida():
        custom_print(
            'AiiDA is not installed. Skipping profile setup.',
            'warn',
        )
        return

    heading(
        'AiiDA Profile Setup',
        'A profile stores your AiiDA database, repository, and daemon settings.\n'
        'You can create a new profile or select an existing one.\n\n'
        'Docs: https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html',
    )

    profiles, default = list_profiles()

    if profiles:
        table = Table(title='Existing profiles', show_lines=False)
        table.add_column('Profile', style='cyan')
        table.add_column('Default', justify='center')
        for p in profiles:
            marker = '*' if p == default else ''
            table.add_row(p, marker)
        rprint(table)
        rprint()

        choice = rp.Prompt.ask(
            'Use an existing profile or create a new one?',
            choices=['existing', 'new'],
            default='existing',
        )
    else:
        custom_print('No AiiDA profiles found.', 'info')
        choice = 'new'

    if choice == 'existing':
        name = rp.Prompt.ask(
            'Profile name',
            choices=profiles,
            default=default or profiles[0],
        )
    else:
        name = rp.Prompt.ask('New profile name', default='atlas')
        custom_print(f'Creating profile "{name}" via verdi presto...', 'info')
        try:
            result = subprocess.run(
                ['verdi', 'presto', '--profile-name', name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                custom_print(
                    f'verdi presto failed:\n{result.stderr.strip()}',
                    'error',
                )
                return
            custom_print(result.stdout.strip(), 'done')
        except FileNotFoundError:
            custom_print(
                '"verdi" command not found. Is aiida-core installed?',
                'error',
            )
            return
        except subprocess.TimeoutExpired:
            custom_print('verdi presto timed out after 120s.', 'error')
            return

    try:
        from aiida import load_profile

        load_profile(name, allow_switch=True)
        custom_print(f'Profile "{name}" loaded.', 'done')
    except Exception as exc:
        custom_print(f'Failed to load profile "{name}": {exc}', 'error')
        return

    if profiles and name != default:
        set_default = rp.Confirm.ask(
            f'Set "{name}" as the default profile?',
            default=False,
        )
        if set_default:
            try:
                from aiida.manage.configuration import get_config

                config = get_config()
                config.set_default_profile(name)
                config.store()
                custom_print(f'Default profile set to "{name}".', 'done')
            except Exception as exc:
                custom_print(f'Failed to set default: {exc}', 'error')
