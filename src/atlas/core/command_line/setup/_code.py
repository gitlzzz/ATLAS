"""AiiDA code setup."""

from __future__ import annotations

import rich.prompt as rp
from rich import print as rprint
from rich.table import Table

from atlas.core.code_utils import custom_print
from atlas.core.command_line.setup._common import (
    heading,
    list_codes,
    list_computers,
    require_profile,
)

PRESETS = {
    'vasp': {
        'label_hint': 'vasp',
        'plugin': 'vasp.vasp',
        'executable_hint': '/opt/vasp/bin/vasp_std',
        'prepend_hint': 'module load vasp/6.4.1',
    },
    'lammps': {
        'label_hint': 'lammps',
        'plugin': 'lammps.raw',
        'executable_hint': '/usr/bin/lmp',
        'prepend_hint': 'module load lammps',
    },
    'mace': {
        'label_hint': 'mace',
        'plugin': 'mace.train',
        'executable_hint': 'mace_run_train',
        'prepend_hint': '',
    },
    'custom': {
        'label_hint': '',
        'plugin': '',
        'executable_hint': '',
        'prepend_hint': '',
    },
}


def setup_code() -> None:
    """Create an AiiDA code interactively."""
    if not require_profile():
        return

    heading(
        'AiiDA Code Setup',
        'A code points to an executable on a configured computer.\n'
        'ATLAS supports VASP, LAMMPS, and MACE presets, or you can add a custom code.\n'
        'All codes are optional, database generation does not require remote codes.'
        '\n\nDocs: '
        'https://aiida.readthedocs.io/projects/aiida-core/en/stable/howto/'
        'run_codes.html#how-to-set-up-a-code',
    )

    computers = list_computers()
    if not computers:
        custom_print(
            'No computers configured. Run "atl_init_setup computer" first.',
            'error',
        )
        return

    existing = list_codes()
    if existing:
        table = Table(title='Existing codes', show_lines=False)
        table.add_column('Full label', style='cyan')
        for c in existing:
            table.add_row(c)
        rprint(table)
        rprint()

    preset_name = rp.Prompt.ask(
        'Code preset',
        choices=list(PRESETS.keys()),
        default='vasp',
    )
    preset = PRESETS[preset_name]

    label = rp.Prompt.ask(
        'Code label',
        default=preset['label_hint'] or None,
    )
    if not label or not label.strip():
        custom_print('Label cannot be empty.', 'error')
        return
    label = label.strip()

    computer_label = rp.Prompt.ask(
        'Computer',
        choices=computers,
        default=computers[0],
    )

    plugin = rp.Prompt.ask(
        'CalcJob plugin entry point',
        default=preset['plugin'] or None,
    )

    filepath = rp.Prompt.ask(
        'Path to executable on the remote',
        default=preset['executable_hint'] or None,
    )
    if not filepath or not filepath.strip():
        custom_print('Executable path cannot be empty.', 'error')
        return
    filepath = filepath.strip()

    prepend = rp.Prompt.ask(
        'Prepend text (e.g. module load ..., empty to skip)',
        default=preset['prepend_hint'],
    )

    try:
        from aiida import orm

        computer = orm.load_computer(computer_label)
        code = orm.InstalledCode(
            label=label,
            computer=computer,
            filepath_executable=filepath,
            default_calc_job_plugin=plugin or None,
            prepend_text=prepend.strip() if prepend else '',
        ).store()

        custom_print(f'Code created: {code.full_label}', 'done')
    except Exception as exc:
        custom_print(f'Failed to create code: {exc}', 'error')
