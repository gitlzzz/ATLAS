"""AiiDA-VASP potential family (POTCAR) setup."""

from __future__ import annotations

import subprocess

import rich.prompt as rp
from rich import print as rprint
from rich.table import Table

from atlas.core.code_utils import custom_print
from atlas.core.command_line.setup._common import heading, require_profile


def _list_potcar_families() -> list[str]:
    """Return names of uploaded POTCAR families."""
    try:
        from aiida import orm

        families = (
            orm.QueryBuilder()
            .append(
                orm.Group,
                filters={'type_string': 'data.vasp.potcar'},
            )
            .all(flat=True)
        )
        return sorted(g.label for g in families)
    except Exception:
        pass
    # Fallback: try via CLI
    try:
        result = subprocess.run(
            ['verdi', 'data', 'vasp.potcar', 'listfamilies'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            return [
                lin.strip() for lin in lines if lin.strip() and not lin.startswith('*')
            ]
    except Exception:
        pass
    return []


def setup_potcar() -> None:
    """Upload a VASP POTCAR family to the AiiDA database."""
    if not require_profile():
        return

    heading(
        'VASP Potential Family (POTCAR)',
        'Upload a set of VASP PAW potentials to AiiDA as a "potential family".\n'
        'You need the POTCAR archive or folder from the VASP portal.\n'
        'Common family names: PBE.54, LDA.54, PBE.52\n\n'
        'Docs: https://aiida-vasp.readthedocs.io/en/latest/getting_started/potentials.html',
    )

    existing = _list_potcar_families()
    if existing:
        table = Table(title='Existing potential families', show_lines=False)
        table.add_column('Family name', style='cyan')
        for f in existing:
            table.add_row(f)
        rprint(table)
        rprint()

    path = rp.Prompt.ask(
        'Path to POTCAR archive or folder\n'
        '(e.g. ~/potpaw_PBE.54.tar or ~/potpaw_PBE.54/)',
    )
    if not path or not path.strip():
        custom_print('Path cannot be empty.', 'error')
        return
    path = path.strip()

    name = rp.Prompt.ask(
        'Family name (e.g. PBE.54)',
        default='PBE.54',
    )
    if not name.strip():
        custom_print('Name cannot be empty.', 'error')
        return
    name = name.strip()

    if name in existing:
        custom_print(f'Family "{name}" already exists.', 'warn')
        add_more = rp.Confirm.ask('Add new potentials to it anyway?', default=False)
        if not add_more:
            return

    description = rp.Prompt.ask(
        'Description',
        default=f'{name} PAW potentials',
    )

    custom_print(f'Uploading potentials from "{path}" as family "{name}"...', 'info')

    try:
        result = subprocess.run(
            [
                'verdi',
                'data',
                'vasp.potcar',
                'uploadfamily',
                f'--path={path}',
                f'--name={name}',
                f'--description={description}',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            custom_print(result.stdout.strip(), 'done')
        else:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            msg = stderr or stdout or 'Unknown error'
            custom_print(f'Upload failed:\n{msg}', 'error')
    except FileNotFoundError:
        custom_print(
            '"verdi" command not found. Is aiida-core installed?',
            'error',
        )
    except subprocess.TimeoutExpired:
        custom_print('Upload timed out after 5 minutes.', 'error')
