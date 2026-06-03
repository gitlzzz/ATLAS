"""Run initial configuration to set up the ATLAS environment."""

import json
import pathlib as pl
import warnings

import rich.prompt as rp
from rich import print as rprint
from rich.panel import Panel

from atlas.core import code_utils as atl_cut

warnings.filterwarnings('ignore')


def run_initial_config():
    atl_cut.init_logger(source=pl.Path(__file__).stem, log_path='/tmp')

    config_file_name = 'secrets.json'
    config_path = atl_cut.get_config_path()
    config_dir = config_path / 'atl'

    # Create config directory if needed
    config_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = config_dir / config_file_name

    if not secrets_path.exists():
        # New setup: prompt for API key interactively
        print()
        panel = Panel(
            'Welcome to ATLAS!\n\n'
            'To get started with database generation,\n'
            'you need a Materials Project API key.\n\n'
            'Get one at: https://next-gen.materialsproject.org/api',
            title='ATLAS Setup',
            border_style='cyan',
        )
        rprint(panel)
        print()

        api_key = rp.Prompt.ask('Enter your MP API key', password=True)
        print()

        if not api_key or not api_key.strip():
            atl_cut.custom_print(
                'API key cannot be empty. Setup aborted.',
                print_type='error',
            )
            return

        # Write secrets file with correct permissions
        secrets_path.write_text(json.dumps({'API_KEY': api_key.strip()}) + '\n')
        import os

        os.chmod(secrets_path, 0o600)

        atl_cut.custom_print(
            f"API key saved to '{secrets_path}'",
            print_type='done',
        )
        atl_cut.custom_print(
            'File permissions set to 0o600 (owner read/write only).',
            print_type='info',
        )
    else:
        atl_cut.custom_print(
            (
                f'Initial configuration already done: '
                f"'{secrets_path}' already exists.\n"
                'To update your MP API key, edit this file '
                'or delete it and re-run setup.'
            ),
            'warn',
        )

    # Create cache directory
    cache_path = atl_cut.get_cache_path()
    cache_dir = cache_path / 'mdb'

    try:
        cache_dir.mkdir(parents=True, exist_ok=False)
        atl_cut.custom_print(
            f"Cache directory created at '{cache_dir}'",
            print_type='info',
        )
    except FileExistsError:
        atl_cut.custom_print('Cache directory already exists. Nothing done.', 'done')
