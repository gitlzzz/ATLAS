"""Materials Project API key setup."""

from __future__ import annotations

import json
import os

import rich.prompt as rp

from atlas.core import code_utils as atl_cut
from atlas.core.command_line.setup._common import heading


def setup_mp_key() -> None:
    """Set up the Materials Project API key interactively."""
    config_dir = atl_cut.get_config_path() / 'atl'
    config_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = config_dir / 'secrets.json'

    if not secrets_path.exists():
        heading(
            'Materials Project API Key',
            'To get started with database generation,\n'
            'you need a Materials Project API key.\n\n'
            'Get one at: https://next-gen.materialsproject.org/api',
        )

        api_key = rp.Prompt.ask('Enter your MP API key', password=True)
        print()

        if not api_key or not api_key.strip():
            atl_cut.custom_print('API key cannot be empty. Skipped.', 'error')
            return

        secrets_path.write_text(json.dumps({'API_KEY': api_key.strip()}) + '\n')
        os.chmod(secrets_path, 0o600)

        atl_cut.custom_print(f"API key saved to '{secrets_path}'", 'done')
        atl_cut.custom_print(
            'File permissions set to 0o600 (owner read/write only).',
            'info',
        )
    else:
        atl_cut.custom_print(
            f"MP API key already configured at '{secrets_path}'.\n"
            'To update it, edit the file or delete it and re-run.',
            'done',
        )

    # Create cache directory
    cache_dir = atl_cut.get_cache_path() / 'mdb'
    try:
        cache_dir.mkdir(parents=True, exist_ok=False)
        atl_cut.custom_print(f"Cache directory created at '{cache_dir}'", 'info')
    except FileExistsError:
        pass
