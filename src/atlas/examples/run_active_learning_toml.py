"""Run an active-learning procedure based on ML-MD using aiida."""

import argparse
import pathlib as pl
import tomllib

from aiida import load_profile
from aiida.engine import run

from atlas.core.command_line.cli_active_learning import create_active_learning_builder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='atl_active_learning',
        description='Launch a ATL active learning loop.',
    )
    parser.add_argument(
        '-c',
        '--config_file',
        help=(
            'path pointing to a TOML settings file. '
            'By default `active_learning_settings.toml` will be searched in the CWD.'
        ),
        type=pl.Path,
        default='./active_learning_settings.toml',
        metavar='PATH',
    )
    # Getting CLI arguments
    args = parser.parse_args()

    # Loading TOML config file
    with open(args.config_file, 'rb') as f:
        toml_dict = tomllib.load(f)

    # Loading default aiida profile
    load_profile(profile=toml_dict['active_learning']['aiida_profile'])

    # Parsing settings from TOML and creating builder for aiida
    builder = create_active_learning_builder(
        toml_dict,
        toml_dict_path=pl.Path(args.config_file).resolve(),
    )

    # TESTING: This should use aiida.engine.submit function once all debugging is done.
    node = run(builder)
