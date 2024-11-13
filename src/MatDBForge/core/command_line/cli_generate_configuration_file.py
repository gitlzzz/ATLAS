"""Generate default configuration files for MDB."""

import argparse
import pathlib as pl
import shutil
import sys
from argparse import RawTextHelpFormatter

from MatDBForge.core import MDB_DATA_DIR


def gen_default_config():
    parser = argparse.ArgumentParser(
        prog='mdb_gen_configuration_file',
        description='Generate MDB default configuration files in the TOML format.',
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        '-t',
        '--config_type',
        help=(
            'Type of the configuration file to be generated. Available types are:\n'
            '\t- active_learning: Configuration file for active learning loop.\n'
            '\t- initial_db: Configuration file for initial database generation.\n'
        ),
        type=str,
        required=True,
        choices=['active_learning', 'initial_db'],
        # default="./active_learning_settings.toml",
        metavar='TYPE',
    )

    parser.add_argument(
        '-p',
        '--path',
        help=(
            'Path in which to store the file.\n'
            'Will use the CWD by default. Folders will be created if necessary.'
        ),
        type=pl.Path,
        default='.',
        metavar='PATH',
    )
    parser.add_argument(
        '-o',
        '--overwrite',
        help=('Whether to overwrite the destination file, if existent.'),
        action='store_const',
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Choosing config file to write.
    if args.config_type == 'active_learning':
        default_config_name = 'active_learning_settings.toml'
    elif args.config_type == 'initial_db':
        default_config_name = 'database_generation_settings.toml'

    config_file_path = pl.Path(MDB_DATA_DIR) / 'input_files' / default_config_name

    # Copying file to path.
    final_path: pl.Path = args.path / default_config_name

    if not final_path.exists() or args.overwrite:
        shutil.copy(config_file_path, args.path)
        print(
            f"Saved file {default_config_name} in path '{final_path.absolute()}'",
        )
    else:
        print(
            'File already exists. Not overwriting as flag -o / --overwrite not set.',
        )
        sys.exit(1)
