"""Generate an initial database from TOML config."""

import argparse
import pathlib as pl
import tomllib
import warnings
from argparse import RawTextHelpFormatter

from rich.traceback import install as traceback_install

# TODO: Remove this once the deprecation warnings are fixed
warnings.filterwarnings('ignore', category=DeprecationWarning, module='spglib')


def gen_initial_database(config_dict: dict):
    """
    Generate a initial database from TOML config.

    A MDB database can be generated from settings
    specified in a TOML file.


    Parameters
    ----------
    toml_path : str
        Path for the TOML configuration file
    """
    from MatDBForge.core.initial_db import cli_run_gen_initial_database

    # Extract parameters from toml
    database_dict = config_dict['database']
    db_path = database_dict['database_path']
    phase_diagram_dict = config_dict['phase_diagram']
    gen_dict = config_dict['generation']
    selected_phases = list(phase_diagram_dict['phase'].keys())

    # Generating the database
    cli_run_gen_initial_database(
        db_path,
        database_dict,
        phase_diagram_dict,
        gen_dict,
        selected_phases,
        config_dict,
    )


def run_gen_initial_database():
    parser = argparse.ArgumentParser(
        prog='mdb_gen_init_db',
        description='Generate and manage MDB initial databases.',
        formatter_class=RawTextHelpFormatter,
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(
        dest='command', help='List of available commands'
    )

    # Create the subparser for the 'report' command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate reports for MatDBForge.',
        usage=(
            'mdb_gen_init_db report [-h]\nGenerate a report for a MatDBForge database'
        ),
    )
    report_parser.add_argument(
        '--db_path',
        '-d',
        help=('Path to the database file.'),
        metavar='<PATH>',
        default=None,
        required=True,
    )

    # Create the subparser for the 'report' command
    init_db_parser = subparsers.add_parser(
        'generate',
        help='Generate an initial database for MatDBForge.',
        usage=(
            'mdb_gen_init_db generate [-h]\n'
            'Generate an initial database for MatDBForge.'
        ),
    )

    init_db_parser.add_argument(
        '-c',
        '--config_file',
        help=(
            'path pointing to a TOML settings file.\n'
            'By default `database_generation_settings.toml` will be '
            'searched in the CWD.'
        ),
        type=pl.Path,
        default='./database_generation_settings.toml',
        metavar='PATH',
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Generate an initial database
    if args.command == 'generate':
        # Loading TOML config file
        try:
            with open(args.config_file, 'rb') as f:
                toml_dict = tomllib.load(f)
        except FileNotFoundError as e:
            error_message = (
                f"The config file '{args.config_file}' does not exist. "
                'Please make sure that is the correct name or input a different path.'
            )
            raise FileNotFoundError(error_message) from e

        # Check if all required sections are present
        from MatDBForge.core.command_line.command_line_utils import validate_config_file

        validate_config_file(config_dict=toml_dict, config_type='database_generation')

        # Calling the function to generate the initial database
        gen_initial_database(config_dict=toml_dict)

    # Generate an initial database report
    elif args.command == 'report':
        db_path = pl.Path(args.db_path).resolve(strict=True)
        if db_path.exists():
            from MatDBForge.core.initial_db import cli_gen_db_report

            # Generating the report
            cli_gen_db_report(database_path=args.db_path)
        else:
            raise FileNotFoundError(
                f"The database '{db_path}' does not exist. "
                'Please make sure that is the correct name or input a different path.'
            )


if __name__ == '__main__':
    # Use rich formatter for tracebacks
    traceback_install(
        width=88,
    )

    run_gen_initial_database()
