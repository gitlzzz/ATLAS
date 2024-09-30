"""Generate an initial database from TOML config."""

import argparse
import pathlib as pl
import sys
from argparse import RawTextHelpFormatter

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import warnings

from MatDBForge.core.command_line.command_line_utils import parse_input_toml
from MatDBForge.core.initial_db import cli_run_gen_initial_database

# TODO: Remove this once the deprecation warnings are fixed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")


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
    # Check if all required sections are present
    parse_input_toml(toml_dict=config_dict, type="generate_database")

    # Extract parameters from toml
    database_dict = config_dict["database"]
    db_path = database_dict["database_path"]
    phase_diagram_dict = config_dict["phase_diagram"]
    gen_dict = config_dict["generation"]
    selected_phases = list(phase_diagram_dict["phase"].keys())

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
        prog="mdb_gen_init_db",
        description="Generate a MDB initial database.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help=(
            "path pointing to a TOML settings file.\n"
            "By default `database_generation_settings.toml` will be "
            "searched in the CWD."
        ),
        type=pl.Path,
        default="./database_generation_settings.toml",
        metavar="PATH",
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Loading TOML config file
    try:
        with open(args.config_file, "rb") as f:
            toml_dict = tomllib.load(f)
    except FileNotFoundError as e:
        error_message = (
            f"The config file '{args.config_file}' does not exist. "
            "Please make sure that is the correct name or input a different path."
        )
        raise FileNotFoundError(error_message) from e

    # Calling the function to generate the initial database
    gen_initial_database(config_dict=toml_dict)
