"""Run VASP DFT calculations for a database of structures using the AiiDA-VASP."""

import argparse
import pathlib as pl
import tomllib


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run VASP DFT calculations for a database using AiiDA-VASP.'
    )
    parser.add_argument(
        '--db_file',
        '-i',
        required=True,
        help='Path to the extxyz file containing the database of structures.',
        metavar='FILE',
    )
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        help='Path to the TOML file with HPC and VASP input configuration.',
        metavar='FILE',
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def main():
    # Parse command-line arguments
    args = parse_arguments()

    from ase.io import read as ase_read

    from atlas.core import code_utils as atl_cut
    from atlas.core import initial_db as atl_indb
    from atlas.core.code_utils import check_atl_version
    from atlas.core.command_line.command_line_utils import validate_config_file
    from atlas.workflows import aiida_utils as atl_aut

    # Load the configuration
    config = load_config(args.config)

    # Start logger
    log_path = config.get('general', {}).get('log_path', '/tmp/')
    logger, log_file_path = atl_cut.init_logger(
        source='run_vasp_database', log_path=log_path
    )

    # Checking version
    check_atl_version(logger=logger)

    # Check if all required sections are present
    any_errors, errors, warnings = validate_config_file(
        config_dict=config, config_type='dft'
    )
    if any_errors:
        return

    source_db = pl.Path(args.db_file)

    # Loading the initial structures dataframe
    match source_db.suffix:
        case '.extxyz' | '.xyz':
            initial_db = ase_read(source_db, format='extxyz', index=':')
        case '.xz':
            initial_db = atl_indb.InitialDatabase(
                database_name=source_db.stem,
                database_path=source_db.parent,
            )

    atl_aut.run_dataframe_vasp_aiida_queue(
        initial_db=initial_db,
        config_dict=config,
        log_file_path=log_file_path,
    )


if __name__ == '__main__':
    main()
