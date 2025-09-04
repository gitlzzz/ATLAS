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

    from MatDBForge.core import code_utils as mdb_cut
    from MatDBForge.core import initial_db as mdb_indb
    from MatDBForge.core.code_utils import check_mdb_version
    from MatDBForge.core.command_line.command_line_utils import validate_config_file
    from MatDBForge.workflows import aiida_utils as mdb_aut

    # Checking version
    check_mdb_version()


    # Load the configuration
    config = load_config(args.config)

    # Check if all required sections are present
    validate_config_file(config_dict=config, config_type='dft')

    # Start logger
    log_path = config.get('general', {}).get('log_path', '/tmp/')
    _, log_file_path = mdb_cut.init_logger(
        source='run_vasp_database', log_path=log_path
    )

    source_db = pl.Path(args.db_file)

    # Loading the initial structures dataframe
    match source_db.suffix:
        case '.extxyz' | '.xyz':
            initial_db = ase_read(source_db, format='extxyz', index=':')
        case '.xz':
            initial_db = mdb_indb.InitialDatabase(
                database_name=source_db.stem,
                database_path=source_db,
            )

    mdb_aut.run_dataframe_vasp_aiida_queue(
        initial_db=initial_db,
        config_dict=config,
        log_file_path=log_file_path,
    )


if __name__ == '__main__':
    main()
