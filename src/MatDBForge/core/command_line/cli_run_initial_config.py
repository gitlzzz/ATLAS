"""Run initial configuration to set up the MatDBForge environment."""

import pathlib as pl
import warnings

from MatDBForge.core import code_utils as mdb_cud

warnings.filterwarnings('ignore')


def run_initial_config():
    mdb_cud.init_logger(source=pl.Path(__file__).stem, log_path='/tmp')

    # Get config directory
    config_path = mdb_cud.get_config_path()

    # Create a mdb folder
    config_dir = mdb_cud.init_config_dir(config_path)

    if config_dir:
        mdb_cud.custom_print(
            f"Enter your materials project API key in '{config_dir/'secrets.json'}'"
            f" to finish the setup process."
            "You can get your API key from https://next-gen.materialsproject.org/api",
            print_type='warn',
        )
    else:
        mdb_cud.custom_print(
            "Initial configuration already done: 'secrets.json' already exists.", 'done'
        )

    # Get cache directory
    cache_path = mdb_cud.get_cache_path()

    # Create an mdb folder inside the ~/.cache directory
    cache_dir = mdb_cud.init_cache_dir(cache_path)

    if cache_dir:
        mdb_cud.custom_print(
            f"Cache directory created at '{cache_dir}'",
            print_type='info',
        )
    else:
        mdb_cud.custom_print('Cache directory already exists. Nothing done.', 'done')
