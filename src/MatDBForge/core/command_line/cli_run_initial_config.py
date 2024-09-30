"""Run initial configuration to set up the MatDBForge environment."""

import pathlib as pl
import warnings

from MatDBForge.core import utils as mdb_ut

warnings.filterwarnings("ignore")


def run_initial_config():
    mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path="/tmp")

    # Get config directory
    config_path = mdb_ut.get_config_path()

    # Create a mdb folder
    config_dir = mdb_ut.init_config_dir(config_path)

    if config_dir:
        mdb_ut.custom_print(
            f"Enter your materials project API key in '{config_dir/'secrets.json'}'"
            f" to finish the setup process."
            "You can get your API key from https://next-gen.materialsproject.org/api",
            print_type="warn",
        )
    else:
        mdb_ut.custom_print(
            "Initial configuration already done: 'secrets.json' already exists.", "done"
        )
