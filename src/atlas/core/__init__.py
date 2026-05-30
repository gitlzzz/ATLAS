"""Main module for the core package of atlas.
This includes structures, initial database creation,
and other core functionalities.
"""

import pathlib as pl

# Core directory
ATL_CORE_DIR = (pl.Path(__file__).parent).resolve()

# Data folder
ATL_DATA_DIR = (pl.Path(__file__).parent.parent / 'data').resolve()
