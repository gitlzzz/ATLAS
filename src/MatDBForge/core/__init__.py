"""Main module for the core package of MatDBForge.
This includes structures, initial database creation,
and other core functionalities.
"""

import pathlib as pl

# Core directory
MDB_CORE_DIR = (pl.Path(__file__).parent).resolve()

# Data folder
MDB_DATA_DIR = (pl.Path(__file__).parent.parent / 'data').resolve()
