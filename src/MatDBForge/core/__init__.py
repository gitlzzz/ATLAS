import pathlib as pl

# Core directory
MDB_CORE_DIR = (pl.Path(__file__).parent).resolve()

# Data folder
MDB_DATA_DIR = (pl.Path(__file__).parent.parent / "data").resolve()