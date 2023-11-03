import pathlib as pl

# Root directory (actually 'core' directory)
ROOT_DIR = (pl.Path(__file__).parent).resolve()

# Data folder
DATA_DIR = (pl.Path(__file__).parent.parent / "data").resolve()