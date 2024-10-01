"""Library for streamlining the structure creation and training for NNPs."""

import pathlib as pl

# Root of package
MDB_ROOT_DIR = (pl.Path(__file__).parent).resolve()

try:
    with open(MDB_ROOT_DIR.parents[1] / "VERSION") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "unknown"
