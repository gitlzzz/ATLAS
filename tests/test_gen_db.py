"""Module for testing the generation of the initial database."""

import filecmp
import os
import sys
import tempfile

import pytest

from atlas.core import ATL_DATA_DIR
from atlas.core.command_line.cli_generate_configuration_file import (
    gen_default_config as atl_gen,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_gen_settings(tmp_dir):
    os.chdir(tmp_dir)

    test_args = ['atl_gen_configuration_file', '-t', 'initial_db']
    sys.argv = test_args
    atl_gen()

    compare = filecmp.cmp(
        f1=f'{tmp_dir}/database_generation_settings.toml',
        f2=f'{ATL_DATA_DIR}/input_files/database_generation_settings.toml',
    )
    assert compare
