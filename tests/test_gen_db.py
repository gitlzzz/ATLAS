"""Module for testing the generation of the initial database."""

import os
import sys
import tempfile
import tomllib

import pytest

from atlas.core.command_line.cli_generate_configuration_file import (
    gen_default_config as atl_gen,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_gen_settings(tmp_dir):
    os.chdir(tmp_dir)

    test_args = ['atl_gen_configuration_file', 'generate', '-t', 'database_generation']
    sys.argv = test_args
    with pytest.raises(SystemExit):
        atl_gen()

    output = f'{tmp_dir}/database_generation_settings.toml'
    assert os.path.exists(output)

    with open(output, 'rb') as f:
        config = tomllib.load(f)

    assert 'database' in config
    assert 'phase_diagram' in config
    assert 'generation' in config
    assert 'vacancies' in config
