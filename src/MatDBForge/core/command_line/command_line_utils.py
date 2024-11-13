#!/usr/bin/env python
"""Run an active-learning procedure based on ML-MD using aiida."""

import warnings

from MatDBForge.core import exceptions as mdb_exc

warnings.filterwarnings('ignore', module='paramiko')


MDB_LOGO = """
  __  __      _   ___  ___ ___
 |  \/  |__ _| |_|   \| _ ) __|__ _ _ __ _ ___
 | |\/| / _` |  _| |) | _ \ _/ _ \ '_/ _` / -_)
 |_|  |_\__,_|\__|___/|___/_|\___/_| \__, \___|
                                     |___/
"""


def parse_input_toml(toml_dict: dict, type: str):
    """
    Parses and validates the input TOML dictionary based on the specified type.

    Parameters
    ----------
    toml_dict : dict
        The input dictionary parsed from a TOML file.
    type : str
        The type of configuration to validate. Currently supports "active_learning".

    Raises
    ------
    MissingMandatoryParameterError
        If any mandatory keys are missing from the input TOML dictionary.
    """
    if type == 'active_learning':
        mandatory_keys_list = ['active_learning', 'md', 'committee_eval', 'dft']

    elif type == 'generate_database':
        mandatory_keys_list = ['database', 'phase_diagram', 'generation']

    for key in mandatory_keys_list:
        if key not in list(toml_dict.keys()):
            raise mdb_exc.MissingMandatoryParameterError(
                f'Input toml file missing mandatory key: {key}.'
            )
