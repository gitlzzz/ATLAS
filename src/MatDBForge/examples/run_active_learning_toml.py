"""Run an active-learning procedure based on ML-MD using aiida."""

import argparse
import pathlib as pl

import tomli
from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int
from aiida.plugins import WorkflowFactory


def create_builder(toml_dict: dict):
    """
    Create builder object for the ActiveLearningWorkChain.

    Parameters
    ----------
    toml_dict : dict
        Dictionary coming from parsing an MDB TOML settings file.

    Returns
    -------
    ProcessBuilder
        A process builder that helps setting up the inputs for
        an ActiveLearningWorkChain.
    """
    # Getting builder for workchain
    al_calculation = WorkflowFactory("mdb-active-learning-base")
    builder = al_calculation.get_builder()

    # General AL settings
    al_conf = toml_dict["active_learning"]
    builder.active_learning.data_path = al_conf["data_path"]
    builder.active_learning.init_db_path = al_conf["init_db_path"]
    builder.active_learning.final_db_path = al_conf["final_db_path"]
    builder.active_learning.max_iterations = Int(al_conf["max_iterations"])
    builder.active_learning.seed_size_frac = al_conf["seed_size_frac"]
    builder.active_learning.md_temperature_K = al_conf["md_temperature_K"]
    builder.active_learning.md_num_steps = al_conf["md_num_steps"]
    builder.active_learning.md_timestep_duration_ps = al_conf["md_timestep_duration_ps"]
    builder.active_learning.commitee_num_models = al_conf["commitee_num_models"]
    builder.active_learning.chem_acc = al_conf["chem_acc"]
    builder.active_learning.chem_acc_multiplier = al_conf["chem_acc_multiplier"]
    builder.active_learning.al_keep_frame_interval_perc = al_conf[
        "al_keep_frame_interval_perc"
    ]

    # MACE training settings
    builder.active_learning.mace_train = Dict(value=toml_dict["mace_train"])

    # LAMMPS-MACE MD Settings
    builder.active_learning.lammps_mace = Dict(value=toml_dict["lammps_mace"])

    # Committee Evaluation Settings
    builder.active_learning.committee_eval = Dict(value=toml_dict["committee_eval"])

    return builder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mdb_active_learning",
        description="Launch a MDB active learning loop.",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help=(
            "path pointing to a TOML settings file. "
            "By default `active_learning_settings.toml` will be searched in the CWD."
        ),
        type=pl.Path,
        default="./active_learning_settings.toml",
        metavar="PATH",
    )
    # Getting CLI arguments
    args = parser.parse_args()

    # Loading TOML config file
    with open(args.config_file, "rb") as f:
        toml_dict = tomli.load(f)

    # Loading default aiida profile
    load_profile(profile=toml_dict["active_learning"]["aiida_profile"])

    # Parsing settings from TOML and creating builder for aiida
    builder = create_builder(toml_dict)

    # TESTING: This should use aiida.engine.submit function once all debugging is done.
    node = run(builder)
