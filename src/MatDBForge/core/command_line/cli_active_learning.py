"""Run active learning loop from a TOML configuration file."""

import argparse
import pathlib as pl
import time
import warnings
from argparse import RawTextHelpFormatter

import tomli

warnings.filterwarnings("ignore")


def create_active_learning_builder(toml_dict: dict):
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
    from aiida.orm import Dict, Int
    from aiida.plugins import WorkflowFactory

    # Getting builder for workchain
    al_calculation = WorkflowFactory("mdb-active-learning-base")
    builder = al_calculation.get_builder()

    ## General AL settings
    al_conf = toml_dict["active_learning"]
    builder.active_learning.run_name = al_conf["run_name"]
    builder.active_learning.load_init_models = al_conf.get("load_init_models")
    builder.active_learning.init_db_path = str(
        pl.Path(al_conf["init_db_path"]).resolve()
    )

    if al_conf.get("results_dir"):
        results_dir = pl.Path(al_conf["results_dir"]).resolve()
    else:
        results_dir = pl.Path("./results").resolve()
    builder.active_learning.results_dir = str(results_dir)

    if al_conf.get("log_path"):
        log_path = pl.Path(al_conf["log_path"]).resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = pl.Path(f"mdb_output_{timestamp}.log").resolve()
    builder.log_path = str(log_path)

    builder.active_learning.final_db_name = al_conf["final_db_name"]
    builder.active_learning.max_iterations = Int(int(al_conf["max_iterations"]))
    builder.active_learning.check_extrapolation = al_conf["check_extrapolation"]

    ## AL seed settings
    builder.active_learning.seed_size_frac = float(
        toml_dict["al_seed"]["seed_size_frac"]
    )
    builder.active_learning.seed_max_num_structs = int(
        toml_dict["al_seed"]["seed_max_num_structs"]
    )

    builder.active_learning.seed_select_settings = toml_dict["al_seed"][
        "seed_select_settings"
    ]

    builder.active_learning.committee_num_models = int(
        toml_dict["committee_eval"]["committee_num_models"]
    )
    builder.active_learning.model_acc_multiplier = float(
        al_conf["model_acc_multiplier"]
    )
    builder.active_learning.al_keep_struct_every_n_ps = float(
        al_conf["al_keep_struct_every_n_ps"]
    )

    ## MD settings
    md_params = toml_dict["md"]["parameters"]
    builder.active_learning.md_temperature_list_K = md_params["temperature_list_K"]
    builder.active_learning.md_max_temp_multiplier = md_params["max_temp_multiplier"]
    builder.active_learning.md_num_steps = int(md_params["num_steps"])
    builder.active_learning.md_timestep_duration_ps = float(
        md_params["timestep_duration_ps"]
    )
    builder.active_learning.gather_traj_cnt_lattice = md_params[
        "gather_traj_cnt_lattice"
    ]
    builder.active_learning.use_kokkos = md_params["use_kokkos"]

    # LAMMPS-MACE MD Settings
    builder.active_learning.lammps_mace = Dict(value=toml_dict["md"]["queue"])

    # MD filters
    builder.active_learning.md_filters = Dict(value=toml_dict["md"].get("filters"))

    ## MACE training settings
    builder.active_learning.mace_train = Dict(value=toml_dict["mace_train"])

    ## Committee Evaluation Settings
    builder.active_learning.committee_eval = Dict(value=toml_dict["committee_eval"])

    ## DFT method selection and settings
    builder.active_learning.dft_method = al_conf["dft_method"]
    if al_conf["dft_method"] == "vasp":
        builder.active_learning.dft_settings = Dict(value=toml_dict["dft"]["vasp"])
    elif al_conf["dft_method"] == "mace":
        # Make sure the path to the MACE potential is absolute
        toml_dict["dft"]["mace"]["mace_potential_path"] = str(
            pl.Path(toml_dict["dft"]["mace"]["mace_potential_path"]).resolve()
        )
        builder.active_learning.dft_settings = Dict(value=toml_dict["dft"]["mace"])

    ## Descriptor settings
    builder.active_learning.descriptor_settings = Dict(value=toml_dict["descriptors"])

    return builder


def run_active_learning():
    parser = argparse.ArgumentParser(
        prog="run_active_learning",
        description="Launch a MDB active learning loop.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help=(
            "path pointing to a TOML settings file.\n"
            "By default `active_learning_settings.toml` will be searched in the CWD."
        ),
        type=pl.Path,
        default="./active_learning_settings.toml",
        # required=True,
        metavar="PATH",
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Create the subparser for the 'gui' command
    gui_parser = subparsers.add_parser(
        "gui", help="Launch a dashboard to keep track of the active learning loop"
    )

    # Add arguments specific to the 'gui' subcommand
    gui_parser.add_argument(
        "--update_interval",
        help=("Refresh time interval in seconds"),
        type=int,
        default=60,
        metavar="n_sec",
    )
    gui_parser.add_argument(
        "--port",
        help=("Port to use for the webapp"),
        type=int,
        default=8000,
        metavar="port",
    )

    gui_parser.add_argument(
        "--debug",
        help=("Enable Flask debug"),
        action="store_const",
        const=True,
        default=False,
    )
    gui_parser.add_argument(
        "--online",
        help=("Enable online"),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Loading TOML config file
    try:
        with open(args.config_file, "rb") as f:
            toml_dict = tomli.load(f)
    except FileNotFoundError as e:
        error_message = (
            f"The config file '{args.config_file}' does not exist. "
            "Please make sure that is the correct name or input a different path."
        )
        raise FileNotFoundError(error_message) from e

    from aiida import load_profile

    from MatDBForge.core import utils as mdb_ut

    try:
        # Loading default aiida profile
        load_profile(profile=toml_dict["active_learning"]["aiida_profile"])
    except Exception as e:
        mdb_ut.custom_print(f"Error loading aiida profile: '{e}'", "error")

    # Parsing settings from TOML and creating builder for aiida
    builder = create_active_learning_builder(toml_dict)

    from aiida.engine import run, submit

    if args.command != "gui":
        node = run(builder)
    else:
        from MatDBForge.core.command_line.cli_dashboard import run_dashboard_app

        node = submit(builder)
        time.sleep(1)

        run_dashboard_app(
            process_id=str(node.pk),
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )
