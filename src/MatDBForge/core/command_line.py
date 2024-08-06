#!/usr/bin/env python
"""Run an active-learning procedure based on ML-MD using aiida."""

import argparse
import pathlib as pl
import shutil
import sys
import warnings
from argparse import RawTextHelpFormatter

import tomli
from aiida import load_profile
from aiida.engine import run, submit
from aiida.orm import Dict, Int
from aiida.plugins import WorkflowFactory
from gunicorn.app.wsgiapp import WSGIApplication

from MatDBForge.active_learning.dashboard.training_dashboard_flask import (
    run_training_dashboard,
)
from MatDBForge.core import MDB_DATA_DIR
from MatDBForge.core import exceptions as mdb_exc
from MatDBForge.core import initial_db as indb
from MatDBForge.core import phase_diagram as mdb_pd
from MatDBForge.core import surfaces as mdb_surf
from MatDBForge.core import utils as mdb_ut

warnings.filterwarnings("ignore")


class MDBDashboardApp(WSGIApplication):
    """Gunicorn application for the MDB dashboard."""

    def __init__(self, app_uri, options=None):
        self.options = options or {}
        self.app_uri = app_uri
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)


def run_dashboard_app(process_id, port, update_interval, debug, online):
    print(
        f"\n\nRunning dashboard to monitor process: {process_id}.\n"
        f"Access: http://127.0.0.1:{port}.\n\n"
    )
    print("Pres Ctrl+C to stop the dashboard.")
    if debug:
        app = run_training_dashboard(
            workchain_node_id=process_id,
            refresh_interval={update_interval},
            port={port},
        )
        if online:
            app.run(debug=True, port=port, host="0.0.0.0")
        else:
            app.run(debug=True, port=port, host="0.0.0.0")
    else:
        app = MDBDashboardApp(
            f"MatDBForge.active_learning.dashboard.training_dashboard_flask"
            f":run_training_dashboard(workchain_node_id={process_id}, "
            f"refresh_interval={update_interval}, port={port})",
        )
        if online:
            app.options["bind"] = f"0.0.0.0:{port}"
            app.load_config()
            app.run()
        else:
            app.options["bind"] = f"127.0.0.1:{port}"
            app.load_config()
            app.run()


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
    builder.active_learning.results_dir = str(pl.Path(al_conf["results_dir"]).resolve())
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

    try:
        # Loading default aiida profile
        load_profile(profile=toml_dict["active_learning"]["aiida_profile"])
    except Exception as e:
        mdb_ut.custom_print(f"Error loading aiida profile: '{e}'", "error")

    # Parsing settings from TOML and creating builder for aiida
    builder = create_active_learning_builder(toml_dict)

    if args.command != "gui":
        node = run(builder)
    else:
        node = submit(builder)

        run_dashboard_app(
            process_id=node,
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )


def gen_default_config():
    parser = argparse.ArgumentParser(
        prog="mdb_conf_gen",
        description="Generate MDB default configuration files in the TOML format.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--config_type",
        help=(
            "Type of the configuration file to be generated. Available types are:\n"
            "\t- active_learning: Configuration file for active learning loop.\n"
            "\t- initial_db: Configuration file for initial database generation.\n"
        ),
        type=str,
        required=True,
        choices=["active_learning", "initial_db"],
        # default="./active_learning_settings.toml",
        metavar="TYPE",
    )

    parser.add_argument(
        "-p",
        "--path",
        help=(
            "Path in which to store the file.\n"
            "Will use the CWD by default. Folders will be created if necessary."
        ),
        type=pl.Path,
        default=".",
        metavar="PATH",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help=("Whether to overwrite the destination file, if existent."),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Choosing config file to write.
    if args.config_type == "active_learning":
        default_config_name = "active_learning_settings.toml"
    elif args.config_type == "initial_db":
        default_config_name = "database_generation_settings.toml"

    config_file_path = pl.Path(MDB_DATA_DIR) / "input_files" / default_config_name

    # Copying file to path.
    final_path: pl.Path = args.path / default_config_name

    if not final_path.exists() or args.overwrite:
        shutil.copy(config_file_path, args.path)
        print(
            f"Saved file {default_config_name} in path '{final_path.absolute()}'",
        )
    else:
        print(
            "File already exists. Not overwriting as flag -o / --overwrite not set.",
        )
        sys.exit(1)


def monitor_al_loop():
    parser = argparse.ArgumentParser(
        prog="monitor_al_loop",
        description="Monitor a MDB active learning loop.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--process_id",
        help=("Process id (pk/uuid) of the WorkChain to monitor.\n"),
        type=str,
        metavar="UUID/PK",
    )
    parser.add_argument(
        "--update_interval",
        help=("Refresh time interval in seconds"),
        type=int,
        default=60,
        metavar="n_sec",
    )
    parser.add_argument(
        "--port",
        help=("Port to use for the webapp"),
        type=int,
        default=8000,
        metavar="port",
    )
    parser.add_argument(
        "--debug",
        help=("Enable Flask debug"),
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--online",
        help=("Enable online"),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    run_dashboard_app(
        process_id=args.process_id,
        port=args.port,
        update_interval=args.update_interval,
        debug=args.debug,
        online=args.online,
    )


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
    if type == "active_learning":
        mandatory_keys_list = ["active_learning", "md", "committee_eval", "dft"]

    elif type == "generate_database":
        mandatory_keys_list = ["system", "generation"]

    for key in mandatory_keys_list:
        if key not in list(toml_dict.keys()):
            raise mdb_exc.MissingMandatoryParameterError(
                f"Input toml file missing mandatory key: {key}."
            )


def run_initial_config():
    mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path="/tmp")

    # Get config directory
    config_path = mdb_ut.get_config_path()

    # Create a mdb folder
    config_dir = mdb_ut.init_config_dir(config_path)

    if config_dir:
        mdb_ut.custom_print(
            f"Enter your materials project API key in '{config_dir/'secrets.json'}'"
            f" to finish the setup process.",
            print_type="info",
        )
    else:
        mdb_ut.custom_print(
            "Initial configuration already done: 'secrets.json' already exists.", "info"
        )


def gen_initial_database(config_dict: dict):
    """
    Generate a initial database from TOML config.

    A MDB database can be generated from settings
    specified in a TOML file.


    Parameters
    ----------
    toml_path : str
        Path for the TOML configuration file
    """
    # Check if all required sections are present
    parse_input_toml(toml_dict=config_dict, type="generate_database")

    # Extract parameters from toml
    sys_dict = config_dict["system"]
    db_path = sys_dict["final_database_path"]
    phase_diagram_dict = config_dict["phase_diagram"]
    gen_dict = config_dict["generation"]
    selected_phases = list(phase_diagram_dict["phase"].keys())

    # Start logger
    log_path = pl.Path(db_path) / "logs"

    if not log_path.exists():
        log_path.mkdir(parents=True)

    mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path=f"{db_path}/logs")

    # Create phase diagram
    phases_list = []
    for _, phase_d in phase_diagram_dict["phase"].items():
        curr_phase = mdb_pd.Phase(
            name=phase_d["name"],
            base_elem=phase_d["base_elem"],
            cluster_elem=phase_d["cluster_elem"],
            base_elem_comp_min=phase_d["base_elem_comp_min"],
            base_elem_comp_max=phase_d["base_elem_comp_max"],
            prototype=phase_d["prototype"],
            offset=phase_d["offset"],
        )
        phases_list.append(curr_phase)

    # Assemble phase diagram
    phase_diagram = mdb_pd.BinaryPhaseDiagram(
        phase_diagram_dict["material_name"],
        *phases_list,
    )

    # Initialize the database
    structures = indb.InitialDatabase(
        database_name=sys_dict["database_name"],
        max_num_atoms=sys_dict["max_num_atoms"],
        phase_diagram=phase_diagram,
    )

    mdb_ut.custom_print(structures, "done")

    read_from_db = True

    if sys_dict.get("relax_struct_path"):
        # Initial structures obtained with DFT relaxation are loaded from a given path
        structures.read_base_structures(
            path=sys_dict["relax_struct_path"],
            target_structures=selected_phases,
        )
    else:
        # Obtain structures from Materials Project
        prototypes = [phase.prototype for phase in phase_diagram.phases]
        structures.gather_base_structures(target_structures=prototypes)
        read_from_db = False

    mdb_ut.custom_print("Generating structures from initial structures...", "debug")

    for phase in selected_phases:
        # Line break for aesthetic purposes
        print()

        # Creating surfaces from the base structures, generating
        # different supercells and applying replacements.
        mdb_ut.custom_print(f"Current phase: {phase}.", "info")

        # Getting phase object
        phase = phase_diagram.get_phase(phase)
        print("command_line phase: ", type(phase))

        if "bulk" in gen_dict:
            mdb_ut.custom_print("Generating bulk structures...", "info")

            # Generating bulk structures.
            structures.generate_bulk_structures(
                prototype=phase.prototype,
                phase=phase,
                num_struct=gen_dict["bulk"]["num_struct"],
                num_repeats=gen_dict["bulk"]["num_repeat"],
                get_different_supercells=True,
                min_num_atoms=sys_dict["min_num_atoms"],
                supercell_max_idx=gen_dict["bulk"]["supercell_max_idx"],
                read=read_from_db,
            )

            mdb_ut.custom_print(structures, "info")

        if "surface" in gen_dict:
            # Generating surface structures.
            mdb_ut.custom_print("Generating slab structures...", "info")

            slabs = mdb_surf.gene_surfaces_diff_miller(
                db_obj=structures,
                phase=phase,
                min_num_atoms=sys_dict["min_num_atoms"],
                overwrite_max_num_atoms=sys_dict["max_num_atoms"],
                min_miller_index=gen_dict["surface"]["min_miller_index"],
                max_miller_index=gen_dict["surface"]["supercell_max_idx"],
                min_slab_size=gen_dict["surface"]["min_slab_size_ang"],
                num_diff_layer_size=gen_dict["surface"]["num_diff_layer_size"],
                min_vacuum_size=gen_dict["surface"]["min_vacuum_size_ang"],
                get_supercells=gen_dict["surface"]["get_supercells"],
                fixed_layers=gen_dict["surface"]["fixed_layers"],
                limit_supercell=gen_dict["surface"]["max_number_supercells"],
                save_in_db=gen_dict["surface"]["save_in_db"],
            )

            mdb_ut.custom_print(
                f"{len(slabs)} slabs generated.",
                "done",
            )

        if "cluster" in gen_dict:
            raise NotImplementedError("Cluster type not implemented yet")

        # Filter small and large structures
        remove_count = structures.remove_structs_out_of_atom_count_range(
            min_num_atoms=sys_dict["min_num_atoms"],
            max_num_atoms=sys_dict["max_num_atoms"],
        )
        mdb_ut.custom_print(
            f"Removed {remove_count} structures out of atom count range.", "info"
        )
        mdb_ut.custom_print(structures, "info")

        # Lattice displacement
        if "displacement" in config_dict:
            displ_dict = config_dict["displacement"]

            mdb_ut.custom_print("Applying displacements to lattices.", "info")

            structures.perturb_min_displacement(
                frac_max=displ_dict["lattice_frac_displ_max"],
                frac_min=displ_dict["lattice_frac_displ_min"],
                repeat=displ_dict["num_repeats"],
            )
            mdb_ut.custom_print(structures, "info")

            remove_count = structures.remove_structs_out_of_cell_size_range(
                min_cell_size=sys_dict["min_cell_size"]
            )
            mdb_ut.custom_print(
                f"Removed {remove_count} structures out of cell size range.", "info"
            )

        if "perturbation" in config_dict:
            perturb_dict = config_dict["perturbation"]
            mdb_ut.custom_print(
                "Applying a random perturbation to the structures...",
                "info",
            )

            mdb_ut.apply_gauss_perturb_db(
                db_obj=structures,
                repeat=perturb_dict["num_repeats"],
                filters=perturb_dict["filter_struct_types"],
                phase=phase,
                limit_num_structures=perturb_dict["limit_max_num_perturbs"],
            )

            mdb_ut.custom_print(structures, "info")

    print()

    mdb_ut.custom_print(structures, "done")
    print()

    structures.save_database(
        path=sys_dict["final_database_path"],
        suffix=sys_dict["database_name"],
    )


def run_gen_initial_database():
    parser = argparse.ArgumentParser(
        prog="mdb_gen_init_db",
        description="Generate a MDB initial database.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help=(
            "path pointing to a TOML settings file.\n"
            "By default `database_generation_settings.toml` will be "
            "searched in the CWD."
        ),
        type=pl.Path,
        default="./database_generation_settings.toml",
        metavar="PATH",
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

    # Calling the function to generate the initial database
    gen_initial_database(config_dict=toml_dict)
