"""General utility functions for the active learning workflows."""

import io
import itertools as it
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slugify
import torch
import wonderwords as ww
from aiida import orm
from aiida.common.log import LOG_LEVEL_REPORT
from aiida.engine import (
    calcfunction,
)
from aiida.plugins import CalculationFactory
from ase import Atoms, geometry, units
from ase.data import atomic_numbers, covalent_radii
from ase.geometry.analysis import Analysis
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, natural_cutoffs
from e3nn.util import jit
from mace.calculators import LAMMPS_MACE, MACECalculator
from pymatgen.core import Structure as pmg_struct
from pymatgen.io.ase import AseAtomsAdaptor
from shapely.geometry import Point, Polygon

from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.core.code_utils import check_mdb_version, custom_print, init_logger
from MatDBForge.workflows import aiida_utils as mdb_aut


def md_apply_temperature_ramp(dyn, total_steps, T_start, T_end):
    """
    Function to compute the temperature ramp during ASE MD simulations.

    Parameters
    ----------
    step : int
        Current step in the MD simulation.
    total_steps : int
        Total number of steps in the MD simulation.
    T_start : float
        Initial temperature of the MD simulation.
    T_end : _type_
        Final temperature of the MD simulation.

    Returns
    -------
    float
        Temperature to set for the current step in the MD simulation.
    """
    # Update the temperature using the ramp function
    current_temperature = T_start + (T_end - T_start) * dyn.nsteps / total_steps

    # Set the temperature in units of energy
    dyn.set_temperature(temperature_K=current_temperature)


def generate_descriptors_mace(model_path: str, database):
    calculator = MACECalculator(
        model_paths=model_path, device="cuda", default_dtype="float64"
    )
    descriptor_dict = {}
    descriptor_list = []
    for struct in database:
        descriptor_dict[struct.info["aiida_uuid"]] = {
            "descriptors": [],
            "latent_space": [],
        }

    for struct in database:
        curr_struct_descriptors = calculator.get_descriptors(struct)
        descriptor_list.append(curr_struct_descriptors)
        descriptor_dict[struct.info["aiida_uuid"]]["descriptors"].append(
            curr_struct_descriptors
        )

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr


def generate_descriptors_soap(database): ...


def run_mace_md_ase(init_conf, md_params, T_start, traj_obj):
    """
    Run MD simulations using ASE and MACE.

    Parameters
    ----------
    init_conf : Atoms
        Initial structure to use for the MD simulation.
    md_params : dict
        Dictionary containing the MD parameters.
    T_start : float
        Initial temperature of the MD simulation.
    traj_obj :
        _description_
    """
    T_multiplier = md_params["max_temp_multiplier"]
    T_end = T_start * T_multiplier

    # Reading structure
    # init_conf = ase_read("curr_structure.xyz", format="extxyz")

    # Load the trained model as an ASE calculator and attach it to the atoms object
    calculator = MACECalculator(
        model_paths="curr_model.model",
        device=md_params["device"],
        default_dtype=md_params["dtype"],
    )
    init_conf.calc = calculator

    # Define the Langevin dynamics
    dyn = Langevin(
        atoms=init_conf,
        timestep=md_params["timestep_duration_ps"]
        * (units.s * 1e-12),  # convert ase fs to ps
        temperature_K=T_start,
        friction=md_params["timestep_duration_ps"] * 100,
        trajectory=traj_obj,
        # logfile=res_folder / f"md-{T_start}.log",
    )

    # Attach the thermostat function to increase the temperature
    # linearly from T_start to T_end
    dyn.attach(
        interval=1,
        function=md_apply_temperature_ramp,
        dyn=dyn,
        total_steps=md_params["num_steps"],
        T_start=T_start,
        T_end=T_end,
    )

    # Run the MD simulation
    dyn.run(md_params["num_steps"])


def plot_al_loop_report(
    ini_db_size, seed_gen_db_sizes, train_db_sizes, mace_e, mace_f, it_idx
):
    # Get unix timestamp for filename
    timestamp = int(time.time())
    filename = Path(f"seed_train_db_sizes_{timestamp}.png").resolve()

    # Define colors from the gruvbox palette
    colors = [
        "#83a598",
        "#b16286",
        "#98971a",
        "#fb4934",
    ]
    line_color = "#28282855"

    # Create a 2x2 figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Plot seed and train db sizes as a stacked bar chart over every iteration
    width = 0.3

    # Adding inital database size as iteration 0
    it_idx = [0] + it_idx
    ind = np.array(it_idx)
    train_db_sizes = [ini_db_size] + train_db_sizes
    seed_gen_db_sizes = [ini_db_size] + seed_gen_db_sizes

    # Plotting seed and train db sizes
    ax[0, 0].bar(ind, train_db_sizes, width=width, label="train_db", color=colors[0])
    ax[0, 0].bar(
        ind + width,
        seed_gen_db_sizes,
        width=width,
        label="seed_gen_db",
        color=colors[1],
    )
    ax[0, 0].set_xticks(ind + width / 2, ind)
    ax[0, 0].set_xlabel("AL Loop Step")
    ax[0, 0].set_ylabel("Number of structures")
    ax[0, 0].legend()
    ax[0, 0].set_title("Seed and Train Database Evolution")

    # Add text labels to top left figure bars
    for idx, seed, train in zip(it_idx, seed_gen_db_sizes, train_db_sizes):
        ax[0, 0].text(idx, train / 2, train, ha="center", va="bottom", rotation=90)
        ax[0, 0].text(
            idx + width, seed / 2, seed, ha="center", va="bottom", rotation=90
        )

    # Plot seed size delta as a bar chart over every iteration
    seed_gen_db_diff, train_db_diff = [], []
    for idx, seed, train in zip(it_idx, seed_gen_db_sizes, train_db_sizes):
        idx = it_idx.index(idx)

        if idx == 0:
            seed_gen_db_diff.append(0)
            train_db_diff.append(0)
        else:
            seed_gen_db_diff.append(seed - seed_gen_db_sizes[idx - 1])
            train_db_diff.append(train - train_db_sizes[idx - 1])

    # Add text labels to bars
    for idx, seed, train in zip(it_idx, seed_gen_db_diff, train_db_diff):
        if idx == 0:
            continue

        # Add sign
        seed_txt = f"{seed}" if seed < 0 else f"+{seed}"
        train_txt = f"{train}" if train < 0 else f"+{train}"

        ax[0, 1].text(idx, train / 2, train_txt, ha="center", va="bottom", rotation=90)
        ax[0, 1].text(
            idx + width, seed / 2, seed_txt, ha="center", va="bottom", rotation=90
        )

    ax[0, 1].bar(ind, train_db_diff, width=width, label="train_db", color=colors[0])
    ax[0, 1].bar(
        ind + width,
        seed_gen_db_diff,
        width=width,
        label="seed_gen_db",
        color=colors[1],
    )
    ax[0, 1].axhline(y=0, color=line_color, linestyle="--")
    ax[0, 1].set_xticks(ind + width / 2, ind)
    ax[0, 1].set_xlabel("AL Loop Step")
    ax[0, 1].set_ylabel(r"$\Delta$ Number of structures")
    ax[0, 1].set_title("Structure count change over iteration")
    ax[0, 1].legend()

    # Plot MACE model energy performance
    ind = np.arange(len(mace_e)) + 1
    ax[1, 0].plot(ind, mace_e, label="MACE Energy", color=colors[2], marker="o")
    ax[1, 0].set_xticks(ind, ind)
    ax[1, 0].set_xlabel("AL Loop Step")
    ax[1, 0].set_ylabel("RMSE E per atom [meV]")
    ax[1, 0].set_title("Evolution of best MACE Model Energy RMSE")

    # Plot MACE model force performance
    ax[1, 1].plot(ind, mace_f, label="MACE Forces", color=colors[3], marker="o")
    ax[1, 1].set_xticks(ind, ind)
    ax[1, 1].set_xlabel("AL Loop Step")
    ax[1, 1].set_ylabel("RMSE F [meV / A]")

    # Add a horizontal line to mark chemical accuracy for energy and forces
    chem_acc = 43.37  # meV

    ax[1, 0].axhline(y=chem_acc, color=line_color, linestyle="--")
    ax[1, 0].text(x=1.5, y=chem_acc, s="Chem. Acc.", color=line_color)
    ax[1, 1].set_title("Evolution of best MACE Model Force RMSE")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    custom_print(f"Saved report to '{filename}'.", "info")
    plt.clf()


def gen_al_loop_report(loop_id: int | str = None, log_path: str = None):
    import re

    from aiida.cmdline.utils.common import get_workchain_report

    # Init logger
    init_logger(source="al_loop_report_gen")

    # Checking version
    check_mdb_version()

    if loop_id:
        al_loop_node = orm.load_node(loop_id)
        report = get_workchain_report(al_loop_node, levelname="REPORT")
    if log_path:
        with open(log_path) as f:
            report = f.read()

    ini_db_line = re.compile(r"initial database containing.*").findall(report)
    ini_db_size = int(ini_db_line[0].split()[3])

    # Match all lines containing the seed_gen_db and train_db sizes
    seed_gen_db_sizes, train_db_sizes, it_idx = [], [], []
    db_lines_re = re.compile(r"Iteration \d\d?: seed_gen_db.*").findall(report)

    # Prepare a list of all seed_gen_db and train_db sizes from db_lines
    for line in db_lines_re:
        it_idx.append(int(line.split()[1].replace(":", "")))
        seed_gen_db_sizes.append(int(line.split()[3].replace(",", "")))
        train_db_sizes.append(int(line.split()[5].replace(",", "")))

    # Match all lines containing the M0 model performance
    mace_e, mace_f = [], []
    lammps_lines = re.compile(r"Generated LAMMPS potential using.*").findall(report)

    # Prepare a list of all mace models generated from lammps_lines
    for line in lammps_lines:
        mace_e.append(float(line.split()[10]))
        mace_f.append(float(line.split()[14]))

    plot_al_loop_report(
        ini_db_size=ini_db_size,
        seed_gen_db_sizes=seed_gen_db_sizes,
        train_db_sizes=train_db_sizes,
        mace_e=mace_e,
        mace_f=mace_f,
        it_idx=it_idx,
    )
    custom_print("Report generation complete.", "done")


def model_res_dict_to_arr(res_dict: dict, dict_type: str) -> np.ndarray:
    """Convert a dictionary of model results to a numpy array.

    Parameters
    ----------
    res_dict : dict
        Dictionary containing the model results.
    dict_type : str
        Type of the dictionary. Either "energy" or "forces".

    Returns
    -------
    np.ndarray
        Numpy array containing the model results.
    """
    res_model_list = []

    # Gathering all trajectories
    for _, res in res_dict.items():
        res_model_list.append(res)

    # Find the maximum length of the inner lists.
    # This length corresponds with the number of gathered frames.
    # All trajs should have the same number of frames. When frames
    # are missing, np.nan is used to pad their lists.
    sublist_lens = set(len(sublist) for sublist in res_model_list)
    max_len = max(sublist_lens)

    # If lists need to be padded, checking all trajs while taking into
    # account if they are energy or forces, as energies will use lists
    # and forces will use arrays.
    if len(sublist_lens) > 1:
        padded_list = []
        for sublist in res_model_list:
            # Pad the energy lists with np.nan
            if dict_type == "energy":
                nan_list = list(it.repeat(np.nan, (max_len - len(sublist))))
                padded_sublist = list(sublist) + nan_list
                padded_list.append(padded_sublist)
            # Pad the forces arrays with (n_at, 3) arrays filled with np.nan
            elif dict_type == "forces":
                nan_list = list(
                    it.repeat(
                        object=np.full(shape=sublist[0].shape, fill_value=np.nan),
                        times=(max_len - len(sublist)),
                    )
                )
                padded_sublist = list(sublist) + nan_list
                padded_list.append(padded_sublist)
        res_model_list = padded_list

    res_model_list = np.array(res_model_list, dtype=float)

    return res_model_list


def get_model_forces_variance(forces_dict: dict) -> np.ndarray:
    """Get the variance of the forces for each structure in the dict."""
    forces_model_list = model_res_dict_to_arr(forces_dict, dict_type="forces")
    forces_var = forces_model_list.var(axis=0)

    return forces_var


def get_model_energies_variance(energies_dict: dict) -> np.ndarray:
    """Get the variance of the energies for each structure in the dict."""
    energies_model_list = model_res_dict_to_arr(energies_dict, dict_type="energy")
    energies_var = energies_model_list.var(axis=0)

    return energies_var


def get_model_forces_std(forces_dict: dict) -> np.ndarray:
    """Get the standard deviation of the forces for each structure in the dict."""
    forces_model_list = model_res_dict_to_arr(forces_dict, dict_type="forces")

    # Calculate the sample standard deviation of the energies
    # for each structure
    forces_std = np.nanstd(forces_model_list, axis=0, ddof=1)

    return forces_std


def get_model_energies_std(energies_dict: dict) -> np.ndarray:
    """Get the standard deviation of the energies for each structure in the dict."""
    energies_model_list: np.ndarray = model_res_dict_to_arr(
        energies_dict, dict_type="energy"
    )

    # Calculate the sample standard deviation of the energies
    # for each structure
    energies_std = np.nanstd(energies_model_list, axis=0, ddof=1)

    return energies_std


def load_database(path: str) -> list[Atoms]:
    """Load an extended xyz file (database) from a given path as a list of ASE Atoms."""
    database = ase_read(
        filename=path,
        format="extxyz",
        index=":",
    )
    return database


def convert_database_to_ase_atoms(
    database: list, deserialize: bool = False
) -> list[Atoms]:
    """Converts a struture list/array into containing both dicts and ase.Atoms
    into a list containing only ase.Atoms.
    """
    upd_database = []
    for struct in database:
        if isinstance(struct, dict):
            if deserialize:
                struct = aiida_serialized_ase_dict_to_atoms(struct)
            else:
                struct = Atoms.fromdict(struct)
        upd_database.append(struct)
    return upd_database


def select_dft_structures(struct_arr, frame_interval):
    """
    Select DFT structures using the interval given as an input of the workchain.

    Parameters
    ----------
    struct_arr : np.array
        Array containing all possible structures to compute.
    frame_interval : orm.Int
        Integer representing the interval between structures to keep.

    Returns
    -------
    np.array
        Array containing only the selected structures.
    """
    slice_step = int(len(struct_arr) * frame_interval)

    if slice_step == 0:
        slice_step = int(len(struct_arr) / 2)

    selected_dft_structs_idxs = range(len(struct_arr))[::slice_step]
    selected_dft_structs = struct_arr[::slice_step]
    selected_high_error = np.nonzero(selected_dft_structs)[0]
    selected_high_error_idxs = np.array(selected_dft_structs_idxs)[selected_high_error]

    return selected_high_error_idxs


def get_max_layer_distance(struct):
    """Get the maximum distance between layers in a structure."""
    # Get the layers and their distance with respect to the origin
    tags, levels = geometry.get_layers(atoms=struct, miller=(0, 0, 1), tolerance=0.1)

    # Compute the maximum layer height
    layer_distances = []
    for layer_index, layer_height in enumerate(levels[1:]):
        layer_height_diff = layer_height - levels[layer_index]
        layer_distances.append(layer_height_diff)

    max_layer_distance = np.max(layer_distances)

    return max_layer_distance


def apply_layer_distance_filter(struct, max_layer_distance_ang):
    """
    Evaluates whether the layer distace is above max_layer_distance_ang.

    Parameters
    ----------
    struct : ase.Atoms
        structure to check.
    max_layer_distance_ang : float
        Maximum distance between layers

    Returns
    -------
    bool
        Returns `True` if the layer distace is above max_layer_distance_ang,
        `False` if otherwise.
    """
    is_structure_wrong = False

    if isinstance(struct, pmg_struct):
        struct = AseAtomsAdaptor().get_atoms(structure=struct)

    max_dist = get_max_layer_distance(struct)

    # Filtering using the max_layer_distance_ang
    if max_dist > max_layer_distance_ang:
        is_structure_wrong = True

    return is_structure_wrong


def apply_filter_no_neighbors(struct):
    """
    Use neighbor list to check if there are any atoms with no neighbors.

    Parameters
    ----------
    struct : ase.Atoms
        structure to check.

    Returns
    -------
    bool
        Returns `True` if there are atoms with no neighbors, `False` if otherwise.
    """
    if isinstance(struct, pmg_struct):
        struct = AseAtomsAdaptor().get_atoms(structure=struct)

    cutoffs: list = natural_cutoffs(struct)
    nl = NeighborList(
        cutoffs,
        skin=0.01,
        sorted=False,
        self_interaction=False,
        bothways=True,
        primitive=NewPrimitiveNeighborList,
    )

    nl.update(struct)
    conn_matr: np.array = nl.get_connectivity_matrix(sparse=False)

    # Check if there is any row in conn_matr that has only zeros
    has_disconnected_atoms: bool = np.any(np.all(conn_matr == 0, axis=1))
    return has_disconnected_atoms


def get_total_num_frames(len_traj, md_tstep_duration_ps, frame_interval):
    """Compute the number of frames to get from the trajectory using user input."""
    # Get total MD time in picoseconds
    total_duration_ps = len_traj * md_tstep_duration_ps

    # Get total number of frames in that time.
    # Frame interval represents every how many ps of MD simulation
    # save a frame.
    total_num_frames = np.ceil(total_duration_ps * 1 / frame_interval)

    return total_num_frames


def select_md_frames_to_keep(
    frame_interval: int,
    # total_n_frames: int,
    md_tstep_duration_ps: float,
    traj,
    steps_E_F_arr: np.array,
    forces: np.array,
):
    """Select MD frames to keep using the frame interval and total number of frames."""
    len_traj = len(traj)
    total_num_frames = get_total_num_frames(
        len_traj, md_tstep_duration_ps, frame_interval
    )

    # Choose the number of frames evenly and create a mask for the arrays
    mask = np.linspace(0, len_traj - 1, total_num_frames, dtype=int)

    # Apply the mask to traj, steps_E_F_arr and forces
    traj_sampled = traj[mask]
    steps_E_F_arr_sampled = steps_E_F_arr[mask]
    forces_sampled = forces[mask]

    return traj_sampled, steps_E_F_arr_sampled, forces_sampled


def get_dft_calc_builder_vasp(
    struct,
    row,
    calc_idx: int,
    group,
    dft_settings: dict,
):
    """Generate a aiida-vasp calculation builder for a given structure and row."""
    struct_type = row["mdb_struct_type"]

    # Gathering row information
    (curr_structure, curr_material_name, curr_unique_id, curr_phase) = (
        mdb_aut.gather_calc_data_from_row(row, curr_structure=struct)
    )

    # Getting default potential mapping
    potential_mapping = mdb_aut.generate_potential_mapping()

    # Updating general INCAR with calc type specific options
    specific_options = dft_settings.get(struct_type)
    if specific_options:
        specific_options = specific_options.get("incar")
        for setting, val in specific_options.items():
            dft_settings["incar"][setting] = val

    builder = mdb_aut.submit_aiida_vasp_calculation(
        index=calc_idx,
        target_structure=struct,
        phase=curr_phase,
        material_name=curr_material_name,
        unique_id=curr_unique_id,
        kspacing_dict=dft_settings["kspacing"],
        calc_type=struct_type,
        queue_dict=dft_settings["queue"],
        potential_family=dft_settings["potential_family"],
        potential_mapping=potential_mapping,
        return_builder=True,
        dry_run=False,
        incar_settings_dict=dft_settings["incar"],
        group=group,
    )
    return builder


def get_dft_calc_builder_mace_list(
    struct_list: list,
    row,
    dft_settings: dict,
):
    """Get a MACE calculation builder for a given structure list and row."""
    updated_struct_list = []

    for idx, curr_struct in enumerate(struct_list):
        curr_struct = struct_list[idx]

        # Gathering row information
        (curr_structure, curr_material_name, curr_unique_id, curr_phase) = (
            mdb_aut.gather_calc_data_from_row(row, curr_structure=curr_struct)
        )
        struct_ase = AseAtomsAdaptor().get_atoms(curr_struct)
        struct_ase.info["mdb_md_node"] = row["mdb_md_node"]
        updated_struct_list.append(struct_ase)

    # Write xyz file into a string captured in the stdout,
    # write it to a temporary file.
    mace_xyz_file = gen_xyz_file_from_traj(updated_struct_list)

    # Prepare GetMACEDescriptorsCalculation
    # Generate builder
    mace_descr_calc = CalculationFactory("mace-eval")
    mace_builder = mace_descr_calc.get_builder()

    mace_builder.mace_settings_dict = dft_settings["settings"]

    # Load model from absolute path
    mace_model_path = Path(dft_settings["mace_potential_path"]).absolute()
    model = orm.SinglefileData(file=mace_model_path)
    mace_builder.model_file = model

    # Load structure as orm.SinglefileData
    mace_builder.configuration_to_evaluate = mace_xyz_file

    # Get code and remove from settings dict
    mace_builder.code = orm.load_code(dft_settings["options"]["code_string"])
    dft_settings["options"].pop("code_string")

    # Load scheduler and resources options
    mace_builder.metadata.options = dft_settings["options"]

    struct_name = curr_material_name
    mace_builder.metadata.label = struct_name

    return mace_builder


def gen_xyz_file_from_traj(struct_list):
    """Generate a temporary xyz file from a list of structures."""
    f = io.StringIO()
    with redirect_stdout(f):
        ase_write(
            filename="-",
            format="extxyz",
            images=struct_list,
        )
    xyz_string = f.getvalue()

    # Generating tmp file
    mace_xyz_file = orm.SinglefileData(
        file=io.BytesIO(str.encode(xyz_string)),
        filename="mace_structures.xyz",
    )

    return mace_xyz_file


def generate_model_name():
    """
    Generate a unique NNP model name combining random words and a number.

    This function creates a unique model name by concatenating randomly selected
    adjective, noun, and verb, followed by a random number. This combination ensures
    the generation of distinctive and memorable names suitable for labeling models
    in simulations or learning tasks.

    Returns
    -------
    str
        A string consisting of a random adjective, noun, and verb followed by a hyphen
        and a random number between 1 and 99, forming a unique model name.
    """
    r = ww.RandomWord()
    randint = np.random.randint(low=1, high=10000)

    adj = r.word(include_parts_of_speech=["adjective"])
    noun = r.word(include_parts_of_speech=["noun"])
    verb = r.word(include_parts_of_speech=["verb"])
    model_name = slugify.slugify(f"{adj}-{noun}-{verb}-{randint}".replace(" ", "_"))

    return model_name


def get_final_db_path(result_dir_path, final_db_name, node):
    """Get the path to the final database file."""
    result_dir_path = Path(result_dir_path)
    caller_uuid = process_call_root(node) if not isinstance(node, str) else node
    curr_run_dir: Path = result_dir_path / f"run_{caller_uuid}"

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()

    # Adding the final database path and the 'mdb_train_db_' prefix
    # used to identifd the final database.
    final_db_path = curr_run_dir / f"mdb_train_db_{final_db_name}.xyz"
    return final_db_path, curr_run_dir


def get_results_dir_path(result_dir_path, node, check_temp_dir=True):
    """Get the path to the results directory."""
    result_dir_path = Path(result_dir_path)

    caller_uuid = process_call_root(node) if not isinstance(node, str) else node
    curr_run_dir: Path = result_dir_path / f"run_{caller_uuid}"

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()
    if check_temp_dir and not (curr_run_dir / "run_tmp_data").exists():
        (curr_run_dir / "run_tmp_data").mkdir()

    return curr_run_dir


def process_call_root(process):
    """Show root process of the call stack for the given process."""
    caller = process.caller

    if caller is None:
        return process.uuid

    while True:
        next_caller = caller.caller
        if next_caller is None:
            break
        caller = next_caller

    return caller.uuid


@calcfunction
def prepare_output_dataframe(md_seed_results_df):
    """Prepare the output dataframe for the active learning workflow."""
    md_seed_results_df.index = md_seed_results_df.index.map(str)
    training_df = orm.Dict(md_seed_results_df.to_dict(orient="index"))
    return training_df


@calcfunction
def update_mace_train_settings_dict(
    settings_dict: dict,
    train_data_path: str,
    curr_model: str,
    curr_iter: int,
    db_size: int,
):
    """Update the MACE training settings dictionary with the new database path."""
    if isinstance(settings_dict, orm.Dict):
        settings_dict: dict = settings_dict.get_dict()

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, orm.Str):
        train_data_path: Path = Path(train_data_path.value)
    elif isinstance(train_data_path, str):
        train_data_path: Path = Path(train_data_path)

    settings_dict["train_file"] = str(train_data_path.name)

    # Updating name to include model and iteration number
    curr_name = settings_dict["name"]

    # For very small datasets (testing), the batch size must be lower than the
    # database size
    if db_size < settings_dict.get("batch_size", 0):
        settings_dict["batch_size"] = db_size // 2

    if isinstance(curr_model, orm.Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, orm.Int):
        curr_iter = curr_iter.value

    settings_dict["name"] = (
        str(curr_model) + "_" + curr_name + "_al-iteration_" + str(curr_iter)
    )

    return orm.Dict(settings_dict)


@calcfunction
def create_mace_lammps_model(model_file: orm.SinglefileData):
    """
    Create a LAMMPS potential from a MACE model.

    Parameters
    ----------
    model_file : orm.SinglefileData
        A MACE model file to convert to a LAMMPS potential.

    Returns
    -------
    orm.SinglefileData
        A LAMMPS potential file generated from the MACE model.
    """
    with model_file.as_path() as model_path:
        # Loading model
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model = model.double().to("cpu")
        lammps_model = LAMMPS_MACE(model)
        lammps_model_compiled = jit.compile(lammps_model)

        # Creating new path
        new_model_path = str(model_path) + "-lammps.pt"

        # Saving LAMMPS model
        lammps_model_compiled.save(new_model_path)

        return orm.SinglefileData(file=new_model_path)


@calcfunction
def check_atom_in_domain(
    concave_hull: orm.ArrayData, descriptors: orm.ArrayData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    concave_hull = concave_hull.get_array()
    descriptors = descriptors.get_array()
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    return orm.Dict(
        {"inside": point_inside, "outside": point_outside, "all": all_points_in_out}
    )


@calcfunction
def plot_concave_hull(
    concave_hull: np.ndarray,
    point_inside: np.ndarray,
    point_outside: np.ndarray,
    latent_space: np.ndarray,
    filename: str = "concave_hull.png",
):
    # Getting arrays from ArrayData objects
    concave_hull = concave_hull.get_array()
    latent_space = latent_space.get_array()
    point_inside = point_inside.get_array()
    point_outside = point_outside.get_array()

    # Plotting the concave hull in 2D space using lines
    plt.plot(concave_hull[:, 0], concave_hull[:, 1], "r-")
    plt.plot(
        latent_space[:, 0],
        latent_space[:, 1],
        "o",
        markersize=2,
        alpha=0.5,
        label="Descriptor in database",
        markeredgewidth=0,
        color="#b16286",
    )
    plt.plot(
        point_inside[:, 0],
        point_inside[:, 1],
        "s",
        label="structure in domain",
        color="#8ec07c",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor="#282828",
    )
    plt.plot(
        point_outside[:, 0],
        point_outside[:, 1],
        "s",
        label="structure out of domain",
        color="#fb4934",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor="#282828",
    )
    plt.title("Concave Hull")
    plt.xlabel("x")
    plt.legend()

    # Create tmp file
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        plt.savefig(f.name, dpi=300)
        plt.close()
        return {"plot": orm.SinglefileData(file=f.name, filename=filename)}


def aiida_serialized_ase_dict_to_atoms(struct_dict: dict) -> Atoms:
    """Convert a serialized Atoms dictionary to an Atoms object."""
    struct_dict["pbc"] = np.array([bool(boo) for boo in struct_dict["pbc"]])

    for key, val in struct_dict.items():
        if key != "pbc" and isinstance(val, list):
            struct_dict[key] = np.array(val)

    if "info" in struct_dict:
        for key, val in struct_dict["info"].items():
            if key != "pbc" and isinstance(val, list):
                struct_dict["info"][key] = np.array(val)

    return Atoms.fromdict(struct_dict)


def serialize_ase(curr_s: dict | Atoms) -> dict:
    """Serialize an ASE Atoms object to a dictionary."""
    if not isinstance(curr_s, dict):
        curr_s = curr_s.todict()

    curr_s["pbc"] = [bool(boo) for boo in curr_s["pbc"]]

    for key, val in curr_s.items():
        if key != "forces" and isinstance(val, np.ndarray):
            curr_s[key] = list(val)

    return curr_s


@calcfunction
def prepare_output_final_training_db(training_db_path):
    """Convert the training database to a orm.SinglefileData object."""
    train_db = orm.SinglefileData(file=training_db_path.value)
    return train_db


@calcfunction
def gather_dft_calcs_vasp(dft_calc_list: list) -> orm.List:
    """
    Collect and preprocess VASP DFT calculation results for active learning input.

    This function takes a list of DFT calculation nodes, extracts the calculation
    results, and processes these results into a format suitable for active learning
    input. Specifically, it converts VASP runs into ASE Atoms objects and collects
    additional calculation data like forces. It also augments the Atoms objects with
    metadata necessary for the active learning workflow. Failed calculations are skipped
    ensuring that only successfully completed CalcJobs are included.
    The function returns a list of serialized ASE Atoms objects, ready for inclusion
    in the active learning database.

    Parameters
    ----------
    dft_calc_list : list
        A list of identifiers for completed DFT calculation nodes.

    Returns
    -------
    orm.List
        An AiiDA orm.List object containing serialized ASE Atoms objects,
        each representing a completed DFT calculation augmented with necessary
        metadata and calculation results.

    Notes
    -----
    - The ASE Atoms objects are serialized to ensure compatibility with AiiDA's data
    storage and manipulation frameworks.
    - Extra care is taken to include forces (and optionally, stress) in the Atoms
    objects, as these are critical for many active learning applications but
    are not included by default in the extxyz format's `Properties` tag.
    - Skips any DFT calculations that encountered errors.
    """
    vasprun_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        finished_dft_calc = orm.load_node(finished_dft_calc)

        try:
            # Gathering the vasprun as an ASE Atoms object. This object won't
            # collect automatically all of the extra information such as forces
            # or energies, and must be collected using methods from ase.calc.u
            vasprun: Atoms = mdb_conv._gather_mace_req_calc_data_from_node(
                finished_dft_calc
            )

        except Exception:
            # If the calculation fails for any reason, skip this calculation
            continue

        # Gathering extra DFT calculation information from vasprun.xml
        calc_info_dict = mdb_conv.gather_calc_data_from_node(
            finished_dft_calc, units="mace"
        )

        # Adding forces manually as an array into the atoms object.
        # This is needed for the atoms object to be able to include the forces in the
        # extxyz format `Properties` tag.
        if "forces" not in vasprun.arrays:
            vasprun.new_array(
                name="forces",
                a=np.array(calc_info_dict["forces"]),
            )

        # if "stress" not in vasprun.arrays.keys():
        #     vasprun.new_array(
        #         name="stress",
        #         a=np.array(calc_info_dict["stress"]),
        #     )

        # Adding the type of structure to the atoms.info dict
        struct_type = mdb_conv.get_struct_type(
            vasprun=vasprun, dft_calc_node=finished_dft_calc
        )
        calc_info_dict["mdb_struct_type"] = struct_type
        vasprun = vasprun_add_info_dict(vasprun, calc_info_dict)

        # Generate a structure name and gathering the aiida_uuid
        vasprun: Atoms = mdb_conv._add_entry_to_mace_input(
            vasprun=vasprun,
            node=finished_dft_calc,
            to_file=False,
            remove_dipole=True,
            remove_stress=False,
        )

        vasprun: dict = serialize_ase(vasprun)
        vasprun_list.append(vasprun)

    return_list = orm.List([val for val in vasprun_list])
    return return_list


@calcfunction
def gather_dft_calcs_mace(
    dft_calc_list: list, results_dir: str, workchain=None
) -> orm.Str:
    """Collect and preprocess MACE DFT calculation results for active learning input."""
    result_list = []
    outlier_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        calc_node = orm.load_node(finished_dft_calc)

        try:
            # Gathering the calculation data from a extxyz stored
            # as a orm.SinglefileData.
            struct_file: orm.SinglefileData = (
                calc_node.outputs.configuration_result_file
            )
            with struct_file.as_path() as struct_file_path:
                result_structures = ase_read(
                    struct_file_path, format="extxyz", index=":"
                )

        # If parsing the calculation fails for any reason, skip it.
        except Exception:
            continue

        curr_struct_res = []
        for structure in result_structures:
            # Gathering extra DFT calculation information from calculation
            # and its extras
            calc_info_dict = {
                "struct_name": calc_node.label,
                "dft_calc_uuid": calc_node.uuid,
                "aiida_uuid": calc_node.base.extras.all["mdb_calc_uuid"],
                "mdb_struct_type": calc_node.base.extras.all["mdb_struct_type"],
                # "mdb_md_node": calc_node.uuid,
            }

            for key, val in calc_info_dict.items():
                structure.info[key] = val

            # Renaming energy key
            structure.info["REF_energy"] = structure.info.pop("mdb_mace_eval_energy")

            # Renaming forces dict
            structure.arrays["REF_forces"] = structure.arrays.pop(
                "mdb_mace_eval_forces"
            )

            # result_list.append(structure)
            curr_struct_res.append(structure)

        # Some calculations that reported high E and F thoughout all
        # steps. Checking for outliers using the bond distance
        for struct in curr_struct_res:
            outlier_flag = False
            min_dists = []

            symbols = set(struct.get_chemical_symbols())

            struct_ana = Analysis(struct)
            struct_ana.clear_cache()

            # Checking all bond distances for all bond types
            for at_A, at_B in it.combinations_with_replacement(symbols, 2):
                # Getting covalent radii from ase.data
                R_atA = covalent_radii[atomic_numbers[at_A]]
                R_atB = covalent_radii[atomic_numbers[at_B]]

                try:
                    # Getting bond distances. This will consider pbc.
                    vals = struct_ana.get_values(
                        struct_ana.get_bonds(at_A, at_B, unique=True)
                    )
                except IndexError:
                    # If there is only one atom of a type (X), there are no
                    # distances for atoms of this type (X-X), and therefore
                    # the bond distance calculation must be skipped for the
                    # current atom pair (X-X).
                    continue

                min_bond_length = np.min(vals[0])
                min_dists.append(min_bond_length)

                # Structures with bond distances below this value will be
                # considered outliers.
                bond_length_threshold = (R_atA + R_atB) * 0.7

                # Check for any outliers below the minimum bond length threshold
                # and add the structure to the outlier list, which will not be
                # part of the final database.
                if np.any(np.array(min_dists) < bond_length_threshold):
                    outlier_list.append(struct)
                    outlier_flag = True
                    break

            if not outlier_flag:
                result_list.append(struct)

    if workchain:
        node = orm.load_node(workchain.value)
        node.logger.log(
            level=LOG_LEVEL_REPORT,
            msg=f"[{node.pk}|{node.process_label}|gather_dft_calcs_mace]:"
            f" Removed {len(outlier_list)} outliers.",
        )

    # Write the results to a temporary file in the calculation directory
    if isinstance(results_dir, orm.Str):
        results_dir = Path(results_dir.value)
    elif isinstance(results_dir, str):
        results_dir = Path(results_dir)

    results_file_path = results_dir / "run_tmp_data" / "gathered_dft_calcs.xyz"
    ase_write(filename=results_file_path, images=result_list)

    # DEBUG: Remove after checking outliers
    if outlier_list:
        outliers_file_path = results_dir / "outliers.extxyz"
        ase_write(filename=outliers_file_path, images=outlier_list)

    # Return the path to the temporary file
    return orm.Str(str(results_file_path))


def iqr_outlier_check(res_list: list) -> np.ndarray:
    """
    Identifies outliers in a list of E/F values using the interquartile range (IQR).

    Parameters
    ----------
    res_list : list
        A list of numerical values to check for outliers.

    Returns
    -------
    np.ndarray
        An array where outliers are replaced with NaN and non-outliers are retained.

    Notes
    -----
    The function calculates the 30th and 70th percentiles of the input list
    to determine the interquartile range (IQR).
    Values outside the range [Q1 - 1.5 * IQR, Q2 + 1.5 * IQR] are considered
    outliers and replaced with NaN.
    """
    q1 = np.percentile(res_list, 30)
    q2 = np.percentile(res_list, 70)
    iqr = q2 - q1

    # Define outlier thresholds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q2 + 1.5 * iqr

    # Identify and remove outliers
    filtered_data = np.array(
        [x if lower_bound <= x <= upper_bound else np.nan for x in res_list]
    )

    return filtered_data


def vasprun_add_info_dict(vasprun_dict: dict, calc_info_dict: dict) -> dict:
    """Add calculation information to the vasprun dictionary."""
    info_list = [
        "stress",
        "dipole",
        "forces",
        "struct_name",
        "energy",
        "aiida_uuid",
        "free_energy",
        "mdb_struct_type",
    ]

    # If forces already in the arrays dictionary, it is not needed in
    # atoms.info
    if "forces" in vasprun_dict.arrays:
        calc_info_dict.pop("forces")

    if not isinstance(vasprun_dict, dict):
        vasprun_dict = Atoms.todict(vasprun_dict)

    if not vasprun_dict.get("info"):
        vasprun_dict["info"] = {}

    for key, val in calc_info_dict.items():
        if key not in vasprun_dict["info"] and key in info_list:
            if key == "free_energy":
                key.replace("free_", "")

            vasprun_dict["info"][key] = val
    return vasprun_dict


@calcfunction
def remove_structs_from_seed_gen_db(
    seed_gen_path: orm.Str, delete_indices: list
) -> orm.List:
    """
    Remove specified structures from a seed generation database based on UUIDs.

    This function iterates over a list of UUIDs (delete_indices) and removes the
    corresponding structures from a seed generation database. The database is accessed
    via the `seed_gen_db` object, which is loaded from the `seed_gen_path` using ASE.
    Each element of the list is an ase.Atoms object with an unique identifier
    (`aiida_uuid`) in the info attribute.
    The function writes the modified list back into `seed_gen_path` after the specified
    ones have been removed. No list is returned into the workchain to avoid having
    to serialize the atoms list.

    Parameters
    ----------
    seed_gen : orm.Str | str
        The path to the seed generation database.
    delete_indices : list
        A list of UUIDs (strings) identifying the structures to be removed
        from the seed generation database.

    """
    if isinstance(seed_gen_path, str):
        seed_gen_db = load_database(seed_gen_path)
    elif isinstance(seed_gen_path, orm.Str):
        seed_gen_db = load_database(seed_gen_path.value)

    if not isinstance(seed_gen_db, list):
        seed_gen_db = seed_gen_path.get_list()

    for curr_uuid in delete_indices:
        for del_idx, struct in enumerate(seed_gen_db):
            struct: Atoms = struct.todict()
            struct_uuid = struct["info"]["aiida_uuid"]
            if curr_uuid == struct_uuid:
                del seed_gen_db[del_idx]

    ase_write(
        filename=seed_gen_path.value,
        images=seed_gen_db,
        format="extxyz",
    )


@calcfunction
def check_md_seed_agreement(return_list_path: str) -> orm.Bool:
    """
    Check if all predictions agree for current seed.

    Parameters
    ----------
    return_list : list
        orm.List containing all calculations for predictions where the
        models disagreed.

    Returns
    -------
    orm.Bool
        True if all the predictions have agreed for the current MD seed
        on the current AL iteration. False if there is no agreement on
        on all structures.
    """
    if len(return_list_path.value) > 0:
        return orm.Bool(False)
    else:
        return orm.Bool(True)
