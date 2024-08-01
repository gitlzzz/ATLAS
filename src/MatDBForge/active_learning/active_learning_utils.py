import io
import itertools as it
import json
import math as m
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import wonderwords as ww
from aiida.common.log import LOG_LEVEL_REPORT
from aiida.engine import (
    calcfunction,
)
from aiida.orm import (
    Bool,
    Dict,
    Int,
    List,
    SinglefileData,
    Str,
    load_code,
    load_node,
)
from aiida.plugins import CalculationFactory
from ase import Atoms, geometry
from ase.data import atomic_numbers, covalent_radii
from ase.geometry.analysis import Analysis
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, natural_cutoffs
from e3nn.util import jit
from mace.calculators import LAMMPS_MACE
from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.workflows import aiida_utils as mdb_aut
from pymatgen.core import Structure as pmg_struct
from pymatgen.io.ase import AseAtomsAdaptor


def model_res_dict_to_arr(res_dict):
    res_model_list = []

    for _, res in res_dict.items():
        res_model_list.append(res)
    res_model_list = np.array(res_model_list, dtype=float)

    return res_model_list


def get_model_forces_variance(forces_dict):
    forces_model_list = model_res_dict_to_arr(forces_dict)
    forces_var = forces_model_list.var(axis=0)

    return forces_var


def get_model_energies_variance(energies_dict):
    energies_model_list = model_res_dict_to_arr(energies_dict)
    energies_var = energies_model_list.var(axis=0)

    return energies_var


def get_model_forces_std(forces_dict):
    forces_model_list = model_res_dict_to_arr(forces_dict)

    # Calculate the sample standard deviation of the energies
    # for each structure
    forces_std = np.nanstd(forces_model_list, axis=0, ddof=1)

    return forces_std


def get_model_energies_std(energies_dict):
    energies_model_list: np.ndarray = model_res_dict_to_arr(energies_dict)

    # Calculate the sample standard deviation of the energies
    # for each structure
    energies_std = np.nanstd(energies_model_list, axis=0, ddof=1)

    return energies_std


def load_database(path: str):
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
    frame_interval : Int
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
        Structure to check.
    max_layer_distance_ang : float
        Maximum distance between layers

    Returns
    -------
    bool
        Returns `True` if the layer distace is above max_layer_distance_ang, `False` if otherwise.
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
        Structure to check.

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


def select_md_frames_to_keep(
    frame_interval: int,
    # total_n_frames: int,
    md_tstep_duration_ps: float,
    traj,
    steps_E_F_arr: np.array,
    forces: np.array,
):
    # Get total MD time in picoseconds
    total_duration_ps = len(traj) * md_tstep_duration_ps

    # Get total number of frames in that time.
    # Frame interval represents every how many ps of MD simulation
    # save a frame.
    total_num_frames = m.ceil(total_duration_ps * 1 / frame_interval)

    # Choose the number of frames evenly and create a mask for the arrays
    mask = np.linspace(0, len(traj) - 1, total_num_frames, dtype=int)

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
    db_row_idx: int,
    group,
    dft_settings: dict,
):
    updated_struct_list = []
    struct_type = row["mdb_struct_type"]

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

    # Load model
    model = SinglefileData(dft_settings["mace_potential_path"])
    mace_builder.model_file = model

    # Load structure as SinglefileData
    mace_builder.configuration_to_evaluate = mace_xyz_file

    # Get code and remove from settings dict
    mace_builder.code = load_code(dft_settings["options"]["code_string"])
    dft_settings["options"].pop("code_string")

    # Load scheduler and resources options
    mace_builder.metadata.options = dft_settings["options"]

    # REMOVE
    # Generating label for the CalcJob
    # struct_formula = curr_structure.formula.replace(" ", "")
    # struct_name = f"{curr_material_name}-{struct_formula}-{db_row_idx}_{struct_type}"

    struct_name = curr_material_name
    mace_builder.metadata.label = struct_name

    return mace_builder


def gen_xyz_file_from_traj(struct_list):
    f = io.StringIO()
    with redirect_stdout(f):
        ase_write(
            filename="-",
            format="extxyz",
            images=struct_list,
        )
    xyz_string = f.getvalue()

    # Generating tmp file
    mace_xyz_file = SinglefileData(
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
    randint = np.random.randint(low=1, high=100)
    adj = r.word(include_parts_of_speech=["adjective"])
    noun = r.word(include_parts_of_speech=["noun"])
    verb = r.word(include_parts_of_speech=["verb"])
    model_name = f"{adj}_{noun}_{verb}-{randint}".replace(" ", "_")

    return model_name


def get_final_db_path(result_dir_path, final_db_name, node):
    result_dir_path = Path(result_dir_path)
    caller_uuid = process_call_root(node) if not isinstance(node, str) else node
    curr_run_dir: Path = result_dir_path / f"run_{caller_uuid}"

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()

    final_db_path = curr_run_dir / f"{final_db_name}.xyz"
    return final_db_path, curr_run_dir


def get_results_dir_path(result_dir_path, node, check_temp_dir=True):
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
    md_seed_results_df.index = md_seed_results_df.index.map(str)
    training_df = Dict(md_seed_results_df.to_dict(orient="index"))
    return training_df


@calcfunction
def load_mace_settings_json(
    settings_path: str, train_data_path: str, curr_model: str, curr_iter: int
):
    if isinstance(settings_path, Str):
        settings_path = settings_path.value

    with open(settings_path) as f:
        training_settings_dict = json.load(f)

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, Str):
        train_data_path: Path = Path(train_data_path.value)
    elif isinstance(train_data_path, str):
        train_data_path: Path = Path(train_data_path)

    training_settings_dict["train_file"] = str(train_data_path.name)

    # Updating name to include model and iteration number
    curr_name = training_settings_dict["name"]

    if isinstance(curr_model, Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, Int):
        curr_iter = curr_iter.value

    training_settings_dict["name"] = (
        str(curr_model) + "_" + curr_name + "_al-iteration_" + str(curr_iter)
    )

    return Dict(training_settings_dict)


@calcfunction
def update_mace_train_settings_dict(
    settings_dict: dict,
    train_data_path: str,
    curr_model: str,
    curr_iter: int,
    db_size: int,
):
    if isinstance(settings_dict, Dict):
        settings_dict: dict = settings_dict.get_dict()

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, Str):
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

    if isinstance(curr_model, Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, Int):
        curr_iter = curr_iter.value

    settings_dict["name"] = (
        str(curr_model) + "_" + curr_name + "_al-iteration_" + str(curr_iter)
    )

    return Dict(settings_dict)


@calcfunction
def create_mace_lammps_model(model_file: SinglefileData):
    """
    Create a LAMMPS potential from a MACE model.

    Parameters
    ----------
    model_file : SinglefileData
        A MACE model file to convert to a LAMMPS potential.

    Returns
    -------
    SinglefileData
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

        return SinglefileData(file=new_model_path)


def aiida_serialized_ase_dict_to_atoms(struct_dict: dict) -> Atoms:
    struct_dict["pbc"] = np.array([bool(boo) for boo in struct_dict["pbc"]])

    for key, val in struct_dict.items():
        if key != "pbc" and isinstance(val, list):
            struct_dict[key] = np.array(val)

    if "info" in struct_dict:
        for key, val in struct_dict["info"].items():
            if key != "pbc" and isinstance(val, list):
                struct_dict["info"][key] = np.array(val)

    return Atoms.fromdict(struct_dict)


def serialize_ase(curr_s) -> dict:
    if not isinstance(curr_s, dict):
        curr_s = curr_s.todict()

    curr_s["pbc"] = [bool(boo) for boo in curr_s["pbc"]]

    for key, val in curr_s.items():
        if key != "forces" and isinstance(val, np.ndarray):
            curr_s[key] = list(val)

    return curr_s


@calcfunction
def prepare_output_final_training_db(training_db_path):
    train_db = SinglefileData(file=training_db_path.value)
    return train_db


@calcfunction
def gather_dft_calcs_vasp(dft_calc_list: list) -> List:
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
    List
        An AiiDA List object containing serialized ASE Atoms objects, each representing
        a completed DFT calculation augmented with necessary metadata and calculation
        results.

    Notes
    -----
    - The ASE Atoms objects are serialized to ensure compatibility with AiiDA's data
    storage and manipulation frameworks.
    - Extra care is taken to include forces (and optionally, stress) in the Atoms objects,
    as these are critical for many active learning applications but are not included by
    default in the extxyz format's `Properties` tag.
    - Skips any DFT calculations that encountered errors.
    """
    vasprun_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        finished_dft_calc = load_node(finished_dft_calc)

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

    return_list = List([val for val in vasprun_list])
    return return_list


@calcfunction
def gather_dft_calcs_mace(dft_calc_list: list, results_dir: str, workchain=None) -> Str:
    """Collect and preprocess MACE DFT calculation results for active learning input."""
    result_list = []
    outlier_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        calc_node = load_node(finished_dft_calc)

        try:
            # Gathering the calculation data from a extxyz stored as a SinglefileData.
            struct_file: SinglefileData = calc_node.outputs.configuration_result_file
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
                "aiida_uuid": calc_node.extras["mdb_calc_uuid"],
                "mdb_struct_type": calc_node.extras["mdb_struct_type"],
                # "mdb_md_node": calc_node.uuid,
            }

            for key, val in calc_info_dict.items():
                structure.info[key] = val

            # Renaming energy key
            structure.info["energy"] = structure.info.pop("mdb_mace_eval_energy")

            # Renaming forces dict
            structure.arrays["forces"] = structure.arrays.pop("mdb_mace_eval_forces")

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
        node = load_node(workchain.value)
        node.logger.log(
            level=LOG_LEVEL_REPORT,
            msg=f"[{node.pk}|{node.process_label}|gather_dft_calcs_mace]:"
            f" Removed {len(outlier_list)} outliers.",
        )

    # Write the results to a temporary file in the calculation directory
    if isinstance(results_dir, Str):
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
    return Str(str(results_file_path))


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


def vasprun_add_info_dict(vasprun_dict, calc_info_dict):
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
    if "forces" in vasprun_dict.arrays.keys():
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
def remove_structs_from_seed_gen_db(seed_gen_path: Str, delete_indices: list) -> List:
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
    seed_gen : Str | str
        The path to the seed generation database.
    delete_indices : list
        A list of UUIDs (strings) identifying the structures to be removed
        from the seed generation database.

    """
    if isinstance(seed_gen_path, str):
        seed_gen_db = load_database(seed_gen_path)
    elif isinstance(seed_gen_path, Str):
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
def check_md_seed_agreement(return_list_path: str) -> Bool:
    """
    Check if all predictions agree for current seed.

    Parameters
    ----------
    return_list : list
        List containing all calculations for predictions where the
        models disagreed.

    Returns
    -------
    Bool
        True if all the predictions have agreed for the current MD seed
        on the current AL iteration. False if there is no agreement on
        on all structures.
    """
    if len(return_list_path.value) > 0:
        return Bool(False)
    else:
        return Bool(True)
