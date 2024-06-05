import io
import json
import math as m
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import wonderwords as ww
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
    StructureData,
    load_code,
    load_node,
)
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from e3nn.util import jit
from mace.calculators import LAMMPS_MACE
from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.workflows import aiida_utils as mdb_aut
from pymatgen.io.ase import AseAtomsAdaptor


def model_res_dict_to_arr(res_dict):
    res_model_list = []
    for _, res in res_dict.items():
        res_model_list.append(res)
    res_model_list = np.array(res_model_list)
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
    forces_std = forces_model_list.std(axis=0)

    return forces_std


def get_model_energies_std(energies_dict):
    energies_model_list: np.ndarray = model_res_dict_to_arr(energies_dict)
    energies_std = energies_model_list.std(axis=0)

    return energies_std


def load_database(path: str):
    database = ase_read(
        filename=path,
        format="extxyz",
        index=":",
    )
    return database


# @calcfunction
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


def select_md_frames_to_keep(
    frame_interval: int,
    total_n_frames: int,
    md_tstep_duration_ps: float,
    traj,
    steps_E_F_arr: np.array,
    forces: np.array,
):
    # Get total MD time in picoseconds
    total_duration_ps = total_n_frames * md_tstep_duration_ps

    # Get total number of frames in that time
    total_num_frames = m.ceil(total_duration_ps * frame_interval)

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


def get_dft_calc_builder_mace(
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

    # Prepare GetMACEDescriptorsCalculation
    # Generate builder
    mace_descr_calc = CalculationFactory("mace-eval")
    mace_builder = mace_descr_calc.get_builder()

    mace_builder.mace_settings_dict = dft_settings["settings"]

    # Load model
    model = SinglefileData(dft_settings["mace_potential_path"])
    mace_builder.model_file = model

    # Load structure as StructureData
    mace_builder.configuration_to_evaluate = StructureData(pymatgen=curr_structure)

    # Get code and remove from settings dict
    mace_builder.code = load_code(dft_settings["options"]["code_string"])
    dft_settings["options"].pop("code_string")

    # Load scheduler and resources options
    mace_builder.metadata.options = dft_settings["options"]

    # Generating label for the CalcJob
    struct_formula = curr_structure.formula.replace(" ", "")
    struct_name = f"{curr_material_name}-{struct_formula}-{calc_idx}_{struct_type}"
    mace_builder.metadata.label = struct_name

    return mace_builder


def get_dft_calc_builder_mace_list(
    struct_list: list,
    row: int,
    db_row_idx,
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

    # Generating label for the CalcJob
    struct_formula = curr_structure.formula.replace(" ", "")
    struct_name = f"{curr_material_name}-{struct_formula}-{db_row_idx}_{struct_type}"
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
    model_name = f"{adj}_{noun}_{verb}-{randint}"
    return model_name


def get_final_db_path(result_dir_path, final_db_name, node):
    result_dir_path = Path(result_dir_path)
    if not isinstance(node, str):
        caller_uuid = process_call_root(node)
    else:
        caller_uuid = node

    curr_run_dir: Path = result_dir_path / f"run_{caller_uuid}"

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()

    final_db_path = curr_run_dir / f"{final_db_name}.xyz"
    return final_db_path, curr_run_dir


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
    settings_dict: dict, train_data_path: str, curr_model: str, curr_iter: int
):
    if isinstance(settings_dict, Dict):
        settings_dict: Dict = settings_dict.get_dict()

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, Str):
        train_data_path: Path = Path(train_data_path.value)
    elif isinstance(train_data_path, str):
        train_data_path: Path = Path(train_data_path)

    settings_dict["train_file"] = str(train_data_path.name)

    # Updating name to include model and iteration number
    curr_name = settings_dict["name"]

    if isinstance(curr_model, Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, Int):
        curr_iter = curr_iter.value

    settings_dict["name"] = (
        str(curr_model) + "_" + curr_name + "_al-iteration_" + str(curr_iter)
    )

    return Dict(settings_dict)


@calcfunction
def create_mace_lammps_model(model_file, rmse_e, rmse_f):
    with model_file.as_path() as model_path:
        # Loading model
        model = torch.load(model_path)
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
def gather_dft_calcs_mace(dft_calc_list: list) -> List:
    """Collect and preprocess MACE DFT calculation results for active learning input."""
    result_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        calc_node = load_node(finished_dft_calc)

        try:
            # Gathering the calculation data as a list of ASE Atoms dicts.
            # This object won't collect automatically all of the extra information
            # such as forces or energies, and must be collected using methods
            # from ase.calc.
            struct_serial_dict_list = calc_node.outputs.configuration_result_list

        except Exception:
            # If the calculation fails for any reason, skip it.
            continue

        for struct_serial_dict in struct_serial_dict_list:
            # Gathering extra DFT calculation information from calculation
            # and its extras
            calc_info_dict = {
                "struct_name": calc_node.label + "_aiida-uuid_" + calc_node.uuid,
                "aiida_uuid": calc_node.extras["mdb_calc_uuid"],
                "mdb_struct_type": calc_node.extras["mdb_struct_type"],
            }

            for key, val in calc_info_dict.items():
                struct_serial_dict["info"][key] = val

            # Renaming energy key
            struct_serial_dict["info"]["energy"] = struct_serial_dict["info"].pop(
                "mdb_mace_eval_energy"
            )

            # Renaming forces dict
            struct_serial_dict["forces"] = struct_serial_dict.pop(
                "mdb_mace_eval_forces"
            )

            # struct = aiida_serialized_ase_dict_to_atoms(struct_serial_dict)
            result_list.append(struct_serial_dict)

    return_list = List([val for val in result_list])
    return return_list


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
def check_md_seed_agreement(return_list: list) -> Bool:
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
    if len(return_list) > 0:
        return Bool(False)
    else:
        return Bool(True)
