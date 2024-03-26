import json
import math as m
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
    load_node,
)
from ase import Atoms
from e3nn.util import jit
from mace.calculators import LAMMPS_MACE
from MatDBForge.training import conversion as mdb_conv
from MatDBForge.workflows import aiida_utils as mdb_aut


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
    energies_model_list = model_res_dict_to_arr(energies_dict)
    energies_std = energies_model_list.std(axis=0)

    return energies_std


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

    print('steps_E_F_arr_sampled: ', steps_E_F_arr_sampled.shape)
    return traj_sampled, steps_E_F_arr_sampled, forces_sampled


def get_dft_calc_builder(struct, row, calc_idx, group):
    struct_type = row["mdb_struct_type"]

    # Gathering row information
    (
        curr_structure,
        curr_material_name,
        curr_unique_id,
        curr_phase,
    ) = mdb_aut.gather_calc_data_from_row(row, curr_structure=struct)

    # HACK
    # TODO: Move this to the central TOML file.
    kspacing_dict = {
        "alpha": 0.135088484104361,
        # "m1": 0.100530964914873,
        "beta-prime": 0.102415920507027,
        # "m2": 0.100530964914873,
        "gamma": 0.141371669411541,
        # "m3": 0.166504410640259,
        "epsilon": 0.105557513160617,
        "eta": 0.0993371597065093,
        # "m4": 0.0948760981384118,
        "delta": 0.0994491889005363,
    }

    # REMOVE # TESTING
    # vasp-std-5.4.4-new@tekla2-new-test
    # vasp-std-5.3.3-new@tekla2-updated-2024
    # ################
    # HACK
    # TODO: Move this to the central TOML file.
    queue_dict = {
        2: {
            "type": "sge",
            "node_cpus": 48,
            "code_string": "vasp-std-5.4.4-new@tekla2-new-test",
            "options_resources": {
                "parallel_env": "c48m256ib_mpi",
                "tot_num_mpiprocs": 48,
            },
            "multiple": 1,
        },
        5: {
            "type": "sge",
            "node_cpus": 48,
            "code_string": "vasp-std-5.4.4-new@tekla2-new-test",
            "options_resources": {
                "parallel_env": "c48m256ib_mpi",
                "tot_num_mpiprocs": 48,
            },
            "multiple": 1,
        },
        40: {
            "type": "sge",
            "node_cpus": 48,
            "code_string": "vasp-std-5.4.4-new@tekla2-new-test",
            "options_resources": {
                "parallel_env": "c48m256ib_mpi",
                "tot_num_mpiprocs": 48,
            },
            "multiple": 1,
        },
    }

    # TESTING # potential_family = "vasp-5.3-PBE"
    # HACK
    # TODO: Move this to the central TOML file.
    potential_family = "vasp-5.4-PBE-2023"
    potential_mapping = mdb_aut.generate_potential_mapping()

    builder = mdb_aut.submit_aiida_calculation(
        index=calc_idx,
        target_structure=struct,
        phase=curr_phase,
        material_name=curr_material_name,
        unique_id=curr_unique_id,
        kspacing_dict=kspacing_dict,
        calc_type=struct_type,
        queue_dict=queue_dict,
        potential_family=potential_family,
        potential_mapping=potential_mapping,
        return_builder=True,
        dry_run=False,
        incar_dict=None,
        group=group,
    )
    return builder


def identify_struct_type(struct):
    ...


def generate_model_name():
    r = ww.RandomWord()
    randint = np.random.randint(low=1, high=100)
    adj = r.word(include_parts_of_speech=["adjective"])
    noun = r.word(include_parts_of_speech=["noun"])
    verb = r.word(include_parts_of_speech=["verb"])
    model_name = f"{adj}_{noun}_{verb}-{randint}"
    return model_name


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
def generate_placeholder_text():
    return Str("placeholder text")


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

    with open(settings_path, "r") as f:
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
def create_mace_lammps_model(model_file):
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

    if "info" in struct_dict.keys():
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
def prepare_output_final_training_db(training_db_list):
    # Converting training_db to aiida types
    struct_list = []
    for ase_struct in training_db_list:
        struct_list.append(ase_struct)

    return List(struct_list)


@calcfunction
def gather_dft_calcs(dft_calc_list: list) -> List:
    vasprun_list = []
    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        finished_dft_calc = load_node(finished_dft_calc)

        # Gathering the vasprun as an ASE Atoms object. This object won't
        # collect automatically all of the extra information such as forces
        # or energies, and must be collected using methods from ase.calc.u
        vasprun: Atoms = mdb_conv._gather_mace_req_calc_data_from_node(
            finished_dft_calc
        )

        # Gathering extra DFT calculation information from vasprun.xml
        calc_info_dict = mdb_conv.gather_calc_data_from_node(
            finished_dft_calc, units="mace"
        )

        # Adding forces manually as an array into the atoms object.
        # This is needed for the atoms object to be able to include the forces in the
        # extxyz format `Properties` tag.
        if "forces" not in vasprun.arrays.keys():
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
        struct_type = mdb_conv.get_struct_type(vasprun)
        calc_info_dict["mdb_struct_type"] = struct_type
        vasprun = vasprun_add_info_dict(vasprun, calc_info_dict)

        # Using this function to generate a structure name and gathering the aiida_uuid.
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
        if key not in vasprun_dict["info"].keys() and key in info_list:
            vasprun_dict["info"][key] = val
    return vasprun_dict


@calcfunction
def remove_structs_from_seed_gen_db(seed_gen: List, delete_indices: list) -> List:
    """
    Remove specified structures from a seed generation database based on UUIDs.

    This function iterates over a list of UUIDs (delete_indices) and removes the
    corresponding structures from a seed generation database. The database is accessed
    and modified via the `seed_gen` object, which is converted to a list.
    Each element of the list is an ase.Atoms object with an unique identifier
    (`aiida_uuid`) in the info attribute.
    The function returns the modified list of structures after the specified ones
    have been removed.

    Parameters
    ----------
    seed_gen : List
        An object that contains the seed generation database.
    delete_indices : list
        A list of UUIDs (strings) identifying the structures to be removed
        from the seed generation database.

    Returns
    -------
    List
        An aiida List of the remaining structures in the seed generation database
        after the specified structures have been removed.
    """
    seed_gen_db = seed_gen.get_list()
    for curr_uuid in delete_indices:
        for del_idx, struct in enumerate(seed_gen_db):
            struct_uuid = struct["info"]["aiida_uuid"]
            if curr_uuid == struct_uuid:
                del seed_gen_db[del_idx]

    return List(seed_gen_db)


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
